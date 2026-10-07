"""Run the context-engineering and interface experiments and compare configurations.

    # Experiment 1+2+3: observation format, history window, browser vs MCP (one model)
    python -m shopbench.harness.experiments run --provider anthropic --model claude-sonnet-4-5 --out runs/exp_sonnet

    # Compare any finished run directories (e.g. base vs LoRA model)
    python -m shopbench.harness.experiments compare runs/qwen_base_mcp runs/qwen_lora_mcp --out runs/qwen_compare.md

Comparisons are paired (same tasks): differences in success are tested with an exact
McNemar test, which is the right test for two systems scored on the same items.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from ..agents.agent import AgentConfig
from .runner import add_common_args, load_tasks, policy_factory, run_config
from .server import StoreServer

GRID = {
    # name: (interface, obs_mode, history, k)
    "browser_html": ("browser", "html", "full", 3),
    "browser_axtree": ("browser", "axtree", "full", 3),
    "browser_axtree_pruned": ("browser", "axtree_pruned", "full", 3),
    "browser_axtree_pruned_last3": ("browser", "axtree_pruned", "last_k", 3),
    "browser_html_last3": ("browser", "html", "last_k", 3),
    "mcp": ("mcp", "axtree_pruned", "full", 3),
    "mcp_last3": ("mcp", "axtree_pruned", "last_k", 3),
}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value; b, c = discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load_run(d: Path) -> tuple[dict, dict[str, dict]]:
    s = json.loads((d / "summary.json").read_text())
    rows = {}
    for line in open(d / "episodes.jsonl"):
        r = json.loads(line)
        rows[r["grade"]["task_id"]] = r["grade"]
    return s, rows


def compare(dirs: list[Path], baseline: int = 0) -> str:
    runs = [(d.name, *load_run(d)) for d in dirs]
    L = ["# ShopBench comparison", "",
         "| Config | Success | 95% CI | Outcome | Safety-violation episodes | Mean steps | Input tok/task | Output tok/task | Cost | Cost/success |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s, _ in runs:
        cost = "n/a" if s["total_cost_usd"] is None else f"${s['total_cost_usd']:.2f}"
        cps = "n/a" if s["cost_per_success_usd"] is None else f"${s['cost_per_success_usd']:.3f}"
        L.append(f"| {name} | {s['success_rate']:.1%} | {s['success_ci95'][0]:.0%}–{s['success_ci95'][1]:.0%} | "
                 f"{s['outcome_rate']:.1%} | {s['safety_violation_rate']:.1%} | {s['mean_steps']:.1f} | "
                 f"{s['mean_input_tokens']:,.0f} | {s['mean_output_tokens']:,.0f} | {cost} | {cps} |")
    bname, _, brows = runs[baseline]
    L += ["", f"## Paired tests vs `{bname}` (exact McNemar on task success)", "",
          "| Config | Wins | Losses | Δ success | p |", "|---|---|---|---|---|"]
    for name, _, rows in runs:
        if name == bname:
            continue
        common = sorted(set(rows) & set(brows))
        wins = sum(1 for t in common if rows[t]["success"] and not brows[t]["success"])
        losses = sum(1 for t in common if brows[t]["success"] and not rows[t]["success"])
        L.append(f"| {name} | {wins} | {losses} | {(wins - losses) / max(1, len(common)):+.1%} | {mcnemar_exact(wins, losses):.3f} |")
    types = sorted({t for _, s, _ in runs for t in s["success_by_type"]})
    L += ["", "## Success by task type", "", "| Type | " + " | ".join(n for n, _, _ in runs) + " |",
          "|---|" + "---|" * len(runs)]
    for t in types:
        L.append(f"| {t} | " + " | ".join(f"{s['success_by_type'].get(t, 0):.0%}" for _, s, _ in runs) + " |")
    fails = sorted({f for _, s, _ in runs for f in s["failures"]})
    L += ["", "## Failure taxonomy (tasks per primary failure)", "", "| Failure | " + " | ".join(n for n, _, _ in runs) + " |",
          "|---|" + "---|" * len(runs)]
    for f in fails:
        L.append(f"| {f} | " + " | ".join(str(s["failures"].get(f, 0)) for _, s, _ in runs) + " |")
    viols = sorted({v for _, s, _ in runs for v in s["violations"]})
    if viols:
        L += ["", "## Safety violations (count)", "", "| Violation | " + " | ".join(n for n, _, _ in runs) + " |",
              "|---|" + "---|" * len(runs)]
        for v in viols:
            L.append(f"| {v} | " + " | ".join(str(s["violations"].get(v, 0)) for _, s, _ in runs) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    add_common_args(r)
    r.add_argument("--configs", nargs="*", default=list(GRID), choices=list(GRID))
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "compare":
        Path(a.out).write_text(compare([Path(d) for d in a.dirs]))
        print(Path(a.out).read_text())
        return
    tasks = load_tasks(a.tasks, a.types, a.limit, a.task_file)
    out = Path(a.out)
    dirs = []
    with StoreServer() as server:
        for name in a.configs:
            iface, obs, hist, k = GRID[name]
            cfg = AgentConfig(iface, obs, hist, k, a.max_steps)
            print(f"== {name}")
            a.interface = iface  # scripted policies need to know the interface
            run_config(policy_factory(a), cfg, tasks, out / name, a.workers, server=server, label=name)
            dirs.append(out / name)
    report = compare(dirs, baseline=0)
    (out / "comparison.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
