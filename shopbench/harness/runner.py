"""Run a benchmark configuration over the task set and write results.

    python -m shopbench.harness.runner --provider anthropic --model claude-sonnet-4-5 \
        --interface browser --obs axtree_pruned --out runs/sonnet_axpruned

    python -m shopbench.harness.runner --policy oracle --interface mcp --out runs/oracle_mcp

Outputs in --out:
  episodes.jsonl   one line per task: run record (trajectory, usage) + grade
  summary.json     aggregate metrics
  report.md        human-readable summary with the failure taxonomy
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..agents.agent import Agent, AgentConfig
from ..agents.llm import make_llm
from ..agents.scripted import ScriptedPolicy
from ..store.catalog import Catalog
from .grader import grade
from .metrics import render_report, summarize
from .server import StoreServer

TASKS_PATH = Path(__file__).resolve().parents[1] / "tasks" / "tasks.json"


def load_tasks(ids: list[str] | None = None, types: list[str] | None = None, limit: int | None = None,
               path: str | Path | None = None) -> list[dict]:
    tasks = json.loads(Path(path or TASKS_PATH).read_text())
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    if types:
        tasks = [t for t in tasks if t["type"] in types]
    return tasks[:limit] if limit else tasks


def run_config(make_policy, cfg: AgentConfig, tasks: list[dict], out: Path, workers: int = 1,
               server: StoreServer | None = None, label: str = "", verbose: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    catalog = Catalog()
    own = server is None
    server = server or StoreServer().__enter__()
    local = threading.local()
    envs: list = []
    lock = threading.Lock()
    results: list[dict] = []
    try:
        def env_for_thread():
            if cfg.interface != "browser":
                return None  # MCP spawns one server process per episode
            if not hasattr(local, "env"):
                from ..agents.browser_env import BrowserEnv
                local.env = BrowserEnv(server.base_url, cfg.obs_mode, cfg.max_obs_chars)
                with lock:
                    envs.append(local.env)
            return local.env

        def one(task: dict) -> dict:
            agent = Agent(make_policy(), cfg)
            rec = agent.run(task, server, env_for_thread())
            g = grade(task, server.snapshot(rec.episode_id), rec.to_dict(), catalog)
            row = {"grade": g, "run": rec.to_dict()}
            if verbose:
                mark = "PASS" if g["success"] else f"FAIL {g['failure']}"
                v = f" violations={g['violations']}" if g["violations"] else ""
                print(f"  [{label}] {task['id']:10s} {mark:32s} steps={rec.steps:2d} "
                      f"tok={rec.usage.input_tokens + rec.usage.cache_read_tokens + rec.usage.output_tokens:7d}{v}", flush=True)
            return row

        with open(out / "episodes.jsonl", "w") as f:
            if workers <= 1:
                for t in tasks:
                    row = one(t)
                    results.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                with ThreadPoolExecutor(workers) as ex:
                    futs = {ex.submit(one, t): t for t in tasks}
                    for fut in as_completed(futs):
                        row = fut.result()
                        results.append(row)
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        if own:
            server.__exit__(None, None, None)
    order = {t["id"]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: order[r["grade"]["task_id"]])
    summary = summarize(results)
    summary["config"] = {**cfg.__dict__, "label": label, "model": results[0]["run"]["config"]["model"] if results else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report.md").write_text(render_report(summary, results))
    return summary


def policy_factory(a) -> callable:
    if a.policy:
        return lambda: ScriptedPolicy(a.policy, a.interface)
    kw = dict(base_url=a.base_url, tool_mode=a.tool_mode, temperature=a.temperature, max_tokens=a.max_tokens)

    def make():
        llm = make_llm(a.provider, a.model, **kw)
        llm.price_in, llm.price_out = a.price_in, a.price_out
        return llm
    return make


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--policy", choices=["oracle", "oracle_noconfirm", "greedy"], help="scripted policy instead of an LLM")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--base-url", default=None, help="API base URL (e.g. http://localhost:8001/v1 for vLLM)")
    ap.add_argument("--tool-mode", default="native", choices=["native", "json"], help="openai provider only")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--price-in", type=float, default=None, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per million output tokens")
    ap.add_argument("--task-file", default=None, help="tasks JSON (default: the evaluation set; "
                    "use shopbench/tasks/tasks_train.json to collect fine-tuning data)")
    ap.add_argument("--tasks", nargs="*", help="task ids")
    ap.add_argument("--types", nargs="*", help="task types")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=40)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--interface", default="browser", choices=["browser", "mcp"])
    ap.add_argument("--obs", default="axtree_pruned", choices=["html", "axtree", "axtree_pruned"])
    ap.add_argument("--history", default="full", choices=["full", "last_k"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = AgentConfig(a.interface, a.obs, a.history, a.k, a.max_steps)
    tasks = load_tasks(a.tasks, a.types, a.limit, a.task_file)
    t0 = time.time()
    s = run_config(policy_factory(a), cfg, tasks, Path(a.out), a.workers, label=Path(a.out).name)
    print(f"\nsuccess {s['success_rate']:.1%}  outcome {s['outcome_rate']:.1%}  "
          f"violations/ep {s['violations_per_episode']:.2f}  mean steps {s['mean_steps']:.1f}  "
          f"cost ${s['total_cost_usd'] or 0:.2f}  ({time.time() - t0:.0f}s) -> {a.out}/report.md")


if __name__ == "__main__":
    main()
