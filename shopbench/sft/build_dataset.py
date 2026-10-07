"""Turn successful benchmark episodes into a supervised fine-tuning dataset.

Each step of a successful trajectory becomes one example: the exact chat context the
agent saw at that step (rendered in the openai json tool-mode format, with the same
history strategy used at evaluation) and the action it took. Training on this format
means the fine-tuned model can be served by any OpenAI-compatible server (vLLM, Ollama,
llama.cpp) and evaluated with `--provider openai --tool-mode json`.

    python -m shopbench.sft.build_dataset runs/teacher_train_mcp --out data/sft_mcp.jsonl

Refuses episodes whose task ids appear in the evaluation set (contamination guard)
and skips episodes produced by privileged scripted policies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..agents.agent import apply_history
from ..agents.llm import OpenAICompatLLM

EVAL_IDS = {t["id"] for t in json.loads((Path(__file__).resolve().parents[1] / "tasks" / "tasks.json").read_text())}


def episode_examples(run: dict, history: str, k: int) -> list[dict]:
    fmt = OpenAICompatLLM("sft", base_url="http://unused", tool_mode="json")
    msgs = run["messages"]
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        # one action per example; a turn with several calls is split in order
        for j, call in enumerate(m["tool_calls"]):
            prefix = msgs[:i]
            if j:  # earlier calls of the same turn and their results are part of the context
                prefix = msgs[:i] + [{"role": "assistant", "text": m.get("text", ""), "tool_calls": m["tool_calls"][:j]}]
                prefix += [x for x in msgs[i + 1:] if x["role"] == "tool"][:j]
            chat = fmt._convert(run["system"], apply_history(prefix, history, k), run["tools"])
            target = {"role": "assistant", "text": m.get("text", "") if j == 0 else "", "tool_calls": [call]}
            chat.append({"role": "assistant", "content": fmt_assistant(target)})
            out.append({"messages": chat, "task_id": run["task_id"], "step": len(out)})
    return out


def fmt_assistant(m: dict) -> str:
    from ..agents.llm import json_assistant_content
    return json_assistant_content(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories containing episodes.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--history", default=None, help="override history strategy (default: the run's own)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--include-unsafe", action="store_true", help="keep correct-outcome episodes with violations")
    a = ap.parse_args()
    n_ep = n_ex = skipped = 0
    with open(a.out, "w") as f:
        for d in a.runs:
            for line in open(Path(d) / "episodes.jsonl"):
                row = json.loads(line)
                g, run = row["grade"], row["run"]
                if run["config"]["model"].startswith("scripted-"):
                    raise SystemExit(f"{d}: scripted policies use privileged information; don't train on them")
                if g["task_id"] in EVAL_IDS:
                    raise SystemExit(f"{d}: contains evaluation task {g['task_id']}; collect training data with "
                                     "--task-file shopbench/tasks/tasks_train.json")
                ok = g["success"] or (a.include_unsafe and g["outcome_correct"])
                if not ok:
                    skipped += 1
                    continue
                cfg = run["config"]
                exs = episode_examples(run, a.history or cfg["history"], a.k or cfg["k"])
                for ex in exs:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_ep += 1
                n_ex += len(exs)
    print(f"{n_ep} successful episodes -> {n_ex} step examples ({skipped} failed episodes skipped) -> {a.out}")


if __name__ == "__main__":
    main()
