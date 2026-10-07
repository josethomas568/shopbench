"""LoRA fine-tuning of a small open model on successful shopping trajectories (PyTorch + PEFT).

    python -m shopbench.sft.train_lora --data data/sft_mcp.jsonl --model Qwen/Qwen2.5-1.5B-Instruct \
        --out checkpoints/qwen1.5b-shop-lora --epochs 2 --max-len 8192

Loss is computed only on the final assistant turn of each example (the action taken at
that step). Long contexts are shortened by keeping the head (system prompt + request)
and the most recent tail, which mirrors what matters at decision time.

Serving the result for evaluation (vLLM example):
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora --lora-modules shop=checkpoints/qwen1.5b-shop-lora \
        --max-model-len 16384 --port 8001
    python -m shopbench.harness.runner --provider openai --base-url http://localhost:8001/v1 \
        --model shop --tool-mode json --interface mcp --out runs/qwen_lora_mcp
and the same command with --model Qwen/Qwen2.5-1.5B-Instruct for the untuned baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class StepDataset(Dataset):
    def __init__(self, rows: list[dict], tok, max_len: int, head: int):
        self.items = []
        self.truncated = 0
        for r in rows:
            msgs = r["messages"]
            prompt_ids = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
            target_ids = tok(msgs[-1]["content"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
            budget = max_len - len(target_ids)
            if budget < head + 64:
                continue  # target alone is too long
            if len(prompt_ids) > budget:
                prompt_ids = prompt_ids[:head] + prompt_ids[-(budget - head):]
                self.truncated += 1
            ids = list(prompt_ids) + list(target_ids)
            labels = [-100] * len(prompt_ids) + list(target_ids)
            self.items.append((ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a)
        lab[i, : len(b)] = torch.tensor(b)
        att[i, : len(a)] = 1
    return ids, att, lab


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for ids, att, lab in loader:
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        k = int((lab != -100).sum())
        tot += out.loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--head", type=int, default=1536, help="prompt tokens always kept from the start")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="cap optimizer steps (smoke tests)")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    rows = [json.loads(l) for l in open(a.data)]
    # split by task so validation measures generalization to unseen tasks
    tasks = sorted({r["task_id"] for r in rows})
    random.shuffle(tasks)
    n_val = max(1, int(len(tasks) * a.val_frac)) if len(tasks) > 1 else 0
    val_tasks = set(tasks[:n_val])
    train_rows = [r for r in rows if r["task_id"] not in val_tasks]
    val_rows = [r for r in rows if r["task_id"] in val_tasks]

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = StepDataset(train_rows, tok, a.max_len, a.head)
    val_ds = StepDataset(val_rows, tok, a.max_len, a.head)
    print(f"device={device} dtype={dtype}  train examples={len(train_ds)} (truncated {train_ds.truncated})  "
          f"val examples={len(val_ds)} from {len(val_tasks)} held-out tasks")

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(device)
    if a.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    targets = [m for m in a.target_modules.split(",") if any(n.endswith(m) for n, _ in model.named_modules())]
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    pad = tok.pad_token_id
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad))
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, collate_fn=lambda b: collate(b, pad))
    steps_total = math.ceil(len(train_dl) * a.epochs / a.grad_accum)
    if a.max_steps:
        steps_total = min(steps_total, a.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total)))))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    model.train()
    step, micro, t0, running = 0, 0, time.time(), 0.0
    if len(val_ds):
        print(f"val loss before training: {evaluate(model, val_dl, device):.4f}")
    while step < steps_total:
        for ids, att, lab in train_dl:
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            (loss / a.grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "loss": running / a.grad_accum, "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            running = 0.0
            if step % 20 == 0 or step == steps_total:
                if len(val_ds):
                    rec["val_loss"] = evaluate(model, val_dl, device)
                print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            if step >= steps_total:
                break
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "shopbench_train_args.json").write_text(json.dumps(vars(a), indent=1))
    print(f"saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
