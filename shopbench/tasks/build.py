"""Compile task specs into tasks/tasks.json with frozen ground truth.

    python -m shopbench.tasks.build            # writes shopbench/tasks/tasks.json
    python -m shopbench.tasks.build --check    # validate only

Each task in the output carries:
  items[i].acceptable   product ids that satisfy item i (ties included), as seen in that task's episode
  items[i].unit_prices  episode prices at checkout time (after price-change traps)
  expect                purchase | no_purchase | cart_only
  traps                 concrete trap config passed to the store when the episode starts
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ..store.catalog import Catalog, Product
from .specs import TASKS

OUT = Path(__file__).with_name("tasks.json")


def matches(p: Product, sel: dict, oos: set[str] = frozenset(), prices: dict[str, float] | None = None) -> bool:
    price = (prices or {}).get(p.id, p.price)
    in_stock = p.in_stock and p.id not in oos
    if "ids" in sel and p.id not in sel["ids"]:
        return False
    if sel.get("category") and p.category != sel["category"]:
        return False
    if sel.get("brand") and p.brand.lower() != sel["brand"].lower():
        return False
    if sel.get("title_re") and not re.search(sel["title_re"], p.title, re.I):
        return False
    if sel.get("min_price") is not None and price < sel["min_price"]:
        return False
    if sel.get("max_price") is not None and price > sel["max_price"]:
        return False
    if sel.get("min_rating") is not None and p.rating < sel["min_rating"]:
        return False
    if sel.get("min_reviews") is not None and p.review_count < sel["min_reviews"]:
        return False
    if sel.get("in_stock", True) and not in_stock:
        return False
    return True


def best(cands: list[Product], objective: str) -> list[Product]:
    if not cands or objective == "any":
        return cands
    key = {"price_asc": lambda p: (p.price,),
           "rating": lambda p: (-p.rating, -p.review_count),
           "reviews": lambda p: (-p.review_count,)}[objective]
    top = min(key(p) for p in cands)
    return [p for p in cands if key(p) == top]


def resolve_price(expr, base: float, budget: float | None, qty: int) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)
    if expr.startswith("x"):
        return round(base * float(expr[1:]), 2)
    if expr.startswith("budget+"):
        assert budget is not None
        # unit price such that qty * price exceeds the budget by the given amount
        return round((budget + float(expr[7:])) / qty, 2)
    raise ValueError(expr)


def compile_task(t: dict, cat: Catalog) -> dict:
    products = cat.all()
    traps = {k: v for k, v in (t.get("traps") or {}).items()}
    oos = set(traps.get("out_of_stock", []))
    expect = t.get("expect", "purchase")
    items_out = []
    for i, it in enumerate(t.get("items", [])):
        sel, obj, qty = it["select"], it.get("objective", "any"), it.get("qty", 1)
        acc = best([p for p in products if matches(p, sel, oos)], obj)
        items_out.append(dict(select=sel, objective=obj, qty=qty, acceptable=sorted(p.id for p in acc),
                              base_prices={p.id: p.price for p in acc}))

    # Resolve price-change traps that point at an item ("$item0") into concrete ids.
    pc = {}
    for k, v in (traps.get("price_change") or {}).items():
        if k.startswith("$item"):
            it = items_out[int(k[5:])]
            for pid in it["acceptable"]:
                pc[pid] = resolve_price(v, it["base_prices"][pid], t.get("budget"), it["qty"])
        else:
            base = cat.get(k).price
            pc[k] = resolve_price(v, base, t.get("budget"), 1)
    if pc:
        traps["price_change"] = pc
    for it in items_out:
        it["unit_prices"] = {pid: pc.get(pid, it["base_prices"][pid]) for pid in it["acceptable"]}

    out = dict(id=t["id"], type=t["type"], instruction=t["instruction"], expect=expect,
               budget=t.get("budget"), items=items_out, traps=traps,
               requires_question=t.get("requires_question", False),
               clarification=t.get("clarification"), user_reply=t.get("user_reply"),
               forbidden_ids=t.get("forbidden_ids", []))
    validate(out, t, cat, oos)
    return out


def validate(task: dict, spec: dict, cat: Catalog, oos: set[str]) -> None:
    tid, expect = task["id"], task["expect"]
    if spec.get("unavailable_re"):
        hits = [p for p in cat.all() if re.search(spec["unavailable_re"], p.title, re.I)]
        assert hits, f"{tid}: unavailable_re matches nothing"
        avail = [p.id for p in hits if p.in_stock and p.id not in oos]
        assert not avail, f"{tid}: target should be unavailable but in stock: {avail}"
    if expect in ("purchase", "cart_only"):
        for i, it in enumerate(task["items"]):
            assert it["acceptable"], f"{tid}: item {i} has no acceptable product"
        cheapest_total = sum(min(it["unit_prices"].values()) * it["qty"] for it in task["items"])
        if task["budget"] is not None:
            assert cheapest_total <= task["budget"] + 1e-9, f"{tid}: correct order {cheapest_total} exceeds budget"
        ids = [set(it["acceptable"]) for it in task["items"]]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                assert not (ids[a] & ids[b]), f"{tid}: items {a} and {b} overlap"
    elif task["type"] == "trap_price":
        tot = sum(min(it["unit_prices"].values()) * it["qty"] for it in task["items"])
        pre = sum(min(it["base_prices"].values()) * it["qty"] for it in task["items"])
        assert pre <= task["budget"] < tot, f"{tid}: price trap should cross budget ({pre} -> {tot}, budget {task['budget']})"
    elif task["type"] == "infeasible":
        for it in task["items"]:
            if it["acceptable"]:
                tot = min(it["unit_prices"].values()) * it["qty"]
                assert task["budget"] is not None and tot > task["budget"], f"{tid}: infeasible task is feasible"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    cat = Catalog()
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    tasks, errors = [], []
    for t in TASKS:
        try:
            tasks.append(compile_task(t, cat))
        except AssertionError as e:
            errors.append(str(e))
    if errors:
        raise SystemExit("task validation failed:\n  " + "\n  ".join(errors))
    if a.verbose:
        for t in tasks:
            parts = []
            for it in t["items"]:
                prices = ", ".join(f"{pid}@{it['unit_prices'][pid]}" for pid in it["acceptable"][:3])
                parts.append(f"{len(it['acceptable'])}x[{prices}] qty={it['qty']}")
            desc = "; ".join(parts)
            print(f"{t['id']:10s} {t['expect']:12s} budget={t['budget']}  {desc}  traps={t['traps'] or ''}")
    by_type: dict[str, int] = {}
    for t in tasks:
        by_type[t["type"]] = by_type.get(t["type"], 0) + 1
    print(f"{len(tasks)} tasks OK: {by_type}")
    if not a.check:
        OUT.write_text(json.dumps(tasks, indent=1))
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
