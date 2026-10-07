"""Scripted policies that plug into the agent loop in place of an LLM.

They exist to validate the benchmark itself:
  oracle            privileged: reads the task's ground truth and acts perfectly (should score 100%, 0 violations)
  oracle_noconfirm  oracle that skips request_confirmation  -> must be flagged purchase_without_confirmation
  greedy            unprivileged baseline: searches the user's words, buys the top result, confirms
                    with the user, never asks questions. Shows what a naive agent scores.
Both interfaces are supported, so the browser and MCP paths are exercised identically.
"""
from __future__ import annotations

import json
import re
import uuid
from urllib.parse import quote_plus

from .llm import LLM, Turn, Usage


def _call(name: str, **args) -> dict:
    return {"id": "call_" + uuid.uuid4().hex[:8], "name": name, "args": args}


def find_ref(obs: str, name: str, role: str | None = None) -> str | None:
    """Locate an element ref by visible name, in either tree or HTML observations."""
    for line in obs.splitlines():
        m = re.search(r'- (\w+) "([^"]*)".*?\[ref=(e\d+)\]', line)
        if m and name.lower() in m.group(2).lower() and (role is None or m.group(1) == role):
            return m.group(3)
    m = re.search(r'<(?:button|a|select|input)\b[^>]*data-ref="(e\d+)"[^>]*>\s*' + re.escape(name), obs, re.I)
    if m:
        return m.group(1)
    if role == "combobox":  # html select located by its label text
        m = re.search(r'<label for="([\w-]+)">\s*' + re.escape(name), obs, re.I)
        if m:
            m2 = re.search(r'<select[^>]*id="' + re.escape(m.group(1)) + r'"[^>]*data-ref="(e\d+)"', obs)
            if m2:
                return m2.group(1)
    return None


def parse_total(obs: str) -> float | None:
    if obs.lstrip().startswith("{"):
        try:
            return float(json.loads(obs)["total"])
        except (ValueError, KeyError):
            return None
    m = re.search(r"Order total:\s*(?:<[^>]+>\s*)*\$([\d,]+\.\d\d)", obs) or re.search(r"Order total:.*?\$([\d,]+\.\d\d)", obs, re.S)
    return float(m.group(1).replace(",", "")) if m else None


class ScriptedPolicy(LLM):
    def __init__(self, kind: str = "oracle", interface: str = "mcp"):
        assert kind in ("oracle", "oracle_noconfirm", "greedy")
        self.kind, self.interface = kind, interface
        self.model = f"scripted-{kind}"
        self.price_in = self.price_out = 0.0

    def bind(self, task: dict, server, eid: str) -> None:
        self.task, self.plan = task, self._make_plan(task)
        self.quote_id = None
        self.total = None

    # ------------------------------------------------------------ planning
    def _targets(self, task: dict) -> list[tuple[str, int]]:
        if self.kind == "greedy":
            return []
        return [(it["acceptable"][0], it["qty"]) for it in task["items"] if it["acceptable"]]

    def _make_plan(self, task: dict) -> list:
        steps: list = []
        if self.kind == "greedy":
            return [("greedy_search",), ("greedy_pick",)] + [("checkout",), ("confirm",), ("place",), ("done",)]
        if task.get("requires_question"):
            steps.append(("ask", "Could you clarify exactly which product you want?"))
        expect = task["expect"]
        if expect == "no_purchase" and task["type"] != "trap_price":
            return steps + [("finish", "not_purchased", "I couldn't buy this as requested.")]
        for pid, qty in self._targets(task):
            steps.append(("add", pid, qty))
        if expect == "cart_only":
            return steps + [("finish", "cart_ready", "The item is in your cart.")]
        steps += [("checkout",)]
        if self.kind != "oracle_noconfirm":
            steps += [("confirm",)]
        return steps + [("place",), ("done",)]

    # ------------------------------------------------------------ acting
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        last = next((m["content"] for m in reversed(messages) if m["role"] in ("tool", "user")), "")
        call = self._next(last)
        return Turn("", [call], Usage(calls=1))

    def _next(self, last: str) -> dict:
        while self.plan:
            step = self.plan[0]
            kind = step[0]
            if kind == "ask":
                self.plan.pop(0)
                return _call("ask_user", question=step[1])
            if kind == "finish":
                self.plan.pop(0)
                return _call("finish", outcome=step[1], message=step[2])
            if kind == "greedy_search":
                self.plan.pop(0)
                words = re.sub(r"[^a-z0-9 ]", " ", self.task["instruction"].lower())
                q = " ".join(w for w in words.split() if w not in _STOP)[:80]
                if self.interface == "mcp":
                    return _call("search_products", query=q, in_stock_only=True)
                return _call("goto", url=f"/search?q={quote_plus(q)}&in_stock=1")
            if kind == "greedy_pick":
                self.plan.pop(0)
                pid = None
                if self.interface == "mcp":
                    try:
                        res = json.loads(last)["results"]
                        pid = res[0]["product_id"] if res else None
                    except (ValueError, KeyError):
                        pid = None
                else:
                    m = re.search(r'href="?/product/(\w+)|\[url=/product/(\w+)\]|link "[^"]*" \[ref=(e\d+)\]', last)
                    if m:
                        pid = m.group(1) or m.group(2)
                        if not pid:  # pruned tree has no urls: click the first result link
                            self.plan.insert(0, ("add_here", 1))
                            return _call("click", ref=self._first_result_ref(last))
                if not pid:
                    self.plan = []
                    return _call("finish", outcome="not_purchased", message="Nothing matched.")
                self.plan.insert(0, ("add", pid, 1))
                continue
            if kind in ("add", "add_here"):
                if self.interface == "mcp":
                    self.plan.pop(0)
                    return _call("add_to_cart", product_id=step[1], qty=step[2])
                # browser: open the product page, set quantity, click Add to cart
                if kind == "add":
                    self.plan[0] = ("add_here", step[2])
                    return _call("goto", url=f"/product/{step[1]}")
                qty = step[1]
                if qty != 1 and not step[-1] == "qty_set":
                    self.plan[0] = ("add_here", qty, "qty_set")
                    return _call("select", ref=find_ref(last, "Quantity", "combobox"), option=str(qty))
                self.plan.pop(0)
                return _call("click", ref=find_ref(last, "Add to cart", "button"))
            if kind == "checkout":
                self.plan.pop(0)
                if self.interface == "mcp":
                    return _call("start_checkout")
                self.plan.insert(0, ("checkout_click",))
                return _call("goto", url="/cart")
            if kind == "checkout_click":
                self.plan.pop(0)
                return _call("click", ref=find_ref(last, "Proceed to checkout", "button"))
            if kind == "confirm":
                self.plan.pop(0)
                if self.interface == "mcp":
                    try:
                        q = json.loads(last)
                        self.quote_id, self.total = q["quote_id"], q["total"]
                    except (ValueError, KeyError):
                        return _call("finish", outcome="not_purchased", message="Checkout failed.")
                else:
                    self.total = parse_total(last)
                    self._checkout_obs = last
                return _call("request_confirmation", summary="Order as requested", total=self.total)
            if kind == "place":
                self.plan.pop(0)
                if "User:" in last and "Yes" not in last:
                    self.plan = []
                    return _call("finish", outcome="not_purchased", message="You declined the purchase at checkout.")
                if self.interface == "mcp":
                    if self.quote_id is None:  # noconfirm path: last is the quote
                        q = json.loads(last)
                        self.quote_id = q["quote_id"]
                    return _call("place_order", quote_id=self.quote_id)
                obs = getattr(self, "_checkout_obs", None) or last
                if "User:" in last:
                    # confirmation reply hides the page; re-read it from the cached checkout observation
                    pass
                return _call("click", ref=find_ref(obs, "Place your order", "button"))
            if kind == "done":
                self.plan.pop(0)
                return _call("finish", outcome="purchased", message="Order placed.")
        return _call("finish", outcome="not_purchased", message="No plan.")

    @staticmethod
    def _first_result_ref(obs: str) -> str | None:
        in_results = False
        for line in obs.splitlines():
            if "result" in line and "heading" in line:
                in_results = True
            if in_results:
                m = re.search(r'link "[^"]*" \[ref=(e\d+)\]', line)
                if m:
                    return m.group(1)
        return None


_STOP = set("""a an the i me my to of for and or but with at least most buy order get purchase please want need
it its that this one ones if is are be than more less under over stars star rating rated or higher better cheapest
highest lowest ratings and dont don t only any as long don't stay keep total budget limit spend much can you""".split())
