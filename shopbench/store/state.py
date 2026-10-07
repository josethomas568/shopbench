"""Per-episode store state: cart, checkout quotes, orders, traps and an event log.

One episode = one agent attempt at one task. All mutations go through `Store`
methods so the HTML site and the JSON API (used by the MCP server) behave
identically, and every action lands in the same event log the grader reads.
"""
from __future__ import annotations

import itertools
import threading
import time
import uuid
from dataclasses import dataclass, field

from .catalog import Catalog, Product


class StoreError(Exception):
    pass


@dataclass
class Traps:
    # product_id -> new unit price, applied the first time that product enters a checkout quote
    price_change: dict[str, float] = field(default_factory=dict)
    # extra product ids that are out of stock in this episode
    out_of_stock: list[str] = field(default_factory=list)
    # product_id -> text appended to its description (prompt-injection trap)
    injection: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Traps":
        d = d or {}
        return cls(price_change={k: float(v) for k, v in d.get("price_change", {}).items()},
                   out_of_stock=list(d.get("out_of_stock", [])),
                   injection=dict(d.get("injection", {})))


@dataclass
class Episode:
    id: str
    task_id: str
    traps: Traps
    cart: dict[str, int] = field(default_factory=dict)  # insertion-ordered
    price_overrides: dict[str, float] = field(default_factory=dict)
    quotes: dict[str, dict] = field(default_factory=dict)
    latest_quote: str | None = None
    orders: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    _seq: itertools.count = field(default_factory=itertools.count)

    def log(self, source: str, type_: str, **data) -> dict:
        ev = {"seq": next(self._seq), "t": time.time(), "source": source, "type": type_, **data}
        self.events.append(ev)
        return ev


class Store:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.episodes: dict[str, Episode] = {}
        self._lock = threading.RLock()

    # ---------- episodes ----------
    def new_episode(self, task_id: str = "", traps: dict | None = None, episode_id: str | None = None) -> Episode:
        with self._lock:
            ep = Episode(id=episode_id or uuid.uuid4().hex[:12], task_id=task_id, traps=Traps.from_dict(traps))
            self.episodes[ep.id] = ep
            ep.log("store", "episode_start", task_id=task_id)
            return ep

    def episode(self, eid: str | None) -> Episode:
        if not eid or eid not in self.episodes:
            raise StoreError("No active shopping session. Open the start link for this task first.")
        return self.episodes[eid]

    # ---------- product views ----------
    def product(self, ep: Episode, pid: str) -> Product:
        p = self.catalog.get(pid)
        if p is None:
            raise StoreError(f"Product {pid} not found.")
        return self._effective(ep, p)

    def _effective(self, ep: Episode, p: Product) -> Product:
        if p.id in ep.price_overrides:
            p.price = ep.price_overrides[p.id]
        if p.id in ep.traps.out_of_stock:
            p.in_stock = False
        if p.id in ep.traps.injection:
            p.description = (p.description + " " + ep.traps.injection[p.id]).strip()
        return p

    def view_product(self, ep: Episode, pid: str, via: str) -> Product:
        p = self.product(ep, pid)
        ep.log("store", "view_product", via=via, product_id=pid, price=p.price, in_stock=p.in_stock)
        return p

    def search(self, ep: Episode, via: str, **kw) -> tuple[list[Product], int]:
        items, total = self.catalog.search(price_overrides=ep.price_overrides, **kw)
        items = [self._effective(ep, p) for p in items]
        ep.log("store", "search", via=via, params={k: v for k, v in kw.items() if v not in (None, "", False)},
               total=total, shown=[p.id for p in items])
        return items, total

    def sponsored(self, ep: Episode, query: str, category: str | None, n: int = 2) -> list[Product]:
        """Ads shown above search results. Like real ads they ignore the shopper's sort and
        filters: the highest-priced in-stock matches for the query."""
        items, _ = self.catalog.search(query=query, category=category, in_stock_only=True, sort="price_desc",
                                       price_overrides=ep.price_overrides)
        ads = [self._effective(ep, p) for p in items[:n]]
        if ads:
            ep.log("store", "sponsored_shown", ids=[p.id for p in ads])
        return ads

    def also_viewed(self, ep: Episode, p: Product, n: int = 8) -> list[Product]:
        import hashlib
        same = [q for q in self.catalog.all(p.category) if q.id != p.id]
        same.sort(key=lambda q: hashlib.md5((p.id + q.id).encode()).hexdigest())
        return [self._effective(ep, q) for q in same[:n]]

    # ---------- cart ----------
    def cart_view(self, ep: Episode) -> dict:
        lines, total = [], 0.0
        for pid, qty in ep.cart.items():
            p = self.product(ep, pid)
            sub = round(p.price * qty, 2)
            total += sub
            lines.append({"product_id": pid, "title": p.title, "unit_price": p.price, "qty": qty,
                          "subtotal": sub, "in_stock": p.in_stock})
        return {"lines": lines, "total": round(total, 2), "item_count": sum(ep.cart.values())}

    def add_to_cart(self, ep: Episode, pid: str, qty: int, via: str) -> dict:
        with self._lock:
            p = self.product(ep, pid)
            if qty < 1 or qty > 20:
                ep.log("store", "add_to_cart_error", via=via, product_id=pid, qty=qty, error="bad_qty")
                raise StoreError("Quantity must be between 1 and 20.")
            if not p.in_stock:
                ep.log("store", "add_to_cart_error", via=via, product_id=pid, qty=qty, error="out_of_stock")
                raise StoreError(f"'{p.title[:60]}' is currently unavailable (out of stock).")
            ep.cart[pid] = ep.cart.get(pid, 0) + qty
            ep.latest_quote = None
            ep.log("store", "add_to_cart", via=via, product_id=pid, qty=qty, cart=dict(ep.cart))
            return self.cart_view(ep)

    def update_cart(self, ep: Episode, pid: str, qty: int, via: str) -> dict:
        with self._lock:
            if pid not in ep.cart:
                raise StoreError(f"{pid} is not in the cart.")
            if qty < 0 or qty > 20:
                raise StoreError("Quantity must be between 0 and 20.")
            if qty == 0:
                del ep.cart[pid]
            else:
                ep.cart[pid] = qty
            ep.latest_quote = None
            ep.log("store", "update_cart", via=via, product_id=pid, qty=qty, cart=dict(ep.cart))
            return self.cart_view(ep)

    # ---------- checkout ----------
    def checkout(self, ep: Episode, via: str) -> dict:
        """Create a price quote for the current cart. Price-change traps fire here."""
        with self._lock:
            if not ep.cart:
                raise StoreError("Your cart is empty.")
            changes = []
            for pid in ep.cart:
                if pid in ep.traps.price_change and pid not in ep.price_overrides:
                    old = self.product(ep, pid).price
                    new = ep.traps.price_change[pid]
                    ep.price_overrides[pid] = new
                    changes.append({"product_id": pid, "old_price": old, "new_price": new})
            cart = self.cart_view(ep)
            unavailable = [l["product_id"] for l in cart["lines"] if not l["in_stock"]]
            qid = "Q" + uuid.uuid4().hex[:8].upper()
            quote = {"quote_id": qid, "lines": cart["lines"], "total": cart["total"],
                     "price_changes": changes, "unavailable": unavailable, "cart": dict(ep.cart)}
            ep.quotes[qid] = quote
            ep.latest_quote = qid
            ep.log("store", "checkout_quote", via=via, quote_id=qid, total=quote["total"],
                   cart=dict(ep.cart), price_changes=changes)
            return quote

    def place_order(self, ep: Episode, quote_id: str, via: str) -> dict:
        with self._lock:
            if not quote_id or quote_id not in ep.quotes:
                ep.log("store", "place_order_error", via=via, quote_id=quote_id, error="unknown_quote")
                raise StoreError("Unknown checkout quote. Start checkout again.")
            if quote_id != ep.latest_quote:
                ep.log("store", "place_order_error", via=via, quote_id=quote_id, error="stale_quote")
                raise StoreError("This checkout quote is out of date (the cart changed). Start checkout again.")
            q = ep.quotes[quote_id]
            if q["unavailable"]:
                raise StoreError("Some items in your cart are unavailable. Remove them and check out again.")
            order = {"order_id": "ORD-" + uuid.uuid4().hex[:6].upper(), "quote_id": quote_id,
                     "lines": q["lines"], "total": q["total"], "t": time.time()}
            ep.orders.append(order)
            ep.cart.clear()
            ep.latest_quote = None
            ep.log("store", "order_placed", via=via, order_id=order["order_id"], quote_id=quote_id,
                   total=order["total"], items={l["product_id"]: l["qty"] for l in q["lines"]})
            return order

    def snapshot(self, ep: Episode) -> dict:
        return {"episode_id": ep.id, "task_id": ep.task_id, "cart": dict(ep.cart), "orders": ep.orders,
                "quotes": ep.quotes, "latest_quote": ep.latest_quote, "price_overrides": ep.price_overrides,
                "events": ep.events}
