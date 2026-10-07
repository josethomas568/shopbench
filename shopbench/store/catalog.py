"""Read-only access to the product catalog (SQLite + FTS5)."""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "catalog.sqlite"
PAGE_SIZE = 20
SORTS = {
    "relevance": None,
    "price_asc": "p.price ASC, p.id",
    "price_desc": "p.price DESC, p.id",
    "rating": "p.rating DESC, p.review_count DESC, p.id",
    "reviews": "p.review_count DESC, p.id",
}


@dataclass
class Product:
    id: str
    category: str
    category_name: str
    title: str
    brand: str
    color: str
    description: str
    bullets: str
    price: float
    list_price: float | None
    rating: float
    review_count: int
    in_stock: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Catalog:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"{self.path} not found; run data/build_catalog.py first")
        self._local = threading.local()
        self.categories: dict[str, str] = dict(self._con().execute("SELECT id, name FROM categories ORDER BY name"))

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    @staticmethod
    def _row(r: sqlite3.Row) -> Product:
        d = dict(r)
        d["in_stock"] = bool(d["in_stock"])
        return Product(**d)

    def get(self, pid: str) -> Product | None:
        r = self._con().execute("SELECT * FROM products WHERE id=?", (pid,)).fetchone()
        return self._row(r) if r else None

    def all(self, category: str | None = None) -> list[Product]:
        q, args = "SELECT * FROM products", ()
        if category:
            q, args = q + " WHERE category=?", (category,)
        return [self._row(r) for r in self._con().execute(q, args)]

    def search(self, query: str = "", category: str | None = None, min_price: float | None = None,
               max_price: float | None = None, min_rating: float | None = None, in_stock_only: bool = False,
               sort: str = "relevance", page: int = 1, price_overrides: dict[str, float] | None = None
               ) -> tuple[list[Product], int]:
        """Returns (products on this page, total matches).

        price_overrides are per-episode prices (e.g. a price that changed at checkout);
        price filters and sorting respect them, so the store stays consistent.
        """
        terms = re.findall(r"[a-z0-9]+", (query or "").lower())
        con = self._con()
        base = "SELECT p.* , {rank} AS _rank FROM products p {join} WHERE 1=1"
        args: list = []
        if terms:
            fts = " AND ".join(f'"{t}"' for t in terms)
            sql = base.format(rank="bm25(products_fts)", join="JOIN products_fts f ON f.id = p.id") + " AND products_fts MATCH ?"
            args.append(fts)
        else:
            sql = base.format(rank="0", join="")
        if category:
            sql += " AND p.category = ?"
            args.append(category)
        if min_rating is not None:
            sql += " AND p.rating >= ?"
            args.append(min_rating)
        if in_stock_only:
            sql += " AND p.in_stock = 1"
        raw = con.execute(sql, args).fetchall()
        ranks = {r["id"]: r["_rank"] for r in raw}
        rows = [self._row({k: r[k] for k in r.keys() if k != "_rank"}) for r in raw]
        overrides = price_overrides or {}
        for p in rows:
            if p.id in overrides:
                p.price = overrides[p.id]
        if min_price is not None:
            rows = [p for p in rows if p.price >= min_price]
        if max_price is not None:
            rows = [p for p in rows if p.price <= max_price]
        if sort == "price_asc":
            rows.sort(key=lambda p: (p.price, p.id))
        elif sort == "price_desc":
            rows.sort(key=lambda p: (-p.price, p.id))
        elif sort == "rating":
            rows.sort(key=lambda p: (-p.rating, -p.review_count, p.id))
        elif sort == "reviews":
            rows.sort(key=lambda p: (-p.review_count, p.id))
        else:  # relevance: bm25 (lower is better), then reviews as a popularity prior
            rows.sort(key=lambda p: (round(ranks.get(p.id, 0.0), 3), -p.review_count, p.id))
        total = len(rows)
        page = max(1, page)
        return rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], total
