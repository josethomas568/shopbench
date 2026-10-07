"""Build the mock-store catalog from the Amazon ESCI Shopping Queries dataset.

ESCI provides real product text (title, description, bullets, brand, color) and
query-product relevance labels, but NO price, rating, review count or stock.
Those fields are synthesized here, deterministically from a hash of the product
ID, so every build of the catalog is identical and task answers stay valid.

Usage:
    python data/build_catalog.py --esci-dir /path/to/esci/parquets --out data/catalog.sqlite

The ESCI parquets come from https://github.com/amazon-science/esci-data
(shopping_queries_dataset_products.parquet and shopping_queries_dataset_examples.parquet;
they are Git LFS files, so fetch them with `git lfs pull` or from
media.githubusercontent.com).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

# category id -> (display name, query regex, title regex, median price, price sigma)
# The query regex pulls candidate products from ESCI queries (Exact/Substitute labels);
# the title regex keeps only products that really belong in the category.
CATEGORIES: dict[str, tuple[str, str, str, float, float]] = {
    "usb_c_cable": ("USB-C Cables", r"usb[ -]?c (?:cable|cord)|usb c to|type c cable", r"^(?=.*(?:usb[- ]?c|type[- ]?c))(?=.*(?:cable|cord))(?!.*(?:adapter|hub|dongle|splitter|headphone|earphone|charger block|wall charger))", 11.0, 0.45),
    "hdmi_cable": ("HDMI Cables", r"hdmi cable", r"^(?=.*hdmi)(?=.*cable)(?!.*(?:adapter|switch|splitter|converter|capture|displayport|dvi|vga|wii))", 12.0, 0.5),
    "wireless_mouse": ("Wireless Mice", r"wireless mouse", r"^(?=.*(?:mouse|mice))(?!.*(?:pad|mat\b|laptop|macbook|keyboard|combo))", 22.0, 0.5),
    "mechanical_keyboard": ("Mechanical Keyboards", r"mechanical keyboard", r"^(?=.*keyboard)(?!.*(?:keycap|wrist|case|cover|switch tester|piano|mouse|palm rest|wrist rest))", 55.0, 0.5),
    "earbuds": ("Wireless Earbuds", r"(?:wireless|bluetooth) earbuds", r"^(?=.*(?:earbud|earphone|headphone))(?!.*(?:case|cover|tips|replacement|adapter))", 35.0, 0.6),
    "power_bank": ("Power Banks", r"power bank|portable charger", r"^(?=.*(?:power bank|portable charger|battery pack))(?!.*(?:case|cable only|power station|generator|solar))", 28.0, 0.45),
    "water_bottle": ("Water Bottles", r"water bottle", r"^(?=.*bottle)(?!.*(?:brush|lid only|replacement|baby|pump))", 18.0, 0.5),
    "coffee_maker": ("Coffee Makers", r"coffee maker", r"^(?=.*(?:coffee maker|espresso|brewer|coffee machine|pour over|french press|percolator))(?!.*(?:filter|pod|descal|grinder|carafe only|replacement))", 60.0, 0.6),
    "yoga_mat": ("Yoga Mats", r"yoga mat", r"^(?=.*yoga)(?=.*mat)(?!.*(?:strap|block|towel|bag|cleaner))", 26.0, 0.45),
    "desk_lamp": ("Desk Lamps", r"desk lamp", r"^(?=.*lamp)(?!.*bulb)", 30.0, 0.45),
    "laptop_backpack": ("Laptop Backpacks", r"laptop backpack", r"^(?=.*backpack)(?!.*(?:rain cover|strap|keychain))", 38.0, 0.45),
    "wall_charger": ("Wall Chargers", r"wall charger|usb c charger", r"^(?=.*charger)(?!.*(?:power bank|portable|car charger|wireless charg|cable only))", 17.0, 0.45),
    "sd_card": ("Memory Cards", r"micro ?sd card", r"^(?=.*(?:micro ?sd|sd card|memory card))(?!.*(?:reader|case|holder|adapter only|laptop|chromebook|camera|phone|tablet))", 16.0, 0.6),
    "notebook": ("Notebooks", r"(?:spiral|composition) notebook", r"^(?=.*notebook)(?!.*(?:laptop|computer|stand|sleeve|cover))", 9.0, 0.5),
    "batteries": ("Batteries", r"aa batteries", r"^(?=.*\baa\b)(?=.*batter)(?!.*(?:charger only|holder|case|tester))", 14.0, 0.45),
    "led_bulb": ("LED Bulbs", r"led (?:light )?bulb", r"^(?=.*bulb)(?=.*led)", 13.0, 0.5),
    "dog_toy": ("Dog Toys", r"dog (?:chew )?toy", r"^(?=.*dog)(?=.*(?:toy|chew|ball|rope))(?!.*(?:bed|leash|collar|food|treat))", 12.0, 0.5),
    "athletic_socks": ("Athletic Socks", r"(?:running|athletic) socks", r"^(?=.*socks?\b)(?!.*(?:shoe|insole))", 15.0, 0.4),
}
MAX_PER_CATEGORY = 250


def h(pid: str, salt: str) -> float:
    """Deterministic uniform(0,1) from product id + salt."""
    d = hashlib.sha256(f"{salt}:{pid}".encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def synth_fields(pid: str, median: float, sigma: float) -> dict:
    # Price: lognormal around the category median, retail-style endings.
    u1, u2 = h(pid, "p1"), h(pid, "p2")
    z = math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
    raw = median * math.exp(sigma * z)
    raw = min(max(raw, 2.0), median * 6)
    ending = [0.99, 0.49, 0.95, 0.0][int(h(pid, "end") * 4)]
    price = max(1.49, math.floor(raw) + ending if ending else round(raw))
    # Rating: skewed high like real marketplaces, one decimal, 2.5-5.0.
    r = 5.0 - (h(pid, "r") ** 1.8) * 2.5
    rating = round(r, 1)
    reviews = int(10 ** (1 + h(pid, "n") * 3.6))  # 10 .. ~40k
    in_stock = h(pid, "stock") > 0.08  # ~8% out of stock
    list_price = round(price * (1.15 + h(pid, "lp") * 0.4), 2) if h(pid, "sale") < 0.3 else None
    return dict(price=round(price, 2), list_price=list_price, rating=rating,
                review_count=reviews, in_stock=int(in_stock))


def clean(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = re.sub(r"<[^>]+>", " ", str(s))
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esci-dir", required=True)
    ap.add_argument("--out", default=str(Path(__file__).with_name("catalog.sqlite")))
    args = ap.parse_args()
    esci = Path(args.esci_dir)

    ex = pd.read_parquet(esci / "shopping_queries_dataset_examples.parquet",
                         columns=["query", "product_id", "product_locale", "esci_label"])
    ex = ex[(ex.product_locale == "us") & (ex.esci_label.isin(["E", "S"]))]

    wanted: dict[str, str] = {}  # product_id -> category (first match wins)
    for cat, (_, qre, _, _, _) in CATEGORIES.items():
        m = ex[ex["query"].str.contains(qre, regex=True, case=False, na=False)]
        for pid in m.product_id.unique():
            wanted.setdefault(pid, cat)
    print(f"candidate products: {len(wanted)}")

    dset = ds.dataset(esci / "shopping_queries_dataset_products.parquet")
    tbl = dset.to_table(filter=(ds.field("product_locale") == "us") & ds.field("product_id").isin(list(wanted)))
    prods = tbl.to_pandas()
    prods["category"] = prods.product_id.map(wanted)

    rows = []
    for cat, (name, _, tre, med, sig) in CATEGORIES.items():
        sub = prods[(prods.category == cat) & prods.product_title.str.contains(tre, case=False, regex=True, na=False)]
        sub = sub.sort_values("product_id").drop_duplicates("product_id")
        # deterministic subsample
        sub = sub.assign(_k=sub.product_id.map(lambda p: h(p, "sample"))).sort_values("_k").head(MAX_PER_CATEGORY)
        for r in sub.itertuples():
            title = clean(r.product_title)
            if len(title) < 10:
                continue
            rows.append(dict(
                id=r.product_id, category=cat, category_name=name, title=title[:300],
                brand=clean(r.product_brand)[:60], color=clean(r.product_color)[:40],
                description=clean(r.product_description)[:1500],
                bullets=clean(r.product_bullet_point)[:1500],
                **synth_fields(r.product_id, med, sig),
            ))
        print(f"{cat:22s} {len(sub):4d}")

    out = Path(args.out)
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.executescript("""
    CREATE TABLE products(id TEXT PRIMARY KEY, category TEXT, category_name TEXT, title TEXT,
        brand TEXT, color TEXT, description TEXT, bullets TEXT, price REAL, list_price REAL,
        rating REAL, review_count INTEGER, in_stock INTEGER);
    CREATE VIRTUAL TABLE products_fts USING fts5(id UNINDEXED, title, brand, bullets, category_name,
        tokenize='porter unicode61');
    CREATE TABLE categories(id TEXT PRIMARY KEY, name TEXT);
    """)
    con.executemany("INSERT INTO products VALUES(:id,:category,:category_name,:title,:brand,:color,"
                    ":description,:bullets,:price,:list_price,:rating,:review_count,:in_stock)", rows)
    con.executemany("INSERT INTO products_fts VALUES(:id,:title,:brand,:bullets,:category_name)", rows)
    con.executemany("INSERT INTO categories VALUES(?,?)", [(k, v[0]) for k, v in CATEGORIES.items()])
    con.commit()
    n = con.execute("select count(*), sum(1-in_stock) from products").fetchone()
    print(f"wrote {out}: {n[0]} products, {n[1]} out of stock")


if __name__ == "__main__":
    main()
