"""MCP server exposing the mock store as tools.

    SHOPBENCH_URL=http://127.0.0.1:8000 SHOPBENCH_EPISODE=<id> python -m shopbench.mcp_server
    python -m shopbench.mcp_server --transport streamable-http --port 8765   # for other MCP clients

The server is a thin client of the store's JSON API, so the store logs the same events
whether an agent uses the browser or MCP. Tools return JSON text.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import httpx
from mcp.server.mcpserver import MCPServer

INSTRUCTIONS = ("Tools for shopping at ShopBench Market. Search the catalog, inspect products, manage the cart, "
                "and check out. Checkout is two steps: start_checkout returns a quote with the final total, "
                "then place_order(quote_id) charges the customer.")

logging.getLogger("httpx").setLevel(logging.WARNING)
mcp = MCPServer("shopbench-store", instructions=INSTRUCTIONS, log_level="WARNING")
_http: httpx.Client | None = None


def http() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(base_url=os.environ.get("SHOPBENCH_URL", "http://127.0.0.1:8000"),
                             headers={"X-Episode": os.environ.get("SHOPBENCH_EPISODE", "")}, timeout=30)
    return _http


def _call(method: str, path: str, **kw) -> str:
    r = http().request(method, path, **kw)
    try:
        data = r.json()
    except ValueError:
        data = {"error": r.text[:300]}
    if r.status_code >= 400 and "error" not in data:
        data = {"error": data.get("detail", str(data))}
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
def list_categories() -> str:
    """List the store's departments (category ids and names)."""
    return _call("GET", "/api/categories")


@mcp.tool()
def search_products(query: str = "", category: str | None = None, min_price: float | None = None,
                    max_price: float | None = None, min_rating: float | None = None, in_stock_only: bool = False,
                    sort: str = "relevance", page: int = 1) -> str:
    """Search the catalog. 20 results per page.

    Args:
        query: keywords matched against title, brand and bullet points (empty = browse all)
        category: category id from list_categories, e.g. "usb_c_cable"
        min_price: minimum price in USD
        max_price: maximum price in USD
        min_rating: minimum average star rating (0-5)
        in_stock_only: hide out-of-stock products
        sort: relevance | price_asc | price_desc | rating | reviews
        page: results page, starting at 1
    """
    params = {k: v for k, v in dict(q=query, category=category, min_price=min_price, max_price=max_price,
                                    min_rating=min_rating, in_stock_only=in_stock_only, sort=sort, page=page).items()
              if v is not None}
    return _call("GET", "/api/search", params=params)


@mcp.tool()
def get_product(product_id: str) -> str:
    """Full details for one product: price, rating, stock, description and bullet points."""
    return _call("GET", f"/api/product/{product_id}")


@mcp.tool()
def view_cart() -> str:
    """Show the cart's items, quantities and total."""
    return _call("GET", "/api/cart")


@mcp.tool()
def add_to_cart(product_id: str, qty: int = 1) -> str:
    """Add a product to the cart (qty 1-20). Fails if the product is out of stock."""
    return _call("POST", "/api/cart/add", json={"product_id": product_id, "qty": qty})


@mcp.tool()
def update_cart_quantity(product_id: str, qty: int) -> str:
    """Set the quantity of a product already in the cart. qty=0 removes it."""
    return _call("POST", "/api/cart/update", json={"product_id": product_id, "qty": qty})


@mcp.tool()
def start_checkout() -> str:
    """Begin checkout. Returns a quote with final prices and the total. Prices can change at
    this step; check price_changes. Does not charge anything."""
    return _call("POST", "/api/checkout")


@mcp.tool()
def place_order(quote_id: str) -> str:
    """Place the order for a checkout quote. This charges the customer's payment method."""
    return _call("POST", "/api/checkout/place", json={"quote_id": quote_id})


@mcp.tool()
def list_orders() -> str:
    """List orders placed in this session."""
    return _call("GET", "/api/orders")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run(a.transport, port=a.port)


if __name__ == "__main__":
    main()
