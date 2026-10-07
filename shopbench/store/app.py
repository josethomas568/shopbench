"""FastAPI mock store.

Three surfaces over one `Store`:
  * HTML pages (/, /search, /product/{id}, /cart, /checkout, /orders) for browser agents.
    The episode is carried in the `sb_ep` cookie, set by /start?episode=<id>.
  * JSON API (/api/...) for the MCP server. The episode is carried in the X-Episode header.
  * Admin API (/admin/...) for the benchmark harness, guarded by X-Admin-Token.
    Agents never see this token, so they cannot read answers or rewrite state.

Run standalone:  python -m shopbench.store.app --port 8000
"""
from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .catalog import PAGE_SIZE, SORTS, Catalog
from .state import Episode, Store, StoreError

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))
COOKIE = "sb_ep"


class AddReq(BaseModel):
    product_id: str
    qty: int = 1


class PlaceReq(BaseModel):
    quote_id: str


class NewEpisodeReq(BaseModel):
    task_id: str = ""
    traps: dict | None = None
    episode_id: str | None = None


class LogReq(BaseModel):
    type: str
    data: dict = {}


def create_app(catalog_path: str | None = None, admin_token: str | None = None, clutter: bool | None = None) -> FastAPI:
    """clutter=True (default; env SHOPBENCH_CLUTTER=0 disables) renders pages the way real stores do:
    mega-menu, cookie banner, sponsored results, carousels, footer, icon SVGs, tracking attributes."""
    catalog = Catalog(catalog_path) if catalog_path else Catalog()
    if clutter is None:
        clutter = os.environ.get("SHOPBENCH_CLUTTER", "1") != "0"
    store = Store(catalog)
    token = admin_token or os.environ.get("SHOPBENCH_ADMIN_TOKEN") or secrets.token_hex(16)
    app = FastAPI(title="ShopBench mock store", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.admin_token = token

    def render(request: Request, name: str, **ctx):
        status = ctx.pop("status_code", 200)
        return TEMPLATES.TemplateResponse(request, name, {"categories": catalog.categories, "clutter": clutter, **ctx},
                                          status_code=status)

    def html_episode(request: Request) -> Episode:
        try:
            return store.episode(request.cookies.get(COOKIE))
        except StoreError as e:
            raise HTTPException(400, str(e))

    def api_episode(x_episode: str | None = Header(default=None)) -> Episode:
        try:
            return store.episode(x_episode)
        except StoreError as e:
            raise HTTPException(400, str(e))

    def admin(x_admin_token: str | None = Header(default=None)) -> None:
        if not x_admin_token or not secrets.compare_digest(x_admin_token, token):
            raise HTTPException(403, "forbidden")

    @app.exception_handler(StoreError)
    async def _store_error(request: Request, exc: StoreError):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": str(exc)}, status_code=400)
        return render(request, "error.html", message=str(exc), status_code=400)

    # ------------------------------------------------------------------ HTML
    @app.get("/start")
    def start(episode: str):
        store.episode(episode)  # validates
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE, episode, httponly=True)
        return resp

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, ep: Episode = Depends(html_episode)):
        return render(request, "home.html", cart_count=sum(ep.cart.values()))

    @app.get("/search", response_class=HTMLResponse)
    def search_page(request: Request, q: str = "", category: str = "", min_price: str = "", max_price: str = "",
                    min_rating: str = "", in_stock: str = "", sort: str = "relevance", page: int = 1,
                    ep: Episode = Depends(html_episode)):
        f = lambda s: float(s) if s not in ("", None) else None  # noqa: E731
        sort = sort if sort in SORTS else "relevance"
        items, total = store.search(ep, "html", query=q, category=category or None, min_price=f(min_price),
                                    max_price=f(max_price), min_rating=f(min_rating), in_stock_only=bool(in_stock),
                                    sort=sort, page=page)
        pages = max(1, -(-total // PAGE_SIZE))
        sponsored = store.sponsored(ep, q, category or None) if clutter and page == 1 else []
        params = dict(q=q, category=category, min_price=min_price, max_price=max_price,
                      min_rating=min_rating, in_stock=in_stock, sort=sort)
        return render(request, "search.html", items=items, total=total, page=page, pages=pages, params=params,
                      sponsored=sponsored,
                      sorts=list(SORTS), cart_count=sum(ep.cart.values()))

    @app.get("/product/{pid}", response_class=HTMLResponse)
    def product_page(request: Request, pid: str, ep: Episode = Depends(html_episode)):
        p = store.view_product(ep, pid, "html")
        also = store.also_viewed(ep, p) if clutter else []
        return render(request, "product.html", p=p, also_viewed=also, cart_count=sum(ep.cart.values()))

    @app.post("/cart/add")
    def cart_add(request: Request, product_id: str = Form(...), qty: int = Form(1),
                 ep: Episode = Depends(html_episode)):
        store.add_to_cart(ep, product_id, qty, "html")
        return RedirectResponse("/cart?added=" + product_id, status_code=303)

    @app.post("/cart/update")
    def cart_update(product_id: str = Form(...), qty: int = Form(...), ep: Episode = Depends(html_episode)):
        store.update_cart(ep, product_id, qty, "html")
        return RedirectResponse("/cart", status_code=303)

    @app.get("/cart", response_class=HTMLResponse)
    def cart_page(request: Request, added: str = "", ep: Episode = Depends(html_episode)):
        return render(request, "cart.html", cart=store.cart_view(ep), added=added,
                      cart_count=sum(ep.cart.values()))

    @app.post("/checkout", response_class=HTMLResponse)
    def checkout_page(request: Request, ep: Episode = Depends(html_episode)):
        quote = store.checkout(ep, "html")
        return render(request, "checkout.html", quote=quote, cart_count=sum(ep.cart.values()))

    @app.post("/checkout/place", response_class=HTMLResponse)
    def place_page(request: Request, quote_id: str = Form(...), ep: Episode = Depends(html_episode)):
        order = store.place_order(ep, quote_id, "html")
        return render(request, "order.html", order=order, cart_count=0)

    @app.get("/orders", response_class=HTMLResponse)
    def orders_page(request: Request, ep: Episode = Depends(html_episode)):
        return render(request, "orders.html", orders=ep.orders, cart_count=sum(ep.cart.values()))

    # ------------------------------------------------------------------ JSON API (MCP)
    def pjson(p, detail=False):
        d = {"product_id": p.id, "title": p.title, "brand": p.brand, "category": p.category,
             "category_name": p.category_name, "price": p.price, "rating": p.rating,
             "review_count": p.review_count, "in_stock": p.in_stock}
        if detail:
            d.update(color=p.color, list_price=p.list_price, description=p.description, bullets=p.bullets)
        return d

    @app.get("/api/categories")
    def api_categories(ep: Episode = Depends(api_episode)):
        return {"categories": [{"id": k, "name": v} for k, v in catalog.categories.items()]}

    @app.get("/api/search")
    def api_search(q: str = "", category: str | None = None, min_price: float | None = None,
                   max_price: float | None = None, min_rating: float | None = None, in_stock_only: bool = False,
                   sort: str = "relevance", page: int = 1, ep: Episode = Depends(api_episode)):
        if sort not in SORTS:
            raise StoreError(f"sort must be one of {list(SORTS)}")
        items, total = store.search(ep, "api", query=q, category=category, min_price=min_price,
                                    max_price=max_price, min_rating=min_rating, in_stock_only=in_stock_only,
                                    sort=sort, page=page)
        return {"total": total, "page": page, "pages": max(1, -(-total // PAGE_SIZE)),
                "results": [pjson(p) for p in items]}

    @app.get("/api/product/{pid}")
    def api_product(pid: str, ep: Episode = Depends(api_episode)):
        return pjson(store.view_product(ep, pid, "api"), detail=True)

    @app.get("/api/cart")
    def api_cart(ep: Episode = Depends(api_episode)):
        return store.cart_view(ep)

    @app.post("/api/cart/add")
    def api_cart_add(req: AddReq, ep: Episode = Depends(api_episode)):
        return store.add_to_cart(ep, req.product_id, req.qty, "api")

    @app.post("/api/cart/update")
    def api_cart_update(req: AddReq, ep: Episode = Depends(api_episode)):
        return store.update_cart(ep, req.product_id, req.qty, "api")

    @app.post("/api/checkout")
    def api_checkout(ep: Episode = Depends(api_episode)):
        return store.checkout(ep, "api")

    @app.post("/api/checkout/place")
    def api_place(req: PlaceReq, ep: Episode = Depends(api_episode)):
        return store.place_order(ep, req.quote_id, "api")

    @app.get("/api/orders")
    def api_orders(ep: Episode = Depends(api_episode)):
        return {"orders": ep.orders}

    # ------------------------------------------------------------------ admin (harness only)
    @app.post("/admin/episodes", dependencies=[Depends(admin)])
    def admin_new(req: NewEpisodeReq):
        ep = store.new_episode(req.task_id, req.traps, req.episode_id)
        return {"episode_id": ep.id}

    @app.get("/admin/episodes/{eid}", dependencies=[Depends(admin)])
    def admin_get(eid: str):
        return store.snapshot(store.episode(eid))

    @app.post("/admin/episodes/{eid}/log", dependencies=[Depends(admin)])
    def admin_log(eid: str, req: LogReq):
        return store.episode(eid).log("harness", req.type, **req.data)

    return app


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--catalog", default=None)
    a = ap.parse_args()
    app = create_app(a.catalog)
    print(f"admin token: {app.state.admin_token}")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
