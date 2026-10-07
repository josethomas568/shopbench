"""Run the store in a background thread and talk to its admin API."""
from __future__ import annotations

import secrets
import socket
import threading
import time

import httpx
import uvicorn

from ..store.app import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StoreServer:
    def __init__(self, port: int | None = None, catalog: str | None = None):
        self.port = port or free_port()
        self.token = secrets.token_hex(16)
        self.app = create_app(catalog, admin_token=self.token)
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.http = httpx.Client(base_url=self.base_url, headers={"X-Admin-Token": self.token}, timeout=30)

    def __enter__(self) -> "StoreServer":
        self._thread.start()
        for _ in range(100):
            try:
                if self.http.post("/admin/episodes", json={"task_id": "__ping__"}).status_code == 200:
                    return self
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        raise RuntimeError("store did not start")

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self.http.close()

    # admin helpers
    def new_episode(self, task: dict) -> str:
        r = self.http.post("/admin/episodes", json={"task_id": task["id"], "traps": task.get("traps") or {}})
        r.raise_for_status()
        return r.json()["episode_id"]

    def snapshot(self, eid: str) -> dict:
        r = self.http.get(f"/admin/episodes/{eid}")
        r.raise_for_status()
        return r.json()

    def log(self, eid: str, type_: str, **data) -> None:
        self.http.post(f"/admin/episodes/{eid}/log", json={"type": type_, "data": data}).raise_for_status()
