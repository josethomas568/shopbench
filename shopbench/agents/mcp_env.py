"""Synchronous wrapper around an MCP stdio client session to the ShopBench MCP server.

The MCP client's anyio contexts must be entered and exited in the same task, so each
episode's session lives in one long-running task that serves tool calls from a queue.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import os
import sys
import threading
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[2]
_CLOSE = object()


class MCPEnv:
    """One MCP server process per episode (the episode id is passed in its environment)."""

    def __init__(self, base_url: str, command: list[str] | None = None, timeout: float = 60):
        self.base_url = base_url
        self.command = command or [sys.executable, "-m", "shopbench.mcp_server"]
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._queue: asyncio.Queue | None = None
        self._task: cf.Future | None = None
        self.tools: list[dict] = []

    async def _serve(self, episode_id: str, ready: cf.Future) -> None:
        params = StdioServerParameters(command=self.command[0], args=self.command[1:], cwd=str(ROOT),
                                       env={**os.environ, "SHOPBENCH_URL": self.base_url,
                                            "SHOPBENCH_EPISODE": episode_id})
        try:
            async with stdio_client(params) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.tools = [{"name": t.name, "description": t.description or "",
                                   "input_schema": t.input_schema} for t in listed.tools]
                    self._queue = asyncio.Queue()
                    ready.set_result(True)
                    while True:
                        item = await self._queue.get()
                        if item is _CLOSE:
                            return
                        name, args, fut = item
                        try:
                            fut.set_result(await session.call_tool(name, args))
                        except Exception as e:  # report to caller, keep serving
                            fut.set_exception(e)
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            raise

    def reset(self, episode_id: str) -> str:
        self._shutdown()
        ready: cf.Future = cf.Future()
        self._task = asyncio.run_coroutine_threadsafe(self._serve(episode_id, ready), self._loop)
        ready.result(self.timeout)
        return ("You are connected to the ShopBench Market MCP server. Use its tools to shop. "
                f"Store tools: {', '.join(t['name'] for t in self.tools)}.")

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        assert self._queue is not None, "call reset() first"
        fut: cf.Future = cf.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, args, fut))
        try:
            r = fut.result(self.timeout)
        except Exception as e:
            return False, json.dumps({"error": f"tool call failed: {e}"})
        text = "\n".join(getattr(c, "text", "") for c in r.content)
        ok = not getattr(r, "is_error", False)
        try:
            ok = ok and "error" not in json.loads(text)
        except (ValueError, TypeError):
            pass
        return ok, text

    def _shutdown(self) -> None:
        if self._task is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _CLOSE)
            try:
                self._task.result(15)
            except Exception:
                pass
        self._task, self._queue = None, None

    def close(self) -> None:
        self._shutdown()
        self._loop.call_soon_threadsafe(self._loop.stop)
