"""LLM providers behind one interface.

Conversation format (provider-neutral):
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Providers:
  anthropic   Anthropic Messages API with native tool use (needs ANTHROPIC_API_KEY)
  openai      any OpenAI-compatible /v1/chat/completions endpoint (OpenAI, vLLM, Ollama, llama.cpp ...)
              tool_mode="native" uses the tools parameter; tool_mode="json" puts tool specs in the
              system prompt and parses a JSON action from the reply (for small or fine-tuned models
              served without a tool-call parser)
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field

import httpx

# USD per million tokens (input, output). Prices change; check your provider and override with
# --price-in/--price-out or by editing this table.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-5": (5.0, 25.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0

    def add(self, o: "Usage") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))


@dataclass
class Turn:
    text: str
    tool_calls: list[dict]
    usage: Usage
    raw: dict = field(default_factory=dict)


class LLM:
    model: str
    price_in: float | None = None
    price_out: float | None = None

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        raise NotImplementedError

    def cost(self, u: Usage) -> float | None:
        pin, pout = self.price_in, self.price_out
        if pin is None or pout is None:
            hit = next((v for k, v in PRICES.items() if self.model.startswith(k)), None)
            if hit is None:
                return None
            pin, pout = hit
        # cache reads bill at 10% and cache writes at 125% of the input price (Anthropic convention)
        return (u.input_tokens * pin + u.cache_read_tokens * pin * 0.1 + u.cache_write_tokens * pin * 1.25
                + u.output_tokens * pout) / 1e6


def _retry(fn, tries: int = 5):
    for i in range(tries):
        try:
            return fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (429, 500, 502, 503, 529) and i < tries - 1:
                time.sleep(2 ** i + 1)
                continue
            raise
        except httpx.TransportError:
            if i < tries - 1:
                time.sleep(2 ** i + 1)
                continue
            raise


class AnthropicLLM(LLM):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0, cache: bool = True):
        self.model = model
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the anthropic provider")
        self.url = (base_url or os.environ.get("SHOPBENCH_ANTHROPIC_URL") or "https://api.anthropic.com").rstrip("/")
        self.max_tokens, self.temperature, self.cache = max_tokens, temperature, cache
        self.http = httpx.Client(timeout=120)

    def _convert(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": m["content"]}]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no output)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and all(b["type"] == "tool_result" for b in out[-1]["content"]):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        if self.cache and out:
            # cache the growing prefix: mark the last block of the final message
            out[-1]["content"][-1] = {**out[-1]["content"][-1], "cache_control": {"type": "ephemeral"}}
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        tools_payload = [dict(t) for t in tools]
        if self.cache and tools_payload:
            tools_payload[-1] = {**tools_payload[-1], "cache_control": {"type": "ephemeral"}}
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "system": [{"type": "text", "text": system, **({"cache_control": {"type": "ephemeral"}} if self.cache else {})}],
                "messages": self._convert(messages), "tools": tools_payload}
        t0 = time.time()

        def call():
            r = self.http.post(f"{self.url}/v1/messages", json=body,
                               headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"})
            r.raise_for_status()
            return r.json()
        data = _retry(call)
        u = data.get("usage", {})
        usage = Usage(u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cache_read_input_tokens", 0) or 0,
                      u.get("cache_creation_input_tokens", 0) or 0, 1, time.time() - t0)
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [{"id": b["id"], "name": b["name"], "args": b.get("input") or {}}
                 for b in data["content"] if b["type"] == "tool_use"]
        return Turn(text, calls, usage, data)


JSON_PROTOCOL = """
## How to act
Reply with your brief reasoning, then exactly one action as a JSON object on its own line:
{"tool": "<tool name>", "args": {...}}
Available tools:
"""


def parse_json_action(text: str) -> dict | None:
    """Find the last {...} object in text that has a "tool" key."""
    for m in reversed(list(re.finditer(r"\{", text))):
        depth, i = 0, m.start()
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and "tool" in obj:
                    return {"id": "call_" + uuid.uuid4().hex[:8], "name": obj["tool"], "args": obj.get("args") or {}}
                break
    return None


def json_assistant_content(m: dict) -> str:
    """Assistant turn as text: reasoning plus one JSON action line (the json tool-mode protocol)."""
    text = (m.get("text") or "").strip()
    if not m.get("tool_calls"):
        return text
    if parse_json_action(text):  # model already wrote its action in-protocol
        return text
    c = m["tool_calls"][0]
    action = json.dumps({"tool": c["name"], "args": c["args"]}, ensure_ascii=False)
    return f"{text}\n{action}" if text else action


def tools_as_text(tools: list[dict]) -> str:
    return "\n".join(f"- {t['name']}: {t['description']} args schema: {json.dumps(t['input_schema'].get('properties', {}))}"
                     for t in tools)


class OpenAICompatLLM(LLM):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 tool_mode: str = "native", max_tokens: int = 1024, temperature: float = 0.0):
        self.model = model
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "none")
        assert tool_mode in ("native", "json")
        self.tool_mode, self.max_tokens, self.temperature = tool_mode, max_tokens, temperature
        self.http = httpx.Client(timeout=300)

    def _convert(self, system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        if self.tool_mode == "json":
            system = system + "\n" + JSON_PROTOCOL + tools_as_text(tools)
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                if self.tool_mode == "json":
                    out.append({"role": "assistant", "content": json_assistant_content(m)})
                else:
                    msg = {"role": "assistant", "content": m.get("text") or None}
                    if m["tool_calls"]:
                        msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                              "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                             for c in m["tool_calls"]]
                    out.append(msg)
            elif m["role"] == "tool":
                if self.tool_mode == "json":
                    out.append({"role": "user", "content": f"Result of {m['name']}:\n{m['content']}"})
                else:
                    out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> Turn:
        body = {"model": self.model, "messages": self._convert(system, messages, tools),
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.tool_mode == "native":
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["input_schema"]}} for t in tools]
        t0 = time.time()

        def call():
            r = self.http.post(f"{self.url}/chat/completions", json=body, headers={"Authorization": f"Bearer {self.key}"})
            r.raise_for_status()
            return r.json()
        data = _retry(call)
        u = data.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        usage = Usage(u.get("prompt_tokens", 0) - cached, u.get("completion_tokens", 0), cached, 0, 1, time.time() - t0)
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        if self.tool_mode == "native":
            calls = []
            for c in msg.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {"__invalid_json__": c["function"]["arguments"]}
                calls.append({"id": c["id"], "name": c["function"]["name"], "args": args})
        else:
            a = parse_json_action(text)
            calls = [a] if a else []
        return Turn(text, calls, usage, data)


def make_llm(provider: str, model: str, **kw) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model, **{k: v for k, v in kw.items() if k in ("api_key", "base_url", "max_tokens", "temperature", "cache")})
    if provider == "openai":
        return OpenAICompatLLM(model, **{k: v for k, v in kw.items() if k in ("api_key", "base_url", "tool_mode", "max_tokens", "temperature")})
    raise ValueError(provider)
