"""Drive full episodes through the Anthropic and OpenAI provider code paths with a mocked HTTP API.

The mock 'model' replays the oracle's actions, so a pass means request/response conversion,
tool-call plumbing and the json action protocol all work end to end."""
import json

import httpx
import pytest

from shopbench.agents.agent import Agent, AgentConfig
from shopbench.agents.llm import AnthropicLLM, OpenAICompatLLM, parse_json_action
from shopbench.agents.scripted import ScriptedPolicy
from shopbench.harness.grader import grade


def _last_observation(msgs: list[dict]) -> str:
    m = msgs[-1]
    c = m["content"]
    if isinstance(c, list):  # anthropic blocks
        b = c[-1]
        return b["content"] if b["type"] == "tool_result" else b["text"]
    return c.split(":\n", 1)[1] if c.startswith("Result of ") else c


def anthropic_handler(policy):
    def handle(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert req.headers["x-api-key"] == "test" and body["tools"] and body["system"]
        call = policy._next(_last_observation(body["messages"]))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "thinking"},
                                                     {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}],
                                         "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 50}})
    return handle


def openai_handler(policy, mode):
    def handle(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        call = policy._next(_last_observation(body["messages"]))
        if mode == "native":
            assert body["tools"]
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}]}
        else:
            assert "tools" not in body and '{"tool":' in body["messages"][0]["content"]
            msg = {"role": "assistant", "content": "Next step.\n" + json.dumps({"tool": call["name"], "args": call["args"]})}
        return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {"prompt_tokens": 120, "completion_tokens": 12}})
    return handle


@pytest.mark.parametrize("provider", ["anthropic", "openai-native", "openai-json"])
@pytest.mark.parametrize("tid", ["std-06", "price-02", "amb-04"])
def test_provider_roundtrip(server, catalog, tasks, provider, tid):
    task = tasks[tid]
    policy = ScriptedPolicy("oracle", "mcp")
    if provider == "anthropic":
        llm = AnthropicLLM("claude-sonnet-4-5", api_key="test")
        llm.http = httpx.Client(transport=httpx.MockTransport(anthropic_handler(policy)))
    else:
        mode = provider.split("-")[1]
        llm = OpenAICompatLLM("local-model", base_url="http://fake/v1", tool_mode=mode)
        llm.http = httpx.Client(transport=httpx.MockTransport(openai_handler(policy, mode)))
    # the mock model needs the oracle's privileged plan; the agent itself only sees the LLM
    orig_run = Agent.run

    agent = Agent(llm, AgentConfig(interface="mcp"))
    eid_holder = {}
    orig_new = server.new_episode

    def new_episode(t):
        eid = orig_new(t)
        policy.bind(t, server, eid)
        eid_holder["eid"] = eid
        return eid
    server.new_episode = new_episode
    try:
        rec = orig_run(agent, task, server)
    finally:
        server.new_episode = orig_new
    g = grade(task, server.snapshot(rec.episode_id), rec.to_dict(), catalog)
    assert g["success"], (g, rec.error)
    assert rec.usage.calls == rec.steps and rec.usage.input_tokens > 0
    if provider == "anthropic":
        assert rec.cost_usd and rec.cost_usd > 0


def test_parse_json_action():
    assert parse_json_action('I will search.\n{"tool": "search_products", "args": {"query": "a {b}"}}')["args"] == {"query": "a {b}"}
    assert parse_json_action("no action here {not json}") is None
