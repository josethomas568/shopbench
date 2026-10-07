import pytest

from shopbench.agents.agent import Agent, AgentConfig
from shopbench.agents.browser_env import BrowserEnv
from shopbench.agents.scripted import ScriptedPolicy
from shopbench.harness.grader import grade


@pytest.mark.parametrize("mode", ["html", "axtree", "axtree_pruned"])
def test_oracle_in_browser(server, catalog, tasks, mode):
    env = BrowserEnv(server.base_url, mode)
    try:
        for tid in ["std-06", "price-01", "amb-02", "cart-01"]:
            rec = Agent(ScriptedPolicy("oracle", "browser"), AgentConfig(interface="browser", obs_mode=mode)).run(
                tasks[tid], server, env)
            g = grade(tasks[tid], server.snapshot(rec.episode_id), rec.to_dict(), catalog)
            assert g["success"], (mode, tid, g, rec.error)
    finally:
        env.close()


def test_pruning_shrinks_observations(server):
    env = BrowserEnv(server.base_url, "html")
    try:
        env.reset(server.new_episode({"id": "t"}))
        env.act("goto", {"url": "/search?q=usb+c+cable"})
        sizes = {}
        for m in ["html", "axtree", "axtree_pruned"]:
            env.obs_mode = m
            sizes[m] = len(env.observe())
        assert sizes["axtree_pruned"] * 4 < sizes["html"]
        assert "[ref=e" in env.observe()
        assert env.act("goto", {"url": "https://example.com"}).ok is False
        assert env.act("goto", {"url": "/admin/episodes/x"}).ok is False
    finally:
        env.close()
