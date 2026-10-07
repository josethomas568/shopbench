import json

import pytest

from shopbench.agents.agent import Agent, AgentConfig
from shopbench.agents.scripted import ScriptedPolicy
from shopbench.sft.build_dataset import episode_examples


def test_examples_match_inference_format(server, tasks):
    rec = Agent(ScriptedPolicy("oracle", "mcp"), AgentConfig(interface="mcp", history="last_k", k=2)).run(tasks["multi-01"], server)
    exs = episode_examples(rec.to_dict(), "last_k", 2)
    assert len(exs) == rec.steps
    for ex in exs:
        m = ex["messages"]
        assert m[0]["role"] == "system" and '{"tool":' in m[0]["content"]
        assert m[-1]["role"] == "assistant" and json.loads(m[-1]["content"].splitlines()[-1])["tool"]
    last = exs[-1]["messages"]
    assert sum("[earlier observation omitted" in x["content"] for x in last) >= 1  # history elision applied


def test_lora_smoke(tmp_path, server, tasks):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    rows = []
    for tid in ["std-01", "cart-01", "price-01"]:
        rec = Agent(ScriptedPolicy("oracle", "mcp"), AgentConfig(interface="mcp")).run(tasks[tid], server)
        rows += episode_examples(rec.to_dict(), "full", 3)
    data = tmp_path / "sft.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))

    # tiny offline model + tokenizer, just to exercise the training loop
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tk.train_from_iterator([m["content"] for r in rows for m in r["messages"]],
                           trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<pad>", "<eos>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<eos>{% endfor %}"
                         "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    mdir = tmp_path / "tiny"
    tok.save_pretrained(mdir)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096)).save_pretrained(mdir)

    import sys
    from shopbench.sft import train_lora
    out = tmp_path / "adapter"
    sys.argv = ["train_lora", "--data", str(data), "--model", str(mdir), "--out", str(out), "--max-len", "2048",
                "--head", "256", "--grad-accum", "2", "--max-steps", "3", "--val-frac", "0.34"]
    train_lora.main()
    assert (out / "adapter_config.json").exists()
    losses = [json.loads(l)["loss"] for l in open(out / "train_log.jsonl")]
    assert len(losses) == 3 and all(l == l for l in losses)
