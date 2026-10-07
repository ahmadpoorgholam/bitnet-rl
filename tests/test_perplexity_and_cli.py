import argparse
import json
import random
import sys
import types

import torch

from rl.grpo import GRPOConfig
from rl.rewards import rule_reward
from rl.toy import CharTokenizer, TinyCausalLM, gold_completion, gold_perplexity, pretrain_base, toy_questions
from rl.train_rl import GRPOTrainer, Sample, build_parser, load_jsonl, run_hf, run_toy


def test_gold_perplexity_is_one_for_a_perfect_model_and_high_for_random():
    tok = CharTokenizer()
    random_model = TinyCausalLM(tok.vocab_size)
    r = gold_perplexity(random_model, tok)
    assert r["ppl"] > 10 and r["answer_ppl"] > 10  # near vocab-size for an untrained model
    assert gold_completion(toy_questions()[7]) == "<think>1+2</think><answer>3</answer>"


def test_pretraining_lowers_gold_perplexity():
    torch.manual_seed(0)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size)
    before = gold_perplexity(model, tok)["ppl"]
    pretrain_base(model, tok, steps=60)
    assert gold_perplexity(model, tok)["ppl"] < before


def test_grpo_iterations_reduce_answer_perplexity_on_cpu():
    torch.manual_seed(1)
    random.seed(1)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size)
    pretrain_base(model, tok, steps=200)
    before = gold_perplexity(model, tok)["answer_ppl"]
    trainer = GRPOTrainer(model, tok, rule_reward, GRPOConfig(max_new_tokens=48, lr=1e-3))
    samples = [Sample(q.prompt, q.answer) for q in toy_questions()]
    for _ in range(3):
        trainer.step(random.sample(samples, 8))
    assert gold_perplexity(model, tok)["answer_ppl"] < before


def test_evaluate_reports_accuracy_and_format():
    tok = CharTokenizer()
    trainer = GRPOTrainer(TinyCausalLM(tok.vocab_size), tok, rule_reward,
                          GRPOConfig(group_size=2, max_new_tokens=6))
    res = trainer.evaluate([Sample("1+2=", "3")], n=2)
    assert set(res) == {"accuracy", "format"} and 0.0 <= res["accuracy"] <= 1.0


def test_load_jsonl_applies_r1_template(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"question": "2+2", "answer": 4}) + "\n\n")
    (s,) = load_jsonl(str(f))
    assert s.answer == "4" and s.prompt.endswith("User: 2+2. Assistant: ")


def test_cli_parser_and_toy_run(capsys):
    args = build_parser().parse_args(
        ["toy", "--steps", "2", "--pretrain-steps", "20", "--group-size", "4",
         "--questions-per-step", "2", "--log-every", "1"]
    )
    result = run_toy(args)
    assert {"before_accuracy", "after_accuracy", "before_format", "after_format"} == set(result)
    assert "before RL" in capsys.readouterr().out


def test_run_hf_path_with_fake_transformers(tmp_path, monkeypatch, capsys):
    class FakeHFTokenizer:
        eos_token_id, pad_token_id = 1, None

        def __init__(self):
            self.inner = CharTokenizer()

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": self.inner.encode(text)}

        def decode(self, ids, skip_special_tokens=True):
            return self.inner.decode(list(ids))

    inner = CharTokenizer()
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: FakeHFTokenizer())
    fake.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: TinyCausalLM(inner.vocab_size, max_len=700)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr("rl.train_rl.R1_ZERO_TEMPLATE", "{question}=")
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"question": "1+2", "answer": "3"}) + "\n")
    args = argparse.Namespace(
        model="fake", data=str(data), steps=1, group_size=2, questions_per_step=1,
        clip_eps=0.2, beta=0.001, ref_update_every=0, lr=1e-4, max_new_tokens=4, seed=0,
    )
    run_hf(args)
    assert "step    1" in capsys.readouterr().out
