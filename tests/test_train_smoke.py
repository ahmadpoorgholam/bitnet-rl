import random

import torch

from bitnet_rl.grpo import GRPOConfig
from bitnet_rl.rewards import rule_reward
from bitnet_rl.toy import CharTokenizer, TinyCausalLM, pretrain_base, toy_questions
from bitnet_rl.train_rl import GRPOTrainer, Sample


def test_trainer_step_updates_policy_and_keeps_reference_frozen():
    torch.manual_seed(0)
    random.seed(0)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size)
    pretrain_base(model, tok, steps=30)
    cfg = GRPOConfig(group_size=8, max_new_tokens=40, lr=1e-3, questions_per_step=4)
    trainer = GRPOTrainer(model, tok, rule_reward, cfg)
    ref_before = [p.clone() for p in trainer.ref.parameters()]
    pol_before = [p.clone() for p in model.parameters()]
    samples = [Sample(q.prompt, q.answer) for q in toy_questions()]
    stats = trainer.step(random.sample(samples, 4))
    assert {"reward", "accuracy", "format", "kl", "clip_frac", "loss"} <= stats.keys()
    assert any(not torch.equal(a, b) for a, b in zip(pol_before, model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(ref_before, trainer.ref.parameters()))


def test_reference_refresh_copies_latest_policy():
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size)
    cfg = GRPOConfig(group_size=4, max_new_tokens=8, ref_update_every=1, questions_per_step=2)
    trainer = GRPOTrainer(model, tok, rule_reward, cfg)
    trainer.step([Sample("1+2=", "3"), Sample("0+1=", "1")])
    assert all(torch.equal(a, b) for a, b in zip(trainer.ref.parameters(), model.parameters()))
