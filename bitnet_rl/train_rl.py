"""GRPO training entry point (DeepSeek-R1-Zero style RL) for BitNet-style models.

Usage (``bitnet-rl`` is installed by ``pip install bitnet-rl``)::

    python -m bitnet_rl.train_rl toy --steps 150 --bitlinear
    python -m bitnet_rl.train_rl hf --model <hf-id> --data train.jsonl --lr 3e-6

``toy`` is a self-contained CPU demo. ``hf`` runs the same trainer on any Hugging Face
causal LM; ``--data`` is JSONL with ``question`` and ``answer`` fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from bitnet_rl.grpo import GRPOConfig, group_advantages, grpo_loss, sample_group, token_logprobs
from bitnet_rl.rewards import R1_ZERO_TEMPLATE, RewardResult, rule_reward


class TokenizerLike(Protocol):
    eos_id: int
    pad_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass
class Sample:
    prompt: str
    answer: str


@dataclass
class Rollout:
    prompt_ids: list[int]
    completion: torch.Tensor
    mask: torch.Tensor
    old_logp: torch.Tensor
    ref_logp: torch.Tensor
    advantages: torch.Tensor
    rewards: list[RewardResult]


class GRPOTrainer:
    """Critic-free GRPO: sample a group per question, normalise rewards within the group."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: TokenizerLike,
        reward_fn: Callable[[str, str], RewardResult],
        cfg: GRPOConfig,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model, self.tok, self.reward_fn, self.cfg = model, tokenizer, reward_fn, cfg
        self.opt = optimizer or torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.ref = self._frozen_copy()
        self.steps = 0

    def _frozen_copy(self) -> torch.nn.Module:
        ref = copy.deepcopy(self.model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    @torch.no_grad()
    def collect(self, sample: Sample) -> Rollout:
        cfg = self.cfg
        prompt_ids = self.tok.encode(sample.prompt)
        completion, mask = sample_group(
            self.model, prompt_ids, cfg.group_size, cfg.max_new_tokens,
            self.tok.eos_id, self.tok.pad_id, cfg.temperature,
        )
        texts = [
            self.tok.decode(row[: int(m.sum())].tolist()) for row, m in zip(completion, mask)
        ]
        rewards = [self.reward_fn(t, sample.answer) for t in texts]
        r = torch.tensor([x.reward for x in rewards], device=completion.device)
        return Rollout(
            prompt_ids, completion, mask,
            token_logprobs(self.model, prompt_ids, completion, cfg.temperature),
            token_logprobs(self.ref, prompt_ids, completion, cfg.temperature),
            group_advantages(r, cfg.adv_eps), rewards,
        )

    def update(self, rollouts: list[Rollout]) -> dict[str, float]:
        cfg = self.cfg
        agg: dict[str, list[float]] = {"loss": [], "kl": [], "clip_frac": [], "ratio": []}
        self.model.train()
        for _ in range(cfg.inner_epochs):
            order = random.sample(range(len(rollouts)), len(rollouts))
            for i in range(0, len(order), cfg.minibatch_questions):
                chunk = [rollouts[j] for j in order[i : i + cfg.minibatch_questions]]
                self.opt.zero_grad()
                for ro in chunk:
                    logp = token_logprobs(self.model, ro.prompt_ids, ro.completion, cfg.temperature)
                    loss, st = grpo_loss(
                        logp, ro.old_logp, ro.ref_logp, ro.advantages, ro.mask,
                        cfg.clip_eps, cfg.beta,
                    )
                    (loss / len(chunk)).backward()
                    agg["loss"].append(float(loss.detach()))
                    for k, v in st.items():
                        agg[k].append(v)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
        return {k: sum(v) / max(len(v), 1) for k, v in agg.items()}

    def step(self, samples: list[Sample]) -> dict[str, float]:
        rollouts = [self.collect(s) for s in samples]
        stats = self.update(rollouts)
        allr = [x for ro in rollouts for x in ro.rewards]
        stats.update(
            reward=sum(x.reward for x in allr) / len(allr),
            accuracy=sum(x.accuracy for x in allr) / len(allr),
            format=sum(x.format for x in allr) / len(allr),
            resp_len=float(sum(ro.mask.sum() for ro in rollouts) / sum(len(ro.mask) for ro in rollouts)),
            zero_std_groups=sum(float(ro.advantages.abs().max() == 0) for ro in rollouts) / len(rollouts),
        )
        self.steps += 1
        if self.cfg.ref_update_every and self.steps % self.cfg.ref_update_every == 0:
            self.ref = self._frozen_copy()
        return stats

    @torch.no_grad()
    def evaluate(self, samples: list[Sample], n: int = 8) -> dict[str, float]:
        results = []
        for s in samples:
            ids = self.tok.encode(s.prompt)
            comp, mask = sample_group(
                self.model, ids, n, self.cfg.max_new_tokens, self.tok.eos_id, self.tok.pad_id,
                self.cfg.temperature,
            )
            for row, m in zip(comp, mask):
                results.append(self.reward_fn(self.tok.decode(row[: int(m.sum())].tolist()), s.answer))
        return {
            "accuracy": sum(r.accuracy for r in results) / len(results),
            "format": sum(r.format for r in results) / len(results),
        }


def _log(step: int, st: dict[str, float]) -> None:
    keys = ["reward", "accuracy", "format", "kl", "clip_frac", "resp_len", "loss"]
    print(f"step {step:4d} | " + " ".join(f"{k}={st[k]:.3f}" for k in keys), flush=True)


def run_toy(args: argparse.Namespace) -> dict[str, float]:
    from bitnet_rl.toy import CharTokenizer, TinyCausalLM, pretrain_base, toy_questions

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size, bitlinear=args.bitlinear)
    base_loss = pretrain_base(model, tok, steps=args.pretrain_steps, seed=args.seed)
    print(f"base policy pretrained (loss {base_loss:.3f}, bitlinear={args.bitlinear})")

    cfg = GRPOConfig(
        group_size=args.group_size, clip_eps=args.clip_eps, beta=args.beta, lr=args.lr,
        max_new_tokens=48, questions_per_step=args.questions_per_step,
        ref_update_every=args.ref_update_every,
    )
    samples = [Sample(q.prompt, q.answer) for q in toy_questions()]
    trainer = GRPOTrainer(model, tok, rule_reward, cfg)
    before = trainer.evaluate(samples)
    print(f"before RL: accuracy={before['accuracy']:.3f} format={before['format']:.3f}")
    for step in range(1, args.steps + 1):
        stats = trainer.step(random.sample(samples, cfg.questions_per_step))
        if step % args.log_every == 0 or step == 1:
            _log(step, stats)
    after = trainer.evaluate(samples)
    print(f"after  RL: accuracy={after['accuracy']:.3f} format={after['format']:.3f}")
    return {"before_accuracy": before["accuracy"], "after_accuracy": after["accuracy"],
            "before_format": before["format"], "after_format": after["format"]}


class _HFTokenizer:
    def __init__(self, tok) -> None:
        self.tok = tok
        self.eos_id = tok.eos_token_id
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


def load_jsonl(path: str) -> list[Sample]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Sample(R1_ZERO_TEMPLATE.format(question=r["question"]), str(r["answer"])) for r in rows]


def run_hf(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    cfg = GRPOConfig(
        group_size=args.group_size, clip_eps=args.clip_eps, beta=args.beta, lr=args.lr,
        max_new_tokens=args.max_new_tokens, questions_per_step=args.questions_per_step,
        ref_update_every=args.ref_update_every,
    )
    samples = load_jsonl(args.data)
    trainer = GRPOTrainer(model, _HFTokenizer(tok), rule_reward, cfg)
    for step in range(1, args.steps + 1):
        _log(step, trainer.step(random.sample(samples, min(cfg.questions_per_step, len(samples)))))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("toy", "hf"):
        s = sub.add_parser(name)
        s.add_argument("--steps", type=int, default=150)
        s.add_argument("--group-size", type=int, default=16)
        s.add_argument("--questions-per-step", type=int, default=8)
        s.add_argument("--clip-eps", type=float, default=0.2)
        s.add_argument("--beta", type=float, default=0.001)
        s.add_argument("--ref-update-every", type=int, default=0)
        s.add_argument("--seed", type=int, default=0)
    toy, hf = sub.choices["toy"], sub.choices["hf"]
    toy.add_argument("--lr", type=float, default=1e-3)
    toy.add_argument("--bitlinear", action="store_true")
    toy.add_argument("--pretrain-steps", type=int, default=400)
    toy.add_argument("--log-every", type=int, default=10)
    hf.add_argument("--lr", type=float, default=3e-6)
    hf.add_argument("--model", required=True)
    hf.add_argument("--data", required=True)
    hf.add_argument("--max-new-tokens", type=int, default=512)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_toy(args) if args.mode == "toy" else run_hf(args)


if __name__ == "__main__":
    main()
