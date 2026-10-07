"""CPU-only validation: does perplexity on correct completions fall after 3 GRPO iterations?

For each seed and policy type (ordinary linear / BitLinear) this builds the toy weak base
policy, measures perplexity on the gold completions, runs a few GRPO iterations, and
measures again. Results are written to ``evidence/cpu_validation.json``.

    python scripts/validate_cpu.py --seeds 5 --iterations 3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.grpo import GRPOConfig  # noqa: E402
from rl.rewards import rule_reward  # noqa: E402
from rl.toy import CharTokenizer, TinyCausalLM, gold_perplexity, pretrain_base, toy_questions  # noqa: E402
from rl.train_rl import GRPOTrainer, Sample  # noqa: E402


def run_one(seed: int, bitlinear: bool, iterations: int, lr: float, questions_per_step: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    tok = CharTokenizer()
    model = TinyCausalLM(tok.vocab_size, bitlinear=bitlinear)
    pretrain_base(model, tok, seed=seed)
    before = gold_perplexity(model, tok)
    cfg = GRPOConfig(max_new_tokens=48, lr=lr, questions_per_step=questions_per_step)
    trainer = GRPOTrainer(model, tok, rule_reward, cfg)
    samples = [Sample(q.prompt, q.answer) for q in toy_questions()]
    for _ in range(iterations):
        trainer.step(random.sample(samples, questions_per_step))
    after = gold_perplexity(model, tok)
    return {"seed": seed, "bitlinear": bitlinear, "before": before, "after": after}


def summarise(runs: list[dict]) -> dict:
    out = {}
    for key, label in ((False, "linear"), (True, "bitlinear")):
        rs = [r for r in runs if r["bitlinear"] is key]
        if not rs:
            continue
        entry = {"n_seeds": len(rs)}
        for metric in ("ppl", "answer_ppl"):
            b = [r["before"][metric] for r in rs]
            a = [r["after"][metric] for r in rs]
            entry[metric] = {
                "before_mean": statistics.fmean(b),
                "after_mean": statistics.fmean(a),
                "relative_change": statistics.fmean(a) / statistics.fmean(b) - 1.0,
                "seeds_improved": sum(x < y for x, y in zip(a, b)),
            }
        out[label] = entry
    return out


def _round(obj, nd: int = 4):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--questions-per-step", type=int, default=8)
    p.add_argument("--out", default="evidence/cpu_validation.json")
    args = p.parse_args()

    t0 = time.time()
    runs = []
    for seed in range(args.seeds):
        for bitlinear in (False, True):
            r = run_one(seed, bitlinear, args.iterations, args.lr, args.questions_per_step)
            runs.append(r)
            print(
                f"seed={seed} bitlinear={bitlinear!s:5} "
                f"ppl {r['before']['ppl']:.3f}->{r['after']['ppl']:.3f} "
                f"answer_ppl {r['before']['answer_ppl']:.3f}->{r['after']['answer_ppl']:.3f}",
                flush=True,
            )
    result = {
        "config": {
            "iterations": args.iterations, "lr": args.lr, "group_size": 16,
            "questions_per_step": args.questions_per_step, "device": "cpu",
            "torch": torch.__version__,
        },
        "summary": summarise(runs),
        "runs": runs,
        "wall_seconds": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_round(result), indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
