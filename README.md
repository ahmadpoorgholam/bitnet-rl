# BitNet RL

[![CI](https://github.com/ahmadpoorgholam/bitnet-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadpoorgholam/bitnet-rl/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fahmadpoorgholam%2Fbitnet-rl%2Fmain%2Fbadges%2Fcoverage.json)](#validation)
[![CPU perplexity](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fahmadpoorgholam%2Fbitnet-rl%2Fmain%2Fbadges%2Fcpu_perplexity.json)](evidence/cpu_validation.json)
[![method](https://img.shields.io/badge/method-GRPO%20(arXiv%3A2501.12948)-blue)](https://arxiv.org/abs/2501.12948)
[![math](https://img.shields.io/badge/math-NumPy%20%E2%87%84%20PyTorch%20gradient--checked-brightgreen)](tests/test_numpy_reference.py)
[![scale](https://img.shields.io/badge/validated%20at-toy%20scale%20(CPU)-orange)](#validation)
[![real model](https://img.shields.io/badge/real%20BitNet%20checkpoint-not%20yet%20validated-lightgrey)](#validation)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

DeepSeek-style **GRPO** reinforcement learning for the vacant RL stage of the [BitNet](https://github.com/microsoft/BitNet) (1.58-bit LLM) training stack, plus a white paper.

BitNet b1.58 2B4T is trained with pre-training → SFT → DPO; its technical report leaves PPO/GRPO-style RL as future work. This repo started as an empty `rl/train_rl.py` marking that gap and now implements the algorithm DeepSeek used for R1-Zero ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).

## What is implemented

- **GRPO** (`rl/grpo.py`): group-normalised advantages (no critic), clipped surrogate, k3 KL penalty to a reference policy, per-token loss averaged per sequence.
- **Rule-based rewards** (`rl/rewards.py`): accuracy + format, equal weight, as in R1-Zero; includes the R1-Zero prompt template.
- **Trainer + CLI** (`rl/train_rl.py`): rollout, update, periodic reference refresh; works with any causal LM whose forward returns logits (including Hugging Face models).
- **BitLinear** (`rl/bitlinear.py`): BitNet-style ternary-weight / int8-activation layer with STE, usable in the toy policy.
- **NumPy reference** (`rl/numpy_ref.py`): independent, torch-free implementation of the same math with a hand-derived gradient, plus a tabular GRPO bandit.
- **Toy task** (`rl/toy.py`): tiny char-level transformer and a gold-completion perplexity metric for CPU demos.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q --cov=rl                     # 51 tests, CPU only
python -m rl.numpy_ref                           # NumPy-only GRPO, no torch needed
python scripts/validate_cpu.py --seeds 8         # 3 GRPO iterations, perplexity before/after
python -m rl.train_rl toy --steps 200            # ordinary linear layers
python -m rl.train_rl toy --steps 200 --bitlinear  # ternary weights, int8 activations
```

On a Hugging Face model with a JSONL file of `{"question": ..., "answer": ...}` rows:

```bash
python -m rl.train_rl hf --model <hf-id> --data train.jsonl --lr 3e-6 --group-size 16
```

## Validation

No GPU is available, so validation is what a CPU can honestly show. The badges above are generated from measured files by `scripts/make_badges.py`, not typed by hand.

| Check | What it shows | Result |
|-------|---------------|--------|
| Method on paper | Loss, advantage, KL and clipping follow R1 Eq. 1-3 | Independent NumPy implementation agrees with PyTorch on loss values and gradients (autograd and finite differences) — `tests/test_numpy_reference.py` |
| Update direction | The GRPO update raises probability / lowers perplexity of the rewarded output | NumPy tabular bandit: perplexity of the correct output 6.0 → ≈2.3 after 3 iterations, 200 of 200 seeds improved |
| 3-iteration CPU perplexity (toy transformer) | Perplexity on the gold completions falls after only 3 GRPO iterations | Answer-digit perplexity: linear −24%, BitLinear −10% (8 seeds each, **16 of 16 runs improved**). Whole-completion perplexity moves only ≈1.5% because most tokens are fixed format characters — `evidence/cpu_validation.json` |
| Coverage | Lines of `rl/` exercised by the tests | 98% (`pytest --cov`) |

What this does **not** show: anything about real BitNet checkpoints, language-model quality, or reasoning ability. The perplexity here is measured on a 25-question arithmetic toy, so the "validity" is that the implementation is mathematically right and learns in the expected direction, not that BitNet gets better at math. Training a real checkpoint needs a GPU; the `hf` path is exercised only with a fake and a tiny random Llama.

## Status and limits

- Generation has no KV cache and groups are processed one question at a time, so it is meant for small models and correctness, not throughput.
- Not implemented: distributed training, the language-consistency reward, R1's multi-stage pipeline (cold-start SFT, rejection sampling).
- Toy end-to-end runs (200 steps) reach evaluation accuracy ≈0.84–0.95 from ≈0.14; they are noisy (few-point run-to-run variation) and not bit-reproducible across thread counts.

See [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) for the design, deviations from DeepSeek's setup, results and open milestones.

Companion inference architecture: [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`).

## License

MIT — see [LICENSE](LICENSE).
