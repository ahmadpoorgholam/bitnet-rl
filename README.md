# BitNet RL: GRPO for the missing reinforcement-learning stage of 1.58-bit LLMs

[![CI](https://github.com/ahmadpoorgholam/bitnet-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadpoorgholam/bitnet-rl/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fahmadpoorgholam%2Fbitnet-rl%2Fmain%2Fbadges%2Fcoverage.json)](#validation)
[![CPU perplexity](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fahmadpoorgholam%2Fbitnet-rl%2Fmain%2Fbadges%2Fcpu_perplexity.json)](evidence/cpu_validation.json)
[![method](https://img.shields.io/badge/method-GRPO%20(arXiv%3A2501.12948)-blue)](https://arxiv.org/abs/2501.12948)
[![math](https://img.shields.io/badge/math-NumPy%20%E2%87%84%20PyTorch%20gradient--checked-brightgreen)](tests/test_numpy_reference.py)
[![scale](https://img.shields.io/badge/validated%20at-toy%20scale%20(CPU)-orange)](#validation)
[![real model](https://img.shields.io/badge/real%20BitNet%20checkpoint-not%20yet%20validated-lightgrey)](#validation)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Keywords:** BitNet, BitNet b1.58, 1-bit LLM, 1.58-bit quantization, ternary weights, quantization-aware training, straight-through estimator, reinforcement learning, GRPO, Group Relative Policy Optimization, DeepSeek-R1, DeepSeek-R1-Zero, DeepSeekMath, RLVR (reinforcement learning with verifiable rewards), rule-based rewards, policy gradient, PPO, critic-free RL, KL regularization, reasoning LLM, post-training, efficient LLM, edge AI, PyTorch, NumPy, reproducible research, master's thesis.

## Summary

BitNet b1.58 is a 1.58-bit large language model: every weight is ternary (-1, 0, +1), which makes inference cheap on CPUs and edge devices. Its public 2B4T technical report describes a three-stage training recipe, pre-training, supervised fine-tuning (SFT) and direct preference optimization (DPO), and lists **reinforcement learning as future work**. The official distribution ships no RL trainer.

This project does two things:

1. It records that gap by starting from an **empty** `rl/train_rl.py` (the first commit).
2. It fills the gap with **GRPO**, the critic-free RL algorithm DeepSeek used for DeepSeekMath and DeepSeek-R1-Zero ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)), implemented from scratch in PyTorch. It then validates the implementation as far as a CPU-only machine allows.

Honest scope: everything here is validated on a tiny arithmetic task. **No real BitNet checkpoint has been trained with this code.** That is the open next step, not a result.

## The white paper

[`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) (v1.1) is the written framing of the project. In order, it covers:

| Section | Content |
|---------|---------|
| 1. Motivation | Why the RL slot in the BitNet stack is empty and why it matters (math and chain-of-thought gains that SFT and DPO do not give) |
| 2. BitNet architecture | BitLinear, absmean ternary weights, int8 activations, straight-through estimator, and what each implies for RL |
| 3. Two RL agendas | **A:** RL *for* BitNet (post-train the LLM). **B:** RL *with* BitNet (frozen 1.58-bit encoder plus small policy heads on edge devices) |
| 4. Implementation | The GRPO objective, rewards, code map, deviations from DeepSeek's setup, toy results, the CPU validation, milestones |
| 5. Theory notes | Why RL is harder at 1.58 bits: structured quantization noise, policy-versus-value asymmetry, exploration/stability trade-off |
| 6. Safety and scope | Reward hacking, no claims beyond toy scale |
| 7. Related work | BitNet b1.58 2B4T, bitnet.cpp, DPO, PPO/GRPO, BitRL |

The white paper is the source of the research questions. The code in this repo implements Agenda A at toy scale and prepares the ground for measuring it on real checkpoints.

## What was done

1. **Located the gap.** Read the BitNet material to confirm RL is absent from the training stack, and made that visible as an empty scaffold.
2. **Studied the method.** Took GRPO from the DeepSeek-R1 paper: group-normalised advantages (no value model), clipped surrogate, k3 KL penalty to a reference policy, and the R1-Zero accuracy plus format rule rewards.
3. **Implemented it** (`rl/`): loss and sampling (`grpo.py`), rewards and R1-Zero prompt template (`rewards.py`), trainer and CLI that also accepts Hugging Face causal LMs (`train_rl.py`), and a BitNet-style `BitLinear` layer with ternary weights, int8 activations and STE (`bitlinear.py`).
4. **Built a CPU test bed** (`toy.py`): a 2-layer char-level transformer, an addition task, and a deliberately weak base policy (about 30% correct) so RL has something to amplify.
5. **Verified the math independently** (`numpy_ref.py`): a torch-free NumPy implementation with a hand-derived gradient, cross-checked against PyTorch autograd and finite differences.
6. **Measured what a CPU can show** (`scripts/validate_cpu.py`): three GRPO iterations, perplexity before and after, 8 seeds, with and without BitLinear.
7. **Made the evidence auditable**: badges are generated from measured files (`scripts/make_badges.py`), CI runs the tests and the checks, and the white paper states the limits next to the results.

## Contributions

What this work contributes, and what it does not.

1. **A reference GRPO implementation for the BitNet setting.** An open, readable, tested trainer that plugs into BitLinear-style models and into any Hugging Face causal LM, filling a stage the official BitNet release leaves empty.
2. **A CPU-checkable validation protocol for RL code when no GPU is available.** Three independent layers of evidence: an independent NumPy implementation that agrees with PyTorch on values and gradients; a tabular bandit showing the update lowers the perplexity of the rewarded output (200 of 200 seeds); and a 3-iteration perplexity check on a toy transformer (16 of 16 runs improved).
3. **Evidence-driven badges.** Coverage and perplexity badges that are computed from stored measurements instead of typed, next to explicit "toy scale" and "not yet validated on a real checkpoint" badges.
4. **An early, small, empirical observation on RL with ternary weights.** BitLinear with STE does not prevent GRPO from learning at toy scale (accuracy about 0.14 to 0.9 in 200 steps). It improved answer perplexity less than full-precision layers in 3 iterations (-10% versus -24%), which is a hypothesis to test at scale, not an established effect.
5. **A documented negative result.** When the base policy answers at random, GRPO collapses to the most frequent answer. RL amplifies capability the base model already has, consistent with R1's premise of starting from a strong base.
6. **A research framing** (white paper): two RL agendas on one architecture, with risks specific to 1.58-bit training.

Not claimed: a new RL algorithm (GRPO is DeepSeek's), any improvement to BitNet itself, or any result on a real language model. Those are the work left for a full thesis (see below).

## Validation

No GPU is available, so validation is what a CPU can honestly show. The badges above are generated from measured files by `scripts/make_badges.py`.

| Check | What it shows | Result |
|-------|---------------|--------|
| Method on paper | Loss, advantage, KL and clipping follow R1 Eq. 1-3 | Independent NumPy implementation agrees with PyTorch on loss values and gradients (autograd and finite differences), see `tests/test_numpy_reference.py` |
| Update direction | The GRPO update raises probability and lowers perplexity of the rewarded output | NumPy tabular bandit: perplexity of the correct output 6.0 to about 2.3 after 3 iterations, 200 of 200 seeds improved |
| 3-iteration CPU perplexity (toy transformer) | Perplexity on the gold completions falls after only 3 GRPO iterations | Answer-digit perplexity: linear -24%, BitLinear -10% (8 seeds each, **16 of 16 runs improved**). Whole-completion perplexity moves only about 1.5% because most tokens are fixed format characters. See `evidence/cpu_validation.json` |
| Coverage | Lines of `rl/` exercised by the tests | 98% (`pytest --cov`), 51 tests |

What this does **not** show: anything about real BitNet checkpoints, language-model quality, or reasoning ability. Perplexity is measured on the 25 training questions of an arithmetic toy (no held-out split), and runs are noisy at the few-point level.

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

## Repository layout

| Path | Role |
|------|------|
| `rl/grpo.py` | Group advantages, k3 KL, clipped loss, sampling, token log-probs |
| `rl/rewards.py` | R1-Zero accuracy and format rewards, prompt template |
| `rl/train_rl.py` | `GRPOTrainer` and CLI (was empty in the first commit) |
| `rl/bitlinear.py` | BitNet-style ternary-weight / int8-activation layer with STE |
| `rl/numpy_ref.py` | Torch-free reference of the same math plus a tabular GRPO bandit |
| `rl/toy.py` | Char-level toy transformer, task and gold-completion perplexity |
| `scripts/` | `validate_cpu.py` (3-iteration check), `make_badges.py` |
| `evidence/`, `badges/` | Measured results and the badge files generated from them |
| `tests/` | Unit, cross-check, perplexity, CLI and smoke tests |
| `docs/WHITE_PAPER.md` | The white paper |

## Limits and future work (what a full thesis would add)

- **Real model (milestone M2):** run `hf` mode on a BF16 BitNet master checkpoint with a verifiable math set. Needs a GPU.
- **Baselines (M4):** compare against the DPO-only model on math benchmarks, and against full-precision RL on the same base.
- **Held-out evaluation** and more seeds with confidence intervals; the current perplexity is on training questions.
- **Speed (M3):** KV-cached or `bitnet.cpp`-served rollouts; latency, memory and energy measurements.
- **Agenda B:** frozen BitNet encoder with small policy/value heads on control tasks.
- Not implemented: distributed training, language-consistency reward, R1's multi-stage pipeline (cold-start SFT, rejection sampling).

## Citation

If you use this work, cite it with the metadata in [`CITATION.cff`](CITATION.cff), and cite the original methods:

- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
- Ma et al., *BitNet b1.58 2B4T Technical Report*, [arXiv:2504.12285](https://arxiv.org/abs/2504.12285).

Companion inference architecture: [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`).

## License

MIT, see [LICENSE](LICENSE).
