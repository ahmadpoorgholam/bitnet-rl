# `bitnet_rl/` — GRPO for BitNet

| File | Role |
|------|------|
| `train_rl.py` | `GRPOTrainer` and CLI (`python -m bitnet_rl.train_rl toy|hf ...`, or `bitnet-rl` once installed). Was empty (as `rl/train_rl.py`) in the first commit. |
| `grpo.py` | Group advantages, k3 KL, clipped loss, group sampling, token log-probs |
| `rewards.py` | R1-Zero accuracy + format rewards and prompt template |
| `bitlinear.py` | BitNet-style `BitLinear` (ternary W, int8 A, STE) |
| `numpy_ref.py` | Torch-free NumPy reference of the GRPO math + tabular bandit (`python -m bitnet_rl.numpy_ref`) |
| `toy.py` | Tiny char-level transformer, toy task and gold-completion perplexity for CPU demos |

Run from the repository root, or `pip install .` so `bitnet_rl` is importable as a package. See [`../docs/WHITE_PAPER.md`](../docs/WHITE_PAPER.md) Section 4 for details.
