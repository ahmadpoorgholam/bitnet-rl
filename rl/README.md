# `rl/` — GRPO for BitNet

| File | Role |
|------|------|
| `train_rl.py` | `GRPOTrainer` and CLI (`python -m rl.train_rl toy|hf ...`). Was empty in the first commit. |
| `grpo.py` | Group advantages, k3 KL, clipped loss, group sampling, token log-probs |
| `rewards.py` | R1-Zero accuracy + format rewards and prompt template |
| `bitlinear.py` | BitNet-style `BitLinear` (ternary W, int8 A, STE) |
| `toy.py` | Tiny char-level transformer and toy task for CPU demos |

Run everything from the repository root so `rl` is importable as a package. See [`../docs/WHITE_PAPER.md`](../docs/WHITE_PAPER.md) Section 4 for details.
