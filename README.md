# BitNet RL

White paper and **empty** RL trainer scaffold for the vacant reinforcement-learning stage of the [BitNet](https://github.com/microsoft/BitNet) (1.58-bit LLM) training stack.

## Why this exists

BitNet b1.58 2B4T is trained with **pre-training → SFT → DPO**. Official docs state that PPO/GRPO-style **reinforcement learning remains future work**. This repository extracts that empty stage as a real file and documents how to fill it.

| Path | Role |
|------|------|
| [`rl/train_rl.py`](rl/train_rl.py) | **Empty** by design — the missing RL entrypoint |
| [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) | White paper: motivation, two agendas, milestones |
| [`rl/README.md`](rl/README.md) | Short note on the empty file |

## Quick start

```bash
git clone https://github.com/ahmadpoorgholam/bitnet-rl.git
cd bitnet-rl
# Read the white paper, then implement rl/train_rl.py
```

Companion inference architecture: [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`).

## License

MIT — see [LICENSE](LICENSE).
