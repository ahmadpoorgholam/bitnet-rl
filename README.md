# BitNet RL

DeepSeek-style **GRPO** reinforcement learning for the vacant RL stage of the [BitNet](https://github.com/microsoft/BitNet) (1.58-bit LLM) training stack, plus a white paper.

BitNet b1.58 2B4T is trained with pre-training → SFT → DPO; its technical report leaves PPO/GRPO-style RL as future work. This repo started as an empty `rl/train_rl.py` marking that gap and now implements the algorithm DeepSeek used for R1-Zero ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).

## What is implemented

- **GRPO** (`rl/grpo.py`): group-normalised advantages (no critic), clipped surrogate, k3 KL penalty to a reference policy, per-token loss averaged per sequence.
- **Rule-based rewards** (`rl/rewards.py`): accuracy + format, equal weight, as in R1-Zero; includes the R1-Zero prompt template.
- **Trainer + CLI** (`rl/train_rl.py`): rollout, update, periodic reference refresh; works with any causal LM whose forward returns logits (including Hugging Face models).
- **BitLinear** (`rl/bitlinear.py`): BitNet-style ternary-weight / int8-activation layer with STE, usable in the toy policy.
- **Toy task** (`rl/toy.py`): tiny char-level transformer for CPU demos.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                              # 18 tests
python -m rl.train_rl toy --steps 200            # ordinary linear layers
python -m rl.train_rl toy --steps 200 --bitlinear  # ternary weights, int8 activations
```

Toy result (single seed, CPU, ~1 min): evaluation accuracy 0.14 → 0.84–0.91 (two runs) and format compliance 0.65 → ~1.00 with `nn.Linear`; 0.135 → 0.945 and 0.70 → 0.995 with `BitLinear` (one run). Runs are not bit-reproducible across thread counts, and evaluation uses only 200 samples, so expect a few points of noise.

On a Hugging Face model with a JSONL file of `{"question": ..., "answer": ...}` rows:

```bash
python -m rl.train_rl hf --model <hf-id> --data train.jsonl --lr 3e-6 --group-size 16
```

## Status and limits

- Validated at toy scale only. The `hf` path runs end to end on a tiny random Llama, but **no real BitNet checkpoint has been trained** with it yet.
- Generation has no KV cache and groups are processed one question at a time, so it is meant for small models and correctness, not throughput.
- Not implemented: distributed training, the language-consistency reward, R1's multi-stage pipeline (cold-start SFT, rejection sampling).

See [`docs/WHITE_PAPER.md`](docs/WHITE_PAPER.md) for the design, deviations from DeepSeek's setup, results and open milestones.

Companion inference architecture: [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`).

## License

MIT — see [LICENSE](LICENSE).
