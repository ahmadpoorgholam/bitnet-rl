# BitNet Reinforcement Learning — White Paper

**Filling the empty RL slot in the 1.58-bit LLM training stack**

| Field | Value |
|-------|--------|
| **Project** | BitNet RL |
| **Author** | Ahmad Poorgholam |
| **Companion architecture** | [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`) / BitNet b1.58 |
| **Document class** | White paper — research framing and implementation agenda |
| **Version** | 1.0 — 2026-10-07 |
| **Empty artifact** | `rl/train_rl.py` (intentionally empty scaffold shipped with this repo) |

---

## Abstract

BitNet b1.58 established a native 1.58-bit Transformer stack (ternary weights $\{-1,0,+1\}$, 8-bit activations) with a three-stage post-training recipe: **pre-training → supervised fine-tuning (SFT) → direct preference optimization (DPO)**. The official BitNet b1.58 2B4T technical report explicitly leaves **reinforcement learning** (PPO, GRPO, and related policy-gradient methods) as **future work**. That vacant stage is the subject of this white paper.

We treat the missing RL stage as a first-class, distributeable artifact: an **empty** `rl/train_rl.py` file that marks where the architecture expects RL to land, and a concrete research program for filling it—covering (1) RL for BitNet *alignment and reasoning* (post-DPO), and (2) RL *with* BitNet as a frozen edge encoder (BitRL-style agents). The goal is a reproducible path from the empty scaffold to working PPO/GRPO and on-device policy learning on top of `bitnet.cpp`.

---

## 1. Motivation — why an empty RL file?

Released BitNet inference code (`bitnet.cpp`) optimizes **forward** kernels for W1.58A8. Training and alignment assets emphasize pre-training, SFT, and DPO. There is **no** shipped PPO/GRPO trainer in the official BitNet distribution.

The BitNet b1.58 2B4T report states that while PPO or GRPO can further improve mathematics and chain-of-thought reasoning, the published model relies solely on pre-training, SFT, and DPO; **exploration of reinforcement learning remains future work**.

This repository therefore **extracts and ships that empty slot** as an explicit file:

```text
rl/train_rl.py   # empty by design — the RL stage not yet filled in BitNet
```

An empty file is stronger than a comment in a paper: it is a contract for contributors and a visible gap in the architecture's training pipeline.

---

## 2. BitNet architecture (constraints for RL)

### 2.1 Core inductive bias

BitNet replaces `nn.Linear` with **BitLinear**:

- Weights quantized to ternary values via absmean quantization.
- Activations quantized to 8-bit (absmax, per-token) in the published W1.58A8 recipe.
- Training uses straight-through estimation (STE) so gradients flow through non-differentiable rounding.

Other components (RoPE, subln, ReLU² FFN in 2B4T, no biases) follow the published BitNet Transformer layout.

### 2.2 Official training stages today

| Stage | Role | Status in BitNet 2B4T |
|-------|------|------------------------|
| Pre-training | World knowledge on ~4T tokens | Done |
| SFT | Instruction / chat | Done |
| DPO | Preference alignment without a separate reward model | Done |
| **RL (PPO / GRPO / …)** | Reasoning / math / tool use via reward signals | **Empty — this project** |

DPO is preference optimization related to RLHF, but it is **not** on-policy reinforcement learning with a learned reward model and policy-gradient updates. The empty file marks that distinction.

### 2.3 Inference stack coupling

Any RL loop that *serves* BitNet for rollouts should prefer **`bitnet.cpp`** (CPU) or the official GPU kernels—not vanilla `transformers` matmul—if latency/energy claims matter. Training of LoRA/heads may still use BF16 master weights (`bitnet-b1.58-2B-4T-bf16`).

---

## 3. Two RL agendas on one architecture

### Agenda A — RL *for* BitNet (post-training the LLM)

**Goal:** Improve BitNet's own generation quality (math, CoT, instruction following) with PPO/GRPO after SFT/DPO.

**Sketch:**

1. Freeze or partially unfreeze BitLinear layers; prefer LoRA/adapters on BF16 masters then re-quantize, or STE-aware updates.
2. Reward model or rule-based verifiers for math/code.
3. On-policy rollouts; advantage estimates; clipped policy objective (PPO) or group-relative advantages (GRPO).
4. Keep KL to SFT/DPO reference to avoid collapse under ternary noise.

**Risks:** Quantization noise in value/advantage estimates; STE bias; need conservative clip / LR vs full-precision RLHF.

### Agenda B — RL *with* BitNet (edge agents)

**Goal:** Use a **frozen** BitNet backbone as a compact state encoder; train tiny policy/value heads on-device (BitRL line of work).

**Sketch:**

1. Serialize observation → text (or token ids).
2. Frozen BitNet → hidden state.
3. Trainable policy and value heads (~tens of thousands of parameters).
4. PPO (or actor–critic) on classic control, MuJoCo, and language-conditioned tasks.
5. Deploy with `bitnet.cpp` on Raspberry Pi–class hardware.

**Reported direction of prior BitRL-style results (literature):** large memory/energy wins with partial retention of FP16 task return; **value estimation** is the main bottleneck under extreme quantization; hybrid precision (e.g. higher-precision critic) is a practical remedy.

This white paper does **not** reproduce third-party numerical claims as our own locks; Agenda B is a research track to be measured in this repo's `evidence/` as experiments land.

---

## 4. Filling `rl/train_rl.py` — proposed API

The empty file should grow into a minimal, honest trainer interface:

```python
# Target shape (not yet implemented — file is empty by design)

def build_bitnet_policy(model_id: str, train_heads_only: bool = True):
    """Load BitNet backbone (+ optional LoRA) and policy/value heads."""

def rollout(env, policy, n_steps: int):
    """Collect on-policy trajectories (prefer bitnet.cpp for edge eval)."""

def ppo_update(batch, policy, clip_eps: float = 0.1, ent_coef: float = 0.01):
    """Clipped surrogate + value loss; conservative defaults for ternary noise."""

def train_rl(config):
    """Main entry: Agenda A (LLM rewards) or Agenda B (env rewards)."""
```

Implementation milestones:

1. **M0** — Empty scaffold + this white paper *(current)*.
2. **M1** — Agenda B CartPole PPO with frozen stub encoder (no BitNet weights required).
3. **M2** — Wire BF16 BitNet backbone + trainable heads; log returns/entropy/value loss.
4. **M3** — `bitnet.cpp` inference path for edge latency/energy measurements.
5. **M4** — Agenda A GRPO on a small math verifier set; compare to DPO-only baseline.

---

## 5. Theoretical notes (why RL is hard at 1.58 bits)

1. **Quantization as structured perturbation.** Ternary weights are not unstructured Gaussian noise; absmean ternary projection induces a bias that STE only partially corrects.
2. **Policy vs value asymmetry.** Policy scores can tolerate more noise than bootstrapped TD value targets; errors compound across temporal-difference updates.
3. **Exploration–stability trade-off.** Quantization entropy can help early exploration and hurt late refinement—suggesting adaptive precision or entropy schedules.
4. **Convergence practice.** Prefer smaller learning rates, tighter PPO clips, gradient clipping, and optionally a higher-precision critic under Agenda B.

---

## 6. Safety and scope

- Do **not** deploy Agenda B agents in safety-critical control without monitors (entropy/value-loss alarms, fallback policies).
- Agenda A RL can amplify reward hacking and unsafe completions; keep preference/safety filters and KL budgets.
- This repo starts from an **empty** trainer: no performance claims until locked metrics exist under `evidence/`.

---

## 7. Related work (pointers)

- **BitNet / BitNet b1.58** — Wang et al.; Ma et al.; BitNet b1.58 2B4T technical report ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)) — training stack and explicit RL-as-future-work statement.
- **bitnet.cpp** — Microsoft — official CPU/GPU inference for 1-bit LLMs.
- **DPO** — Rafailov et al. — preference stage currently used instead of full RLHF.
- **PPO / GRPO** — Schulman et al.; Shao et al. — candidate methods for the empty stage.
- **BitRL** — reinforcement learning with 1-bit quantized LMs for edge deployment ([arXiv:2604.24273](https://arxiv.org/abs/2604.24273)) — Agenda B inspiration; cite when reproducing numbers.

---

## 8. Final goal

**Ship a filled `rl/train_rl.py` that makes BitNet's missing RL stage executable**—first for edge policy heads on frozen 1.58-bit backbones, then for verifier-driven GRPO/PPO on BitNet itself—measured with locked returns, latency, memory, and energy, and kept honest about value-function limits under ternary quantization.

---

## Appendix — Repository layout

```text
README.md
LICENSE
docs/WHITE_PAPER.md      # this document
rl/train_rl.py          # EMPTY — extracted vacant RL stage of the BitNet stack
rl/README.md            # how to fill the empty file
evidence/               # reserved for future locked metrics
```

---

*End of BitNet RL White Paper v1.0*
