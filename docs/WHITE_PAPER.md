# BitNet Reinforcement Learning — White Paper

**Filling the empty RL slot in the 1.58-bit LLM training stack with DeepSeek-style GRPO**

| Field | Value |
|-------|--------|
| **Project** | BitNet RL |
| **Author** | Ahmad Poorgholam |
| **Companion architecture** | [microsoft/BitNet](https://github.com/microsoft/BitNet) (`bitnet.cpp`) / BitNet b1.58 |
| **Document class** | White paper — research framing, implementation notes, and toy-scale results |
| **Version** | 1.1 — 2026-10-07 (v1.0 shipped `rl/train_rl.py` empty; v1.1 implements it) |
| **Implementation** | `rl/train_rl.py` + `rl/grpo.py` — Group Relative Policy Optimization from DeepSeekMath / DeepSeek-R1 |

---

## Abstract

BitNet b1.58 established a native 1.58-bit Transformer stack (ternary weights $\{-1,0,+1\}$, 8-bit activations) with a three-stage post-training recipe: **pre-training → supervised fine-tuning (SFT) → direct preference optimization (DPO)**. The official BitNet b1.58 2B4T technical report explicitly leaves **reinforcement learning** (PPO, GRPO, and related policy-gradient methods) as **future work**. That vacant stage is the subject of this white paper.

This repository first shipped that slot as an **empty** `rl/train_rl.py`. It is now filled with a from-scratch PyTorch implementation of **GRPO** as described in the DeepSeek-R1 paper (arXiv:2501.12948): critic-free group-relative advantages, a clipped surrogate, the k3 KL penalty to a reference policy, and the rule-based accuracy + format rewards of R1-Zero. A self-contained CPU demo exercises it on a tiny transformer built either from ordinary linear layers or from BitNet-style `BitLinear` layers (ternary weights, int8 activations, straight-through estimator). Two research agendas remain: (1) RL *for* BitNet reasoning post-DPO, and (2) RL *with* BitNet as a frozen edge encoder (BitRL-style agents). Real BitNet checkpoints have **not** yet been trained with this code.

---

## 1. Motivation — the empty RL slot

Released BitNet inference code (`bitnet.cpp`) optimizes **forward** kernels for W1.58A8. Training and alignment assets emphasize pre-training, SFT, and DPO. There is **no** shipped PPO/GRPO trainer in the official BitNet distribution.

The BitNet b1.58 2B4T report states that while PPO or GRPO can further improve mathematics and chain-of-thought reasoning, the published model relies solely on pre-training, SFT, and DPO; **exploration of reinforcement learning remains future work**.

This repository therefore began by shipping that slot as an explicit, empty `rl/train_rl.py` — a visible gap in the training pipeline — and now fills it with GRPO (Section 4).

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
| **RL (PPO / GRPO / …)** | Reasoning / math / tool use via reward signals | **Empty upstream; GRPO implemented here (toy-validated)** |

DPO is preference optimization related to RLHF, but it is **not** on-policy reinforcement learning with a learned reward model and policy-gradient updates. This repository's trainer implements the on-policy variant.

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

## 4. Implementation — GRPO in `rl/`

### 4.1 Algorithm (from the DeepSeek-R1 paper, Sec. 2.1)

For each question $q$, sample a group of $G$ outputs from the current policy and maximise

$$\mathcal{J}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\Big(\min\big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\varepsilon,1+\varepsilon)A_i\big)-\beta\,\mathbb{D}_{KL}(\pi_\theta\Vert\pi_{ref})\Big)$$

with ratio $\rho_i=\pi_\theta(o_i|q)/\pi_{\theta_{old}}(o_i|q)$, the unbiased KL estimator $\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$, and the **critic-free advantage** $A_i=(R_i-\mathrm{mean}(R))/\mathrm{std}(R)$ computed within the group. There is no value model, which removes the component the paper identifies as most memory-hungry and most sensitive to tuning in PPO — and, under extreme quantization, the component most prone to bootstrapping error (Section 5).

### 4.2 Rewards (R1-Zero, Sec. 2.2)

`Reward = Reward_acc + Reward_format`, equal weight, purely rule-based: accuracy checks the extracted `<answer>` (or `\boxed{}`) against ground truth, format checks that reasoning and answer sit in `<think>…</think><answer>…</answer>`. No neural reward model is used. The R1-Zero prompt template is provided as `R1_ZERO_TEMPLATE`.

### 4.3 Code map

| File | Role |
|------|------|
| `rl/grpo.py` | Group advantages, k3 KL, clipped GRPO loss, group sampling, token log-probs |
| `rl/rewards.py` | Accuracy and format rewards, R1-Zero template |
| `rl/train_rl.py` | `GRPOTrainer` (collect → update → optional reference refresh) and the CLI (`toy`, `hf`) |
| `rl/bitlinear.py` | BitNet b1.58-style `BitLinear`: absmean ternary weights, absmax int8 activations, STE |
| `rl/numpy_ref.py` | Torch-free NumPy reference of the same math, with a hand-derived gradient and a tabular GRPO bandit |
| `rl/toy.py` | Char-level tiny transformer, toy task, weak base policy and gold-completion perplexity for CPU demos |
| `scripts/validate_cpu.py`, `scripts/make_badges.py` | 3-iteration CPU perplexity check; badge JSON generated from measured files |
| `tests/` | 51 tests (loss math, NumPy cross-checks, perplexity, CLI, trainer); 98% line coverage of `rl/` |

### 4.4 Deviations from DeepSeek's setup

R1-Zero used $\eta=3\times10^{-6}$, $\beta=0.001$, $G=16$, temperature 1, 32 questions per step, a reference refresh every 400 steps, and $\varepsilon=10$ in the first RL stage of R1. The defaults here follow $\beta$, $G$ and temperature; `clip_eps` defaults to the conventional 0.2 and the refresh interval is configurable (`--ref-update-every`). Not implemented: distributed rollout/training infrastructure, KV-cached generation, sequence packing, the language-consistency reward, and R1's multi-stage pipeline (cold-start SFT, rejection sampling). Loss aggregation is per-token ratio/KL averaged within each sequence, then across the group (the DeepSeekMath form).

### 4.5 Toy-scale results (single seed, CPU)

A 2-layer, 64-dim char-level transformer is pre-trained as a *weak base policy*: it knows the answer format but answers single-digit addition questions ($a+b$ with $0\le a,b\le4$) correctly only ~30% of the time and breaks the format ~30% of the time. GRPO then runs for 200 steps (8 questions × 16 samples per step, lr $10^{-3}$, seed 0). Evaluation samples 8 completions for each of the 25 questions.

| Policy layers | Accuracy before → after | Format compliance before → after |
|---------------|-------------------------|----------------------------------|
| `nn.Linear` | 0.140 → 0.905 (rerun: 0.840) | 0.650 → 1.000 (rerun: 0.995) |
| `BitLinear` (ternary W, int8 A) | 0.135 → 0.945 | 0.700 → 0.995 |

The two `nn.Linear` figures are the same seed on different thread counts (runs are not bit-reproducible); in-training accuracy at step 200 was 0.961 and 0.977. Evaluation is only 200 samples, so differences of a few points are noise.

Reproduce with `python -m rl.train_rl toy --steps 200` and `... --bitlinear`. Section 4.6 adds the NumPy cross-checks and the 3-iteration perplexity check.

**What this does and does not show.** It shows the implementation learns from rule-based group-relative rewards and that BitLinear layers with STE do not prevent it. It does not show anything about BitNet-scale language models, reasoning emergence, or the value-bottleneck effects discussed in Section 5; those need real checkpoints and compute. One seed (with at most two runs per configuration) is not a statistical claim.

**A useful failure.** When the base policy answered *randomly* (never correct more often than chance), GRPO collapsed to answering the single most frequent sum for every question (accuracy plateau ≈ 0.2 for all learning rates tried) — a genuine reward optimum for a policy whose answer token carries no information about the question. RL amplifies capability the base policy already has; it did not create it here. This matches the R1 paper's premise of starting from a strong base model.

### 4.6 Validation without a GPU

**Math on paper.** `rl/numpy_ref.py` re-implements the objective in pure NumPy, including a hand-derived gradient (surrogate gradient $\rho A$ unless clipping is active; KL gradient $1-e^{\log\pi_{ref}-\log\pi_\theta}$). The tests check it against PyTorch for loss values (several clip/KL settings) and gradients (autograd and central finite differences).

**Update direction.** A one-context softmax policy trained with exactly this update (`tabular_grpo`) lowers the perplexity $1/p(\text{correct})$ of the rewarded output from 6.0 to about 2.3 in 3 iterations; all 200 seeds tried improved. Groups with identical rewards leave the policy unchanged, as the advantage is zero.

**Three iterations on the toy transformer.** `scripts/validate_cpu.py` pre-trains the weak base policy, measures perplexity on the *correct, well-formatted* completion of every question, runs 3 GRPO iterations (8 questions x 16 samples, lr $10^{-3}$) and measures again, for 8 seeds each:

| Policy layers | Answer-digit perplexity (mean) | Whole-completion perplexity (mean) | Seeds improved |
|---------------|--------------------------------|------------------------------------|----------------|
| `nn.Linear` | 4.20 → 3.20 (−24%) | 1.052 → 1.036 (−1.5%) | 8 of 8 (both metrics) |
| `BitLinear` | 4.58 → 4.11 (−10%) | 1.054 → 1.041 (−1.3%) | 8 of 8 (both metrics) |

Whole-completion perplexity is dominated by fixed format characters the base model already predicts, so the answer-digit perplexity is the informative metric. Raw per-seed numbers are in `evidence/cpu_validation.json`; the README badges are generated from that file and from the coverage report.

**Limits.** This is a 25-question arithmetic toy, so it validates the implementation and the update direction, not BitNet-scale behaviour. Perplexity is measured on the training questions (there is no held-out split). BitLinear improves less than full precision within 3 iterations; with 8 seeds that gap is suggestive, not established.

### 4.7 Milestones

1. **M0** — Empty scaffold + white paper. *Done.*
2. **M1** — GRPO core, rule-based rewards, toy task with ordinary and BitLinear layers, NumPy reference, CPU perplexity check, tests. *Done (this version).*
3. **M2** — Run `hf` mode on a BF16 BitNet master checkpoint with a verifiable math set (the code path is exercised with a tiny random Llama; real BitNet weights untested; BitNet 2B4T needs the model card's `transformers` revision). *Open.*
4. **M3** — KV-cached or `bitnet.cpp`-served rollouts for speed; latency/energy measurements. *Open.*
5. **M4** — Compare against the DPO-only baseline on math benchmarks. *Open.*

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
- Results in this repo are toy-scale; no performance claims about BitNet checkpoints exist until locked metrics are added under `evidence/`.

---

## 7. Related work (pointers)

- **BitNet / BitNet b1.58** — Wang et al.; Ma et al.; BitNet b1.58 2B4T technical report ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)) — training stack and explicit RL-as-future-work statement.
- **bitnet.cpp** — Microsoft — official CPU/GPU inference for 1-bit LLMs.
- **DPO** — Rafailov et al. — preference stage currently used instead of full RLHF.
- **PPO / GRPO** — Schulman et al.; Shao et al. — candidate methods for the empty stage.
- **BitRL** — reinforcement learning with 1-bit quantized LMs for edge deployment ([arXiv:2604.24273](https://arxiv.org/abs/2604.24273)) — Agenda B inspiration; cite when reproducing numbers.

---

## 8. Final goal

**Make BitNet's missing RL stage executable at real scale** (the trainer now exists; real checkpoints are the open step)—first for edge policy heads on frozen 1.58-bit backbones, then for verifier-driven GRPO/PPO on BitNet itself—measured with locked returns, latency, memory, and energy, and kept honest about value-function limits under ternary quantization.

---

## Appendix — Repository layout

```text
README.md
LICENSE
requirements.txt
docs/WHITE_PAPER.md      # this document
rl/train_rl.py           # GRPOTrainer + CLI (was empty in v1.0)
rl/grpo.py               # GRPO math and sampling
rl/rewards.py            # rule-based accuracy + format rewards
rl/bitlinear.py          # BitNet-style BitLinear (ternary W, int8 A, STE)
rl/toy.py                # toy model/task for CPU demos
rl/numpy_ref.py          # torch-free NumPy reference + tabular GRPO
scripts/                 # validate_cpu.py, make_badges.py
badges/                  # shields.io endpoint JSON (generated)
evidence/                # cpu_validation.json (measured)
tests/                   # 51 unit, cross-check and smoke tests
.github/workflows/ci.yml # CPU CI
```

---

*End of BitNet RL White Paper v1.1*
