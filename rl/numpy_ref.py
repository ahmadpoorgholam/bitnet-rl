"""Pure-NumPy reference for the GRPO math (no torch, no GPU).

Serves two purposes:

1. An independent implementation of the DeepSeek-R1 objective (arXiv:2501.12948,
   Eq. 1-3) with an analytic gradient, used in ``tests/test_numpy_reference.py`` to
   cross-check ``rl.grpo`` (PyTorch) numerically.
2. ``tabular_grpo``: a one-context softmax-policy bandit trained with exactly the GRPO
   update, small enough to run anywhere in milliseconds. It checks "on paper" that
   the update direction raises the probability (lowers the perplexity) of the
   rewarded output.

Run ``python -m rl.numpy_ref`` for a 3-iteration demonstration.
"""

from __future__ import annotations

import numpy as np


def group_advantages(rewards: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    if r.shape[-1] < 2:
        raise ValueError("GRPO needs a group size of at least 2")
    # ddof=1 matches torch.Tensor.std, which rl.grpo uses
    return (r - r.mean(-1, keepdims=True)) / (r.std(-1, ddof=1, keepdims=True) + eps)


def kl_k3(logp: np.ndarray, ref_logp: np.ndarray) -> np.ndarray:
    d = ref_logp - logp
    return np.exp(d) - d - 1.0


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps, beta) -> float:
    """Negative GRPO objective for one group; arrays are ``[G, T]`` (advantages ``[G]``)."""
    ratio = np.exp(logp - old_logp)
    a = np.asarray(advantages)[:, None]
    surrogate = np.minimum(ratio * a, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    per_token = surrogate - beta * kl_k3(logp, ref_logp)
    per_seq = (per_token * mask).sum(-1) / np.maximum(mask.sum(-1), 1.0)
    return float(-per_seq.mean())


def grpo_loss_grad(logp, old_logp, ref_logp, advantages, mask, clip_eps, beta) -> np.ndarray:
    """Analytic d(loss)/d(logp), derived by hand (not by autodiff).

    Surrogate: gradient ``ratio * A`` unless clipping is active (``A>0`` and ratio above
    ``1+eps``, or ``A<0`` and ratio below ``1-eps``). KL term: ``d k3 / d logp = 1 - exp(ref - logp)``.
    """
    ratio = np.exp(logp - old_logp)
    a = np.asarray(advantages)[:, None]
    active = ~(((a > 0) & (ratio > 1 + clip_eps)) | ((a < 0) & (ratio < 1 - clip_eps)))
    d_obj = ratio * a * active - beta * (1.0 - np.exp(ref_logp - logp))
    n_seq = logp.shape[0]
    return -(d_obj * mask) / (np.maximum(mask.sum(-1, keepdims=True), 1.0) * n_seq)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def tabular_grpo(
    n_actions: int = 6,
    correct: int = 2,
    group_size: int = 16,
    iters: int = 3,
    lr: float = 1.0,
    beta: float = 0.001,
    clip_eps: float = 0.2,
    seed: int = 0,
) -> dict[str, list[float]]:
    """GRPO on a single-context softmax policy; reward is 1 for the ``correct`` action.

    ``perplexity`` is that of the rewarded output, ``1 / p(correct)``.
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros(n_actions)
    ref_logp_all = np.log(_softmax(theta))
    hist = {"p_correct": [], "perplexity": [], "mean_reward": []}

    def record() -> None:
        p = _softmax(theta)[correct]
        hist["p_correct"].append(float(p))
        hist["perplexity"].append(float(1.0 / p))

    record()
    for _ in range(iters):
        p = _softmax(theta)
        acts = rng.choice(n_actions, size=group_size, p=p)
        rewards = (acts == correct).astype(np.float64)
        hist["mean_reward"].append(float(rewards.mean()))
        adv = group_advantages(rewards)
        logp = np.log(p)[acts][:, None]
        grad_logp = grpo_loss_grad(
            logp, logp, ref_logp_all[acts][:, None], adv, np.ones_like(logp), clip_eps, beta
        )[:, 0]
        # d logp(a) / d theta = onehot(a) - p
        onehot = np.eye(n_actions)[acts]
        theta = theta - lr * (grad_logp[:, None] * (onehot - p)).sum(0)
        record()
    return hist


if __name__ == "__main__":
    for seed in range(3):
        h = tabular_grpo(seed=seed)
        ppl = " -> ".join(f"{x:.3f}" for x in h["perplexity"])
        print(f"seed {seed}: perplexity of rewarded output over 3 iterations: {ppl}")
