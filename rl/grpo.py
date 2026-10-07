"""Group Relative Policy Optimization (GRPO), as used to train DeepSeek-R1(-Zero).

References
----------
- Shao et al., 2024, "DeepSeekMath" (introduces GRPO).
- DeepSeek-AI, 2025, "DeepSeek-R1" (arXiv:2501.12948), Sec. 2.1 / Supp. A.3.

Objective (R1 paper, Eq. 1-3), maximised per question ``q`` with a group of
``G`` outputs sampled from the old policy::

    J = 1/G sum_i [ min(r_i A_i, clip(r_i, 1-eps, 1+eps) A_i) - beta * KL(pi_theta || pi_ref) ]
    r_i = pi_theta(o_i|q) / pi_theta_old(o_i|q)
    KL  = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1          (Schulman's k3 estimator)
    A_i = (R_i - mean(R)) / std(R)                            (no value model / critic)

Following DeepSeekMath, the ratio, KL and clipping are applied per token and
averaged over each output's tokens before averaging over the group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    group_size: int = 16
    clip_eps: float = 0.2
    beta: float = 0.001
    lr: float = 1e-5
    temperature: float = 1.0
    max_new_tokens: int = 64
    questions_per_step: int = 8
    minibatch_questions: int = 4
    inner_epochs: int = 1
    max_grad_norm: float = 1.0
    ref_update_every: int = 0
    adv_eps: float = 1e-4


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Group-normalised advantages. ``rewards`` has shape ``[..., G]`` with ``G >= 2``."""
    if rewards.shape[-1] < 2:
        raise ValueError("GRPO needs a group size of at least 2")
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token unbiased, non-negative KL(pi_theta || pi_ref) estimator (R1 Eq. 2)."""
    d = ref_logp - logp
    return d.exp() - d - 1.0


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Negative GRPO objective for one group.

    ``logp`` / ``old_logp`` / ``ref_logp`` / ``mask``: ``[G, T]`` per-token values;
    ``advantages``: ``[G]``. ``old_logp`` and ``ref_logp`` must not require grad.
    """
    ratio = (logp - old_logp).exp()
    adv = advantages.unsqueeze(-1)
    surrogate = torch.min(ratio * adv, ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv)
    kl = kl_k3(logp, ref_logp)
    per_token = surrogate - beta * kl

    denom = mask.sum(dim=-1).clamp(min=1.0)
    per_seq = (per_token * mask).sum(dim=-1) / denom
    loss = -per_seq.mean()

    with torch.no_grad():
        n_tok = mask.sum().clamp(min=1.0)
        clipped = ((ratio > 1 + clip_eps) & (adv > 0)) | ((ratio < 1 - clip_eps) & (adv < 0))
        stats = {
            "kl": float((kl * mask).sum() / n_tok),
            "clip_frac": float((clipped.float() * mask).sum() / n_tok),
            "ratio": float((ratio * mask).sum() / n_tok),
        }
    return loss, stats


def _logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def sample_group(
    model: torch.nn.Module,
    prompt_ids: list[int],
    group_size: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``group_size`` completions for one prompt.

    Returns ``(completion_ids [G, T], mask [G, T])``; the mask includes the EOS token.
    Works with any causal LM whose forward returns logits (or an object with ``.logits``).
    No KV cache is used, so cost per token grows with sequence length.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = torch.tensor(prompt_ids, device=device).unsqueeze(0).repeat(group_size, 1)
    finished = torch.zeros(group_size, dtype=torch.bool, device=device)
    tokens, masks = [], []
    for _ in range(max_new_tokens):
        logits = _logits(model, ids)[:, -1].float() / temperature
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        tokens.append(nxt)
        masks.append(~finished)
        finished = finished | (nxt == eos_id)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        if bool(finished.all()):
            break
    model.train(was_training)
    return torch.stack(tokens, dim=1), torch.stack(masks, dim=1).float()


def token_logprobs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    completion: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token log-probabilities of ``completion`` ``[G, T]`` given the prompt."""
    group, _ = completion.shape
    prompt = torch.tensor(prompt_ids, device=completion.device).unsqueeze(0).repeat(group, 1)
    ids = torch.cat([prompt, completion], dim=1)
    logits = _logits(model, ids)[:, len(prompt_ids) - 1 : -1].float() / temperature
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, completion.unsqueeze(-1)).squeeze(-1)
