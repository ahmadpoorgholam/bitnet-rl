"""BitLinear: W1.58A8 linear layer with straight-through estimators (BitNet b1.58).

Weights are quantised to {-1, 0, +1} * scale with absmean scaling and activations to
int8 with per-token absmax scaling. Full-precision master weights receive gradients
through the STE, so the layer can be trained (and RL fine-tuned) directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().mean().clamp(min=1e-5)
    return (w / scale).round().clamp(-1, 1), scale


def quantize_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127), scale


class BitLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, w_scale = quantize_weights(self.weight)
        x_q, x_scale = quantize_activations(x)
        w_eff = self.weight + (w_q * w_scale - self.weight).detach()
        x_eff = x + (x_q / x_scale - x).detach()
        return F.linear(x_eff, w_eff, self.bias)

    def ternary_weights(self) -> torch.Tensor:
        return quantize_weights(self.weight.detach())[0]
