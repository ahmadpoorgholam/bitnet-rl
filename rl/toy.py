"""Self-contained toy setting to exercise GRPO end to end on CPU.

A tiny char-level causal transformer (optionally built from ``BitLinear`` layers) is
first given a *base* policy that knows the R1 output format but answers at random
and sometimes breaks the format. GRPO with rule-based rewards must then discover
correct answers and reliable formatting - a miniature of R1-Zero's recipe.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rl.bitlinear import BitLinear

_CHARS = "0123456789+-=<>/ abcdefghijklmnopqrstuvwxyz"


class CharTokenizer:
    pad_id = 0
    eos_id = 1

    def __init__(self) -> None:
        self.itos = ["<pad>", "<eos>"] + list(_CHARS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i != self.pad_id:
                out.append(self.itos[i])
        return "".join(out)


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, linear) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = linear(d, 3 * d, bias=False), linear(d, d, bias=False)
        self.up, self.down = linear(d, 4 * d, bias=False), linear(4 * d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.relu(self.up(self.ln2(x))) ** 2)


class TinyCausalLM(nn.Module):
    def __init__(
        self, vocab: int, d: int = 64, layers: int = 2, heads: int = 4, max_len: int = 96,
        bitlinear: bool = False,
    ) -> None:
        super().__init__()
        linear = BitLinear if bitlinear else nn.Linear
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(_Block(d, heads, linear) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


@dataclass
class ToySample:
    prompt: str
    answer: str


def toy_questions(n: int = 5) -> list[ToySample]:
    return [ToySample(f"{a}+{b}=", str(a + b)) for a in range(n) for b in range(n)]


def _base_completion(sample: ToySample, rng: random.Random, n: int, p_correct: float) -> str:
    a, b = sample.prompt.rstrip("=").split("+")
    guess = sample.answer if rng.random() < p_correct else str(rng.randint(0, 2 * (n - 1)))
    kind = rng.random()
    if kind < 0.7:
        return f"<think>{a}+{b}</think><answer>{guess}</answer>"
    if kind < 0.85:
        return f"<think>{a}+{b}<answer>{guess}</answer>"
    return f"<think>{a}+{b}</think>{guess}"


def gold_completion(sample: ToySample) -> str:
    a, b = sample.prompt.rstrip("=").split("+")
    return f"<think>{a}+{b}</think><answer>{sample.answer}</answer>"


@torch.no_grad()
def gold_perplexity(model: TinyCausalLM, tok: CharTokenizer, n: int = 5) -> dict[str, float]:
    """Perplexity of the policy on the correct, well-formatted completion of every question.

    ``ppl`` is over all completion tokens (incl. EOS); ``answer_ppl`` only over the
    answer digits. Lower is better; RL toward the rule-based reward should reduce both.
    """
    from rl.grpo import token_logprobs

    was_training = model.training
    model.eval()
    nll_all, nll_ans = [], []
    for s in toy_questions(n):
        text = gold_completion(s)
        ids = torch.tensor([tok.encode(text) + [tok.eos_id]])
        nll = -token_logprobs(model, tok.encode(s.prompt), ids)[0]
        start = text.index("<answer>") + len("<answer>")
        nll_all.extend(nll.tolist())
        nll_ans.extend(nll[start : start + len(s.answer)].tolist())
    model.train(was_training)
    return {
        "ppl": math.exp(sum(nll_all) / len(nll_all)),
        "answer_ppl": math.exp(sum(nll_ans) / len(nll_ans)),
    }


def pretrain_base(
    model: TinyCausalLM, tok: CharTokenizer, n: int = 5, steps: int = 400, seed: int = 0,
    batch: int = 64, lr: float = 3e-3, p_correct: float = 0.3,
) -> float:
    """Fit a weak base policy: right answer with prob ``p_correct``, else random.

    Like a real pre-trained LM, the base solves a fraction of problems, which is
    what gives RL a non-degenerate reward signal to amplify (R1-Zero starts from
    DeepSeek-V3-Base, not from a model that never succeeds).
    """
    rng = random.Random(seed)
    qs = toy_questions(n)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_val = math.nan
    for _ in range(steps):
        seqs, cmask = [], []
        for s in (rng.choice(qs) for _ in range(batch)):
            p, c = tok.encode(s.prompt), tok.encode(_base_completion(s, rng, n, p_correct)) + [tok.eos_id]
            seqs.append(p + c)
            cmask.append([0] * len(p) + [1] * len(c))
        width = max(map(len, seqs))
        ids = torch.tensor([x + [tok.pad_id] * (width - len(x)) for x in seqs])
        m = torch.tensor([x + [0] * (width - len(x)) for x in cmask], dtype=torch.float)
        logits = model(ids[:, :-1])
        ce = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
        loss = (ce * m[:, 1:]).sum() / m[:, 1:].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
    return loss_val
