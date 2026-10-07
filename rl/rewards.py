"""Rule-based rewards from DeepSeek-R1-Zero (R1 paper, Sec. 2.2).

``Reward_rule = Reward_acc + Reward_format`` with equal weight. No neural reward
model is used, which is how R1 avoids reward hacking on reasoning tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

R1_ZERO_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>. User: {question}. Assistant: "
)

_FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


@dataclass
class RewardResult:
    reward: float
    accuracy: float
    format: float


def format_reward(completion: str) -> float:
    """1.0 if reasoning and answer are enclosed in the required tags, in order."""
    return 1.0 if _FORMAT_RE.match(completion) else 0.0


def extract_answer(completion: str) -> str | None:
    """Final answer from the last ``<answer>`` block, falling back to ``\\boxed{}``."""
    found = _ANSWER_RE.findall(completion)
    if found:
        return found[-1].strip()
    boxed = _BOXED_RE.findall(completion)
    return boxed[-1].strip() if boxed else None


def _normalise(text: str) -> str:
    return text.strip().strip("$").replace(",", "").replace(" ", "").rstrip(".").lower()


def _as_number(text: str) -> Fraction | None:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def accuracy_reward(completion: str, ground_truth: str) -> float:
    """1.0 if the extracted answer matches the ground truth (numerically if possible)."""
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    pred, truth = _normalise(pred), _normalise(ground_truth)
    if pred == truth:
        return 1.0
    p, t = _as_number(pred), _as_number(truth)
    return 1.0 if p is not None and t is not None and p == t else 0.0


def rule_reward(completion: str, ground_truth: str) -> RewardResult:
    acc = accuracy_reward(completion, ground_truth)
    fmt = format_reward(completion)
    return RewardResult(reward=acc + fmt, accuracy=acc, format=fmt)
