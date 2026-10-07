"""Generate shields.io endpoint JSON badges from *measured* artifacts.

    python -m pytest --cov=bitnet_rl --cov-report=json     # writes coverage.json
    python scripts/validate_cpu.py                  # writes evidence/cpu_validation.json
    python scripts/make_badges.py                   # writes badges/*.json

Badge values are never typed by hand: coverage comes from ``coverage.json`` and the
perplexity badge from ``evidence/cpu_validation.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(name: str, label: str, message: str, color: str) -> None:
    out = ROOT / "badges" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"schemaVersion": 1, "label": label, "message": message, "color": color}, indent=2) + "\n")
    print(f"{name}: {label} | {message} | {color}")


def coverage_badge() -> None:
    pct = json.loads((ROOT / "coverage.json").read_text())["totals"]["percent_covered"]
    color = "brightgreen" if pct >= 90 else "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
    _write("coverage", "coverage", f"{pct:.0f}%", color)


def perplexity_badge() -> None:
    ev = json.loads((ROOT / "evidence" / "cpu_validation.json").read_text())
    s, cfg = ev["summary"], ev["config"]
    parts, all_improved, any_worse = [], True, False
    for label, key in (("linear", "linear"), ("bitlinear", "bitlinear")):
        m = s[key]["answer_ppl"]
        parts.append(f"{m['relative_change']:+.0%}")
        all_improved &= m["seeds_improved"] == s[key]["n_seeds"]
        any_worse |= m["relative_change"] >= 0
    n = s["linear"]["n_seeds"]
    color = "red" if any_worse else "brightgreen" if all_improved else "yellow"
    _write("cpu_perplexity", f"CPU answer-ppl, {cfg['iterations']} GRPO iters",
           f"{' / '.join(parts)} ({n} seeds, fp/ternary)", color)


if __name__ == "__main__":
    coverage_badge()
    perplexity_badge()
