"""Cheap preflop hand features for Short Deck PLO5.

`features()` is pure Python (a few microseconds).  The linear weights in
preflop_model.json are fitted offline by tools/fit_preflop.py to Monte Carlo
all-in equity versus ONE uniformly random hand, so `equity_vs_random()` is an
approximation of that number, and `percentile()` ranks a hand among random
hands.  Equity versus a random hand is not equity versus a real raising range.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right

# Straight windows in short deck (rank 0 = six ... 8 = ace); A-6-7-8-9 is the wheel.
_WINDOWS = [frozenset({8, 0, 1, 2, 3})] + [frozenset(range(lo, lo + 5)) for lo in range(5)]

FEATURE_NAMES = (
    "bias", "rank_sum", "max_rank", "n_pairs", "pair_rank", "has_trips", "has_quads",
    "suits_2plus", "suits_3plus", "suited_ace", "suited_high", "double_suited",
    "best_window", "window_pairs", "n_distinct", "gaps_small",
)


def features(hole) -> list[float]:
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    rc = [0] * 9
    for r in ranks:
        rc[r] += 1
    sc = [0] * 4
    for s in suits:
        sc[s] += 1
    pairs = [r for r in range(9) if rc[r] == 2]
    distinct = {r for r in range(9) if rc[r]}
    suited_high = 0
    suited_ace = 0
    for s in range(4):
        if sc[s] >= 2:
            top = max(r for r, t in zip(ranks, suits) if t == s)
            suited_high += top / 8.0
            if top == 8:
                suited_ace = 1
    best_window = 0
    window_pairs = 0
    for w in _WINDOWS:
        k = len(distinct & w)
        best_window = max(best_window, k)
        window_pairs += k * (k - 1) // 2
    sr = sorted(distinct)
    gaps_small = sum(1 for a, b in zip(sr, sr[1:]) if b - a <= 2)
    return [
        1.0,
        sum(ranks) / 40.0,
        max(ranks) / 8.0,
        float(len(pairs)),
        (max(pairs) / 8.0) if pairs else 0.0,
        float(any(v == 3 for v in rc)),
        float(any(v == 4 for v in rc)),
        float(sum(1 for v in sc if v >= 2)),
        float(sum(1 for v in sc if v >= 3)),
        float(suited_ace),
        suited_high,
        float(sum(1 for v in sc if v == 2) == 2),
        float(best_window),
        window_pairs / 10.0,
        float(len(distinct)),
        float(gaps_small),
    ]


_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preflop_model.json")
_WEIGHTS: list[float] | None = None
_QUANTILES: list[float] | None = None


def load(path: str = _MODEL_PATH) -> None:
    global _WEIGHTS, _QUANTILES
    with open(path, encoding="utf-8") as fh:
        model = json.load(fh)
    if model.get("features") != list(FEATURE_NAMES):
        raise ValueError("preflop_model.json was fitted with different features")
    _WEIGHTS = [float(w) for w in model["weights"]]
    _QUANTILES = [float(q) for q in model["quantiles"]]


def equity_vs_random(hole) -> float:
    if _WEIGHTS is None:
        load()
    return sum(w * f for w, f in zip(_WEIGHTS, features(hole)))


def percentile(hole) -> float:
    """Fraction of random hands with a lower predicted equity (0..1)."""
    if _QUANTILES is None:
        load()
    q = _QUANTILES
    return bisect_right(q, equity_vs_random(hole)) / len(q)
