"""Micro-benchmark of decision costs (no HTTP).

    .\\.venv\\Scripts\\python.exe tools/bench_decision.py

Reports startup build time, preflop feature cost, Monte Carlo cost and noise
per street and sample count, and full choose_action() cost per street from
self-play-like states (random opponent).
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from evaluator import warmup  # noqa: E402
from game import AgentConfig, Game, Street, make_deal  # noqa: E402
from opsbot import baseline, fasteval, preflop  # noqa: E402


def main() -> None:
    t = time.perf_counter(); warmup(); t_w = time.perf_counter() - t
    t = time.perf_counter(); baseline.warmup(); t_b = time.perf_counter() - t
    print(f"startup: evaluator table {t_w:.2f}s, opsbot tables+model {t_b:.2f}s")

    rng = random.Random(0)
    hands = [rng.sample(range(36), 5) for _ in range(20000)]
    t = time.perf_counter()
    for h in hands:
        preflop.percentile(h)
        preflop.equity_vs_random(h)
    print(f"preflop features+lookup: {(time.perf_counter() - t) / len(hands) * 1e6:.1f} us")

    g = np.random.default_rng(0)
    for nb in (3, 4, 5):
        for n in (64, 128, 256, 384, 512):
            spots = [rng.sample(range(36), 5 + nb) for _ in range(150)]
            t = time.perf_counter()
            means = []
            for c in spots:
                o, _ = fasteval.sample_equity(c[:5], c[5:], n, g)
                means.append(o.mean())
            dt = (time.perf_counter() - t) / len(spots)
            se = statistics.mean((m * (1 - m) / n) ** 0.5 for m in means)
            print(f"board={nb} samples={n:<4} {dt * 1e3:6.3f} ms   typical MC s.e. {se:.3f}")

    # Full decisions by street.
    memory = {}
    baseline.new_half(AgentConfig(), memory)
    by_street = {s: [] for s in Street}
    r = random.Random(1)
    for hand in range(600):
        game = Game(make_deal("bench", hand), hand % 2, hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                t = time.perf_counter()
                a = baseline.choose_action(obs, memory)
                by_street[obs.street].append(time.perf_counter() - t)
            else:
                a = r.choice(obs.legal_actions)
            game.apply(a)
        baseline.on_hand_end(game.hand_result(0), memory)
    for s, v in by_street.items():
        if v:
            v.sort()
            print(f"choose_action {s.name:<8} n={len(v):<5} mean {statistics.mean(v) * 1e3:.3f} ms  "
                  f"p99 {v[int(0.99 * (len(v) - 1))] * 1e3:.3f} ms  max {v[-1] * 1e3:.3f} ms")
    print(f"total think over 600 hands: {memory['think']:.3f}s ({memory['decisions']} decisions)")


if __name__ == "__main__":
    main()
