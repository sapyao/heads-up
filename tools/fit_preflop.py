"""Fit opsbot/preflop_model.json: linear preflop features -> MC equity vs a random hand.

    .\\.venv\\Scripts\\python.exe tools/fit_preflop.py --hands 6000 --samples 2000 --seed 1

Offline only (never run during a match).  Prints in-sample and held-out R^2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from evaluator import warmup  # noqa: E402
from opsbot import fasteval, preflop  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(preflop.__file__), "preflop_model.json"))
    args = ap.parse_args()

    warmup()
    fasteval.build()
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    X, y = [], []
    for _ in range(args.hands):
        hole = sorted(rng.choice(36, 5, replace=False).tolist())
        outcome, _ = fasteval.sample_equity(hole, [], args.samples, rng)
        X.append(preflop.features(hole))
        y.append(float(outcome.mean()))
    X, y = np.array(X), np.array(y)
    n_train = int(0.8 * len(y))
    w, *_ = np.linalg.lstsq(X[:n_train], y[:n_train], rcond=None)

    def r2(a, b):
        return 1.0 - float(((a - b) ** 2).sum()) / float(((b - b.mean()) ** 2).sum())

    pred = X @ w
    noise_var = 0.25 / args.samples  # rough upper bound on MC label variance
    print(f"fit {args.hands} hands x {args.samples} samples in {time.perf_counter() - t0:.1f}s")
    print(f"R2 train {r2(pred[:n_train], y[:n_train]):.3f}  held-out {r2(pred[n_train:], y[n_train:]):.3f}  "
          f"label std {y.std():.4f}  MC noise std <= {noise_var ** 0.5:.4f}")
    print(f"held-out MAE {np.abs(pred[n_train:] - y[n_train:]).mean():.4f}")
    quantiles = np.quantile(pred, np.linspace(0, 1, 201)[1:-1]).tolist()
    model = {
        "description": "Linear fit of preflop features to MC all-in equity vs one uniformly random hand",
        "rules": "short-deck 36 cards, PLO5 exactly 2 hole + 3 board, flush > full house, trips > straight",
        "features": list(preflop.FEATURE_NAMES),
        "weights": w.tolist(),
        "quantiles": quantiles,
        "fit": {"hands": args.hands, "samples": args.samples, "seed": args.seed,
                "r2_heldout": r2(pred[n_train:], y[n_train:])},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
