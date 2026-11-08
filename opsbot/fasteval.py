"""Vectorised Short Deck PLO5 evaluation built from the official evaluator.

The official `evaluator.get_table()` maps every 5-card mask to a score.  We
re-index it as a dense numpy table

    SC[pair_id, triple_id] -> int16 strength rank (higher is better)

where pair_id names 2 hole cards and triple_id names 3 board cards.  An Omaha
hand value is then max over 10 hole pairs x 10 board triples, which numpy
evaluates for hundreds of sampled opponents/runouts in one gather.  Ranks are
the position of the official score among all distinct scores, so comparisons
(win / tie / loss) are identical to `omaha_score`.  test_opsbot.py checks this.
"""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np

from evaluator import NUM_CARDS, get_table

PAIRS5 = np.array(list(combinations(range(5), 2)), dtype=np.intp)    # (10, 2)
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.intp)  # (10, 3)

PAIR_ID: np.ndarray | None = None   # (36, 36) -> pair id, symmetric, -1 on diagonal
TRIP_ID: np.ndarray | None = None   # (36, 36, 36) -> triple id, all orders
SC: np.ndarray | None = None        # (630, 7140) int16 strength rank, -1 = overlapping cards
RANK_CATEGORY: np.ndarray | None = None  # strength rank -> official category 0..8


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fasteval_tables.npz")
_SHAPES = {"PAIR_ID": ((NUM_CARDS, NUM_CARDS), np.intp), "TRIP_ID": ((NUM_CARDS,) * 3, np.intp),
           "SC": ((630, 7140), np.int16), "RANK_CATEGORY": (None, np.int8)}


def build(cache_path: str | None = CACHE_PATH) -> None:
    """Load the tables from the committed cache (~20 ms), else build them (~0.6 s CPU).

    Startup must finish within 15 s even on a loaded machine, so the cache
    matters; a missing, unreadable or malformed cache silently falls back to
    building.  test_opsbot.py checks the committed cache equals a fresh build.
    """
    global PAIR_ID, TRIP_ID, SC, RANK_CATEGORY
    if SC is not None:
        return
    if cache_path and _load_cache(cache_path):
        return
    _build_from_evaluator()


def _load_cache(path: str) -> bool:
    global PAIR_ID, TRIP_ID, SC, RANK_CATEGORY
    try:
        with np.load(path, allow_pickle=False) as z:
            arrs = {}
            for name, (shape, dtype) in _SHAPES.items():
                a = np.ascontiguousarray(z[name]).astype(dtype, copy=False)
                if shape is not None and a.shape != shape:
                    return False
                arrs[name] = a
    except Exception:
        return False
    if int(arrs["SC"].max()) >= len(arrs["RANK_CATEGORY"]):
        return False
    PAIR_ID, TRIP_ID, SC, RANK_CATEGORY = (arrs[k] for k in ("PAIR_ID", "TRIP_ID", "SC", "RANK_CATEGORY"))
    return True


def save_cache(path: str = CACHE_PATH) -> None:
    """Regenerate the committed cache from the official evaluator."""
    global SC
    SC = None
    _build_from_evaluator()
    np.savez_compressed(path, PAIR_ID=PAIR_ID.astype(np.int16), TRIP_ID=TRIP_ID.astype(np.int16),
                        SC=SC, RANK_CATEGORY=RANK_CATEGORY)


def _build_from_evaluator() -> None:
    global PAIR_ID, TRIP_ID, SC, RANK_CATEGORY
    table = get_table()

    combos5 = np.array(list(combinations(range(NUM_CARDS), 5)), dtype=np.int64)
    masks = (np.int64(1) << combos5).sum(axis=1)
    scores = np.fromiter((table[m] for m in masks.tolist()), dtype=np.int64, count=len(masks))
    distinct = np.unique(scores)
    rank_of_combo = np.searchsorted(distinct, scores).astype(np.int16)
    RANK_CATEGORY = (distinct >> 20).astype(np.int8)

    binom = np.zeros((NUM_CARDS + 1, 6), dtype=np.int64)
    for n in range(NUM_CARDS + 1):
        for k in range(6):
            binom[n, k] = _comb(n, k)
    colex_of = lambda sorted_cards: sum(binom[sorted_cards[..., i], i + 1] for i in range(sorted_cards.shape[-1]))
    s5 = np.empty(int(binom[NUM_CARDS, 5]), dtype=np.int16)
    s5[colex_of(combos5)] = rank_of_combo

    pairs = np.array(list(combinations(range(NUM_CARDS), 2)), dtype=np.int8)
    trips = np.array(list(combinations(range(NUM_CARDS), 3)), dtype=np.int8)
    pid = np.full((NUM_CARDS, NUM_CARDS), -1, dtype=np.int16)
    pid[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
    pid[pairs[:, 1], pairs[:, 0]] = np.arange(len(pairs))
    tid = np.full((NUM_CARDS,) * 3, -1, dtype=np.int16)
    ar = np.arange(len(trips))
    for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        tid[trips[:, a], trips[:, b], trips[:, c]] = ar

    sc = np.full((len(pairs), len(trips)), -1, dtype=np.int16)
    for start in range(0, len(pairs), 90):  # chunk to bound peak memory (~60 MB)
        p = pairs[start:start + 90]
        u = np.concatenate([np.broadcast_to(p[:, None, :], (len(p), len(trips), 2)),
                            np.broadcast_to(trips[None, :, :], (len(p), len(trips), 3))], axis=2)
        u = np.sort(u.astype(np.int64), axis=2)
        ok = np.all(u[..., 1:] != u[..., :-1], axis=2)
        idx = colex_of(np.where(ok[..., None], u, np.arange(5)))
        sc[start:start + len(p)] = np.where(ok, s5[idx], -1)

    PAIR_ID, TRIP_ID, SC = pid.astype(np.intp), tid.astype(np.intp), sc


def _comb(n: int, k: int) -> int:
    from math import comb
    return comb(n, k) if 0 <= k <= n else 0


def hand_rank(hole, board) -> int:
    """Strength rank of a 5-card hole hand on a full 5-card board."""
    h = np.asarray(hole, dtype=np.intp)
    b = np.asarray(board, dtype=np.intp)
    pids = PAIR_ID[h[PAIRS5[:, 0]], h[PAIRS5[:, 1]]]
    tids = TRIP_ID[b[TRIPLES5[:, 0]], b[TRIPLES5[:, 1]], b[TRIPLES5[:, 2]]]
    return int(SC[pids[:, None], tids[None, :]].max())


def made_category(hole, board) -> int:
    """Official category (0..8) of our best hand on a visible board of 3-5 cards."""
    h = np.asarray(hole, dtype=np.intp)
    pids = PAIR_ID[h[PAIRS5[:, 0]], h[PAIRS5[:, 1]]]
    return int(RANK_CATEGORY[SC[pids[:, None], visible_triples(board)[None, :]].max()])


def visible_triples(board) -> np.ndarray:
    """Triple ids available on the visible board (1 on flop, 4 on turn, 10 on river)."""
    return np.array([TRIP_ID[a, b, c] for a, b, c in combinations(board, 3)], dtype=np.intp)


def draw(hole, board, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (opponent 5 cards, full 5-card board) pairs from the unseen cards.

    Each row is a uniformly random subset drawn without replacement from the
    36 - 5 - len(board) cards not in hole or board.
    """
    dead = set(hole) | set(board)
    unseen = np.array([c for c in range(NUM_CARDS) if c not in dead], dtype=np.intp)
    need = 5 + (5 - len(board))
    # argpartition of iid uniforms = the first `need` items of a random permutation.
    picks = np.argpartition(rng.random((n, len(unseen))), need - 1, axis=1)[:, :need]
    drawn = unseen[picks]
    if len(board) < 5:
        full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.intp), (n, len(board))),
                               drawn[:, 5:]], axis=1)
    else:
        full = np.broadcast_to(np.array(board, dtype=np.intp), (n, 5))
    return drawn[:, :5], full


def sample_equity(hole, board, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo showdown outcomes against uniformly random opponent hands.

    Card removal: opponents and runouts are drawn without replacement from
    the 36 - 5 - len(board) unseen cards, so no card appears twice.
    Returns (outcome, opp_strength):
        outcome[i]      1.0 win, 0.5 tie, 0.0 loss for sample i
        opp_strength[i] opponent i's percentile (0..1] on the VISIBLE board,
                        bumped up for flush draws; used to weight ranges.
                        It never uses the sampled future runout.
    """
    hole = list(hole)
    board = list(board)
    opp, full = draw(hole, board, n, rng)

    tids = TRIP_ID[full[:, TRIPLES5[:, 0]], full[:, TRIPLES5[:, 1]], full[:, TRIPLES5[:, 2]]]  # (n,10)
    h = np.array(hole, dtype=np.intp)
    hero_pids = PAIR_ID[h[PAIRS5[:, 0]], h[PAIRS5[:, 1]]]                                      # (10,)
    opp_pids = PAIR_ID[opp[:, PAIRS5[:, 0]], opp[:, PAIRS5[:, 1]]]                             # (n,10)
    if len(board) == 5:
        hero = int(SC[hero_pids[:, None], tids[0][None, :]].max())
    else:
        hero = SC[hero_pids[None, :, None], tids[:, None, :]].max(axis=(1, 2))
    opp_final = SC[opp_pids[:, :, None], tids[:, None, :]].max(axis=(1, 2))
    outcome = (hero > opp_final).astype(np.float64) + 0.5 * (hero == opp_final)

    # Opponent strength from information on the visible board only.
    if not board:  # preflop: no board information (used only offline)
        return outcome, np.ones(n)
    if len(board) == 5:
        cur = opp_final
    else:
        vt = visible_triples(board)
        cur = SC[opp_pids[:, :, None], vt[None, None, :]].max(axis=(1, 2))
    order = np.argsort(cur, kind="stable")
    pct = np.empty(n, dtype=np.float64)
    # Percentile of each sample's current strength; ties share the higher value.
    sorted_cur = cur[order]
    last_of_group = np.searchsorted(sorted_cur, sorted_cur, side="right")
    pct[order] = last_of_group / n
    if len(board) < 5:
        bsuits = np.bincount(np.array(board) & 3, minlength=4)
        osuits = np.zeros((n, 4), dtype=np.int64)
        for s in range(4):
            osuits[:, s] = (opp & 3 == s).sum(axis=1)
        fd = ((bsuits[None, :] == 2) & (osuits >= 2)).any(axis=1)
        fd_floor = 0.75 if len(board) == 3 else 0.65
        pct = np.where(fd, np.maximum(pct, fd_floor), pct)
    return outcome, pct
