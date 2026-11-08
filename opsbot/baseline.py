"""Phase 1 equity baseline for heads-up Short Deck PLO5.

Decision outline
----------------
Preflop: a fitted feature model gives equity vs a random hand and a percentile.
  Opens/3-bets use percentile thresholds; calls compare a range-discounted
  equity with pot odds.
Postflop: vectorised Monte Carlo against sampled opponent hands (exact card
  removal, half credit for ties). Opponent samples are weighted by their
  visible-board strength raised to a power that grows with the opponent's
  aggression this hand, scaled by how selective this opponent has been
  (an always-raise bot's bets carry little information).
  Actions are compared by a one-step EV model:
    check  : E * real * pot
    call   : E * real * (pot + to_call) - to_call
    raise  : f * pot + (1 - f) * (E_called * real * final_pot - added)
  where f is a fold probability (prior blended with this opponent's observed
  folds to our bets) and E_called is our equity against the strongest (1 - f)
  of the weighted range, i.e. the hands that continue.  `real` < 1 discounts
  equity we will not fully realise before the river, more so out of position.
  The model ignores re-raises and later-street betting; it is a heuristic.

Every answer is checked against obs.legal_actions; any exception or illegal
choice falls back to check-or-fold.  Thinking time is tracked in memory and
the sample count shrinks if the half's pace would exceed the budget.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

from game import Action, AgentConfig, HandResult, Observation, Street

from . import fasteval, preflop

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

CFG = {
    # Monte Carlo samples per postflop decision (before pacing).
    "samples": {Street.FLOP: 128, Street.TURN: 128, Street.RIVER: 192},
    "min_samples": 48,
    # Time budget for one half.  The judge's 8 s bank also charges HTTP/JSON
    # handling that choose_action() cannot see, so the estimate of charged
    # time is a sum over requests, like the judge's, of
    #     max(0, think_i + outside_cpu_i + overhead_per_call_s - grace_s)
    # think_i     wall time inside choose_action
    # outside_cpu_i  this process's CPU time since our previous answer (JSON,
    #             FastAPI, on_hand_end) -- measured, so it scales when the
    #             machine is slow or throttled
    # grace_s     the judge forgives the first 1 ms of every request; we
    #             credit only 0.8 ms to stay conservative.
    "overhead_per_call_s": 0.0002,
    "grace_s": 0.0008,
    # The judge charges WALL time per request.  On a busy machine much of it is
    # scheduling delay we cannot see: with the CPUs 2x oversubscribed the judge
    # charged 2.2-2.4x our estimate (no load: 0.3-0.8x).  So the tiers switch
    # early enough that a 2.5x under-count still fits the 8 s bank.  On an idle
    # Linux box a full half estimates ~0.3 s, so these never trigger there.
    # The cheap policy costs tens of microseconds, so after the switch the bank
    # grows only by unavoidable per-request overhead.
    "soft_budget_s": 1.5,       # pacing: shrink Monte Carlo samples when ahead of this pace
    "hard_budget_s": 2.5,       # beyond this: no Monte Carlo, cheap made-hand heuristic
    "emergency_budget_s": 6.5,  # beyond this: check-or-fold only (no computation)
    "max_logged_exceptions": 5,
    # Preflop percentile thresholds.
    "sb_open_raise": 0.45,
    "sb_open_call": 0.08,
    "bb_iso_raise": 0.70,
    "threebet": 0.85,
    "fourbet": 0.93,
    "pre_raise_penalty": 0.035,   # equity lost vs a raising range, per opponent raise
    # Postflop.
    "real_ip": 0.95,
    "real_oop": 0.88,
    "fold_prior": 0.40,
    "fold_prior_weight": 8.0,
    "aggr_prior": 0.30,
    "aggr_prior_weight": 10.0,
    "k_raise": {Street.PREFLOP: 0.25, Street.FLOP: 0.7, Street.TURN: 0.9, Street.RIVER: 1.1},
    "k_call": 0.2,
}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def warmup() -> None:
    fasteval.build()
    preflop.load()
    # Exercise every hot path once so the first real decision pays no first-call cost.
    rng = np.random.default_rng(0)
    hole = (0, 5, 10, 15, 20)
    for board in ((25, 30, 35), (25, 30, 35, 1), (25, 30, 35, 1, 6)):
        fasteval.sample_equity(hole, board, 16, rng)
        fasteval.made_category(hole, board)
    preflop.percentile(hole)


def new_half(config: AgentConfig, memory: dict) -> None:
    seed = int(os.environ.get("OPS_SEED", "20261107"))
    memory.update(
        rng=np.random.default_rng(seed),
        num_hands=max(1, int(config.num_hands)),
        think=0.0, decisions=0, fallbacks=0, exceptions=0, illegal=0,
        cheap=0, emergency=0, mc_decisions=0, hand_end_exceptions=0,
        outside_cpu=0.0, cpu_exit=None, charged=0.0,
        opp_actions=0, opp_raises=0,
        faced=0, folds=0,
        hist_hand=-1, hist_seen=0,
        pending_raise=None,
    )


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        if memory.get("pending_raise") == result.hand_index:
            memory["faced"] += 1
            if not result.went_to_showdown and result.winner == "me":
                memory["folds"] += 1
        memory["pending_raise"] = None
    except Exception:
        memory["hand_end_exceptions"] = memory.get("hand_end_exceptions", 0) + 1
        _log_exception(memory, "on_hand_end")
    try:
        if result.hand_index == memory.get("num_hands", 0) - 1:
            emit_stats(memory)
    except Exception:
        pass  # diagnostics only; never worth an error


STAT_KEYS = ("decisions", "fallbacks", "exceptions", "illegal", "cheap", "emergency",
             "mc_decisions", "hand_end_exceptions", "think", "faced", "folds",
             "opp_actions", "opp_raises", "outside_cpu", "charged")


def stats(memory: dict) -> dict:
    out = {k: memory.get(k) for k in STAT_KEYS}
    out["think"] = round(float(out["think"] or 0.0), 4)
    out["est_charged_s"] = round(estimated_charged(memory), 4)
    return out


def emit_stats(memory: dict) -> None:
    """One machine-readable line on stderr (read by tools/run_eval.py)."""
    print("OPS_STATS " + json.dumps(stats(memory)), file=sys.stderr, flush=True)


def _log_exception(memory: dict, where: str) -> None:
    logged = memory.get("logged_exceptions", 0)
    if logged < CFG["max_logged_exceptions"]:
        memory["logged_exceptions"] = logged + 1
        print(f"OPS_EXCEPTION in {where}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def estimated_charged(memory: dict) -> float:
    """Estimate of the judge-charged time so far this half."""
    return memory.get("charged", 0.0)


def safe_action(obs: Observation) -> Action:
    """Bounded fallback: free check, otherwise fold (always legal)."""
    legal = obs.legal_actions
    if obs.to_call == 0 and Action.CHECK_CALL in legal:
        return Action.CHECK_CALL
    if Action.FOLD in legal:
        return Action.FOLD
    return legal[0] if legal else Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    if "rng" not in memory:          # defensive: new_half not called
        new_half(AgentConfig(), memory)
    spent = estimated_charged(memory)
    action = None
    if spent >= CFG["emergency_budget_s"]:
        memory["emergency"] += 1
        action = safe_action(obs)
    else:
        try:
            _observe_opponent(obs, memory)
            if spent >= CFG["hard_budget_s"]:
                memory["cheap"] += 1
                action = cheap_action(obs, memory)
            else:
                action = decide(obs, memory)
        except Exception:
            memory["exceptions"] += 1
            _log_exception(memory, "choose_action")
            action = None
        if action is not None and action not in obs.legal_actions:
            memory["illegal"] += 1
        if action not in obs.legal_actions:
            memory["fallbacks"] += 1
            action = safe_action(obs)
    if action in RAISES:
        memory["pending_raise"] = obs.hand_index
    memory["decisions"] += 1
    inside = time.perf_counter() - t0
    outside = max(0.0, cpu0 - memory["cpu_exit"]) if memory.get("cpu_exit") is not None else 0.0
    memory["think"] += inside
    memory["outside_cpu"] += outside
    memory["charged"] = memory.get("charged", 0.0) + max(
        0.0, inside + outside + CFG["overhead_per_call_s"] - CFG["grace_s"])
    memory["cpu_exit"] = time.process_time()
    return action


# ---------------------------------------------------------------------------
# Opponent statistics (only from what we observed)
# ---------------------------------------------------------------------------

def _observe_opponent(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory["hist_hand"]:
        memory["hist_hand"] = obs.hand_index
        memory["hist_seen"] = 0
    for a in obs.action_history[memory["hist_seen"]:]:
        if a.is_me:
            continue
        memory["opp_actions"] += 1
        if a.action in RAISES:
            memory["opp_raises"] += 1
        if memory["pending_raise"] == obs.hand_index:
            memory["faced"] += 1          # the opponent answered our raise without folding
            memory["pending_raise"] = None
    memory["hist_seen"] = len(obs.action_history)


def _selectivity(memory: dict) -> float:
    """1.0 for a typical opponent; small when the opponent raises nearly always."""
    w = CFG["aggr_prior_weight"]
    aggr = (memory["opp_raises"] + CFG["aggr_prior"] * w) / (memory["opp_actions"] + w)
    return min(1.2, max(0.15, (1.0 - aggr) / 0.7))


def _fold_rate(memory: dict) -> float:
    w = CFG["fold_prior_weight"]
    return (memory["folds"] + CFG["fold_prior"] * w) / (memory["faced"] + w)


def _range_power(obs: Observation, memory: dict) -> float:
    k = 0.0
    for a in obs.action_history:
        if a.is_me:
            continue
        if a.action in RAISES:
            k += CFG["k_raise"][a.street]
        elif a.action == Action.CHECK_CALL and a.street != Street.PREFLOP and a.amount > 0:
            k += CFG["k_call"]
    return k * _selectivity(memory)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def decide(obs: Observation, memory: dict) -> Action:
    if obs.street == Street.PREFLOP:
        return _decide_preflop(obs, memory)
    return _decide_postflop(obs, memory)


def _pick(obs: Observation, want: Action) -> Action:
    """Return `want` if legal, else the nearest legal substitute."""
    legal = obs.legal_actions
    if want in legal:
        return want
    if want in RAISES:
        other = Action.RAISE_POT if want == Action.RAISE_HALF else Action.RAISE_HALF
        if other in legal:
            return other
        return Action.CHECK_CALL
    return safe_action(obs)


def _decide_preflop(obs: Observation, memory: dict) -> Action:
    pct = preflop.percentile(obs.hole_cards)
    eq = preflop.equity_vs_random(obs.hole_cards)
    opp_raises = sum(1 for a in obs.action_history if not a.is_me and a.action in RAISES)
    sel = _selectivity(memory)
    can_raise = Action.RAISE_POT in obs.legal_actions

    if obs.to_call == 0:  # BB after a limp
        if can_raise and pct >= CFG["bb_iso_raise"]:
            return Action.RAISE_POT
        return Action.CHECK_CALL

    if opp_raises == 0:   # SB first decision: to_call is the 1-chip completion
        if can_raise and pct >= CFG["sb_open_raise"]:
            return Action.RAISE_POT
        if pct >= CFG["sb_open_call"]:
            return Action.CHECK_CALL
        return Action.FOLD

    # Facing a raise.  Less selective raisers -> smaller equity discount and
    # lower re-raise thresholds.
    eq_adj = eq - CFG["pre_raise_penalty"] * opp_raises * sel
    real = CFG["real_ip"] if obs.is_button else CFG["real_oop"]
    reraise_thr = (CFG["threebet"] if opp_raises == 1 else CFG["fourbet"]) - 0.25 * (1.0 - min(sel, 1.0))
    if can_raise and pct >= reraise_thr:
        return Action.RAISE_POT
    price = obs.to_call / (obs.pot + obs.to_call)
    if eq_adj * real >= price:
        return Action.CHECK_CALL
    return Action.FOLD


def _n_samples(obs: Observation, memory: dict) -> int:
    n = CFG["samples"][obs.street]
    budget = CFG["soft_budget_s"]
    progress = (obs.hand_index + 1) / memory["num_hands"]
    allowed = budget * progress + 0.25 * budget
    spent = estimated_charged(memory)
    if spent > allowed:
        n = max(CFG["min_samples"], n // 4)
    elif spent > 0.8 * allowed:
        n = max(CFG["min_samples"], n // 2)
    return n


def cheap_action(obs: Observation, memory: dict) -> Action:
    """No-Monte-Carlo policy used after the hard budget (tens of microseconds).

    Preflop uses the (already cheap) feature model.  Postflop uses only our
    made-hand category on the visible board: trips or better (trips beat
    straights in short deck) bets pot / calls; two pair or a straight calls
    bets up to half pot; anything else checks or folds.
    """
    if obs.street == Street.PREFLOP:
        return _decide_preflop(obs, memory)
    cat = fasteval.made_category(obs.hole_cards, obs.board)
    if cat >= 4:
        return _pick(obs, Action.RAISE_POT)
    if cat >= 2 and (obs.to_call == 0 or obs.to_call <= 0.5 * obs.pot):
        return Action.CHECK_CALL
    return safe_action(obs)


def postflop_estimates(obs: Observation, memory: dict, n: int):
    """(weighted equity, weights and outcomes sorted strongest-opponent first)."""
    outcome, strength = fasteval.sample_equity(obs.hole_cards, obs.board, n, memory["rng"])
    k = _range_power(obs, memory)
    w = strength ** k if k > 0 else np.ones_like(strength)
    order = np.argsort(-strength, kind="stable")  # strongest opponent hands first
    return float((w * outcome).sum() / w.sum()), w[order], outcome[order]


def _equity_vs_continuing(w_sorted: np.ndarray, out_sorted: np.ndarray, fold: float) -> float:
    """Equity against the strongest (1 - fold) share of the weighted range."""
    cw = np.cumsum(w_sorted)
    keep = max(1, int(np.searchsorted(cw, (1.0 - fold) * cw[-1])) + 1)
    ww = w_sorted[:keep]
    return float((ww * out_sorted[:keep]).sum() / ww.sum())


def _decide_postflop(obs: Observation, memory: dict) -> Action:
    n = _n_samples(obs, memory)
    memory["mc_decisions"] += 1
    eq, w_sorted, out_sorted = postflop_estimates(obs, memory, n)
    pot, to_call = obs.pot, obs.to_call
    real = 1.0 if obs.street == Street.RIVER else (CFG["real_ip"] if obs.is_button else CFG["real_oop"])

    evs = {}
    if to_call == 0:
        evs[Action.CHECK_CALL] = eq * real * pot
    else:
        evs[Action.FOLD] = 0.0
        evs[Action.CHECK_CALL] = eq * real * (pot + to_call) - to_call

    base_fold = _fold_rate(memory)
    if to_call > 0:
        base_fold *= 0.7          # a bettor who is re-raised folds less often
    for a in RAISES:
        if a not in obs.legal_actions:
            continue
        raise_to = obs.raise_to[a]
        added = raise_to - obs.my_bet
        # Both stacks start equal each hand, so the opponent can always cover.
        opp_call = raise_to - obs.opp_bet
        f = min(0.85, max(0.0, base_fold * (0.9 if a == Action.RAISE_HALF else 1.1)))
        e_called = _equity_vs_continuing(w_sorted, out_sorted, f)
        evs[a] = f * pot + (1.0 - f) * (e_called * real * (pot + added + opp_call) - added)
    best = max(evs, key=evs.get)
    return _pick(obs, best)
