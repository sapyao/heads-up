"""Tests for the opsbot strategy package (run with: python -m pytest -q)."""

import random
import time
from itertools import combinations

import numpy as np
import pytest

from evaluator import cards_from_str, omaha_score, warmup
from game import Action, AgentConfig, Game, HandResult, Street, make_deal
from opsbot import baseline, fasteval, preflop

warmup()
baseline.warmup()


def test_fast_table_matches_official_evaluator():
    rng = random.Random(11)
    ties = 0
    for _ in range(4000):
        c = rng.sample(range(36), 15)
        a, b, board = c[:5], c[5:10], c[10:]
        sa, sb = omaha_score(a, board), omaha_score(b, board)
        ra, rb = fasteval.hand_rank(a, board), fasteval.hand_rank(b, board)
        assert (sa > sb) == (ra > rb) and (sa == sb) == (ra == rb)
        ties += sa == sb
    assert ties > 0  # ties occur and are classified identically


@pytest.mark.parametrize("nboard", [0, 3, 4, 5])
def test_draw_respects_card_removal(nboard):
    rng = np.random.default_rng(3)
    cards = random.Random(nboard).sample(range(36), 5 + nboard)
    hole, board = cards[:5], cards[5:]
    opp, full = fasteval.draw(hole, board, 2000, rng)
    assert opp.shape == (2000, 5) and full.shape == (2000, 5)
    assert (full[:, :nboard] == np.array(board, dtype=np.intp)).all()
    rows = np.concatenate([opp, full], axis=1)
    for row in rows[:500]:
        assert len(set(row.tolist())) == 10            # no duplicate inside a deal
        assert not set(row.tolist()) & set(hole)       # hero cards removed
    # Every unseen card is drawable for the opponent (roughly uniform).
    unseen = sorted(set(range(36)) - set(hole) - set(board))
    counts = np.bincount(opp.ravel(), minlength=36)[unseen]
    assert counts.min() > 0.6 * counts.mean()


def test_river_equity_matches_exhaustive_enumeration_with_half_credit_ties():
    # A board with a straight on it makes ties common.
    hole = cards_from_str("Tc Jd 6h 6s Ac")
    board = cards_from_str("7c 8d 9h Ks Qh")
    my = omaha_score(hole, board)
    unseen = [c for c in range(36) if c not in hole + board]
    win = tie = total = 0
    for opp in combinations(unseen, 5):
        s = omaha_score(opp, board)
        win += my > s
        tie += my == s
        total += 1
    exact = (win + 0.5 * tie) / total
    assert tie > 0
    outcome, _ = fasteval.sample_equity(hole, board, 40000, np.random.default_rng(5))
    assert abs(outcome.mean() - exact) < 0.01


def test_flop_equity_matches_reference_monte_carlo():
    hole = cards_from_str("Ah Kh Qd Jd 9c")
    board = cards_from_str("Th 7h 6c")
    rng = random.Random(9)
    unseen = [c for c in range(36) if c not in hole + board]
    ref = 0.0
    n = 4000
    for _ in range(n):
        d = rng.sample(unseen, 7)
        full = board + d[5:]
        a, b = omaha_score(hole, full), omaha_score(d[:5], full)
        ref += 1.0 if a > b else 0.5 if a == b else 0.0
    ref /= n
    outcome, strength = fasteval.sample_equity(hole, board, 40000, np.random.default_rng(1))
    assert abs(outcome.mean() - ref) < 0.025   # ~3 standard errors of the reference
    assert strength.min() > 0 and strength.max() <= 1.0


def test_preflop_model_orders_obvious_hands():
    strong = cards_from_str("As Ah Ks Kh Qd")
    weak = cards_from_str("6c 6d 6h 7s 9c")
    assert preflop.percentile(strong) > 0.9
    assert preflop.percentile(weak) < 0.3
    assert preflop.equity_vs_random(strong) > preflop.equity_vs_random(weak)


def _play(num_hands, seed, opp_policy):
    """Seat 0 = baseline (called directly), seat 1 = opp_policy(obs, rng)."""
    memory = {}
    baseline.new_half(AgentConfig(num_hands=num_hands), memory)
    rng = random.Random(seed)
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                action = baseline.choose_action(obs, memory)
                assert action in obs.legal_actions
            else:
                action = opp_policy(obs, rng)
            game.apply(action)
        baseline.on_hand_end(game.hand_result(0), memory)
    return memory


def _random_opp(obs, rng):
    return rng.choice(obs.legal_actions)


def _maniac(obs, rng):
    return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL


def _folder(obs, rng):
    return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL


@pytest.mark.parametrize("policy", [_random_opp, _maniac, _folder])
def test_baseline_always_legal_without_fallback(policy):
    memory = _play(300, f"legal-{policy.__name__}", policy)
    assert memory["fallbacks"] == 0      # decide() never crashed or chose illegally
    assert memory["decisions"] >= 150   # at least one decision in half the hands


def test_fold_statistics_track_opponent():
    m_fold = _play(150, "folds", _folder)
    m_maniac = _play(150, "maniac", _maniac)
    assert m_fold["faced"] > 10 and m_fold["folds"] == m_fold["faced"]
    assert m_maniac["faced"] > 10 and m_maniac["folds"] == 0
    assert baseline._selectivity(m_maniac) < baseline._selectivity(m_fold)


def test_exception_falls_back_to_legal_action(monkeypatch):
    def boom(obs, memory):
        raise RuntimeError("simulated bug")
    monkeypatch.setattr(baseline, "decide", boom)
    memory = _play(20, "fallback", _random_opp)
    assert memory["fallbacks"] == memory["decisions"] > 0


def test_illegal_choice_falls_back(monkeypatch):
    monkeypatch.setattr(baseline, "decide", lambda obs, memory: Action.RAISE_POT)
    memory = _play(40, "illegal", _maniac)   # maniac often leaves raises illegal (cap / all-in)
    assert memory["fallbacks"] > 0
    assert memory["illegal"] == memory["fallbacks"] and memory["exceptions"] == 0


def test_decision_time_is_small():
    memory = _play(200, "timing", _random_opp)
    per = memory["think"] / memory["decisions"]
    assert per < 0.004, f"{per * 1e3:.2f} ms per decision"


# ---------------------------------------------------------------------------
# Regression checks: thinking budget tiers, visibility of fallbacks
# ---------------------------------------------------------------------------

def _play_with_spent(monkeypatch, spent, num_hands=120, seed="budget"):
    """Play as if `spent` seconds were already charged this half."""
    real_new_half = baseline.new_half

    def new_half_spent(config, memory):
        real_new_half(config, memory)
        memory["charged"] = spent
    monkeypatch.setattr(baseline, "new_half", new_half_spent)
    return _play(num_hands, seed, _random_opp)


def test_hard_budget_uses_cheap_policy_without_monte_carlo(monkeypatch):
    def no_mc(*a, **k):
        raise AssertionError("Monte Carlo must not run after the hard budget")
    monkeypatch.setattr(fasteval, "sample_equity", no_mc)
    memory = _play_with_spent(monkeypatch, baseline.CFG["hard_budget_s"] + 0.01)
    assert memory["cheap"] == memory["decisions"] > 0
    assert memory["mc_decisions"] == 0
    assert memory["exceptions"] == 0 and memory["fallbacks"] == 0


def test_emergency_budget_is_check_or_fold_only(monkeypatch):
    def no_compute(*a, **k):
        raise AssertionError("no strategy code after the emergency budget")
    monkeypatch.setattr(baseline, "decide", no_compute)
    monkeypatch.setattr(baseline, "cheap_action", no_compute)
    memory = _play_with_spent(monkeypatch, baseline.CFG["emergency_budget_s"] + 0.01, num_hands=60)
    assert memory["emergency"] == memory["decisions"] > 0
    assert memory["exceptions"] == 0


def test_budget_estimate_applies_grace_per_request(monkeypatch):
    memory = {}
    baseline.new_half(AgentConfig(), memory)
    obs = Game(make_deal("grace", 0), 0, 0).observation(0)
    clock = iter([0.0, 0.0001, 1.0, 1.0101])   # perf_counter: a 0.1 ms and a 10.1 ms decision
    monkeypatch.setattr(baseline.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(baseline.time, "process_time", lambda: 0.0)
    monkeypatch.setattr(baseline, "decide", lambda o, m: Action.CHECK_CALL)
    baseline.choose_action(obs, memory)
    assert memory["charged"] == 0.0           # 0.1 + 0.2 ms floor is inside the grace
    baseline.choose_action(obs, memory)
    expected = 0.0101 + baseline.CFG["overhead_per_call_s"] - baseline.CFG["grace_s"]
    assert memory["charged"] == pytest.approx(expected)
    assert baseline.estimated_charged(memory) == memory["charged"]


def test_cpu_spent_between_decisions_is_charged():
    memory = _play(20, "outside", _random_opp)
    before = memory["outside_cpu"]
    t = time.process_time()
    while time.process_time() - t < 0.2:   # simulate 0.2 s of plumbing work between requests
        pass
    charged_before = memory["charged"]
    obs = Game(make_deal("outside", 99), 0, 99).observation(0)
    baseline.choose_action(obs, memory)
    assert memory["outside_cpu"] - before >= 0.15
    assert memory["charged"] - charged_before >= 0.14   # counted against the budget


def test_soft_pacing_shrinks_samples():
    game = Game(make_deal("pace", 0), 0, 0)
    while game.street == Street.PREFLOP:
        game.apply(Action.CHECK_CALL)          # limp + check -> flop
    obs = game.observation(game.to_act)
    memory = {}
    baseline.new_half(AgentConfig(), memory)
    full = baseline._n_samples(obs, memory)
    memory["charged"] = baseline.CFG["soft_budget_s"]   # far ahead of pace on hand 0
    assert baseline._n_samples(obs, memory) < full
    assert baseline._n_samples(obs, memory) >= baseline.CFG["min_samples"]


def test_cheap_action_always_legal():
    memory = {}
    baseline.new_half(AgentConfig(), memory)
    rng = random.Random(4)
    checked = 0
    for hand in range(300):
        game = Game(make_deal("cheap", hand), hand % 2, hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                a = baseline.cheap_action(obs, memory)
                assert a in obs.legal_actions
                checked += 1
            else:
                a = rng.choice(obs.legal_actions)
            game.apply(a)
    assert checked > 300


def test_exceptions_are_counted_and_logged(monkeypatch, capsys):
    def boom(obs, memory):
        raise RuntimeError("simulated bug")
    monkeypatch.setattr(baseline, "decide", boom)
    memory = _play(30, "exc-log", _random_opp)
    err = capsys.readouterr().err
    assert memory["exceptions"] == memory["fallbacks"] == memory["decisions"] > 0
    assert memory["illegal"] == 0
    assert 1 <= err.count("OPS_EXCEPTION") <= baseline.CFG["max_logged_exceptions"]
    assert "simulated bug" in err


def test_stats_line_emitted_once_at_end_of_half(capsys):
    import json
    memory = _play(25, "stats", _random_opp)
    lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("OPS_STATS ")]
    assert len(lines) == 1
    st = json.loads(lines[0][len("OPS_STATS "):])
    assert st["decisions"] == memory["decisions"] and st["fallbacks"] == 0 and st["exceptions"] == 0


def test_on_hand_end_exception_is_contained(capsys):
    memory = {"pending_raise": 0, "num_hands": 1000}   # missing counters -> KeyError inside
    res = HandResult(hand_index=0, my_net=3, pot=6, went_to_showdown=False, board=(), opp_hole=None, winner="me")
    baseline.on_hand_end(res, memory)            # must not raise
    assert memory["hand_end_exceptions"] == 1
    assert "OPS_EXCEPTION in on_hand_end" in capsys.readouterr().err


def test_committed_table_cache_equals_fresh_build():
    fasteval.build()                                   # whatever the bot uses at startup
    cached = (fasteval.PAIR_ID, fasteval.TRIP_ID, fasteval.SC, fasteval.RANK_CATEGORY)
    assert fasteval._load_cache(fasteval.CACHE_PATH), "opsbot/fasteval_tables.npz missing or malformed"
    fasteval._build_from_evaluator()
    try:
        for a, b in zip(cached, (fasteval.PAIR_ID, fasteval.TRIP_ID, fasteval.SC, fasteval.RANK_CATEGORY)):
            assert np.array_equal(a, b)
    finally:
        fasteval.PAIR_ID, fasteval.TRIP_ID, fasteval.SC, fasteval.RANK_CATEGORY = cached


def test_bad_cache_falls_back_to_build(tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a numpy file")
    assert not fasteval._load_cache(str(bad))
    assert not fasteval._load_cache(str(tmp_path / "missing.npz"))
