"""
sparring_tag.py - tougher practice opponent: a tight-aggressive heuristic (TAGBot).

Written independently of opsbot/ (no Monte Carlo, no shared code) so it is not
a mirror of our own strategy. It folds weak hands, value-bets made hands,
semi-bluffs draws, c-bets and bluffs at seeded random frequencies, and
re-raises its strongest hands, so it exercises fold equity, aggression from a
selective opponent, and check-raises that the four fixed baseline bots never do.
Built from the baseline-bot template (unchanged server plumbing), PORT 5009:
    python baseline-bots/sparring_tag.py
"""

import os
import sys
import traceback

# This file lives in baseline-bots/, so look one folder up for game.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import get_table, omaha_score, warmup
from game import Action, AgentConfig, HandResult, Observation, Street

PARTICIPANT = "Challenges Team"
AGENT_NAME = "TAGBot"


# ===== YOUR CODE HERE =====

import random
from itertools import combinations

warmup()
_TABLE = get_table()
_PAIRS = list(combinations(range(5), 2))
STRAIGHT, TRIPS, FULL_HOUSE = 3, 4, 5
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def _pre_score(hole) -> float:
    ranks = [c >> 2 for c in hole]
    rc = [ranks.count(r) for r in range(9)]
    sc = [sum(1 for c in hole if c & 3 == s) for s in range(4)]
    s = sum(ranks) / 8.0
    for r in range(9):
        if rc[r] == 2:
            s += 0.6 + r / 8.0
        elif rc[r] >= 3:
            s -= 0.8
    for su in range(4):
        if sc[su] in (2, 3):
            top = max(c >> 2 for c in hole if c & 3 == su)
            s += 0.5 + (0.5 if top == 8 else 0.0)
        elif sc[su] >= 4:
            s -= 0.5
    d = sorted({r for r in ranks})
    s += 0.25 * sum(1 for a, b in zip(d, d[1:]) if b - a <= 2)
    return s


_QS = sorted(_pre_score(random.Random(i).sample(range(36), 5)) for i in range(20000))


def _pct(hole) -> float:
    from bisect import bisect_right
    return bisect_right(_QS, _pre_score(hole)) / len(_QS)


def _cat(hole, board) -> int:
    """Best category using exactly 2 hole + 3 of the visible board cards."""
    best = 0
    tri = [(1 << a) | (1 << b) | (1 << c) for a, b, c in combinations(board, 3)]
    for i, j in _PAIRS:
        pm = (1 << hole[i]) | (1 << hole[j])
        for t in tri:
            v = _TABLE[pm | t]
            if v > best:
                best = v
    return best >> 20


def _outs(hole, board, cat_now) -> int:
    """Unseen cards that would give a straight or better and improve the category."""
    dead = set(hole) | set(board)
    target = max(STRAIGHT, cat_now + 1)
    return sum(1 for c in range(36) if c not in dead and _cat(hole, list(board) + [c]) >= target)


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = random.Random(7)


def _bet(obs, size):
    want = Action.RAISE_POT if size == "pot" else Action.RAISE_HALF
    if want in obs.legal_actions:
        return want
    for a in RAISES:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _fold_or_check(obs):
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    rng = memory["rng"]
    act = _decide(obs, rng)
    return act if act in obs.legal_actions else _fold_or_check(obs)


def _decide(obs, rng):
    hole, board = list(obs.hole_cards), list(obs.board)
    pot, call = obs.pot, obs.to_call
    if obs.street == Street.PREFLOP:
        p = _pct(hole)
        r = sum(1 for a in obs.action_history if not a.is_me and a.action in RAISES)
        if call == 0:
            return _bet(obs, "pot") if p > 0.75 else Action.CHECK_CALL
        if r == 0:
            if p > 0.5:
                return _bet(obs, "pot")
            return Action.CHECK_CALL if p > 0.15 else Action.FOLD
        if p > 0.9 and obs.raises_this_street < 4:
            return _bet(obs, "pot")
        if p > 0.4 + 0.15 * (r - 1) and (call <= 0.5 * obs.my_stack or p > 0.75):
            return Action.CHECK_CALL
        return Action.FOLD

    i_raised_pre = any(a.is_me and a.action in RAISES and a.street == Street.PREFLOP
                       for a in obs.action_history)
    cat = omaha_score(hole, board) >> 20 if len(board) == 5 else _cat(hole, board)
    if obs.street == Street.RIVER:
        if cat >= FULL_HOUSE:
            return _bet(obs, "pot")
        if cat >= STRAIGHT:
            if call == 0:
                return _bet(obs, "half")
            return Action.CHECK_CALL if call <= 0.75 * pot else Action.FOLD
        if cat == 2:
            return Action.CHECK_CALL if call <= 0.5 * pot else Action.FOLD
        if call == 0 and rng.random() < 0.15:
            return _bet(obs, "pot")
        return _fold_or_check(obs)

    outs = _outs(hole, board, cat)
    if cat >= TRIPS:
        return _bet(obs, "pot") if obs.raises_this_street < 2 else Action.CHECK_CALL
    if cat == STRAIGHT or outs >= 8:
        if outs >= 12 and rng.random() < 0.25:
            return _bet(obs, "pot")
        if call == 0:
            return _bet(obs, "half")
        return Action.CHECK_CALL
    if cat == 2:
        if call == 0:
            return _bet(obs, "half")
        return Action.CHECK_CALL if call <= 0.5 * pot else Action.FOLD
    if outs >= 4 and call <= 0.4 * pot:
        return Action.CHECK_CALL
    if call == 0 and rng.random() < (0.3 if i_raised_pre else 0.1):
        return _bet(obs, "half")
    return _fold_or_check(obs)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Nothing to learn."""

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
# Keep your per-half state in `memory`: other global variables are NOT reset
# between halves.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5009 for your own testing convenience
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
