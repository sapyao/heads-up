"""Duplicate-match evaluation harness (tournament-style: fresh bot processes per half).

    .\\.venv\\Scripts\\python.exe tools/run_eval.py --agent1 agent.py ^
        --opponents baseline-bots/baseline_random.py,baseline-bots/baseline_calling_station.py ^
        --seeds 101,102,103 --deals 1000 --json results.json

For every (opponent, seed) it plays half 1 and half 2 with the official
judge_engine.play_half(), restarting BOTH bots before each half, then pairs the
halves with combine_halves().  Bots are reached at explicit http://127.0.0.1 URLs.
Agent variants can be selected with --agent1-env KEY=VALUE (passed to agent1 only).

Reports per match and pooled per opponent: bb/deal with a 95% CI, errors,
forfeits, time-outs and the judge-charged thinking time of agent1 per half,
plus agent1's own OPS_STATS line (internal exceptions, illegal choices that
were replaced by the fallback, cheap/emergency budget decisions).  Those are
invisible to the judge because the fallback hides them.  --strict makes the
exit code 1 if any of them, or any judge error, occurred.

Safety: before launching, each port must be free (nothing answers on
127.0.0.1:port), each launched process gets its own random AGENT_TOKEN,
and the harness checks the launched process is still alive after the health
check, so it never plays against an unrelated server already running.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import time

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUNDLE)

from evaluator import warmup  # noqa: E402
from game import AgentConfig  # noqa: E402
from judge_engine import (STARTUP_TIMEOUT_SECONDS, HttpAgentHandle,  # noqa: E402
                          combine_halves, play_half)


class HarnessError(RuntimeError):
    pass


def ensure_port_free(port: int) -> None:
    """Refuse to start if anything already accepts connections on the port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            pass
    except OSError:
        return
    raise HarnessError(f"port {port} is already in use by another process; stop it or pick "
                       f"another --port1/--port2 (refusing to play against an unknown server)")


def start_bot(script: str, port: int, extra_env: dict[str, str], log_path: str):
    """Launch a bot on 127.0.0.1:port. Returns (process, token, log path)."""
    ensure_port_free(port)
    token = secrets.token_hex(16)
    env = {**os.environ, "PORT": str(port), "PYTHONPATH": BUNDLE, **extra_env, "AGENT_TOKEN": token}
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([sys.executable, os.path.join(BUNDLE, script)], cwd=BUNDLE,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, token


def wait_own_bot(proc: subprocess.Popen, handle: HttpAgentHandle, log_path: str) -> float:
    """Wait until OUR process answers GET /; fail fast if it exits (e.g. port clash)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < STARTUP_TIMEOUT_SECONDS:
        if proc.poll() is not None:
            raise HarnessError(f"bot exited with code {proc.returncode} before answering; "
                               f"log {log_path}:\n{_tail(log_path)}")
        if handle.wait_healthy(0.5):
            if proc.poll() is not None:  # something else answered while ours died
                raise HarnessError(f"bot process exited; an unrelated server answered on {handle.base_url}")
            return time.perf_counter() - t0
    raise HarnessError(f"bot did not answer within {STARTUP_TIMEOUT_SECONDS}s; log {log_path}")


def _tail(path: str, n: int = 15) -> str:
    try:
        with open(path, "rb") as fh:
            return b"\n".join(fh.read().splitlines()[-n:]).decode("utf-8", "replace")
    except OSError:
        return "(no log)"


def read_ops_stats(log_path: str):
    """Last OPS_STATS line the bot wrote, plus OPS_EXCEPTION count, or None if absent."""
    stats, exc_logged = None, 0
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("OPS_STATS "):
                    stats = json.loads(line[len("OPS_STATS "):])
                elif line.startswith("OPS_EXCEPTION"):
                    exc_logged += 1
    except OSError:
        return None
    if stats is not None:
        stats["exceptions_logged"] = exc_logged
    return stats


def stop_bot(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def play_one_half(args, opponent: str, seed: str, half: int, env1: dict, log_dir: str):
    """Returns (HalfResult, agent1 startup seconds, agent1 OPS_STATS or None)."""
    config = AgentConfig(num_hands=args.deals)
    tag = f"{os.path.splitext(os.path.basename(opponent))[0]}_s{seed}_h{half}"
    log1 = os.path.join(log_dir, f"agent1_{tag}.log")
    log2 = os.path.join(log_dir, f"agent2_{tag}.log")
    procs = []
    try:
        p1, tok1 = start_bot(args.agent1, args.port1, env1, log1)
        procs.append(p1)
        p2, tok2 = start_bot(opponent, args.port2, {}, log2)
        procs.append(p2)
        b1 = HttpAgentHandle(f"http://127.0.0.1:{args.port1}", config, tok1)
        b2 = HttpAgentHandle(f"http://127.0.0.1:{args.port2}", config, tok2)
        startup = wait_own_bot(p1, b1, log1)
        wait_own_bot(p2, b2, log2)
        result = play_half(b1, b2, seed, half, args.deals)
    finally:
        for p in procs:
            stop_bot(p)
    return result, startup, read_ops_stats(log1)


def pooled(paired: list[float]) -> tuple[float, float, float]:
    mean = sum(paired) / len(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired)) if len(paired) > 1 else 0.0
    return mean, mean - 1.96 * se, mean + 1.96 * se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent1", default="agent.py")
    ap.add_argument("--agent1-env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--opponents", default="baseline-bots/baseline_random.py")
    ap.add_argument("--seeds", default="101")
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--port1", type=int, default=5108)
    ap.add_argument("--port2", type=int, default=5109)
    ap.add_argument("--json")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any judge error, forfeit, time-out, internal exception or fallback")
    ap.add_argument("--no-ops-stats", action="store_true",
                    help="agent1 does not emit OPS_STATS (e.g. bench/starter_agent.py)")
    args = ap.parse_args()
    env1 = dict(kv.split("=", 1) for kv in args.agent1_env)
    warmup()
    log_dir = tempfile.mkdtemp(prefix="ops_eval_")
    report = {"agent1": args.agent1, "agent1_env": env1, "deals": args.deals, "matches": [], "pooled": {}}
    print(f"agent1={args.agent1} env={env1} deals={args.deals} (bot logs: {log_dir})", flush=True)
    problems = 0
    for opponent in args.opponents.split(","):
        all_paired = []
        for seed in args.seeds.split(","):
            try:
                h1, s1, o1 = play_one_half(args, opponent, seed, 1, env1, log_dir)
                h2, s2, o2 = play_one_half(args, opponent, seed, 2, env1, log_dir)
            except HarnessError as exc:
                # Not scored: a bot failed to start (in the tournament that is a forfeit).
                problems += 1
                report["matches"].append({"opponent": os.path.basename(opponent), "seed": seed,
                                          "harness_error": str(exc)})
                print(f"{os.path.basename(opponent):<30} seed={seed:<6} NOT PLAYED <-- PROBLEM: {exc}",
                      flush=True)
                continue
            m = combine_halves(h1, h2)
            st = m.agent_stats["agent1"]
            row = {
                "opponent": os.path.basename(opponent), "seed": seed,
                "bb_per_deal": round(m.mean_bb, 4), "ci": [round(m.ci[0], 4), round(m.ci[1], 4)],
                "verdict": m.verdict,
                "errors": [s.errors for s in st], "forfeited": [s.forfeited for s in st],
                "timed_out": [s.timed_out for s in st],
                "think_s": [round(s.time_seconds, 3) for s in st],
                "calls": [s.calls for s in st],
                "opp_errors": [s.errors for s in m.agent_stats["agent2"]],
                "startup_s": [round(s1, 2), round(s2, 2)],
                "error_messages": [msg for s in st for msg in s.error_messages][:3],
                "ops_stats": [o1, o2],
            }
            internal = _internal_summary(row["ops_stats"], not args.no_ops_stats)
            row["internal"] = internal
            bad = (sum(row["errors"]) or any(row["forfeited"]) or any(row["timed_out"])
                   or internal["missing"] or internal["exceptions"] or internal["fallbacks"]
                   or internal["hand_end_exceptions"])
            problems += bool(bad)
            report["matches"].append(row)
            all_paired.extend(m.paired_bb)
            print(f"{row['opponent']:<30} seed={seed:<6} {row['bb_per_deal']:+8.3f} bb/deal "
                  f"CI[{row['ci'][0]:+.3f},{row['ci'][1]:+.3f}] {m.verdict:<11} err={row['errors']} "
                  f"forfeit={row['forfeited']} think={row['think_s']}s calls={row['calls']} "
                  f"startup={row['startup_s']}s", flush=True)
            print(f"   internal: exceptions={internal['exceptions']} illegal={internal['illegal']} "
                  f"fallbacks={internal['fallbacks']} cheap={internal['cheap']} "
                  f"emergency={internal['emergency']} est_charged={internal['est_charged_s']}s"
                  + ("  (OPS_STATS missing for a half)" if internal["missing"] else "")
                  + ("  <-- PROBLEM" if bad else ""), flush=True)
            if row["error_messages"]:
                print("   first error:", row["error_messages"][0][:500], flush=True)
        if not all_paired:
            continue
        mean, lo, hi = pooled(all_paired)
        report["pooled"][os.path.basename(opponent)] = {"bb_per_deal": mean, "ci": [lo, hi], "deals": len(all_paired)}
        print(f"POOLED {os.path.basename(opponent):<23} {mean:+8.3f} bb/deal CI[{lo:+.3f},{hi:+.3f}] "
              f"over {len(all_paired)} paired deals", flush=True)
    report["problem_matches"] = problems
    print(f"matches with problems: {problems}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 1 if (args.strict and problems) else 0


def _internal_summary(per_half: list, expect_stats: bool = True) -> dict:
    """Sum agent1's OPS_STATS over halves. A half without the line counts as
    missing (a problem) unless expect_stats is False (e.g. the starter)."""
    out = {k: 0 for k in ("exceptions", "illegal", "fallbacks", "cheap", "emergency", "hand_end_exceptions")}
    out["est_charged_s"] = []
    out["missing"] = False
    for st in per_half:
        if st is None:
            out["missing"] = out["missing"] or expect_stats
            continue
        for k in ("exceptions", "illegal", "fallbacks", "cheap", "emergency", "hand_end_exceptions"):
            out[k] += int(st.get(k) or 0)
        out["est_charged_s"].append(st.get("est_charged_s"))
    return out



if __name__ == "__main__":
    sys.exit(main())
