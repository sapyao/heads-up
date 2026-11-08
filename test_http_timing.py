"""Real-server timing checks: agent.py under uvicorn, refereed by judge_engine.

Unlike test_agent.py (in-process TestClient), this measures what the judge
actually charges: wall time per HTTP request minus the 1 ms grace, including
JSON parsing and FastAPI overhead.  Limits are set at half the real ones so a
pass leaves a comfortable margin:
    projected charged time per 1000-hand half  < 4.0 s   (bank 8 s)
    slowest single request                      < 0.5 s   (limit 1 s)
It also checks that the bot's own charged-time estimate (which drives its
budget fallbacks) does not under-count the judge's figure.
Timing depends on the machine: on a laptop running on battery, Windows may
throttle background processes enough to fail the budget assertions (measured
9-11 s projected on battery vs 0.4-0.7 s plugged in).  Run plugged in and idle.
"""

import os
import socket
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import run_eval  # noqa: E402
from judge_engine import HttpAgentHandle, play_half  # noqa: E402

DEALS = 300


class TimedHandle(HttpAgentHandle):
    """Judge handle that also records every /act latency."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.latencies = []

    def act(self, obs):
        t = time.perf_counter()
        try:
            return super().act(obs)
        finally:
            self.latencies.append(time.perf_counter() - t)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.parametrize("opponent", ["baseline-bots/baseline_always_half.py",
                                      "baseline-bots/sparring_tag.py"])
def test_real_server_time_budget(tmp_path, opponent):
    config = run_eval.AgentConfig(num_hands=DEALS)
    procs = []
    try:
        p1_port, p2_port = _free_port(), _free_port()
        log1, log2 = str(tmp_path / "a1.log"), str(tmp_path / "a2.log")
        p1, tok1 = run_eval.start_bot("agent.py", p1_port, {}, log1)
        procs.append(p1)
        p2, tok2 = run_eval.start_bot(opponent, p2_port, {}, log2)
        procs.append(p2)
        b1 = TimedHandle(f"http://127.0.0.1:{p1_port}", config, tok1)
        b2 = HttpAgentHandle(f"http://127.0.0.1:{p2_port}", config, tok2)
        startup = run_eval.wait_own_bot(p1, b1, log1)
        run_eval.wait_own_bot(p2, b2, log2)
        result = play_half(b1, b2, "http-timing", 1, DEALS)
    finally:
        for p in procs:
            run_eval.stop_bot(p)

    st = result.stats_agent1
    assert st.errors == 0 and not st.forfeited and not st.timed_out, st.error_messages[:1]
    projected = st.time_seconds * 1000 / DEALS
    lat = sorted(b1.latencies)
    p99 = lat[int(0.99 * (len(lat) - 1))]
    print(f"\n{os.path.basename(opponent)}: startup {startup:.2f}s, {st.calls} calls, "
          f"charged {st.time_seconds:.3f}s -> {projected:.2f}s per 1000 hands, "
          f"mean {sum(lat) / len(lat) * 1e3:.2f} ms, p99 {p99 * 1e3:.2f} ms, max {lat[-1] * 1e3:.2f} ms")
    stats = run_eval.read_ops_stats(log1)
    assert stats is not None and stats["exceptions"] == 0 and stats["fallbacks"] == 0
    print(f"   bot's own estimate {stats['est_charged_s']:.3f}s vs judge-charged {st.time_seconds:.3f}s "
          f"(think {stats['think']:.3f}s, outside_cpu {stats['outside_cpu']:.3f}s)")
    # The budget tiers act on the bot's own estimate, so it must not badly
    # under-count what the judge charges (this holds on slow machines too).
    assert stats["est_charged_s"] >= 0.75 * st.time_seconds
    assert startup < 7.5                 # limit 15 s
    assert projected < 4.0, f"projected {projected:.2f}s of the 8 s bank"
    assert lat[-1] < 0.5, f"slowest request {lat[-1]:.3f}s"
