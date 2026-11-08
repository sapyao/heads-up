"""Regression checks for tools/run_eval.py safety and reporting."""

import argparse
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import run_eval  # noqa: E402


def _listening_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def test_refuses_port_already_in_use():
    s, port = _listening_socket()
    try:
        with pytest.raises(run_eval.HarnessError, match="already in use"):
            run_eval.ensure_port_free(port)
    finally:
        s.close()
    run_eval.ensure_port_free(port)  # free again -> no error


def test_play_one_half_never_uses_an_existing_server(tmp_path):
    s, port = _listening_socket()  # stands in for an unrelated server (e.g. a stale agent.py)
    try:
        args = argparse.Namespace(agent1="agent.py", port1=port, port2=port + 1, deals=5)
        with pytest.raises(run_eval.HarnessError, match="already in use"):
            run_eval.play_one_half(args, "baseline-bots/baseline_random.py", "1", 1, {}, str(tmp_path))
    finally:
        s.close()


def test_bot_that_exits_is_detected_fast(tmp_path):
    script = tmp_path / "dies.py"
    script.write_text("import sys; print('boom'); sys.exit(3)\n")
    s, port = _listening_socket()
    s.close()
    log = str(tmp_path / "dies.log")
    proc, token = run_eval.start_bot(str(script), port, {}, log)
    handle = run_eval.HttpAgentHandle(f"http://127.0.0.1:{port}", run_eval.AgentConfig(), token)
    with pytest.raises(run_eval.HarnessError, match="exited with code 3"):
        run_eval.wait_own_bot(proc, handle, log)
    assert len(token) == 32


def test_reads_last_ops_stats_and_counts_exceptions(tmp_path):
    log = tmp_path / "a.log"
    log.write_text('INFO started\nOPS_EXCEPTION in choose_action:\nTraceback...\n'
                   'OPS_STATS {"decisions": 5, "fallbacks": 1, "exceptions": 1}\n', encoding="utf-8")
    st = run_eval.read_ops_stats(str(log))
    assert st["fallbacks"] == 1 and st["exceptions_logged"] == 1
    assert run_eval.read_ops_stats(str(tmp_path / "missing.log")) is None


def test_internal_summary_flags_missing_stats_and_fallbacks():
    good = {"exceptions": 0, "illegal": 0, "fallbacks": 0, "cheap": 2, "emergency": 0,
            "hand_end_exceptions": 0, "est_charged_s": 1.0}
    bad = dict(good, exceptions=2, fallbacks=2)
    s = run_eval._internal_summary([good, bad])
    assert s["exceptions"] == 2 and s["fallbacks"] == 2 and s["cheap"] == 4 and not s["missing"]
    assert run_eval._internal_summary([good, None])["missing"]
    assert not run_eval._internal_summary([None, None], expect_stats=False)["missing"]


def test_main_records_startup_failure_and_continues(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_half(args, opponent, seed, half, env1, log_dir):
        calls.append(seed)
        raise run_eval.HarnessError("bot did not answer")
    monkeypatch.setattr(run_eval, "play_one_half", fake_half)
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--seeds", "1,2", "--strict", "--json", str(out)])
    assert run_eval.main() == 1                 # strict -> failure exit code
    assert calls == ["1", "2"]                  # second seed still attempted
    assert "NOT PLAYED" in capsys.readouterr().out
    import json
    assert json.loads(out.read_text())["problem_matches"] == 2
