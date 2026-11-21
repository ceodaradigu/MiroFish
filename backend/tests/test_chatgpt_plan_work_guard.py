"""Account isolation while a local plan simulation owns a live child process."""

from __future__ import annotations

import json
import threading

from app.config import Config
from app.services import simulation_runner as runner_module
from app.services.plan_work_guard import plan_work_guard
from app.services.simulation_runner import SimulationRunner


def test_plan_session_stays_fixed_if_monitor_returns_before_process_exit(tmp_path, monkeypatch):
    """A monitor failure must not free the account while the child still runs."""

    simulation_id = "sim-plan-guard-monitor-failure"
    run_dir = tmp_path / "runs" / simulation_id
    run_dir.mkdir(parents=True)
    (run_dir / "simulation_config.json").write_text(
        json.dumps({"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}),
        encoding="utf-8",
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_twitter_simulation.py").write_text("pass\n", encoding="utf-8")
    process_exit = threading.Event()

    class Process:
        pid = 12345

        def poll(self):
            return 0 if process_exit.is_set() else None

        def wait(self, timeout=None):
            assert process_exit.wait(5)
            return 0

    monkeypatch.setattr(Config, "CHATGPT_PLAN_MODE", True)
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(SimulationRunner, "SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(SimulationRunner, "_monitor_simulation", classmethod(lambda _cls, *_args: None))
    termination_requested = threading.Event()
    monkeypatch.setattr(
        SimulationRunner,
        "_terminate_process",
        classmethod(lambda _cls, *_args: termination_requested.set()),
    )
    monkeypatch.setattr(SimulationRunner, "_sync_simulation_status", classmethod(lambda _cls, *_args: None))

    assert not plan_work_guard.has_work()
    monitor = None
    try:
        SimulationRunner.start_simulation(simulation_id, platform="twitter")
        monitor = SimulationRunner._monitor_threads[simulation_id]
        assert termination_requested.wait(5)
        assert plan_work_guard.has_work()
        assert monitor.is_alive()  # Waiting for the real child, not the monitor state.
    finally:
        process_exit.set()
        if monitor is not None:
            monitor.join(timeout=5)
        stream = SimulationRunner._stdout_files.pop(simulation_id, None)
        if stream is not None:
            stream.close()
        SimulationRunner._run_states.pop(simulation_id, None)
        SimulationRunner._processes.pop(simulation_id, None)
        SimulationRunner._monitor_threads.pop(simulation_id, None)
        SimulationRunner._action_queues.pop(simulation_id, None)
        SimulationRunner._stderr_files.pop(simulation_id, None)
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)
    assert not plan_work_guard.has_work()
