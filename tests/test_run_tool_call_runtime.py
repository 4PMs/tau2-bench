from __future__ import annotations

from copy import deepcopy
from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest

from tau2.data_model.message import ToolMessage
from tau2.data_model.simulation import RunConfig, SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import run_domain, run_task, run_tasks


def make_simulation(task_id: str, message: ToolMessage | None = None) -> SimulationRun:
    return SimulationRun(
        id=f"simulation-{task_id}",
        task_id=task_id,
        start_time="start",
        end_time="end",
        duration=0.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[] if message is None else [message],
    )


def test_run_task_injects_executor_and_preserves_tool_error(
    domain_name: str,
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = object()
    captured: dict[str, Any] = {}
    blocked = ToolMessage(
        id="blocked-call",
        role="tool",
        content="blocked",
        requestor="assistant",
        error=True,
    )

    class RecordingOrchestrator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def run(self) -> SimulationRun:
            return make_simulation(base_task.id, blocked)

    monkeypatch.setattr("tau2.run.Orchestrator", RecordingOrchestrator)
    monkeypatch.setattr(
        "tau2.run.evaluate_simulation", lambda **_: SimpleNamespace(reward=0.0)
    )

    simulation = run_task(
        domain=domain_name,
        task=base_task,
        agent="llm_agent",
        user="user_simulator",
        llm_agent="unused",
        llm_args_agent={},
        llm_user="unused",
        llm_args_user={},
        evaluation_type=EvaluationType.ENV,
        tool_call_executor=executor,
    )

    assert captured["tool_call_executor"] is executor
    assert isinstance(simulation.messages[0], ToolMessage)
    assert simulation.messages[0].error is True


def test_run_tasks_creates_one_executor_per_task_trial(
    domain_name: str,
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_task = deepcopy(base_task)
    second_task.id = f"{base_task.id}-second"
    created: list[tuple[str, int, object]] = []
    received: list[object] = []
    lock = Lock()

    def factory(task_id: str, trial: int) -> object:
        executor = object()
        with lock:
            created.append((task_id, trial, executor))
        return executor

    def fake_run_task(
        *, task: Task, tool_call_executor: object, **_: Any
    ) -> SimulationRun:
        with lock:
            received.append(tool_call_executor)
        return make_simulation(task.id)

    monkeypatch.setattr("tau2.run.run_task", fake_run_task)

    results = run_tasks(
        domain=domain_name,
        tasks=[base_task, second_task],
        agent="llm_agent",
        user="user_simulator",
        llm_args_agent={},
        llm_args_user={},
        num_trials=2,
        max_concurrency=4,
        console_display=False,
        evaluation_type=EvaluationType.ENV,
        tool_call_executor_factory=factory,
    )

    assert len(created) == 4
    assert len(received) == 4
    assert {(task_id, trial) for task_id, trial, _ in created} == {
        (base_task.id, 0),
        (base_task.id, 1),
        (second_task.id, 0),
        (second_task.id, 1),
    }
    assert {id(executor) for _, _, executor in created} == {
        id(executor) for executor in received
    }
    assert len({id(executor) for executor in received}) == 4
    assert "tool_call_executor" not in results.model_dump()
    assert "tool_call_executor_factory" not in results.model_dump()


def test_run_tasks_without_factory_passes_no_executor(
    domain_name: str,
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object | None] = []

    def fake_run_task(
        *, task: Task, tool_call_executor: object | None = None, **_: Any
    ) -> SimulationRun:
        received.append(tool_call_executor)
        return make_simulation(task.id)

    monkeypatch.setattr("tau2.run.run_task", fake_run_task)

    results = run_tasks(
        domain=domain_name,
        tasks=[base_task],
        agent="llm_agent",
        user="user_simulator",
        llm_args_agent={},
        llm_args_user={},
        console_display=False,
        evaluation_type=EvaluationType.ENV,
    )

    assert received == [None]
    assert set(results.model_dump()) == {"timestamp", "info", "tasks", "simulations"}


def test_run_tasks_emits_completed_simulation_to_agent_report_sink(
    domain_name: str,
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[SimulationRun] = []

    def fake_run_task(*, task: Task, **_: Any) -> SimulationRun:
        return make_simulation(task.id)

    monkeypatch.setattr("tau2.run.run_task", fake_run_task)

    results = run_tasks(
        domain=domain_name,
        tasks=[base_task],
        agent="llm_agent",
        user="user_simulator",
        llm_args_agent={},
        llm_args_user={},
        console_display=False,
        evaluation_type=EvaluationType.ENV,
        agent_report_sink=received.append,
    )

    assert received == results.simulations
    assert received[0].trial == 0
    assert "agent_report_sink" not in results.model_dump()


def test_agent_report_sink_failure_aborts_batch(
    domain_name: str,
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_task(*, task: Task, **_: Any) -> SimulationRun:
        return make_simulation(task.id)

    def failing_sink(_: SimulationRun) -> None:
        raise RuntimeError("raw report persistence failed")

    monkeypatch.setattr("tau2.run.run_task", fake_run_task)

    with pytest.raises(RuntimeError, match="raw report persistence failed"):
        run_tasks(
            domain=domain_name,
            tasks=[base_task],
            agent="llm_agent",
            user="user_simulator",
            llm_args_agent={},
            llm_args_user={},
            console_display=False,
            evaluation_type=EvaluationType.ENV,
            agent_report_sink=failing_sink,
        )


def test_resumed_simulation_does_not_create_runtime_or_emit_report(
    domain_name: str,
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_run_task(*, task: Task, seed: int, **_: Any) -> SimulationRun:
        simulation = make_simulation(task.id)
        simulation.seed = seed
        return simulation

    monkeypatch.setattr("tau2.run.run_task", fake_run_task)
    save_to = tmp_path / "resume.json"
    common = {
        "domain": domain_name,
        "tasks": [base_task],
        "agent": "llm_agent",
        "user": "user_simulator",
        "llm_args_agent": {},
        "llm_args_user": {},
        "console_display": False,
        "evaluation_type": EvaluationType.ENV,
        "save_to": save_to,
        "seed": 300,
    }
    run_tasks(**common)
    monkeypatch.setattr(
        "tau2.run.ConsoleDisplay.console.input",
        lambda *_args, **_kwargs: "y",
    )
    created: list[tuple[str, int]] = []
    reports: list[SimulationRun] = []

    def factory(task_id: str, trial: int) -> Any:
        created.append((task_id, trial))
        return object()

    resumed = run_tasks(
        **common,
        tool_call_executor_factory=factory,
        agent_report_sink=reports.append,
    )

    assert len(resumed.simulations) == 1
    assert created == []
    assert reports == []


def test_run_domain_forwards_runtime_factory(
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = RunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        task_ids=[base_task.id],
        llm_agent="unused",
        llm_args_agent={},
        llm_user="unused",
        llm_args_user={},
        save_to=None,
    )

    def factory(task_id: str, trial: int) -> object:
        return object()

    reports: list[SimulationRun] = []

    captured: dict[str, Any] = {}

    monkeypatch.setattr("tau2.run.get_tasks", lambda *_args, **_kwargs: [base_task])

    def fake_run_tasks(**kwargs: Any):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("tau2.run.run_tasks", fake_run_tasks)
    monkeypatch.setattr("tau2.run.compute_metrics", lambda _: {})
    monkeypatch.setattr("tau2.run.ConsoleDisplay.display_run_config", lambda _: None)
    monkeypatch.setattr("tau2.run.ConsoleDisplay.display_agent_metrics", lambda _: None)

    result = run_domain(
        run_config,
        tool_call_executor_factory=factory,
        agent_report_sink=reports.append,
    )

    assert result is not None
    assert captured["tool_call_executor_factory"] is factory
    assert captured["agent_report_sink"] == reports.append
