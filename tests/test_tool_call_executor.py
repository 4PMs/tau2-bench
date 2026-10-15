from collections.abc import Callable

import pytest

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.orchestrator.tool_call_executor import ToolCallExecutor
from tau2.user.user_simulator import UserSimulator


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def execute(
        self,
        tool_call: ToolCall,
        dispatch: Callable[[ToolCall], ToolMessage],
    ) -> ToolMessage:
        self.calls.append(tool_call)
        return dispatch(tool_call)


class FailingExecutor:
    def execute(
        self,
        tool_call: ToolCall,
        dispatch: Callable[[ToolCall], ToolMessage],
    ) -> ToolMessage:
        raise RuntimeError("executor failed")


def make_orchestrator(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
    executor: ToolCallExecutor | None = None,
) -> Orchestrator:
    environment = get_environment()
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )
    user = UserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )
    return Orchestrator(
        domain=domain_name,
        agent=agent,
        user=user,
        environment=environment,
        task=base_task,
        tool_call_executor=executor,
    )


def arm_live_tool_step(orchestrator: Orchestrator, tool_calls: list[ToolCall]) -> None:
    orchestrator.initialize()
    message = AssistantMessage(role="assistant", tool_calls=tool_calls)
    orchestrator.trajectory.append(message)
    orchestrator.message = message
    orchestrator.from_role = Role.AGENT
    orchestrator.to_role = Role.ENV


def test_live_tool_call_is_wrapped_exactly_once(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
) -> None:
    executor = RecordingExecutor()
    orchestrator = make_orchestrator(domain_name, get_environment, base_task, executor)
    tool_call = ToolCall(
        id="call-1", name="get_users", arguments={}, requestor="assistant"
    )
    arm_live_tool_step(orchestrator, [tool_call])

    orchestrator.step()

    assert executor.calls == [tool_call]
    assert isinstance(orchestrator.message, ToolMessage)
    assert orchestrator.message.id == tool_call.id


def test_empty_live_tool_call_id_is_generated_before_executor_and_trajectory(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
) -> None:
    executor = RecordingExecutor()
    orchestrator = make_orchestrator(domain_name, get_environment, base_task, executor)
    tool_call = ToolCall(
        id="", name="get_users", arguments={}, requestor="assistant"
    )
    arm_live_tool_step(orchestrator, [tool_call])

    orchestrator.step()

    assert tool_call.id
    assert executor.calls[0].id == tool_call.id
    assert isinstance(orchestrator.message, ToolMessage)
    assert orchestrator.message.id == tool_call.id
    trajectory_message = orchestrator.trajectory[-2]
    assert isinstance(trajectory_message, AssistantMessage)
    assert trajectory_message.tool_calls is not None
    assert trajectory_message.tool_calls[0].id == tool_call.id


def test_default_path_matches_environment_dispatch(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
) -> None:
    orchestrator = make_orchestrator(domain_name, get_environment, base_task)
    tool_call = ToolCall(
        id="call-default", name="get_users", arguments={}, requestor="assistant"
    )
    expected = get_environment().get_response(tool_call)
    arm_live_tool_step(orchestrator, [tool_call])

    orchestrator.step()

    assert orchestrator.message == expected


def test_multi_tool_order_and_ids_are_preserved(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
) -> None:
    executor = RecordingExecutor()
    orchestrator = make_orchestrator(domain_name, get_environment, base_task, executor)
    tool_calls = [
        ToolCall(
            id="call-first", name="get_users", arguments={}, requestor="assistant"
        ),
        ToolCall(
            id="call-second", name="get_users", arguments={}, requestor="assistant"
        ),
    ]
    arm_live_tool_step(orchestrator, tool_calls)

    orchestrator.step()

    assert executor.calls == tool_calls
    assert isinstance(orchestrator.message, MultiToolMessage)
    assert [message.id for message in orchestrator.message.tool_messages] == [
        "call-first",
        "call-second",
    ]


def test_executor_failure_does_not_fall_back_to_dispatch(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = make_orchestrator(
        domain_name, get_environment, base_task, FailingExecutor()
    )
    tool_call = ToolCall(
        id="call-fail", name="get_users", arguments={}, requestor="assistant"
    )
    arm_live_tool_step(orchestrator, [tool_call])
    dispatch_count = 0

    def unexpected_dispatch(tool_call: ToolCall) -> ToolMessage:
        nonlocal dispatch_count
        dispatch_count += 1
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content="unexpected",
            requestor=tool_call.requestor,
        )

    monkeypatch.setattr(orchestrator.environment, "get_response", unexpected_dispatch)

    with pytest.raises(RuntimeError, match="executor failed"):
        orchestrator.step()

    assert dispatch_count == 0


def test_initialization_action_is_not_instrumented(
    domain_name: str,
    get_environment: Callable[..., Environment],
    task_with_initialization_actions: Task,
) -> None:
    executor = RecordingExecutor()
    orchestrator = make_orchestrator(
        domain_name,
        get_environment,
        task_with_initialization_actions,
        executor,
    )

    orchestrator.initialize()

    assert executor.calls == []


def test_message_history_replay_is_not_instrumented(
    domain_name: str,
    get_environment: Callable[..., Environment],
    task_with_message_history: Task,
) -> None:
    executor = RecordingExecutor()
    orchestrator = make_orchestrator(
        domain_name,
        get_environment,
        task_with_message_history,
        executor,
    )

    orchestrator.initialize()

    assert executor.calls == []


def test_evaluator_replay_is_not_instrumented(
    domain_name: str,
    get_environment: Callable[..., Environment],
    task_with_env_assertions: Task,
) -> None:
    executor = RecordingExecutor()
    orchestrator = make_orchestrator(
        domain_name,
        get_environment,
        task_with_env_assertions,
        executor,
    )
    tool_call = ToolCall(
        id="call-evaluate",
        name="create_task",
        arguments={"user_id": "user_1", "title": "Test task"},
        requestor="assistant",
    )
    arm_live_tool_step(orchestrator, [tool_call])
    orchestrator.step()
    simulation = SimulationRun(
        id="simulation-1",
        task_id=task_with_env_assertions.id,
        start_time="start",
        end_time="end",
        duration=0.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=orchestrator.get_trajectory(),
    )

    evaluate_simulation(
        simulation=simulation,
        task=task_with_env_assertions,
        evaluation_type=EvaluationType.ENV,
        solo_mode=False,
        domain=domain_name,
    )

    assert executor.calls == [tool_call]


def test_executor_response_must_preserve_call_identity(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
) -> None:
    class MismatchedExecutor:
        def execute(
            self,
            tool_call: ToolCall,
            dispatch: Callable[[ToolCall], ToolMessage],
        ) -> ToolMessage:
            return ToolMessage(
                id="wrong-id",
                role="tool",
                content="blocked",
                requestor=tool_call.requestor,
            )

    orchestrator = make_orchestrator(
        domain_name, get_environment, base_task, MismatchedExecutor()
    )
    tool_call = ToolCall(
        id="call-identity", name="get_users", arguments={}, requestor="assistant"
    )
    arm_live_tool_step(orchestrator, [tool_call])

    with pytest.raises(ValueError, match="must preserve tool call id"):
        orchestrator.step()


def test_executor_cannot_mutate_originating_call_identity(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
) -> None:
    class MutatingExecutor:
        def execute(
            self,
            tool_call: ToolCall,
            dispatch: Callable[[ToolCall], ToolMessage],
        ) -> ToolMessage:
            tool_call.id = "changed-id"
            tool_call.requestor = "user"
            return ToolMessage(
                id=tool_call.id,
                role="tool",
                content="blocked",
                requestor=tool_call.requestor,
            )

    orchestrator = make_orchestrator(
        domain_name, get_environment, base_task, MutatingExecutor()
    )
    tool_call = ToolCall(
        id="original-id", name="get_users", arguments={}, requestor="assistant"
    )
    arm_live_tool_step(orchestrator, [tool_call])

    with pytest.raises(ValueError, match="must preserve tool call id"):
        orchestrator.step()

    assert tool_call.id == "original-id"
    assert tool_call.requestor == "assistant"


def test_user_live_call_can_be_short_circuited_without_dispatch(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = RecordingExecutor()

    def short_circuit(
        tool_call: ToolCall,
        dispatch: Callable[[ToolCall], ToolMessage],
    ) -> ToolMessage:
        executor.calls.append(tool_call)
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content="blocked",
            requestor=tool_call.requestor,
            error=True,
        )

    monkeypatch.setattr(executor, "execute", short_circuit)
    orchestrator = make_orchestrator(domain_name, get_environment, base_task, executor)
    tool_call = ToolCall(
        id="call-user", name="get_users", arguments={}, requestor="user"
    )
    orchestrator.initialize()
    message = UserMessage(role="user", tool_calls=[tool_call])
    orchestrator.trajectory.append(message)
    orchestrator.message = message
    orchestrator.from_role = Role.USER
    orchestrator.to_role = Role.ENV

    def unexpected_dispatch(tool_call: ToolCall) -> ToolMessage:
        raise AssertionError(f"unexpected dispatch: {tool_call.id}")

    monkeypatch.setattr(orchestrator.environment, "get_response", unexpected_dispatch)

    orchestrator.step()

    assert executor.calls == [tool_call]
    assert isinstance(orchestrator.message, ToolMessage)
    assert orchestrator.message.id == tool_call.id
    assert orchestrator.message.requestor == "user"
    assert orchestrator.message.content == "blocked"
    assert orchestrator.message.error is True
    assert orchestrator.message.model_dump()["error"] is True


def test_later_multi_tool_failure_does_not_dispatch_failed_call(
    domain_name: str,
    get_environment: Callable[..., Environment],
    base_task: Task,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailSecondExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(
            self,
            tool_call: ToolCall,
            dispatch: Callable[[ToolCall], ToolMessage],
        ) -> ToolMessage:
            self.calls.append(tool_call.id)
            if len(self.calls) == 2:
                raise RuntimeError("second call failed")
            return dispatch(tool_call)

    executor = FailSecondExecutor()
    orchestrator = make_orchestrator(domain_name, get_environment, base_task, executor)
    tool_calls = [
        ToolCall(
            id="call-first", name="get_users", arguments={}, requestor="assistant"
        ),
        ToolCall(
            id="call-second", name="get_users", arguments={}, requestor="assistant"
        ),
    ]
    arm_live_tool_step(orchestrator, tool_calls)
    dispatched: list[str] = []

    def record_dispatch(tool_call: ToolCall) -> ToolMessage:
        dispatched.append(tool_call.id)
        return ToolMessage(
            id=tool_call.id,
            role="tool",
            content="ok",
            requestor=tool_call.requestor,
        )

    monkeypatch.setattr(orchestrator.environment, "get_response", record_dispatch)

    with pytest.raises(RuntimeError, match="second call failed"):
        orchestrator.step()

    assert executor.calls == ["call-first", "call-second"]
    assert dispatched == ["call-first"]
