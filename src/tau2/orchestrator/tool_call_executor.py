from collections.abc import Callable
from typing import Protocol

from tau2.data_model.message import ToolCall, ToolMessage

ToolCallDispatch = Callable[[ToolCall], ToolMessage]


class ToolCallExecutor(Protocol):
    """Optional wrapper for live orchestrator tool-call dispatch.

    Implementations may observe, govern, or short-circuit a live call. Returned
    messages must preserve the originating tool-call ``id`` and ``requestor``.
    Raising an exception aborts the step; the orchestrator never falls back to
    an unwrapped dispatch.
    """

    def execute(
        self,
        tool_call: ToolCall,
        dispatch: ToolCallDispatch,
    ) -> ToolMessage:
        """Execute one live tool call, optionally delegating to ``dispatch``."""
        ...


ToolCallExecutorFactory = Callable[[], ToolCallExecutor]
