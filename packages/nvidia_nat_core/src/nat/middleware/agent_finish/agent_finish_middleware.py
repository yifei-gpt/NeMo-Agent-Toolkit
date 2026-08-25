"""The boundary that turns a `finish` call into the answer of the agent that made it.

An agent whose menu holds no way to stop keeps choosing the best tool it has: one coordinator read
the same file 33 times with P=1.00 on every choice, because verifying it was genuinely the right
move and answering was not on offer. `finish` puts stopping on the menu; this middleware makes it
end one agent rather than the whole run.
"""
import logging
from collections.abc import AsyncIterator
from typing import Any

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from nat.middleware.function_middleware import CallNext
from nat.middleware.function_middleware import CallNextStream
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.function_middleware import FunctionMiddlewareContext

from .agent_finish_middleware_config import AgentFinishMiddlewareConfig

logger = logging.getLogger(__name__)


class AgentFinished(BaseException):
    """Carries the answer past a framework's agent loop. BaseException, so no `except Exception`
    inside that loop swallows it and hands the agent back its own stop."""

    def __init__(self, answer: str) -> None:
        super().__init__(answer)
        self.answer = answer


def _usable(value: Any) -> Any:
    """An agent out of turns hands back its last raw ReAct block, and a caller reads that monologue
    as an answer, gets nothing from it, and sends the same request again. Say what it is instead."""
    if not isinstance(value, str):
        return value
    body = value.strip().strip("`").strip()
    if body.startswith("Thought:") and "Action:" in body:
        return ("[this specialist ran out of steps and did not finish; what it had established is "
                "below. Treat it as partial -- sending it the same request returns the same thing.]"
                "\n\n" + body)
    return value


class AgentFinishMiddleware(FunctionMiddleware):

    def __init__(self, config: AgentFinishMiddlewareConfig) -> None:
        super().__init__()
        self._config = config

    async def function_middleware_invoke(self, *args: Any, call_next: CallNext,
                                         context: FunctionMiddlewareContext, **kwargs: Any) -> Any:
        try:
            return _usable(await call_next(*args, **kwargs))
        except AgentFinished as done:
            logger.info("%s finished", context.name)
            return done.answer

    async def function_middleware_stream(self, *args: Any, call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        try:
            async for chunk in call_next(*args, **kwargs):
                yield chunk
        except AgentFinished as done:
            yield done.answer


@register_middleware(config_type=AgentFinishMiddlewareConfig)
async def agent_finish_middleware(config: AgentFinishMiddlewareConfig, builder: Builder):
    yield AgentFinishMiddleware(config)
