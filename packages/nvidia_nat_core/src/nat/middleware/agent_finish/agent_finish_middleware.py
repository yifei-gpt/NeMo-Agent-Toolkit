"""The boundary that turns a `finish` call into the answer of the agent that made it.

An agent whose menu holds no way to stop keeps choosing the best tool it has: one coordinator read
the same file 33 times with P=1.00 on every choice, because verifying it was genuinely the right
move and answering was not on offer. `finish` puts stopping on the menu; this middleware makes it
end one agent rather than the whole run.
"""
import hashlib
import json
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


def _usable(value: Any, name: str = "this agent") -> Any:
    """A ReAct block is a monologue, not an answer: read as one it yields nothing and the caller
    asks again. Said with an Action the agent ran out of turns; said without one it never acted --
    under load CrewAI returned a bare Thought as the final answer, and the run produced no file."""
    if not isinstance(value, str):
        return value
    body = value.strip().strip("`").strip()
    # Empty reads as a broken call and gets resent
    if not body:
        return f"[{name} ran out of turns without writing an answer; see the workspace for what it did.]"
    if not body.startswith(("Thought:", "Action:", "Action Input:")):
        return value
    # `Action:` alone is the one a zero-argument tool provokes: CrewAI's parser wants an
    # `Action Input:` line beside it, the model leaves it off for a tool that needs none, and
    # after its retries the fragment comes back as the answer.
    note = ("ran out of steps and did not finish" if body.startswith("Thought:") and "Action:" in body
            else "answered with its plan and never called a tool" if body.startswith("Thought:")
            else "named an action its framework could not parse and never called a tool")
    return (f"[this specialist {note}; what it had established is below. Treat it as partial "
            "-- sending it the same request returns the same thing.]\n\n" + body)


# One call this often is a loop (worst honest repeat over 125 runs: 26)
CIRCLING = 40


def repeated(name: str, calls: int, same: int, content: str) -> str:
    """`content` after `calls` identical calls (incl. this), `same` identical answers: flag 3rd, withhold 6th+."""
    if calls < 2 or same < 2:
        return content
    if calls >= 6:
        return (f"[this is call {calls} of the same {name} with the same arguments, and the answer above has not "
                "changed. It is not being repeated here; work from the one you already have.]")
    return "[you have had this exact answer twice already and it will not change; try something else.]\n" + content


class Repeats:
    """One run's calls and answers, keyed on args too: `think` answers every call alike."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], list[str]] = {}

    def note(self, name: str, args, output):
        seen = self._seen.setdefault((name, json.dumps(args, sort_keys=True, default=str)), [])
        digest = hashlib.sha1(str(output).encode("utf-8", "replace")).hexdigest()
        said = repeated(name, len(seen) + 1, seen.count(digest), output) if isinstance(output, str) else output
        seen.append(digest)
        return said

    @property
    def circling(self) -> bool:
        return max(map(len, self._seen.values()), default=0) >= CIRCLING


class AgentFinishMiddleware(FunctionMiddleware):

    def __init__(self, config: AgentFinishMiddlewareConfig) -> None:
        super().__init__()
        self._config = config

    async def function_middleware_invoke(self, *args: Any, call_next: CallNext,
                                         context: FunctionMiddlewareContext, **kwargs: Any) -> Any:
        try:
            return _usable(await call_next(*args, **kwargs), context.name or "this agent")
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
