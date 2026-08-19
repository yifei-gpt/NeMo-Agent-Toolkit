# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Middleware that stops the two ways agent loops crash: context blowup and repeat loops.

Both agent loops append tool output to state without trimming or deduplicating, so the
bounds have to be enforced around the tool call itself.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.cli.register_workflow import register_middleware
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext

logger = logging.getLogger(__name__)

# One instance is shared by every function that names it and lives for the whole eval, so the
# per-run tables have to evict instead of keeping a question's history forever.
_MAX_RUNS = 16
_MAX_KEYS_PER_RUN = 128
_MAX_CACHED_CHARS = 8000


class OutputLimitConfig(FunctionMiddlewareBaseConfig, name="tool_output_limit"):
    """Truncates tool output so a long transcript cannot exceed the context window."""

    max_chars: int = Field(default=8000, gt=0, description="Maximum characters returned by an intercepted call")
    # Per call, the cap says nothing about the transcript, and the run then dies mid-way with
    # nothing to show. Measured on the runs that overflowed a 256k-token window: 260k chars of tool
    # output (~70k tokens) but 246k input tokens, because every turn resends the whole history --
    # the agent's own reasoning and the tool schemas dwarf the outputs. So the cap has to sit well
    # under what the outputs alone would allow.
    max_chars_per_run: int = Field(default=200_000, ge=0,
                                   description="Total characters of tool output per run; 0 disables")
    notice: str = Field(default="\n...[output truncated; narrow your query to see more]",
                        description="Appended when output is truncated")


class RepeatBreakerConfig(FunctionMiddlewareBaseConfig, name="repeat_call_breaker"):
    """Answers an identical repeated call with a nudge instead of running it again."""

    max_repeats: int = Field(default=2, ge=1, description="Identical calls allowed before the loop is broken")
    max_calls_per_run: int = Field(default=0, ge=0,
                                   description="Total calls allowed per workflow run; 0 disables the budget")
    notice: str = Field(default=("You already ran this exact call and got the same result. Do not repeat it. "
                                 "Either try a materially different query, or answer with what you have."),
                        description="Returned in place of the repeated call's output")
    budget_notice: str = Field(default=("You have used your tool budget for this question. Do not call any more "
                                        "tools. Answer now with the information you already have."),
                               description="Returned once the per-run budget is spent")
    # Frameworks that expose no iteration cap ignore the notice and keep calling, so the
    # budget needs a hard edge that raises rather than another message they can discard.
    hard_stop_multiple: int = Field(default=3, ge=1,
                                    description="Raise once calls exceed this multiple of the budget")


def _evict(table: OrderedDict, limit: int) -> None:
    while len(table) > limit:
        table.popitem(last=False)


def _stable(value: Any) -> Any:
    # A plain object's repr carries its address, which hashes two identical calls apart and
    # silently disables dedup, so arguments are reduced to values before hashing.
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        return [(str(k), _stable(v)) for k, v in sorted(value.items(), key=lambda item: str(item[0]))]
    if isinstance(value, (set, frozenset)):
        return sorted(repr(_stable(item)) for item in value)
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    fields = getattr(value, "__dict__", None)
    return (type(value).__name__, _stable(fields) if fields else repr(value))


class _RunState:

    def __init__(self) -> None:
        self.seen: OrderedDict[str, int] = OrderedDict()
        self.results: OrderedDict[str, Any] = OrderedDict()
        self.totals: dict[str, int] = defaultdict(int)
        self.last_key: str | None = None


class OutputLimitMiddleware(FunctionMiddleware):

    def __init__(self, config: OutputLimitConfig) -> None:
        super().__init__()
        self._config = config
        self._spent: OrderedDict[str, int] = OrderedDict()

    def _room(self, taken: int) -> int:
        """Characters this call may return, once the run's own transcript budget is counted."""
        cap = self._config.max_chars
        if not self._config.max_chars_per_run:
            return cap
        run_id = Context.get().workflow_run_id or "no-run"
        spent = self._spent.pop(run_id, 0)
        room = max(0, min(cap, self._config.max_chars_per_run - spent))
        # Charge what the caller will actually see, not what it asked for, or a run spends its
        # budget on characters that were truncated away.
        self._spent[run_id] = spent + min(taken, room)
        _evict(self._spent, _MAX_RUNS)
        return room

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        # A dict or model output fills the transcript just as fast as a long string does.
        text = context.output if isinstance(context.output, str) else str(context.output)
        room = self._room(len(text))
        if len(text) <= room:
            return None
        logger.info("Truncated %s output from %d to %d chars",
                    context.function_context.name,
                    len(text),
                    room)
        context.output = text[:room] + self._config.notice
        return context

    async def function_middleware_stream(self, *args: Any, call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        # The inherited stream path runs post_invoke per chunk, which caps each chunk instead
        # of the call, so the remaining budget is tracked across the whole stream here.
        remaining = self._room(self._config.max_chars)
        async for chunk in call_next(*args, **kwargs):
            text = chunk if isinstance(chunk, str) else str(chunk)
            if len(text) > remaining:
                logger.info("Truncated %s stream at its remaining budget", context.name)
                yield text[:remaining] + self._config.notice
                return
            remaining -= len(text)
            yield chunk


class RepeatBreakerMiddleware(FunctionMiddleware):

    def __init__(self, config: RepeatBreakerConfig) -> None:
        super().__init__()
        self._config = config
        self._runs: OrderedDict[str, _RunState] = OrderedDict()

    def _run_state(self) -> _RunState:
        # Counting per workflow run keeps one question's repeats from affecting the next.
        run_id = Context.get().workflow_run_id or "no-run"
        state = self._runs.pop(run_id, None) or _RunState()
        self._runs[run_id] = state
        _evict(self._runs, _MAX_RUNS)
        return state

    def _key(self, name: str | None, args: tuple, kwargs: dict) -> str:
        return hashlib.sha256(repr(_stable([name, args, kwargs])).encode()).hexdigest()

    def _budget_spent(self, state: _RunState, name: str) -> str | None:
        # A hard per-run budget is what actually guarantees termination: paraphrased queries
        # slip past exact-match dedup and would otherwise run to the graph recursion limit.
        if not self._config.max_calls_per_run:
            return None
        # Counted per function because one instance serves every function it is attached to,
        # and counted before dedup so an agent spamming one identical call still hard-stops.
        state.totals[name] += 1
        total = state.totals[name]
        if total > self._config.max_calls_per_run * self._config.hard_stop_multiple:
            raise RuntimeError(f"Tool budget exceeded {self._config.hard_stop_multiple}x on {name} "
                               f"({total} calls); the agent ignored the budget notice.")
        if total > self._config.max_calls_per_run:
            logger.info("Tool budget spent on %s (%d calls)", name, total)
            return self._config.budget_notice
        return None

    def _repeated(self, state: _RunState, key: str) -> bool:
        # Only back-to-back repeats are loops. An identical call with a different one in between may
        # see state that call changed -- listing a directory after writing to it -- and answering
        # that from cache serves the pre-write world as if it were current.
        consecutive, state.last_key = state.last_key == key, key
        # Re-inserting keeps a hot repeat loop at the young end, so eviction never resets it.
        count = (state.seen.pop(key, 0) + 1) if consecutive else 1
        state.seen[key] = count
        _evict(state.seen, _MAX_KEYS_PER_RUN)
        return count > self._config.max_repeats

    def _remember(self, state: _RunState, key: str, result: Any) -> None:
        # Oversized results are dropped rather than cached; a miss only costs the notice below.
        if len(result if isinstance(result, str) else str(result)) > _MAX_CACHED_CHARS:
            return
        state.results[key] = result
        _evict(state.results, _MAX_KEYS_PER_RUN)

    async def function_middleware_invoke(self, *args: Any, call_next: CallNext,
                                         context: FunctionMiddlewareContext, **kwargs: Any) -> Any:
        state = self._run_state()
        spent = self._budget_spent(state, context.name)
        if spent is not None:
            return spent

        key = self._key(context.name, args, kwargs)
        if self._repeated(state, key):
            logger.info("Broke repeat loop on %s", context.name)
            # Handing back the result this exact call already produced lets the caller move on;
            # a bare notice leaves it with nothing, so it calls again and burns the budget.
            return state.results.get(key, self._config.notice)
        result = await call_next(*args, **kwargs)
        self._remember(state, key, result)
        return result

    async def function_middleware_stream(self, *args: Any, call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        # The inherited stream path only runs pre/post_invoke, so a streaming tool would
        # otherwise get neither dedup nor budget.
        state = self._run_state()
        spent = self._budget_spent(state, context.name)
        if spent is not None:
            yield spent
            return

        if self._repeated(state, self._key(context.name, args, kwargs)):
            logger.info("Broke repeat loop on %s", context.name)
            # Replaying a stream would mean buffering every chunk, so a repeat gets the notice.
            yield self._config.notice
            return
        async for chunk in call_next(*args, **kwargs):
            yield chunk


@register_middleware(config_type=OutputLimitConfig)
async def tool_output_limit(config: OutputLimitConfig, builder: Builder):
    yield OutputLimitMiddleware(config=config)


@register_middleware(config_type=RepeatBreakerConfig)
async def repeat_call_breaker(config: RepeatBreakerConfig, builder: Builder):
    yield RepeatBreakerMiddleware(config=config)
