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
"""Middleware that breaks identical-call loops and enforces a per-run call budget."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from nat.builder.context import Context
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.repeat_breaker.repeat_breaker_middleware_config import RepeatBreakerMiddlewareConfig

logger = logging.getLogger(__name__)


class ToolBudgetExceeded(BaseException):
    """Outside Exception on purpose: an agent framework answers a tool's Exception with an error
    string and keeps calling, which is the one thing a spent budget must not allow."""

# One instance serves every function for a whole evaluation, so per-run tables evict.
_MAX_RUNS = 64
_MAX_KEYS_PER_RUN = 128
_MAX_CACHED_CHARS = 8000
# Not a function name, so it cannot collide with one in the same counter.
_RUN_TOTAL = "<run>"


def _evict(table: OrderedDict, limit: int, what: str = "") -> None:
    while len(table) > limit:
        dropped, _ = table.popitem(last=False)
        if what:
            # Evicting a run that is still going hands it a fresh budget, so it must not be silent.
            logger.warning("dropped %s for run %s past %d live runs; its budget restarts", what, dropped, limit)


def _stable(value: Any) -> Any:
    # An object repr carries its address, hashing identical calls apart and disabling dedup.
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
        # Counted whether or not the calls touch: alternating two useless calls repeats neither.
        self.lifetime: OrderedDict[str, int] = OrderedDict()
        self.last_key: str | None = None
        # Said once per run: a second warning is noise on every later observation.
        self.warned: bool = False


_RUNS: "OrderedDict[str, _RunState]" = OrderedDict()


class RepeatBreakerMiddleware(FunctionMiddleware):
    """Deduplicates back-to-back identical calls and stops a run that ignores its budget."""

    def __init__(self, config: RepeatBreakerMiddlewareConfig) -> None:
        super().__init__()
        self._config = config
        # Shared, not per instance: one middleware object is built per function that names it, so
        # a per-instance counter gave a fifteen-agent workflow fifteen separate "per run" budgets.
        # A run made 266 calls against a limit of 100 without one notice firing.
        self._runs = _RUNS

    def _run_state(self) -> _RunState:
        # Counting per workflow run keeps one question's repeats from affecting the next.
        run_id = Context.get().workflow_run_id or "no-run"
        state = self._runs.pop(run_id, None) or _RunState()
        self._runs[run_id] = state
        _evict(self._runs, _MAX_RUNS, "the tool budget")
        return state

    def _key(self, name: str | None, args: tuple, kwargs: dict) -> str:
        return hashlib.sha256(repr(_stable([name, args, kwargs])).encode()).hexdigest()

    def _budget_spent(self, state: _RunState, name: str) -> str | None:
        # Paraphrases slip past exact-match dedup; only a hard per-run budget guarantees termination.
        if not self._config.max_calls_per_run:
            return None
        # Counted before dedup so spam still hard-stops; per function alone, N tools bought N budgets.
        state.totals[name] += 1
        state.totals[_RUN_TOTAL] += 1
        run_total = state.totals[_RUN_TOTAL]
        if run_total > self._config.max_calls_per_run * self._config.hard_stop_multiple:
            raise ToolBudgetExceeded(f"Tool budget exceeded {self._config.hard_stop_multiple}x "
                               f"({run_total} calls this run, {state.totals[name]} of them on {name}); "
                               f"the agent ignored the budget notice.")
        if run_total > self._config.max_calls_per_run:
            logger.info("Tool budget spent (%d calls this run, %d on %s)", run_total, state.totals[name], name)
            return self._config.budget_notice
        return None

    def _budget_warning(self, state: _RunState) -> str | None:
        """Said once, while there is still room to act on it. Told only at exhaustion, a run spends
        everything gathering and then has nothing left to write the answer with."""
        limit = self._config.max_calls_per_run
        share = self._config.warn_at_fraction
        if not limit or not share or state.warned:
            return None
        spent = state.totals[_RUN_TOTAL]
        if spent < limit * share:
            return None
        state.warned = True
        return self._config.warning.format(spent=spent, total=limit)

    def _repeated(self, state: _RunState, key: str) -> int:
        # Only back-to-back repeats are loops; after a different call, state may have changed.
        consecutive, state.last_key = state.last_key == key, key
        # Re-inserting keeps a hot repeat loop at the young end, so eviction never resets it.
        count = (state.seen.pop(key, 0) + 1) if consecutive else 1
        state.seen[key] = count
        state.lifetime[key] = state.lifetime.pop(key, 0) + 1
        _evict(state.seen, _MAX_KEYS_PER_RUN)
        _evict(state.lifetime, _MAX_KEYS_PER_RUN)
        return count

    def _loop_reply(self, state: _RunState, key: str, over: int) -> str:
        """A reply that changes with the count, because an identical one is what sustains the loop."""
        if over >= self._config.max_loop_replies:
            return self._config.budget_notice
        if over == 1 and key in state.results:
            return f"{state.results[key]}\n\n[{self._config.notice}]"
        return (f"{self._config.notice} That was repeat {over} of the same call and its result "
                f"will not change, so repeating it again cannot move the task forward.")

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
        repeats = self._repeated(state, key)
        # The run-long count stands in when the repeats are spread out; one URL was fetched 36
        # times with a search between each, so no two were adjacent and none was ever broken.
        spread = self._config.max_identical_per_run
        if spread and state.lifetime.get(key, 0) > spread:
            # Monotone in the lifetime count: one different call in between used to reset the
            # ladder, and the reply at the bottom of it is the cached result -- a fixed point.
            repeats = max(repeats, state.lifetime.get(key, 0) - spread + self._config.max_repeats)
        over = repeats - self._config.max_repeats
        if over > 0:
            # An agent that ignores every escalation would otherwise spin out the whole wall clock.
            if over >= self._config.max_loop_replies * self._config.hard_stop_multiple:
                raise ToolBudgetExceeded(f"{context.name} was called identically {repeats} times in a row")
            logger.info("Broke repeat loop on %s (%d in a row)", context.name, repeats)
            return self._loop_reply(state, key, over)
        result = await call_next(*args, **kwargs)
        self._remember(state, key, result)
        warning = self._budget_warning(state)
        return f"{result}\n\n[{warning}]" if warning and isinstance(result, str) else result

    async def function_middleware_stream(self, *args: Any, call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        # The inherited stream path skips pre/post, so a streaming tool would get neither guard.
        state = self._run_state()
        spent = self._budget_spent(state, context.name)
        if spent is not None:
            yield spent
            return

        over = self._repeated(state, self._key(context.name, args, kwargs)) - self._config.max_repeats
        if over > 0:
            # Replaying a stream would mean buffering every chunk, so a repeat gets the notice.
            logger.info("Broke repeat loop on %s", context.name)
            yield self._loop_reply(state, "", over)
            return
        async for chunk in call_next(*args, **kwargs):
            yield chunk
        warning = self._budget_warning(state)
        if warning:
            yield f"\n\n[{warning}]"
