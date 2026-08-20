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
"""Middleware that truncates tool output, per call and per workflow run."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from nat.builder.context import Context
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.middleware.output_limit.output_limit_middleware_config import OutputLimitMiddlewareConfig

logger = logging.getLogger(__name__)

# One instance serves every function for a whole evaluation, so per-run tables evict.
_MAX_RUNS = 64


def _evict(table: OrderedDict, limit: int, what: str = "") -> None:
    while len(table) > limit:
        dropped, _ = table.popitem(last=False)
        if what:
            # Evicting a run that is still going hands it a fresh budget, so it must not be silent.
            logger.warning("dropped %s for run %s past %d live runs; its budget restarts", what, dropped, limit)


class OutputLimitMiddleware(FunctionMiddleware):
    """Caps what a tool call returns; an optional per-run budget caps the whole transcript."""

    def __init__(self, config: OutputLimitMiddlewareConfig) -> None:
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
        # Charge what the caller sees, or the budget is spent on characters truncated away.
        self._spent[run_id] = spent + min(taken, room)
        _evict(self._spent, _MAX_RUNS, "the output budget")
        return room

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        # A dict or model output fills the transcript just as fast as a long string does.
        text = context.output if isinstance(context.output, str) else str(context.output)
        room = self._room(len(text))
        if len(text) <= room:
            return None
        logger.info("Truncated %s output from %d to %d chars", context.function_context.name, len(text), room)
        # 'Narrow your query' cannot be taken once the budget is gone; it kept agents searching to the cap.
        context.output = (self._config.spent_notice if room <= 0 else text[:room] + self._config.notice)
        return context

    async def function_middleware_stream(self, *args: Any, call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        # The inherited path caps per chunk, so the whole-stream budget is tracked here.
        remaining = self._room(self._config.max_chars)
        async for chunk in call_next(*args, **kwargs):
            text = chunk if isinstance(chunk, str) else str(chunk)
            if len(text) > remaining:
                logger.info("Truncated %s stream at its remaining budget", context.name)
                yield (self._config.spent_notice if remaining <= 0 else text[:remaining] + self._config.notice)
                return
            remaining -= len(text)
            yield chunk
