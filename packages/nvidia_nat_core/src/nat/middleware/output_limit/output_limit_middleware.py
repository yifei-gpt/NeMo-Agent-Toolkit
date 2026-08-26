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
"""Middleware that truncates what one tool call returns."""

from __future__ import annotations

import collections
import logging
from collections.abc import AsyncIterator
from typing import Any

from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNextStream
from nat.middleware.middleware import FunctionMiddlewareContext
from nat.middleware.middleware import InvocationContext
from nat.middleware.output_limit.output_limit_middleware_config import OutputLimitMiddlewareConfig

logger = logging.getLogger(__name__)

# Counted, not just logged: a run cut on half its calls reads exactly like one that
# was never cut, and the summary is where anyone would look.
FIRED: "collections.Counter[str]" = collections.Counter()


class OutputLimitMiddleware(FunctionMiddleware):
    """Caps what a tool call returns, and says how to read past the cut."""

    def __init__(self, config: OutputLimitMiddlewareConfig) -> None:
        super().__init__()
        self._config = config

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        # A dict or model output fills the transcript just as fast as a long string does.
        text = context.output if isinstance(context.output, str) else str(context.output)
        cap = self._config.max_chars
        if len(text) <= cap:
            return None
        FIRED[context.function_context.name or "?"] += 1
        logger.info("Truncated %s output from %d to %d chars",
                    context.function_context.name, len(text), cap)
        context.output = text[:cap] + self._config.notice
        return context

    async def function_middleware_stream(self, *args: Any, call_next: CallNextStream,
                                         context: FunctionMiddlewareContext,
                                         **kwargs: Any) -> AsyncIterator[Any]:
        # The inherited path caps per chunk, so the whole-stream length is tracked here.
        remaining = self._config.max_chars
        async for chunk in call_next(*args, **kwargs):
            text = chunk if isinstance(chunk, str) else str(chunk)
            if len(text) > remaining:
                FIRED[context.name or "?"] += 1
                logger.info("Truncated %s stream at its remaining budget", context.name)
                yield text[:remaining] + self._config.notice
                return
            remaining -= len(text)
            yield chunk
