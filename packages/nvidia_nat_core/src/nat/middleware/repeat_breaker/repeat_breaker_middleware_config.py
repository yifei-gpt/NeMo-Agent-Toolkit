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
"""Configuration for repeat breaker middleware."""

from pydantic import Field

from nat.data_models.middleware import FunctionMiddlewareBaseConfig


class RepeatBreakerMiddlewareConfig(FunctionMiddlewareBaseConfig, name="repeat_call_breaker"):
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
    warn_at_fraction: float = Field(default=0.7, ge=0.0, le=1.0,
                                    description="Warn once this share of the budget is spent; 0 disables")
    warning: str = Field(default=("You have used {spent} of your {total} tool calls. Spend what is left "
                                  "on finishing the answer, not on gathering more."),
                         description="Appended once, to the result of the call that crosses the threshold")
    # An identical reply to an identical call is a fixed point, so the nudge escalates by count
    # and, past this many rejected repeats, stops asking for a different call and asks for an answer.
    max_loop_replies: int = Field(default=8, ge=1,
                                  description="Rejected repeats before the loop is told to stop calling tools")

    # Frameworks without an iteration cap ignore notices; the budget needs an edge that raises.
    hard_stop_multiple: int = Field(default=3, ge=1,
                                    description="Raise once calls exceed this multiple of the budget")
