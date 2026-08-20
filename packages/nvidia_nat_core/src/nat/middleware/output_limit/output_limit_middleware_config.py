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
"""Configuration for output limit middleware."""

from pydantic import Field

from nat.data_models.middleware import FunctionMiddlewareBaseConfig


class OutputLimitMiddlewareConfig(FunctionMiddlewareBaseConfig, name="tool_output_limit"):
    """Truncates tool output so a long transcript cannot exceed the context window."""

    max_chars: int = Field(default=8000, gt=0, description="Maximum characters returned by an intercepted call")
    max_chars_per_run: int = Field(default=0, ge=0,
                                   description="Total characters of tool output per run; 0 disables")
    notice: str = Field(default="\n...[output truncated; narrow your query to see more]",
                        description="Appended when output is truncated")
    spent_notice: str = Field(default=("[This run has read as much tool output as it is allowed. "
                                       "Reading more will return nothing, however the query is "
                                       "written. Answer now with what you already have.]"),
                              description="Returned instead once the run's whole budget is spent")
