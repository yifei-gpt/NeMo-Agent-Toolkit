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
    """Truncates what one tool call returns, so no single result floods the transcript."""

    max_chars: int = Field(default=8000, gt=0, description="Maximum characters returned by an intercepted call")
    notice: str = Field(default=("\n...[cut here at the output limit. Asking again the same way "
                                 "returns the same cut: ask for one part at a time, or have the "
                                 "tool write to a file and read that.]"),
                        description="Appended when output is truncated")
