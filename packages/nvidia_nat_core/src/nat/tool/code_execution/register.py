# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class CodeExecutionToolConfig(FunctionBaseConfig, name="code_execution"):
    """
    Tool for executing python code in a remotely hosted sandbox environment.
    """
    uri: HttpUrl = Field(default=HttpUrl("http://127.0.0.1:6000"),
                         description="URI for the code execution sandbox server")
    sandbox_type: Literal["local", "piston"] = Field(default="local", description="The type of code execution sandbox")
    timeout: float = Field(default=10.0, description="Number of seconds to wait for a code execution request")
    max_output_characters: int = Field(default=1000, description="Maximum number of characters that can be returned")


@register_function(config_type=CodeExecutionToolConfig)
async def code_execution_tool(config: CodeExecutionToolConfig, builder: Builder):
    from nat.tool.code_execution.code_sandbox import get_sandbox

    class CodeExecutionInputSchema(BaseModel):
        generated_code: str = Field(description="String containing the code to be executed")

    # Create sandbox without working_directory
    sandbox_kwargs = {"uri": config.uri}

    sandbox = get_sandbox(sandbox_type=config.sandbox_type, **sandbox_kwargs)
    logger.info(f"[DEBUG] Created sandbox of type: {config.sandbox_type}")

    def _in_workspace(code: str) -> str:
        # The sandbox starts in its own container directory, so an unqualified write lands somewhere
        # the agent can never read back while the tool still reports success.
        root = os.environ.get("NAT_WORKSPACE_DIR")
        # umask too: the sandbox runs as root, and a 644 file it leaves behind is one the workspace
        # tools can then never rewrite.
        # No try: a workspace the sandbox cannot enter must fail loudly, not run in the container.
        return code if not root else (f"import os\nos.umask(0)\nos.chdir({root!r})\n" + code)

    async def _execute_code(generated_code: str) -> dict:
        logger.info("Executing code in the sandbox at %s", config.uri)
        try:
            output = await sandbox.execute_code(
                generated_code=_in_workspace(generated_code),
                language="python",
                timeout_seconds=config.timeout,
                max_output_characters=config.max_output_characters,
            )
        except Exception as e:
            logger.exception("Error when executing code in the sandbox, %s", e)
            output = {"process_status": "error", "stdout": "", "stderr": str(e)}
        # A silently absent sandbox is worse than a loud one: the agent answers from memory instead.
        if output.get("process_status") == "error":
            output["stdout"] = ("THE SANDBOX DID NOT RUN THIS CODE. Do not answer from your own "
                                "arithmetic -- retry, use another tool, or report the failure.")
        text = output.get("stdout") or ""
        cap = config.max_output_characters
        if output.get("process_status") == "timeout" and not text:
            output["stdout"] = ("No output: the sandbox times out when a single run prints more than "
                                "about 60000 characters. Print a summary, not the whole thing.")
        elif len(text) > cap:
            output["stdout"] = text[:cap] + f"\n... {len(text) - cap} more characters, print less"
        return output

    yield FunctionInfo.from_fn(
        fn=_execute_code,
        input_schema=CodeExecutionInputSchema,
        description="""Runs `generated_code` as python and returns its stdout, stderr and status.
        Print what you want to see -- nothing is returned otherwise, and no variable survives to the
        next call. The workspace is the working directory, so relative paths read and write the same
        files the workspace tools see.""")
