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

import json
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

# The same counter the output cap keeps: a tool that cuts itself loses just as much.
from nat.middleware.output_limit.output_limit_middleware import FIRED


class CodeExecutionToolConfig(FunctionBaseConfig, name="code_execution"):
    """
    Tool for executing python code in a remotely hosted sandbox environment.
    """
    uri: HttpUrl = Field(default=HttpUrl("http://127.0.0.1:6000"),
                         description="URI for the code execution sandbox server")
    sandbox_type: Literal["local", "piston"] = Field(default="local", description="The type of code execution sandbox")
    timeout: float = Field(default=10.0, description="Number of seconds to wait for a code execution request")
    max_output_characters: int = Field(default=1000, description="Maximum number of characters that can be returned")
    # Declared, because pydantic ignores what it does not declare: a task set's own wording for
    # this tool was being dropped without a word, and the model read the default instead.
    description: str | None = Field(default=None, description="What the model is told this tool is")


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
        # A container sandbox already starts where the task lives and accepts shell as well as
        # python; a host chdir prepended there is both the wrong path and the wrong language.
        # Only a bridge the harness actually opened counts: the placeholder set for config
        # validation would otherwise skip the chdir and claim a container session that is not there.
        bridge = os.environ.get("NAT_BRIDGE_URL") if os.environ.get("NAT_BRIDGE_READY") else None
        if not root or bridge:
            logger.info("sandbox preamble off (root=%s bridge=%s uri=%s)", bool(root), bridge, config.uri)
            return code
        # No try: a workspace the sandbox cannot enter must fail loudly, not run in the container.
        return f"import os\nos.umask(0)\nos.chdir({root!r})\n" + code

    def _record(code: str, output: dict | None) -> None:
        """What the agent actually submitted, when asked for. The decision journal records which
        tool was chosen and never the code, and that is where a stuck or mistaken run hides."""
        trail = os.environ.get("NAT_CODE_LOG")
        if not trail:
            return
        entry = {"code": code[:800]} if output is None else {
            "status": output.get("process_status"), "out": str(output.get("stdout", ""))[:400]}
        with open(trail, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    async def _execute_code(generated_code: str) -> dict:
        logger.info("Executing code in the sandbox at %s", config.uri)
        _record(generated_code, None)
        try:
            output = await sandbox.execute_code(
                generated_code=_in_workspace(generated_code),
                language="python",
                timeout_seconds=config.timeout,
                max_output_characters=config.max_output_characters,
            )
        except Exception as e:
            logger.exception("Error when executing code in the sandbox, %s", e)
            output = {"process_status": "unreachable", "stdout": "", "stderr": str(e)}
        _record("", output)
        # A silently absent sandbox is worse than a loud one: the agent answers from memory instead.
        # Only when nothing ran, though: the sandbox reports "error" for code that raised, and that
        # code has usually printed real results first -- telling the agent they never happened
        # throws away the work and is untrue.
        if output.get("process_status") == "unreachable":
            output["stdout"] = ("THE SANDBOX DID NOT RUN THIS CODE. Do not answer from your own "
                                "arithmetic -- retry, use another tool, or report the failure.")
        text = output.get("stdout") or ""
        # The traceback is what tells the agent which line to fix; without it the failure is mute.
        if output.get("process_status") == "error":
            tail = str(output.get("stderr") or "").strip()[-600:]
            text = (text + "\n\n[the run raised after this output]\n" + tail).strip()
            output["stdout"] = text
        cap = config.max_output_characters
        if output.get("process_status") == "timeout" and not text:
            FIRED[config.type] += 1
            output["stdout"] = (f"No output: the run was stopped at the {config.timeout:g}s limit, and "
                                "anything it printed is lost with it. Do less in one call.")
        elif len(text) > cap:
            FIRED[config.type] += 1
            output["stdout"] = text[:cap] + f"\n... {len(text) - cap} more characters, print less"
        return output

    yield FunctionInfo.from_fn(
        fn=_execute_code,
        input_schema=CodeExecutionInputSchema,
        description=(config.description or ("Runs `generated_code` in the task's own container and returns its stdout, "
                     "stderr and status. A shell command line works as well as python -- send "
                     "whichever suits the step. The session persists, so a directory you enter "
                     "and a file you write are still there on the next call."
                     if os.environ.get("NAT_BRIDGE_READY") else
                     """Runs `generated_code` as python and returns its stdout, stderr and status.
        Print what you want to see -- nothing is returned otherwise, and no variable survives to the
        next call. The workspace is the working directory, so relative paths read and write the same
        files the workspace tools see.""")))
