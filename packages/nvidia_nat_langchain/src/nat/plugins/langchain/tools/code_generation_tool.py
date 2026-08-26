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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)


class CodeGenerationTool(FunctionBaseConfig, name="code_generation"):
    """
    Tool for generating code using the configured LLM.
    """
    llm_name: LLMRef
    verbose: bool = False
    programming_language: str = "Python"
    description: str = ("Useful to generate Python code. For any questions about code generation, you must only use "
                        "this tool!")


@register_function(config_type=CodeGenerationTool, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def code_generation_tool(config: CodeGenerationTool, builder: Builder):
    from langchain_core.prompts.chat import ChatPromptTemplate

    log.info('Initializing code generation tool\nGetting tool LLM from config')
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    system_prompt = """
You are a helpful code assistant that can teach a junior developer how to code.  Your language of
 choice is {programming_language}. Don't explain the code, just generate the code block itself.
"""
    user_prompt = """
{question}
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
    log.info("Filling tool's prompt variable from config")
    prompt = prompt.partial(programming_language=config.programming_language)
    tool = prompt | llm
    log.info('Initialized code generation tool')

    async def _inner(query: str, path: str = "") -> str:
        log.info('Running code generation tool')
        response = await tool.ainvoke({"question": query})
        code = response.text()
        if config.verbose:
            log.debug('Tool input was: %s\nTool output is: \n%s', query, response)
        if not path:
            return code
        # A file has no output limit: a complete implementation returned as text was cut at the
        # cap and asked for again, twelve times, each one a minute.
        from nat.tool.workspace_tools import _resolve
        body = code.strip()
        if body.startswith("```"):
            body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body.strip() + "\n", encoding="utf-8")
        # Re-reading its own code invented flaws and it rewrote the file 3 times; running it
        # reports the real ones, which workspace_edit can then fix in place.
        return (f"wrote {path} ({len(body)} chars). Run it to see what it actually does, and "
                "fix what breaks with workspace_edit.")

    yield FunctionInfo.from_fn(_inner, description=config.description)
