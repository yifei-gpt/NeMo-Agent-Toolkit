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
"""One minimal single-agent workflow per framework, taking llm_name + tool_names and nothing
else -- the shipped examples are task-specific and cannot run a benchmark dataset."""

import logging
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = ("You are a careful analyst. Use the available tools to look up facts and to compute "
                  "values; never guess. End with a short, direct answer.")


class AdkProbeConfig(FunctionBaseConfig, name="adk_probe"):
    """Single Google ADK agent over NAT tools."""

    llm_name: LLMRef = Field(description="Model to use via the ADK wrapper")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="NAT tools exposed to the agent")
    system_prompt: str = Field(default=DEFAULT_PROMPT, description="Agent instructions")


class AutogenProbeConfig(FunctionBaseConfig, name="autogen_probe"):
    """Single AutoGen assistant over NAT tools."""

    llm_name: LLMRef = Field(description="Model to use via the AutoGen wrapper")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="NAT tools exposed to the agent")
    system_prompt: str = Field(default=DEFAULT_PROMPT, description="Agent instructions")
    max_turns: int = Field(default=20, description="Maximum assistant turns")


@register_function(config_type=AdkProbeConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def adk_probe(config: AdkProbeConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from google.adk.agents import Agent
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.ADK)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.ADK)
    agent = Agent(name="analyst", model=llm, description="Analyst", instruction=config.system_prompt, tools=tools)

    session_service = InMemorySessionService()
    runner = Runner(app_name="analyst",
                    agent=agent,
                    artifact_service=InMemoryArtifactService(),
                    session_service=session_service)

    async def _run(inputs: str) -> str:
        # A fresh session per question keeps benchmark items independent.
        session = await session_service.create_session(app_name="analyst", user_id="bench")
        content = types.Content(role="user", parts=[types.Part.from_text(text=inputs)])
        parts: list[str] = []
        async for event in runner.run_async(user_id="bench", session_id=session.id, new_message=content):
            if event.content and event.content.parts:
                parts.extend(p.text for p in event.content.parts if p.text)
        return "".join(parts)

    yield FunctionInfo.from_fn(_run, description="Run a single Google ADK agent over the configured NAT tools")


@register_function(config_type=AutogenProbeConfig, framework_wrappers=[LLMFrameworkEnum.AUTOGEN])
async def autogen_probe(config: AutogenProbeConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from autogen_agentchat.agents import AssistantAgent

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AUTOGEN)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.AUTOGEN)

    async def _run(inputs: str) -> str:
        agent = AssistantAgent(name="analyst",
                               model_client=llm,
                               tools=tools,
                               system_message=config.system_prompt,
                               max_tool_iterations=config.max_turns,
                               reflect_on_tool_use=True)
        result = await agent.run(task=inputs)
        messages = getattr(result, "messages", None) or []
        return str(messages[-1].content) if messages else ""

    yield FunctionInfo.from_fn(_run, description="Run a single AutoGen assistant over the configured NAT tools")


