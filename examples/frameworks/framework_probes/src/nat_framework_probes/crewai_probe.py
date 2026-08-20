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
"""A minimal CrewAI workflow (NAT ships no example): exercises the nvidia-nat-crewai
LLM client, tool wrapper, and callback handler."""

import asyncio
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


class CrewAIProbeConfig(FunctionBaseConfig, name="crewai_probe"):
    """Configuration for a single-agent CrewAI crew backed by NAT tools."""

    llm_name: LLMRef = Field(description="Model to use via the CrewAI wrapper")
    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="NAT tools exposed to the CrewAI agent")
    role: str = Field(default="Research Analyst", description="The CrewAI agent's role")
    goal: str = Field(default="Answer the user's question accurately, using tools rather than guessing",
                      description="The CrewAI agent's goal")
    backstory: str = Field(
        default=("You are a meticulous analyst. You never invent numbers or dates: "
                 "when a tool can produce a fact, you call the tool."),
        description="The CrewAI agent's backstory")
    expected_output: str = Field(default="A short, direct answer that states which tool results it relies on.",
                                 description="The expected output description for the CrewAI task")
    verbose: bool = Field(default=True, description="Verbose CrewAI logging")
    max_iter: int = Field(default=15, description="Hard cap on the agent's reasoning iterations")


@register_function(config_type=CrewAIProbeConfig, framework_wrappers=[LLMFrameworkEnum.CREWAI])
async def crewai_probe(config: CrewAIProbeConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Build a one-agent, one-task CrewAI crew that can call NAT tools."""

    from crewai import Agent
    from crewai import Crew
    from crewai import Process
    from crewai import Task

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.CREWAI)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.CREWAI)

    async def _run(inputs: str) -> str:
        agent = Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            llm=llm,
            tools=tools,
            verbose=config.verbose,
            allow_delegation=False,
            max_iter=config.max_iter,
        )
        task = Task(description=inputs, expected_output=config.expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=config.verbose)

        # Off the loop thread: the NAT CrewAI tool wrapper calls back into this loop and would deadlock.
        result = await asyncio.to_thread(crew.kickoff)
        return str(result)

    yield FunctionInfo.from_fn(_run, description="Run a single-agent CrewAI crew over the configured NAT tools")
