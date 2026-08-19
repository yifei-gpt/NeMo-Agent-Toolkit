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
"""A two-agent CrewAI crew, which is the shape CrewAI actually exists for.

The single-agent probe only exercises the LLM wrapper, not the crew orchestration.
"""

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


class CrewAICrewProbeConfig(FunctionBaseConfig, name="crewai_crew_probe"):
    """Researcher plus analyst crew running CrewAI's sequential process."""

    llm_name: LLMRef = Field(description="Model to use via the CrewAI wrapper")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="NAT tools exposed to the researcher")
    process: str = Field(default="sequential", description="CrewAI process: 'sequential' or 'hierarchical'")
    verbose: bool = Field(default=False, description="Verbose CrewAI logging")
    max_iter: int = Field(default=15, description="Hard cap on each agent's reasoning iterations")


@register_function(config_type=CrewAICrewProbeConfig, framework_wrappers=[LLMFrameworkEnum.CREWAI])
async def crewai_crew_probe(config: CrewAICrewProbeConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from crewai import Agent
    from crewai import Crew
    from crewai import Process
    from crewai import Task

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.CREWAI)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.CREWAI)
    process = Process.hierarchical if config.process == "hierarchical" else Process.sequential

    async def _run(inputs: str) -> str:
        researcher = Agent(role="Researcher",
                           goal="Find every fact the question depends on, using the tools rather than memory",
                           backstory="You gather evidence and quote the exact figures and dates you find.",
                           llm=llm,
                           tools=tools,
                           verbose=config.verbose,
                           allow_delegation=False,
                           max_iter=config.max_iter)
        analyst = Agent(role="Analyst",
                        goal="Combine the gathered facts into the final answer",
                        backstory="You reason over evidence someone else collected and commit to one answer.",
                        llm=llm,
                        tools=tools,
                        verbose=config.verbose,
                        allow_delegation=False,
                        max_iter=config.max_iter)

        research_task = Task(description=f"Gather the facts needed to answer: {inputs}",
                             expected_output="A list of the facts found, each with its source.",
                             agent=researcher)
        answer_task = Task(description=f"Using the gathered facts, answer: {inputs}",
                           expected_output="A short, direct answer covering every part of the question.",
                           agent=analyst,
                           context=[research_task])

        crew = Crew(agents=[researcher, analyst],
                    tasks=[research_task, answer_task],
                    process=process,
                    verbose=config.verbose)

        # Must run off the loop thread; the NAT CrewAI tool wrapper calls back into
        # this loop and would deadlock otherwise.
        return str(await asyncio.to_thread(crew.kickoff))

    yield FunctionInfo.from_fn(_run, description="Run a two-agent CrewAI crew over the configured NAT tools")
