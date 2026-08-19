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
"""Native multi-agent workflows, one per framework.

Each framework has its own team primitive, and the single-agent probes never touch them.
"""

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

RESEARCH_ROLE = ("Find every fact the question depends on. Use the search tool for each entity "
                 "separately and never guess a number or a date.")
ANSWER_ROLE = ("Combine the facts the researcher gathered into the final answer. Answer every part "
               "the question asks for, briefly.")


class AdkTeamConfig(FunctionBaseConfig, name="adk_team_probe"):
    """Two Google ADK agents in sequence: one gathers facts, the other answers from them."""

    llm_name: LLMRef = Field(description="Model to use via the ADK wrapper")
    brief: str = Field(default="", description="What the task set asks of any agent working it")
    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="NAT tools exposed to both agents")


@register_function(config_type=AdkTeamConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def adk_team_probe(config: AdkTeamConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from google.adk.agents import Agent
    from google.adk.agents import SequentialAgent
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.ADK)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.ADK)

    def _with_brief(role):
        # What the task set asks of anyone, then what this seat is for.
        return " ".join(x for x in (config.brief, role) if x)

    researcher = Agent(name="researcher", model=llm, description="Gathers facts.",
                       instruction=_with_brief(RESEARCH_ROLE),
                       tools=tools)
    analyst = Agent(name="analyst", model=llm, description="Writes the answer.",
                    instruction=_with_brief(ANSWER_ROLE))
    team = SequentialAgent(name="research_then_answer", sub_agents=[researcher, analyst])

    session_service = InMemorySessionService()
    runner = Runner(app_name="team",
                    agent=team,
                    artifact_service=InMemoryArtifactService(),
                    session_service=session_service)

    async def _run(inputs: str) -> str:
        session = await session_service.create_session(app_name="team", user_id="bench")
        content = types.Content(role="user", parts=[types.Part.from_text(text=inputs)])
        parts: list[str] = []
        async for event in runner.run_async(user_id="bench", session_id=session.id, new_message=content):
            if event.content and event.content.parts:
                parts.extend(p.text for p in event.content.parts if p.text)
        return "".join(parts)

    yield FunctionInfo.from_fn(_run, description="Run a Google ADK sequential two-agent team")


