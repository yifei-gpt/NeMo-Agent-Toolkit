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
"""Two solvers answer, read each other, and may revise before a judge decides.

The ensemble topology here votes over answers produced in isolation, so correlated
mistakes survive. Here each side sees the other's reasoning and can change its mind.
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

OPENING = ("Question: {question}\n\nAnswer it using your tools. State the facts you relied on, "
           "then give a short, direct answer.")

REBUTTAL = ("Question: {question}\n\nYour previous answer:\n{mine}\n\nThe other analyst answered:\n{theirs}\n\n"
            "Check the points where you disagree, using your tools where a fact is in doubt. "
            "Then give your final short answer, changed or not.")

VERDICT = ("Question: {question}\n\nAnalyst A: {a}\n\nAnalyst B: {b}\n\n"
           "Decide the correct answer. If they agree, state it. If they differ, pick the one whose "
           "evidence holds up. Reply with the answer only, no preamble.")


class DebateTopologyConfig(FunctionBaseConfig, name="debate_topology"):
    """Two tool-using solvers exchange positions before a judge rules."""

    llm_name: LLMRef = Field(description="Model for both analysts")
    judge_llm_name: LLMRef = Field(description="Model that rules on the exchange")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="Tools both analysts may use")
    rounds: int = Field(default=1, ge=0, description="Rebuttal rounds after the opening answers")
    max_iterations: int = Field(default=10, description="Tool-call cap per analyst turn")
    verbose: bool = Field(default=False, description="Verbose agent logging")


@register_function(config_type=DebateTopologyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def debate_topology(config: DebateTopologyConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from langchain_core.messages import HumanMessage

    from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

    judge = await builder.get_llm(config.judge_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tools = [str(t) for t in config.tool_names]

    analysts = {}
    for side in ("a", "b"):
        analysts[side] = await builder.add_function(
            f"debate_{side}",
            ToolCallAgentWorkflowConfig(tool_names=tools,
                                        llm_name=config.llm_name,
                                        verbose=config.verbose,
                                        max_iterations=config.max_iterations,
                                        handle_tool_errors=True,
                                        description=f"Debate analyst {side.upper()}"))

    async def _turn(side: str, prompt: str) -> str:
        try:
            return str(await analysts[side].acall_invoke(prompt))
        except Exception as exc:  # noqa: BLE001 - a failed turn should not sink the debate
            logger.warning("Analyst %s failed: %s", side, exc)
            return ""

    async def _run(inputs: str) -> str:
        opening = OPENING.format(question=inputs)
        pos = {"a": await _turn("a", opening), "b": await _turn("b", opening)}

        for round_no in range(config.rounds):
            logger.info("Rebuttal round %d", round_no + 1)
            revised = {}
            for side, other in (("a", "b"), ("b", "a")):
                revised[side] = await _turn(
                    side, REBUTTAL.format(question=inputs, mine=pos[side][:3000], theirs=pos[other][:3000]))
            pos = {s: v or pos[s] for s, v in revised.items()}

        verdict = await judge.ainvoke(
            [HumanMessage(content=VERDICT.format(question=inputs, a=pos["a"][:4000], b=pos["b"][:4000]))])
        return str(getattr(verdict, "content", verdict))

    yield FunctionInfo.from_fn(_run, description="Answer through two analysts who read and rebut each other")
