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
"""Specialists built per question that read and write one shared record.

The agent-as-a-tool topologies here are star shaped: a finding only reaches another
specialist if the coordinator repeats it. Here each specialist sees what the previous
ones wrote, so evidence accumulates instead of being relayed.
"""

import asyncio
import json
import logging
import re
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

try:
    from markagentx import decision_point
except ImportError:  # the topology runs unmarked when MarkAgentX is not installed
    # Said once at import: an unmarked run is otherwise indistinguishable from a marked one.
    logger.warning("markagentx is not importable; this topology runs UNMARKED")

    def decision_point(step_type, *, native, **_):   # **_ so it cannot drift from the real signature
        return native

ROLE_LIBRARY: dict[str, dict] = {
    "entity_researcher": {
        "tools": ["wikipedia_search"],
        "brief": "Look up the entities the question names and record their dates, figures and relations.",
        "max_iterations": 8,
    },
    "date_resolver": {
        "tools": ["wikipedia_search", "current_datetime"],
        "brief": "Pin down the dates or intervals the question depends on.",
        "max_iterations": 5,
    },
    "calculator_specialist": {
        "tools": ["calculator"],
        "brief": "Do the arithmetic the question needs, using figures already on the record.",
        "max_iterations": 4,
    },
    "comparison_analyst": {
        "tools": ["wikipedia_search", "calculator"],
        "brief": "Compare the quantities in question, filling any gap the record leaves.",
        "max_iterations": 8,
    },
}

PLANNER_PROMPT = ("You are staffing a team for one question. Roles available:\n{roles}\n\n"
                  "Question: {question}\n\n"
                  "Reply with only a JSON array of role names, in the order they should work, "
                  "at most {limit}.")

TURN_PROMPT = ("Question: {question}\n\n"
               "Shared record so far:\n{record}\n\n"
               "You are the {role}. {brief} Add only what is missing; do not repeat what is "
               "already recorded. Reply with your findings in one short paragraph.")

COMPOSE_PROMPT = ("Question: {question}\n\n"
                  "Shared record from the team:\n{record}\n\n"
                  "Answer the question directly and briefly, covering every part it asks.")


class BlackboardTopologyConfig(FunctionBaseConfig, name="blackboard_topology"):
    """Specialists chosen per question that accumulate findings on a shared record."""

    llm_name: LLMRef = Field(description="Model for the planner, the specialists and the composer")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="Tools the specialists may use")
    max_roles: int = Field(default=3, ge=1, description="Most specialists to staff for one question")
    max_record_chars: int = Field(default=6000, description="Cap on the shared record handed to each turn")
    verbose: bool = Field(default=False, description="Verbose agent logging")


def _parse_roles(text: str, limit: int) -> list[str]:
    """Reads the planner's ordered role list, falling back to a name scan."""
    match = re.search(r"\[.*?\]", text or "", re.DOTALL)
    names: list[str] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                names = [str(n) for n in parsed]
        except json.JSONDecodeError:
            names = []
    if not names:
        names = [role for role in ROLE_LIBRARY if role in (text or "")]
    kept = [n for n in names if n in ROLE_LIBRARY]
    return kept[:limit] or ["entity_researcher"]


@register_function(config_type=BlackboardTopologyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def blackboard_topology(config: BlackboardTopologyConfig,
                              builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from langchain_core.messages import HumanMessage

    from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    available = {str(t) for t in config.tool_names}
    catalogue = "\n".join(f"- {name}: {spec['brief']}" for name, spec in ROLE_LIBRARY.items())
    built: dict[str, object] = {}
    # Questions are evaluated concurrently, so every add_function call is serialised.
    build_lock = asyncio.Lock()

    async def _specialist(role: str):
        # A marking method may hand back a name that was never on the menu; losing that one
        # specialist is recoverable, taking the whole question down with a KeyError is not.
        spec = ROLE_LIBRARY.get(role)
        if spec is None:
            logger.warning("Ignoring unknown role %r", role)
            return None
        tools = [t for t in spec["tools"] if t in available]
        if not tools:
            return None
        async with build_lock:
            if role not in built:
                built[role] = await builder.add_function(
                    f"bb_{role}",
                    ToolCallAgentWorkflowConfig(tool_names=tools,
                                                llm_name=config.llm_name,
                                                verbose=config.verbose,
                                                max_iterations=spec["max_iterations"],
                                                handle_tool_errors=True,
                                                description=f"Specialist: {role}"))
                logger.info("Built specialist %s with tools %s", role, tools)
        return built[role]

    async def _run(inputs: str) -> str:
        plan = await llm.ainvoke(
            [HumanMessage(content=PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles))])
        roles = _parse_roles(getattr(plan, "content", str(plan)), config.max_roles)
        # Staffing is the widest choice this workflow makes, so it is offered for marking;
        # with nothing marking, decision_point returns the planner's own pick.
        eligible = [n for n, s in ROLE_LIBRARY.items() if any(x in available for x in s["tools"])]
        prompt = PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles)
        roles = list(dict.fromkeys(decision_point("role_selection", candidates=eligible, native=r,
                                                  context=[{"role": "user", "content": prompt}])
                                   for r in roles))
        logger.info("Team staffed with %s", roles)

        record: list[str] = []
        for role in roles:
            agent = await _specialist(role)
            if agent is None:
                continue
            shared = "\n\n".join(record)[-config.max_record_chars:] or "(empty)"
            turn = TURN_PROMPT.format(question=inputs,
                                      record=shared,
                                      role=role,
                                      brief=ROLE_LIBRARY[role]["brief"])
            try:
                finding = str(await agent.acall_invoke(turn))
            except Exception as exc:  # noqa: BLE001 - one specialist failing must not sink the run
                logger.warning("Specialist %s failed: %s", role, exc)
                continue
            record.append(f"[{role}] {finding.strip()}")

        shared = "\n\n".join(record)[-config.max_record_chars:] or "(empty)"
        final = await llm.ainvoke([HumanMessage(content=COMPOSE_PROMPT.format(question=inputs, record=shared))])
        return str(getattr(final, "content", final))

    yield FunctionInfo.from_fn(_run, description="Answer through per-question specialists sharing one record")
