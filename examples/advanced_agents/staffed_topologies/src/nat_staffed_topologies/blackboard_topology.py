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
"""Specialists built per question over one shared record: each sees what the previous wrote,
so evidence accumulates instead of being relayed through a coordinator."""

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

from .roles import ROLE_LIBRARY
from .roles import parse_roles
from .roles import pick_staff
from .roles import validate

logger = logging.getLogger(__name__)


WORKSPACE_TOOLS = ["workspace_list", "workspace_read", "workspace_write", "workspace_search"]

PLANNER_PROMPT = ("You are staffing a team for one question. Roles available:\n{roles}\n\n"
                  "Question: {question}\n\n"
                  "Reply with only a JSON array of exactly {limit} role names, in the order they "
                  "should work. Name only roles from the list above.")

TURN_PROMPT = ("Question: {question}\n\n"
               "Shared record so far:\n{record}\n\n"
               "You are the {role}. {brief} Add only what is missing; do not repeat what is "
               "already recorded. Use your tools to establish what you report -- never answer "
               "from the task text alone -- then give your findings in one short paragraph.")

COMPOSE_PROMPT = ("Question: {question}\n\n"
                  "Shared record from the team:\n{record}\n\n"
                  "{brief} Answer the question, covering every part it asks.")


class BlackboardTopologyConfig(FunctionBaseConfig, name="blackboard_topology"):
    """Specialists chosen per question that accumulate findings on a shared record."""

    llm_name: LLMRef = Field(description="Model for the planner, the specialists and the composer")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="Tools the specialists may use")
    max_roles: int = Field(default=3, ge=1, description="Most specialists to staff for one question")
    role_middleware: list[str] = Field(default_factory=list,
                                       description="Middleware every specialist carries")
    brief: str = Field(default="", description="What the task set asks of any agent working it")
    roles: dict = Field(default_factory=dict,
                        description="Role library override: name -> "
                                    "{tools, brief, max_iterations}; empty uses the built-in one")
    max_record_chars: int = Field(default=6000, description="Cap on the shared record handed to each turn")
    verbose: bool = Field(default=False, description="Verbose agent logging")


@register_function(config_type=BlackboardTopologyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def blackboard_topology(config: BlackboardTopologyConfig,
                              builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from langchain_core.messages import HumanMessage

    from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    available = {str(t) for t in config.tool_names}
    # A tool set naming none of these leaves nobody eligible and the staffing menu empty.
    library = validate(config.roles or ROLE_LIBRARY, "brief")
    generic = not any(x in available for s in library.values() for x in s["tools"])
    if generic:
        logger.warning("No role names any of the %d offered tools; staffing on briefs alone", len(available))
    # Only staffable roles: the rest spend a slot on a specialist that _specialist drops.
    eligible = [n for n, s in library.items() if any(x in available for x in s["tools"])] \
        or list(library)
    catalogue = "\n".join(f"- {name}: {library[name]['brief']}" for name in eligible)
    built: dict[str, object] = {}
    # Questions are evaluated concurrently, so every add_function call is serialised.
    build_lock = asyncio.Lock()

    async def _specialist(role: str):
        # A method may return a name never on the menu; losing one specialist beats a KeyError.
        spec = library.get(role)
        if spec is None:
            logger.warning("Ignoring unknown role %r", role)
            return None
        tools = [t for t in spec["tools"] if t in available] or (sorted(available) if generic else [])
        if not tools:
            return None
        # The cap fits the tools a role names; elsewhere the longest cap is honest.
        iters = max(s["max_iterations"] for s in library.values()) if generic else spec["max_iterations"]
        async with build_lock:
            if role not in built:
                built[role] = await builder.add_function(
                    f"bb_{role}",
                    ToolCallAgentWorkflowConfig(tool_names=tools,
                                                llm_name=config.llm_name,
                                                verbose=config.verbose,
                                                max_iterations=iters,
                                                handle_tool_errors=True,
                                                middleware=list(config.role_middleware),
                                                # The task set's ask, then the role's own.
                                                additional_instructions=" ".join(
                                                    x for x in (config.brief, spec["brief"]) if x),
                                                description=f"Specialist: {role}"))
                logger.info("Built specialist %s with tools %s", role, tools)
        return built[role]

    async def _run(inputs: str) -> str:
        plan = await llm.ainvoke(
            [HumanMessage(content=PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles))])
        roles = parse_roles(getattr(plan, "content", str(plan)), config.max_roles, eligible)
        # The widest choice here, so it is offered for marking; unmarked keeps the plan.
        prompt = PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles)
        roles = pick_staff(roles, eligible, prompt)
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
                                      brief=library[role]["brief"])
            try:
                finding = str(await agent.acall_invoke(turn))
            except Exception as exc:
                logger.warning("Specialist %s failed: %s", role, exc)
                continue
            record.append(f"[{role}] {finding.strip()}")

        shared = "\n\n".join(record)[-config.max_record_chars:] or "(empty)"
        prompt = COMPOSE_PROMPT.format(question=inputs, record=shared, brief=config.brief)
        final = await llm.ainvoke([HumanMessage(content=prompt)])
        return str(getattr(final, "content", final))

    yield FunctionInfo.from_fn(_run, description="Answer through per-question specialists sharing one record")
