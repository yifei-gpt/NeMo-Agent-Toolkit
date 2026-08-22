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
"""Specialists created per question, not declared in config: the model picks the roles,
only those are built with `Builder.add_function`, and a coordinator delegates."""

import asyncio
import hashlib
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


# Each entry is a role the planner may ask for: the tools it gets and how it is told to work.
GENERIC_NOTE = (" The tools and file paths named above belong to a different environment. Use the"
                " tools you actually have, and do the job the instruction describes rather than the"
                " literal steps.")

PLANNER_PROMPT = ("You are routing a question to specialists. The available roles are:\n"
                  "{roles}\n\n"
                  "Question: {question}\n\n"
                  "Reply with only a JSON array of exactly {limit} role names, most important "
                  "first. Name only roles from the list above.")


class DynamicTopologyConfig(FunctionBaseConfig, name="dynamic_topology"):
    """Builds the specialists a question needs, then answers through them."""

    llm_name: LLMRef = Field(description="Model used by the planner, the specialists and the composer")
    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="Tools the specialists may be given; names must match "
                                          "the role library")
    max_roles: int = Field(default=3, ge=1, description="Most specialists to build for one question")
    brief: str = Field(default="", description="What the task set asks of any agent working it")
    roles: dict = Field(default_factory=dict,
                        description="Role library override: name -> "
                                    "{tools, instructions, max_iterations}; empty uses the built-in one")
    coordinator_max_iterations: int = Field(default=12, description="Cap on the coordinator's delegation loop")
    role_middleware: list[str] = Field(default_factory=list,
                                       description="Middleware applied to every runtime-built role and to the "
                                       "coordinator over them")
    verbose: bool = Field(default=False, description="Verbose agent logging")


@register_function(config_type=DynamicTopologyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def dynamic_topology(config: DynamicTopologyConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from langchain_core.messages import HumanMessage

    from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    available = {str(t) for t in config.tool_names}
    # A tool set naming none of these (APEX's MCP, say) leaves nobody eligible and the run unmarked.
    library = validate(config.roles or ROLE_LIBRARY, "instructions")
    generic = not any(x in available for s in library.values() for x in s["tools"])
    if generic:
        logger.warning("No role names any of the %d offered tools; routing on instructions alone",
                       len(available))
    eligible = [n for n, s in library.items() if any(t in available for t in s["tools"])] \
        or list(library)
    catalogue = "\n".join(f"- {name}: {library[name]['instructions']}" for name in eligible)
    built: set[str] = set()
    coordinators: dict[str, object] = {}
    # Questions are evaluated concurrently, so every add_function call is serialised.
    build_lock = asyncio.Lock()

    async def _ensure_role(role: str, task: str = "") -> str | None:
        """Builds a specialist for the task it will work on; the same task reuses it.

        A specialist used to see only what the coordinator wrote to it, and a paraphrase drops
        exactly what these tasks are scored on -- which quarter, which percentage, which file.
        """
        # A method may return a name never on the menu; losing one specialist beats a KeyError.
        spec = library.get(role)
        if spec is None:
            logger.warning("Ignoring unknown role %r", role)
            return None
        tools = [t for t in spec["tools"] if t in available] or (sorted(available) if generic else [])
        # The cap was tuned for the tools a role names; elsewhere the library's longest cap is honest.
        iters = max(s["max_iterations"] for s in library.values()) if generic else spec["max_iterations"]
        # Briefs name tools of the set they were written for; saying so beats editing them per set.
        brief = " ".join(x for x in (config.brief, spec["instructions"]) if x) \
            + (GENERIC_NOTE if generic else "")
        if task:
            # The whole task, not a summary of it, and the shared directory that lets the
            # specialists hand work to each other instead of through the coordinator's retelling.
            brief += ("\n\nThe task in full, as it was given:\n" + task
                      + "\n\nYou share a working directory with the other specialists. Read what "
                        "they have already left there before repeating their work, and write what "
                        "you produce to a file, naming it in your reply.")
        if not tools:
            return None
        named = f"{role}__{hashlib.sha256(task.encode()).hexdigest()[:8]}" if task else role
        async with build_lock:
            if named in built:
                return named
            await builder.add_function(
                named,
                ToolCallAgentWorkflowConfig(
                    tool_names=tools,
                    llm_name=config.llm_name,
                    verbose=config.verbose,
                    max_iterations=iters,
                    handle_tool_errors=True,
                    truncation_retry={'max_retries': 4, 'token_scaling': 1.25},
                    middleware=list(config.role_middleware),
                    description=f"Specialist: {role}",
                    additional_instructions=brief,
                ))
            built.add(named)
            logger.info("Built specialist %s with tools %s", named, tools)
        return named

    async def _run(inputs: str) -> str:
        plan = await llm.ainvoke([
            HumanMessage(content=PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles))
        ])
        roles = parse_roles(getattr(plan, "content", str(plan)), config.max_roles, eligible)
        # The widest choice this workflow makes, so it is offered for marking; unmarked keeps the plan.
        prompt = PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles)
        picked = pick_staff(list(dict.fromkeys(roles)), eligible, prompt)
        chosen = [r for r in [await _ensure_role(role, inputs) for role in dict.fromkeys(picked)] if r]
        logger.info("Question routed to %s", chosen)

        # An unbuildable plan once yielded "", scoring as wrong instead of failure; answer directly.
        if not chosen:
            logger.warning("No specialist could be built for this question; answering directly")
            return str(getattr(await llm.ainvoke([HumanMessage(content=inputs)]), "content", ""))
        # One coordinator per role set, built once: re-adding raises a duplicate-name error.
        key = "_".join(chosen)
        async with build_lock:
            if key not in coordinators:
                coordinators[key] = await builder.add_function(
                    f"coordinator_{key}",
                    ToolCallAgentWorkflowConfig(
                        tool_names=chosen,
                        llm_name=config.llm_name,
                        verbose=config.verbose,
                        # Not scaled by the number of specialists: given the room, this
                        # coordinator spends it asking one of them forty-seven times.
                        max_iterations=config.coordinator_max_iterations,
                        handle_tool_errors=True,
                        truncation_retry={'max_retries': 4, 'token_scaling': 1.25},
                        # Middleware attaches per function: without this the delegation loop ran unbudgeted.
                        middleware=list(config.role_middleware),
                        description="Coordinates the specialists built for this question.",
                        additional_instructions=(config.brief + " " if config.brief else "") + (
                            "Send each part of the question to the specialist that fits it, then "
                            "answer, covering every part asked. "
                            "Call each specialist ONCE. If one returns a notice instead of work, "
                            "move on -- never call it again -- and answer with what you have. "
                            "When the task names a file to produce, confirm it exists with a "
                            "workspace tool before you say it was written, and write it yourself "
                            "if no specialist did."),
                    ))
        result = await coordinators[key].acall_invoke(inputs)
        return str(result)

    yield FunctionInfo.from_fn(_run, description="Answer through specialists chosen and built per question")
