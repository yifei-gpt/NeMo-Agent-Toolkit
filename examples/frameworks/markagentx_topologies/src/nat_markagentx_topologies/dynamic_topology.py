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
"""A topology whose specialists are created per question, not declared in the config.

Every other topology here fixes its nodes in YAML; this one asks the model which roles
the question needs, builds only those with `Builder.add_function`, then delegates.
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

# Each entry is a role the planner may ask for: the tools it gets and how it is told to work.
GENERIC_NOTE = (" The tools and file paths named above belong to a different environment. Use the"
                " tools you actually have, and do the job the instruction describes rather than the"
                " literal steps.")

ROLE_LIBRARY: dict[str, dict] = {
    # Workspace roles: eligible only when the config exposes the workspace tools, since
    # a role whose tools are all missing is skipped when the planner picks it.
    "workspace_surveyor": {
        "max_iterations": 45,
        "tools": ["workspace_list", "workspace_read", "workspace_write", "workspace_search"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Survey the workspace, then write what you found to _notes/survey.md with "
                         "workspace_write. Reply with one sentence naming that file, never the findings."),
    },
    "workspace_author": {
        "max_iterations": 45,
        "tools": ["workspace_list", "workspace_read", "workspace_write", "workspace_search"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Read _notes/survey.md first, then write the deliverable with workspace_write "
                         "under the exact filename the task names. Reply with the filename only."),
    },
    "workspace_auditor": {
        "max_iterations": 45,
        "tools": ["workspace_list", "workspace_read", "workspace_write", "workspace_search"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Read the deliverable back and rewrite it if a requirement or the format is "
                         "missed. Reply with one sentence, never the file contents."),
    },
    # Generic roles: the benchmark supplies helpers under these names, so one library serves
    # code, planning and tool-graph tasks without a role per benchmark.
    "drafter": {
        "max_iterations": 12,
        "tools": ["write_draft"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Send the task to write_draft and return its answer verbatim, in the exact format "
                         "the task demands. Call write_draft once."),
    },
    "critic": {
        "max_iterations": 12,
        "tools": ["list_defects"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Send the draft to list_defects and report only the concrete defects it names. "
                         "Never rewrite the answer yourself."),
    },
    "finalizer": {
        "max_iterations": 12,
        "tools": ["emit_final"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Send the draft and any criticism to emit_final, then reply with its output "
                         "verbatim and nothing else."),
    },
    "entity_researcher": {
        # It is asked for dates, so it gets the clock: one tool leaves it re-searching until the budget goes.
        "tools": ["wikipedia_search", "calculator", "current_datetime"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Look up one named entity at a time and report the dates, figures and relations "
                         "you find. Do not attempt the wider question."),
        "max_iterations": 8,
    },
    "date_resolver": {
        "tools": ["wikipedia_search", "calculator", "current_datetime"],
        "instructions": "Establish the single date or interval asked for and state it plainly.",
        "max_iterations": 5,
    },
    "calculator_specialist": {
        "tools": ["wikipedia_search", "calculator", "current_datetime"],
        "instructions": "Do the arithmetic on the numbers you are given. Never look anything up.",
        "max_iterations": 4,
    },
    "comparison_analyst": {
        "tools": ["wikipedia_search", "calculator", "current_datetime"],
        "instructions": ("Reply in at most three sentences and never restate what you were given. ""Compare the quantities the question asks about, gathering any figure you are "
                         "missing before deciding."),
        "max_iterations": 8,
    },
}

PLANNER_PROMPT = ("You are routing a question to specialists. The available roles are:\n"
                  "{roles}\n\n"
                  "Question: {question}\n\n"
                  "Reply with only a JSON array of the role names needed, most important first, "
                  "at most {limit}. Example: [\"entity_researcher\", \"calculator_specialist\"]")


class DynamicTopologyConfig(FunctionBaseConfig, name="dynamic_topology"):
    """Builds the specialists a question needs, then answers through them."""

    llm_name: LLMRef = Field(description="Model used by the planner, the specialists and the composer")
    tool_names: list[FunctionRef] = Field(default_factory=list,
                                          description="Tools the specialists may be given; names must match "
                                          "the role library")
    max_roles: int = Field(default=3, ge=1, description="Most specialists to build for one question")
    coordinator_max_iterations: int = Field(default=12, description="Cap on the coordinator's delegation loop")
    role_middleware: list[str] = Field(default_factory=list,
                                       description="Middleware applied to every runtime-built role and to the "
                                       "coordinator over them")
    verbose: bool = Field(default=False, description="Verbose agent logging")


def _parse_roles(text: str, limit: int, eligible: list[str]) -> list[str]:
    """Reads the planner's role list, falling back to a name scan when the JSON is malformed."""
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
        names = [role for role in eligible if role in (text or "")]
    # The planner sometimes repeats a role; a duplicate adds a turn without adding coverage.
    kept = list(dict.fromkeys(n for n in names if n in eligible))
    kept = kept[:limit] or eligible[:1]
    # A one-specialist plan is the degenerate case this topology exists to avoid, and it leaves the
    # coordinator a menu of one. Where the cap allows a second, take one; limit=1 stays single by design.
    for role in eligible:
        if len(kept) >= min(2, limit):
            break
        if role not in kept:
            kept.append(role)
    return kept


@register_function(config_type=DynamicTopologyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def dynamic_topology(config: DynamicTopologyConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from langchain_core.messages import HumanMessage

    from nat.plugins.langchain.agent.tool_calling_agent.register import ToolCallAgentWorkflowConfig

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    available = {str(t) for t in config.tool_names}
    # Offering a role whose tools this config never exposes produces a plan nothing can build.
    # A tool set this library names nothing from -- APEX's MCP tools, say -- leaves nobody eligible,
    # and the routing choice would then be made over an empty menu and the run silently unmarked.
    generic = not any(x in available for s in ROLE_LIBRARY.values() for x in s["tools"])
    if generic:
        logger.warning("No role names any of the %d offered tools; routing on instructions alone",
                       len(available))
    eligible = [n for n, s in ROLE_LIBRARY.items() if any(t in available for t in s["tools"])] \
        or list(ROLE_LIBRARY)
    catalogue = "\n".join(f"- {name}: {ROLE_LIBRARY[name]['instructions']}" for name in eligible)
    built: set[str] = set()
    coordinators: dict[str, object] = {}
    # Questions are evaluated concurrently, so every add_function call is serialised.
    build_lock = asyncio.Lock()

    async def _ensure_role(role: str) -> str | None:
        """Builds a specialist on first use; later questions reuse the same instance."""
        # A marking method may hand back a name that was never on the menu; losing that one
        # specialist is recoverable, taking the whole question down with a KeyError is not.
        spec = ROLE_LIBRARY.get(role)
        if spec is None:
            logger.warning("Ignoring unknown role %r", role)
            return None
        tools = [t for t in spec["tools"] if t in available] or (sorted(available) if generic else [])
        # A role's iteration cap was tuned for the tools it names; on a tool set it does not name,
        # the library's own longest cap is the only honest choice.
        iters = max(s["max_iterations"] for s in ROLE_LIBRARY.values()) if generic else spec["max_iterations"]
        # A brief names the tools and hand-off files of the tool set it was written for; on any other
        # one those do not exist. Said plainly beats editing the brief, which loses its referents.
        brief = spec["instructions"] + (GENERIC_NOTE if generic else "")
        if not tools:
            return None
        async with build_lock:
            if role in built:
                return role
            await builder.add_function(
                role,
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
            built.add(role)
            logger.info("Built specialist %s with tools %s", role, tools)
        return role

    async def _run(inputs: str) -> str:
        plan = await llm.ainvoke([
            HumanMessage(content=PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles))
        ])
        roles = _parse_roles(getattr(plan, "content", str(plan)), config.max_roles, eligible)
        # Which specialists exist is the widest choice this workflow makes, so it is offered
        # for marking; with nothing marking, decision_point returns the planner's own pick.
        prompt = PLANNER_PROMPT.format(roles=catalogue, question=inputs, limit=config.max_roles)
        picked: list[str] = []
        for role in dict.fromkeys(roles):
            # Each pick is "which specialist to add next", so a role already taken is off the menu:
            # otherwise two picks can land on one and the plan silently loses a specialist.
            menu = [r for r in eligible if r not in picked]
            if not menu:
                break
            picked.append(decision_point("role_selection", candidates=menu,
                                         native=role if role in menu else menu[0],
                                         context=[{"role": "user", "content": prompt}]))
        chosen = [r for r in [await _ensure_role(role) for role in dict.fromkeys(picked)] if r]
        logger.info("Question routed to %s", chosen)

        # An unbuildable plan used to yield an empty string, which scores as a wrong answer
        # rather than as the failure it is; answering directly at least reports the model.
        if not chosen:
            logger.warning("No specialist could be built for this question; answering directly")
            return str(getattr(await llm.ainvoke([HumanMessage(content=inputs)]), "content", ""))
        # Concurrent questions often pick the same roles, so one coordinator per role set is
        # built once and reused; adding it again raises a duplicate-name error.
        key = "_".join(chosen)
        async with build_lock:
            if key not in coordinators:
                coordinators[key] = await builder.add_function(
                    f"coordinator_{key}",
                    ToolCallAgentWorkflowConfig(
                        tool_names=chosen,
                        llm_name=config.llm_name,
                        verbose=config.verbose,
                        max_iterations=config.coordinator_max_iterations,
                        handle_tool_errors=True,
                        truncation_retry={'max_retries': 4, 'token_scaling': 1.25},
                        # Middleware attaches per function, so leaving it off here left the
                        # delegation loop unbudgeted while every specialist under it was capped.
                        middleware=list(config.role_middleware),
                        description="Coordinates the specialists built for this question.",
                        additional_instructions=(
                            "Send each part of the question to the specialist that fits it, then "
                            "answer with a short, direct statement covering every part asked. "
                            "Call each specialist ONCE. If one returns a notice instead of work, "
                            "move on -- never call it again -- and answer with what you have."),
                    ))
        result = await coordinators[key].acall_invoke(inputs)
        return str(result)

    yield FunctionInfo.from_fn(_run, description="Answer through specialists chosen and built per question")
