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

import contextlib
import logging
import warnings
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.middleware.agent_finish.agent_finish_middleware import CIRCLING
from nat.middleware.agent_finish.agent_finish_middleware import Repeats
from nat.plugins.langchain.agent.tool_calling_agent.agent import _CLOSING
from nat.plugins.langchain.agent.tool_calling_agent.agent import CLEAR_CHUNK
from nat.plugins.langchain.agent.tool_calling_agent.agent import chars_per_token
from nat.plugins.langchain.agent.tool_calling_agent.agent import clear_trigger

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = ("You are a careful analyst. Use the available tools to look up facts and to compute "
                  "values; never guess. End with a short, direct answer.")


# The langchain agent's rule, so all three frameworks trim alike.
KEEP, CLEARED = 3, "[cleared]"


def _to_clear(held, size, over, chars_per_token):
    """The oldest results that free `over` rounded up to CLEAR_CHUNK, never the newest KEEP."""
    need, out = -(-over // CLEAR_CHUNK) * CLEAR_CHUNK, set()
    for x in held[:-KEEP]:
        if need <= 0:
            break
        need -= (size(x) - len(CLEARED)) / chars_per_token
        out.add(id(x))
    if out:
        logger.warning("history near the context window: cleared the %d oldest of %d tool results", len(out), len(held))
    return out


def _bare(part):
    # Unwrap ADK's {'result': text}: LiteLlm sends it as JSON and the model copied the escapes into edit_file
    reply = part.function_response
    if reply is None or not isinstance(reply.response, dict) or list(reply.response) != ["result"] \
            or not isinstance(reply.response["result"], str):
        return part
    # model_copy skips validation on purpose: typed dict, but LiteLlm sends a str as is
    text = reply.response["result"]
    return part.model_copy(update={"function_response": reply.model_copy(update={"response": text})})


def _adk_callbacks(runs):
    # Request is rebuilt, so the session keeps every result; state per invocation: questions run at once
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError

    def state(ctx):
        return runs.setdefault(ctx.invocation_id, {"chars_per_token": 4.0, "chars": 0, "repeats": Repeats()})

    def before(callback_context, llm_request):
        seen = state(callback_context)
        if seen["repeats"].circling:
            # As the cap: a tool-less closing turn follows
            raise LlmCallsLimitExceededError(f"one call made {CIRCLING} times over")
        chars = sum(len(c.model_dump_json()) for c in llm_request.contents)
        over = chars / seen["chars_per_token"] - clear_trigger()
        if over > 0:
            held = [p for c in llm_request.contents for p in (c.parts or []) if p.function_response]
            old = _to_clear(held, lambda p: len(str(p.function_response.response)), over, seen["chars_per_token"])
            def gone(p):
                return p.model_copy(update={"function_response": p.function_response.model_copy(
                    update={"response": {"result": CLEARED}})})

            llm_request.contents = [c.model_copy(update={"parts": [gone(p) if id(p) in old else p
                                                                   for p in c.parts or []]})
                                    for c in llm_request.contents]
            chars = sum(len(c.model_dump_json()) for c in llm_request.contents)
        seen["chars"] = chars
        llm_request.contents = [c.model_copy(update={"parts": [_bare(p) for p in c.parts]}) if c.parts else c
                                for c in llm_request.contents]

    def after(callback_context, llm_response):
        seen = state(callback_context)
        if sent := getattr(llm_response.usage_metadata, "prompt_token_count", None):
            seen["chars_per_token"] = chars_per_token(seen["chars"], sent)

    def after_tool(tool, args, tool_context, tool_response):
        said = state(tool_context)["repeats"].note(tool.name, args, tool_response)
        return {"result": said} if said is not tool_response else None

    return before, after, after_tool


def _autogen_context():
    """That rule as an AutoGen model context; get_messages copies, so the history keeps every result."""
    from autogen_core.model_context import UnboundedChatCompletionContext
    from autogen_core.models import FunctionExecutionResultMessage

    class Cleared(UnboundedChatCompletionContext):
        chars_per_token, chars = 4.0, 0

        def observe(self, prompt_tokens):
            # This agent's own reply: the client's total is shared
            if self.chars and prompt_tokens:
                self.chars_per_token = chars_per_token(self.chars, prompt_tokens)

        async def get_messages(self):
            messages = await super().get_messages()
            over = sum(len(m.model_dump_json()) for m in messages) / self.chars_per_token - clear_trigger()
            if over > 0:
                held = [r for m in messages if isinstance(m, FunctionExecutionResultMessage) for r in m.content]
                old = _to_clear(held, lambda r: len(r.model_dump_json()), over, self.chars_per_token)
                def gone(r):
                    return r.model_copy(update={"content": CLEARED}) if id(r) in old else r

                messages = [m.model_copy(update={"content": [gone(r) for r in m.content]})
                            if isinstance(m, FunctionExecutionResultMessage) else m for m in messages]
            self.chars = sum(len(m.model_dump_json()) for m in messages)
            return messages

    return Cleared()


class AdkProbeConfig(FunctionBaseConfig, name="adk_probe"):
    """Single Google ADK agent over NAT tools."""

    llm_name: LLMRef = Field(description="Model to use via the ADK wrapper")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="NAT tools exposed to the agent")
    system_prompt: str = Field(default=DEFAULT_PROMPT, description="Agent instructions")
    max_iter: int = Field(default=250, description="Model calls before the run must end")


class AutogenProbeConfig(FunctionBaseConfig, name="autogen_probe"):
    """Single AutoGen assistant over NAT tools."""

    llm_name: LLMRef = Field(description="Model to use via the AutoGen wrapper")
    tool_names: list[FunctionRef] = Field(default_factory=list, description="NAT tools exposed to the agent")
    system_prompt: str = Field(default=DEFAULT_PROMPT, description="Agent instructions")
    # 250 like its three siblings: worker_config injects the topology's number over this, so a
    # smaller default never bites at run time -- it only misleads whoever reads it, and it does
    # bite anyone who builds this config directly.
    max_turns: int = Field(default=250, description="Model rounds before the run must end")


@register_function(config_type=AdkProbeConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def adk_probe(config: AdkProbeConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from google.adk.agents import Agent
    from google.adk.agents.invocation_context import LlmCallsLimitExceededError
    from google.adk.agents.run_config import RunConfig
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    # The closer sees the analyst's turns by design
    logging.getLogger("google_adk.google.adk.runners").addFilter(
        lambda record: not str(record.msg).startswith("Event from an unknown agent"))
    # _bare puts a str in a dict field
    warnings.filterwarnings("ignore", message=r"(?s)Pydantic serializer warnings:.*field_name='response'",
                            category=UserWarning)

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.ADK)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.ADK)
    runs: dict = {}
    before, after, after_tool = _adk_callbacks(runs)
    agent = Agent(name="analyst", model=llm, description="Analyst", instruction=config.system_prompt, tools=tools,
                  before_model_callback=before, after_model_callback=after, after_tool_callback=after_tool)
    # Tool-less twin for the answer at the cap; own name: ADK passes the analyst's turns as context, not tool calls
    closer = Agent(name="closer", model=llm, description="Closer", instruction=config.system_prompt,
                   before_model_callback=before, after_model_callback=after)

    session_service = InMemorySessionService()
    artifacts = InMemoryArtifactService()
    runner = Runner(app_name="analyst", agent=agent, artifact_service=artifacts, session_service=session_service)
    closing = Runner(app_name="analyst", agent=closer, artifact_service=artifacts, session_service=session_service)

    async def _close(session_id: str) -> str:
        said = ""
        ask = types.Content(role="user", parts=[types.Part.from_text(text=_CLOSING)])
        async for event in closing.run_async(user_id="bench", session_id=session_id, new_message=ask,
                                             run_config=RunConfig(max_llm_calls=2)):
            if event.is_final_response() and event.content and event.content.parts:
                said = "".join(p.text for p in event.content.parts if p.text) or said
        return said

    async def _run(inputs: str) -> str:
        # A fresh session per question keeps benchmark items independent.
        session = await session_service.create_session(app_name="analyst", user_id="bench")
        content = types.Content(role="user", parts=[types.Part.from_text(text=inputs)])
        parts: list[str] = []
        answered = ""
        invocations: set[str] = set()
        # The only turn cap ADK offers; without it this is the one framework a run cannot bound.
        limit = RunConfig(max_llm_calls=config.max_iter)
        try:
            async for event in runner.run_async(user_id="bench", session_id=session.id,
                                                new_message=content, run_config=limit):
                invocations.add(event.invocation_id)
                if not (event.content and event.content.parts):
                    continue
                said = "".join(p.text for p in event.content.parts if p.text)
                if not said:
                    continue
                parts.append(said)
                # Text beside a function call is the preamble to it; only a final turn is an answer.
                if event.is_final_response():
                    answered = said
        except LlmCallsLimitExceededError as stop:
            # Raising graded as no answer; preambles alone are not one
            logger.warning("adk stopped (%s); asking for the answer with no tools", stop)
            try:
                answered = await _close(session.id) or answered
            except Exception as exc:  # noqa: BLE001 -- the work so far still beats nothing
                logger.warning("adk's closing turn failed (%s); returning the work so far", exc)
        finally:
            for invocation in invocations:
                runs.pop(invocation, None)
        return answered or "".join(parts)

    yield FunctionInfo.from_fn(_run, description="Run a single Google ADK agent over the configured NAT tools")


@register_function(config_type=AutogenProbeConfig, framework_wrappers=[LLMFrameworkEnum.AUTOGEN])
async def autogen_probe(config: AutogenProbeConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    from autogen_agentchat.agents import AssistantAgent

    from nat.plugins.autogen.tool_wrapper import FINISHED_ON
    from nat.plugins.autogen.tool_wrapper import REPEATS_ON

    # Logs every message whole at INFO: 10x the rest of the log
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.AUTOGEN)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.AUTOGEN)

    async def _run(inputs: str) -> str:
        agent = AssistantAgent(name="analyst",
                               model_client=llm,
                               tools=tools,
                               system_message=config.system_prompt,
                               max_tool_iterations=config.max_turns,
                               reflect_on_tool_use=True,
                               model_context=(context := _autogen_context()))
        # A holder per question, set before the run so every tool task inherits the same object.
        held: list[str] = []
        FINISHED_ON.set(held)
        REPEATS_ON.set(memory := Repeats())
        # Streamed rather than awaited, so the run can be left the moment `finish` answers. Awaited,
        # AutoGen has no point at which to stop: a tool cannot raise out of it, and telling the model
        # to stop did not -- it called `finish` 236 times in one run, 27 on average.
        last = ""
        # Closed on leaving: in-flight tool calls must not write while the answer is graded
        async with contextlib.aclosing(agent.run_stream(task=inputs)) as stream:
            async for message in stream:
                if usage := getattr(message, "models_usage", None):
                    context.observe(usage.prompt_tokens)
                if held:
                    return held[0]
                said = getattr(message, "content", None)
                if isinstance(said, str) and said:
                    last = said
                if memory.circling:
                    break
        if held:
            return held[0]
        if memory.circling:
            logger.warning("autogen made one call %d times over; asking for the answer with no tools", CIRCLING)
            closer = AssistantAgent(name="analyst", model_client=llm, tools=[], system_message=config.system_prompt,
                                    model_context=context)
            with contextlib.suppress(Exception):    # the work so far still beats nothing
                result = await closer.run(task=_CLOSING)
                said = getattr(result.messages[-1], "content", None) if result.messages else None
                last = said if isinstance(said, str) and said else last
        return last

    yield FunctionInfo.from_fn(_run, description="Run a single AutoGen assistant over the configured NAT tools")


