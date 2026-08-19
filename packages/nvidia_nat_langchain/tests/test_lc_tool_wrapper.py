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

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from nat.builder.builder import Builder
from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.function import EmptyFunctionConfig
from nat.plugins.langchain.tool_wrapper import langchain_tool_wrapper


def _content_text(content: object) -> str:
    if isinstance(content, list):
        first_content = content[0]
        if isinstance(first_content, dict):
            return first_content["text"]
        return str(first_content.text)
    return str(content)


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_maps_string_to_input_message() -> None:

    async def _echo(chat_request_or_message: ChatRequestOrMessage) -> str:
        if chat_request_or_message.input_message is not None:
            return chat_request_or_message.input_message
        return _content_text(chat_request_or_message.messages[-1].content)  # type: ignore[index, union-attr]

    info = FunctionInfo.from_fn(_echo, description="Echo input")
    fn = LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="echo")
    tool = langchain_tool_wrapper("echo", fn, MagicMock(spec=Builder))

    assert await tool.ainvoke("hello") == "hello"
    assert await tool.ainvoke({"messages": [{"role": "user", "content": "hi"}]}) == "hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": "json hi"}]'}) == "json hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": {"type": "text", "text": "json object hi"}}]'}
                              ) == "json object hi"


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_maps_string_to_chat_request() -> None:

    async def _echo(chat_request: ChatRequest) -> str:
        return _content_text(chat_request.messages[-1].content)

    info = FunctionInfo.from_fn(_echo, description="Echo input")
    fn = LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name="echo")
    tool = langchain_tool_wrapper("echo", fn, MagicMock(spec=Builder))

    assert await tool.ainvoke("hello") == "hello"
    assert await tool.ainvoke({"messages": [{"role": "user", "content": "hi"}]}) == "hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": "json hi"}]'}) == "json hi"
    assert await tool.ainvoke({"messages": '[{"role": "user", "content": {"type": "text", "text": "json object hi"}}]'}
                              ) == "json object hi"


class NestedRequest(BaseModel):
    """The object an MCP style tool asks for."""

    action: str
    ticker: str | None = None


class NestedInput(BaseModel):
    """An input schema whose only field is the request object."""

    request: NestedRequest


class TwoNestedInput(BaseModel):
    """An input schema where two fields could hold the same arguments."""

    request: NestedRequest
    fallback: NestedRequest


def _nested_tool(input_schema: type[BaseModel], name: str):
    """Wrap a real NAT function which takes a single nested request object."""

    async def _lookup(tool_input: BaseModel | None = None, **kwargs) -> str:
        return kwargs["request"].action

    info = FunctionInfo.create(single_fn=_lookup, description="Look a company up", input_schema=input_schema)
    fn = LambdaFunction.from_info(config=EmptyFunctionConfig(), info=info, instance_name=name)

    return langchain_tool_wrapper(name, fn, MagicMock(spec=Builder))


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_keeps_wrapped_arguments() -> None:
    tool = _nested_tool(NestedInput, "lc_wrapped")

    assert await tool.ainvoke({"request": {"action": "cik", "ticker": "KVUE"}}) == "cik"


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_wraps_flat_arguments() -> None:
    tool = _nested_tool(NestedInput, "lc_flat")

    assert await tool.ainvoke({"action": "cik", "ticker": "KVUE"}) == "cik"


@pytest.mark.asyncio
async def test_langchain_tool_wrapper_keeps_the_original_error() -> None:
    tool = _nested_tool(NestedInput, "lc_no_match")

    with pytest.raises(ValidationError) as exc_info:
        await tool.ainvoke({"nonsense": "KVUE"})

    assert "validation error for NestedInput" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)

    ambiguous = _nested_tool(TwoNestedInput, "lc_ambiguous")

    with pytest.raises(ValidationError) as exc_info:
        await ambiguous.ainvoke({"action": "cik"})

    assert "2 validation errors for TwoNestedInput" in str(exc_info.value)


@pytest.mark.asyncio
async def test_langchain_tool_node_runs_a_flat_call() -> None:
    """The agent calls tools through a ToolNode, which turns a refusal into a message nobody sees."""
    from langchain_core.messages import AIMessage
    from langgraph.prebuilt import ToolNode
    from langgraph.runtime import DEFAULT_RUNTIME

    tool = _nested_tool(NestedInput, "lc_tool_node")
    node = ToolNode([tool], handle_tool_errors=True)
    call = AIMessage(content="", tool_calls=[{"name": "lc_tool_node", "args": {"action": "cik"}, "id": "call_1"}])

    # The same config the agent uses, since a ToolNode called outside its graph has no runtime.
    response = await node.ainvoke({"messages": [call]}, config={"configurable": {"__pregel_runtime": DEFAULT_RUNTIME}})
    message = response["messages"][-1]

    assert message.content == "cik"
    assert message.status == "success"
