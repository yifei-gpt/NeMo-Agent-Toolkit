# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tool Wrapper file"""
import json
import logging
import types
from collections.abc import AsyncIterator
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any
from typing import Union
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import Function
from nat.cli.register_workflow import register_tool_wrapper

logger = logging.getLogger(__name__)


def _is_text(annotation: Any) -> bool:
    """Whether this field is declared as a string -- `str`, or an optional/union of it.

    A `list[str]` is not text: its JSON form is what the model must send, so it still decodes.
    """
    import types
    import typing
    if annotation is str:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return bool(args) and all(a is str for a in args)
    return False


def resolve_type(t: Any) -> Any:
    """Return the non-None member of a Union/PEP 604 union;
    otherwise return the type unchanged.

    Args:
        t (Any): The type to resolve.

    Returns:
        Any: The resolved type.
    """
    # `typing.Union[...]` as well as `X | Y`: pydantic builds the former from a JSON Schema
    # `anyOf`, and ADK raises rather than degrading when a union reaches its declaration parser.
    origin = get_origin(t)
    if origin is types.UnionType or origin is Union:
        members = [arg for arg in get_args(t) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
        # A real sum type has no declaration ADK can express, and naming one member would
        # advertise a schema the tool rejects, so it is passed as a plain object instead.
        return dict[str, Any] if members else t
    return t


def _shape_text(input_schema: Any, names: list[str]) -> str:
    """The JSON Schema of the parameters ADK had to flatten, small enough to sit in a docstring."""
    try:
        schema = input_schema.model_json_schema()
    except Exception:  # noqa: BLE001
        return ""
    props = {n: schema.get("properties", {}).get(n) for n in names}
    body = {"properties": {k: v for k, v in props.items() if v}, "$defs": schema.get("$defs", {})}
    text = json.dumps(body, separators=(",", ":"))
    return f"Argument shape (one of these variants must match exactly): {text[:4000]}"


def _expressible(annotation: Any, func_name: str) -> Any:
    """ADK refuses a declaration it cannot express and takes the whole tool down with it, so the
    type it rejects becomes a plain object -- the function validates its own arguments anyway."""
    import inspect as _inspect

    from google.adk.tools import _function_parameter_parse_util as _parse
    from google.adk.utils.variant_utils import get_google_llm_variant
    try:
        _parse._parse_schema_from_parameter(
            get_google_llm_variant(),
            _inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation),
            func_name)
        return annotation
    except (ValueError, TypeError) as refusal:
        # ADK phrases every refusal as one of these. Anything else means its private parser moved,
        # and reading that as "not expressible" would flatten every tool in the fleet in silence.
        if "parameter" not in str(refusal) and "supported" not in str(refusal):
            logger.warning("ADK's declaration parser did not answer as expected (%s); %s is being "
                           "passed as a plain object", refusal, annotation)
        logger.debug("ADK cannot declare %s for %s; passing it as an object", annotation, func_name)
        return dict[str, Any]


@register_tool_wrapper(wrapper_type=LLMFrameworkEnum.ADK)
def google_adk_tool_wrapper(
    name: str,
    fn: Function,
    _builder: Builder  # pylint: disable=W0613
) -> Any:  # Changed from Callable[..., Any] to Any to allow FunctionTool return
    """Wrap a NAT `Function` as a Google ADK `FunctionTool`.

    Args:
        name (str): The name of the tool.
        fn (Function): The NAT `Function` to wrap.
        _builder (Builder): The NAT `Builder` (not used).
    Returns:
        A Google ADK `FunctionTool` wrapping the NAT `Function`.
    """
    import inspect

    def _decoded(kwargs: dict[str, Any]) -> dict[str, Any]:
        """ADK hands a structured argument through as the JSON text the model emitted, and the
        function's own schema then rejects it, so it is decoded back before the call."""
        fields = getattr(fn.input_schema, "model_fields", {}) or {}
        out = dict(kwargs)
        for key, value in kwargs.items():
            if not isinstance(value, str) or key not in fields:
                continue
            if not value.lstrip()[:1] in ("{", "["):
                continue
            # A field declared as text keeps the text. Decoding it turned a message that merely
            # began with "{" into a mapping, and the schema then read it as two inputs at once:
            # "Either messages or input_message must be provided, not both".
            if _is_text(fields[key].annotation):
                continue
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                pass                      # a string that merely looks like JSON stays a string
        # A one-of schema shows the model both fields, and it fills both: "Either messages or
        # input_message must be provided, not both". The plain string is what a tool call means.
        if out.get("messages") is not None and out.get("input_message") is not None:
            out.pop("messages")
        return out

    async def callable_ainvoke(*args: Any, **kwargs: Any) -> Any:
        """Async function to invoke the NAT function.

        Args:
            *args: Positional arguments to pass to the NAT function.
            **kwargs: Keyword arguments to pass to the NAT function.
        Returns:
            Any: The result of invoking the NAT function.
        """
        return await fn.acall_invoke(*args, **_decoded(kwargs))

    async def callable_astream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Async generator to stream results from the NAT function.

        Args:
            *args: Positional arguments to pass to the NAT function.
            **kwargs: Keyword arguments to pass to the NAT function.
        Yields:
            Any: Streamed items from the NAT function.
        """
        async for item in fn.acall_stream(*args, **_decoded(kwargs)):
            yield item

    def nat_function(
        func: Callable[..., Any] | None = None,
        *,
        name: str = name,
        description: str | None = fn.description,
        input_schema: Any = fn.input_schema,
    ) -> Callable[..., Any]:
        """
        Decorator to wrap a function as a NAT function.

        Args:
            func (Callable): The function to wrap.
            name (str): The name of the function.
            description (str): The description of the function.
            input_schema (BaseModel): The Pydantic model defining the input schema.

        Returns:
            Callable[..., Any]: The wrapped function.
        """
        if func is None:
            raise ValueError("'func' must be provided.")

        # If input_schema is a dataclass, convert it to a Pydantic model
        if input_schema is not None and is_dataclass(input_schema):
            input_schema = BaseModel.model_validate(input_schema)

        def decorator(func_to_wrap: Callable[..., Any]) -> Callable[..., Any]:
            """
            Decorator to set metadata on the function.
            """
            # Set the function's metadata
            if name is not None:
                func_to_wrap.__name__ = name
            if description is not None:
                func_to_wrap.__doc__ = description

            # Set signature only if input_schema is provided
            params: list[inspect.Parameter] = []
            degraded: list[str] = []
            if input_schema is not None:
                annotations = getattr(input_schema, "__annotations__", {}) or {}
                fields = getattr(input_schema, "model_fields", {}) or {}
                needed, optional = [], []
                for param_name, param_annotation in annotations.items():
                    usable = _expressible(resolve_type(param_annotation), name)
                    # Compared by value, not identity: `resolve_type` builds its own permissive
                    # object for a union, and an identity test would miss that one silently.
                    if usable == dict[str, Any] and param_annotation != dict[str, Any]:
                        degraded.append(param_name)
                    # ADK reads mandatory off the SIGNATURE, not the schema: a parameter with no
                    # default is one it will refuse the call for. Synthesised without defaults,
                    # every optional parameter became mandatory, and a model that correctly left
                    # one out was told to try again -- 25 of 142 task_list calls on ADK.
                    field = fields.get(param_name)
                    if field is not None and not field.is_required():
                        optional.append(
                            inspect.Parameter(param_name,
                                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                              annotation=usable,
                                              default=field.get_default(call_default_factory=True)))
                    else:
                        needed.append(
                            inspect.Parameter(param_name,
                                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                              annotation=usable))
                # Defaults last, or Signature() refuses the order.
                params = needed + optional
            # A parameter passed as a bare object tells the model nothing about which fields go
            # together, and the server still rejects the wrong combination. The declaration cannot
            # carry a union, but the description can, so the shape goes there instead.
            if degraded:
                shape = _shape_text(input_schema, degraded)
                if shape:
                    func_to_wrap.__doc__ = f"{func_to_wrap.__doc__ or ''}\n\n{shape}".strip()
            setattr(func_to_wrap, "__signature__", inspect.Signature(parameters=params))

            return func_to_wrap

        # If func is None, return the decorator itself to be applied later
        if func is None:
            return decorator
        # Otherwise, apply the decorator to the provided function
        return decorator(func)

    from google.adk.tools.function_tool import FunctionTool

    if fn.has_streaming_output and not fn.has_single_output:
        logger.debug("Creating streaming FunctionTool for: %s", name)
        callable_tool = nat_function(func=callable_astream)
    else:
        logger.debug("Creating non-streaming FunctionTool for: %s", name)
        callable_tool = nat_function(func=callable_ainvoke)
    return FunctionTool(callable_tool)
