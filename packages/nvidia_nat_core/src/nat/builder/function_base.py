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
"""Base class for NAT functions providing type handling and schema management.

This module contains the FunctionBase abstract base class which provides core functionality
for NAT functions including type handling via generics, schema management for inputs and outputs,
and type conversion capabilities.
"""

import logging
import typing
from abc import ABC
from collections.abc import Callable
from types import NoneType

from pydantic import BaseModel
from pydantic import ValidationError

from nat.utils.type_converter import TypeConverter
from nat.utils.type_utils import DecomposedType
from nat.utils.type_utils import read_only_cached_property

InputT = typing.TypeVar("InputT")
StreamingOutputT = typing.TypeVar("StreamingOutputT")
SingleOutputT = typing.TypeVar("SingleOutputT")

logger = logging.getLogger(__name__)

# Names of the functions already reported as taking flat arguments, so the repair is
# announced once per function instead of once per call.
_flat_input_logged: set[str] = set()


class FunctionBase(typing.Generic[InputT, StreamingOutputT, SingleOutputT], ABC):
    """
    Abstract base class providing core functionality for NAT functions.

    This class provides type handling via generics, schema management for inputs and outputs,
    and type conversion capabilities.

    Parameters
    ----------
    InputT : TypeVar
        The input type for the function
    StreamingOutputT : TypeVar
        The output type for streaming results
    SingleOutputT : TypeVar
        The output type for single results

    Notes
    -----
    FunctionBase is the foundation of the NAT function system, providing:
    - Type handling via generics
    - Schema management for inputs and outputs
    - Type conversion capabilities
    - Abstract interface that concrete function classes must implement
    """

    def __init__(self,
                 *,
                 input_schema: type[BaseModel] | None = None,
                 streaming_output_schema: type[BaseModel] | type[None] | None = None,
                 single_output_schema: type[BaseModel] | type[None] | None = None,
                 converters: list[Callable[[typing.Any], typing.Any]] | None = None):

        converters = converters or []

        self._converter_list = converters

        final_input_schema = input_schema or DecomposedType(self.input_type).get_pydantic_schema(converters)

        assert not issubclass(final_input_schema, NoneType)

        self._input_schema = final_input_schema

        if streaming_output_schema is not None:
            self._streaming_output_schema = streaming_output_schema
        else:
            self._streaming_output_schema = DecomposedType(self.streaming_output_type).get_pydantic_schema(converters)

        if single_output_schema is not None:
            self._single_output_schema = single_output_schema
        else:
            self._single_output_schema = DecomposedType(self.single_output_type).get_pydantic_schema(converters)

        self._converter: TypeConverter = TypeConverter(converters)

    @read_only_cached_property
    def input_type(self) -> type[InputT]:
        """
        Get the input type of the function. The input type is determined by the generic parameters of the class.

        For example, if a function is defined as `def my_function(input: list[int]) -> str`, the `input_type` is
        `list[int]`.

        Returns
        -------
        type[InputT]
            The input type specified in the generic parameters

        Raises
        ------
        ValueError
            If the input type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:

            base_cls_args = typing.get_args(base_cls)

            if len(base_cls_args) == 3:
                return base_cls_args[0]

        raise ValueError("Could not find input schema")

    @read_only_cached_property
    def input_class(self) -> type:
        """
        Get the python class of the input type. This is the class that can be used to check if a value is an instance of
        the input type. It removes any generic or annotation information from the input type.

        For example, if a function is defined as `def my_function(input: list[int]) -> str`, the `input_class` is
        `list`.

        Returns
        -------
        type
            The python type of the input type
        """

        if (self.input_type is typing.Any):
            return object

        input_origin = typing.get_origin(self.input_type)

        if (input_origin is None):
            return self.input_type

        return input_origin

    @read_only_cached_property
    def input_schema(self) -> type[BaseModel]:
        """
        Get the Pydantic model schema for validating inputs. The schema must be pydantic models. This allows for
        type validation and coercion, and documenting schema properties of the input value. If the input type is
        already a pydantic model, it will be returned as is.

        For example, if a function is defined as `def my_function(input: list[int]) -> str`, the `input_schema` is::

            class InputSchema(BaseModel):
                input: list[int]


        Returns
        -------
        type[BaseModel]
            The Pydantic model class for input validation
        """
        return self._input_schema

    @property
    def converter_list(self) -> list[Callable[[typing.Any], typing.Any]]:
        """
        Get the list of type converters used by this function.

        Returns
        -------
        list[Callable[[typing.Any], typing.Any]]
            List of converter functions that transform input types
        """
        return self._converter_list

    @read_only_cached_property
    def streaming_output_type(self) -> type[StreamingOutputT]:
        """
        Get the streaming output type of the function. The streaming output type is determined by the generic parameters
        of the class.

        For example, if a function is defined as `def my_function(input: int) -> AsyncGenerator[dict[str, Any]]`,
        the `streaming_output_type` is `dict[str, Any]`.

        Returns
        -------
        type[StreamingOutputT]
            The streaming output type specified in the generic parameters

        Raises
        ------
        ValueError
            If the streaming output type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:

            base_cls_args = typing.get_args(base_cls)

            if len(base_cls_args) == 3:
                return base_cls_args[1]

        raise ValueError("Could not find output schema")

    @read_only_cached_property
    def streaming_output_class(self) -> type:
        """
        Get the python class of the output type. This is the class that can be used to check if a value is an instance
        of the output type. It removes any generic or annotation information from the output type.

        For example, if a function is defined as `def my_function(input: int) -> AsyncGenerator[dict[str, Any]]`,
        the `streaming_output_class` is `dict`.

        Returns
        -------
        type
            The python type of the output type
        """

        if (self.streaming_output_type is typing.Any):
            return object

        output_origin = typing.get_origin(self.streaming_output_type)

        if (output_origin is None):
            return self.streaming_output_type

        return output_origin

    @read_only_cached_property
    def streaming_output_schema(self) -> type[BaseModel] | type[None]:
        """
        Get the Pydantic model schema for validating streaming outputs. The schema must be pydantic models. This allows
        for type validation and coercion, and documenting schema properties of the output value. If the output type is
        already a pydantic model, it will be returned as is.

        For example, if a function is defined as `def my_function(input: int) -> AsyncGenerator[dict[str, Any]]`,
        the `streaming_output_schema` is::

            class StreamingOutputSchema(BaseModel):
                value: dict[str, Any]

        Returns
        -------
        type[BaseModel] | type[None]
            The Pydantic model class for streaming output validation, or NoneType if no streaming output.
        """
        return self._streaming_output_schema

    @read_only_cached_property
    def single_output_type(self) -> type[SingleOutputT]:
        """
        Get the single output type of the function. The single output type is determined by the generic parameters
        of the class. Returns NoneType if no single output is supported.

        For example, if a function is defined as `def my_function(input: int) -> list[str]`, the `single_output_type` is
        `list[str]`.

        Returns
        -------
        type[SingleOutputT]
            The single output type specified in the generic parameters

        Raises
        ------
        ValueError
            If the single output type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:

            base_cls_args = typing.get_args(base_cls)

            if len(base_cls_args) == 3:
                return base_cls_args[2]

        raise ValueError("Could not find output schema")

    @read_only_cached_property
    def single_output_class(self) -> type:
        """
        Get the python class of the output type. This is the class that can be used to check if a value is an instance
        of the output type. It removes any generic or annotation information from the output type.

        For example, if a function is defined as `def my_function(input: int) -> list[str]`, the `single_output_class`
        is `list`.

        Returns
        -------
        type
            The python type of the output type
        """

        if (self.single_output_type is typing.Any):
            return object

        output_origin = typing.get_origin(self.single_output_type)

        if (output_origin is None):
            return self.single_output_type

        return output_origin

    @read_only_cached_property
    def single_output_schema(self) -> type[BaseModel] | type[None]:
        """
        Get the Pydantic model schema for validating single outputs. The schema must be pydantic models. This allows for
        type validation and coercion, and documenting schema properties of the output value. If the output type is
        already a pydantic model, it will be returned as is.

        For example, if a function is defined as `def my_function(input: int) -> list[str]`, the `single_output_schema`
        is::

            class SingleOutputSchema(BaseModel):
                value: list[str]

        Returns
        -------
        type[BaseModel] | type[None]
            The Pydantic model class for single output validation, or None if no single output
        """
        return self._single_output_schema

    @property
    def has_streaming_output(self) -> bool:
        """
        Check if this function supports streaming output.

        Returns
        -------
        bool
            True if the function supports streaming output, False otherwise
        """
        # Override in derived classes if this needs to return False. Assumption is, if not overridden, it has streaming
        # output because the ABC has it.
        return True

    @property
    def has_single_output(self) -> bool:
        """
        Check if this function supports single output.

        Returns
        -------
        bool
            True if the function supports single output, False otherwise
        """
        # Override in derived classes if this needs to return False. Assumption is, if not overridden, it has single
        # output because the ABC has it.
        return True

    def _convert_input(self, value: typing.Any) -> InputT:
        if (DecomposedType(self.input_type).is_instance(value)):
            return value

        # No converter, try to convert to the input schema
        if (isinstance(value, dict)):
            try:
                value = self.input_schema.model_validate(value)
            except ValidationError as e:
                value = wrap_flat_input(self.input_schema,
                                        value,
                                        e,
                                        getattr(self, "instance_name", self.input_schema.__name__))

            if (self.input_type == self.input_schema):
                return value

        if (isinstance(value, self.input_schema)):

            # Get the first value from the schema object
            first_key = next(iter(self.input_schema.model_fields.keys()))

            return getattr(value, first_key)

        # If the value is None bypass conversion to avoid raising an error.
        if value is None:
            return value

        # Fallback to the converter
        try:
            return self._converter.convert(value, to_type=self.input_class)
        except ValueError as e:
            # Input parsing should yield a TypeError instead of a ValueError
            raise TypeError from e


def wrap_flat_input(schema: type[BaseModel], value: dict, error: ValidationError, name: str) -> BaseModel:
    """
    Validate arguments which were supplied without the wrapper object their schema asks for.

    A caller which sends the inner fields of a single-object schema directly has left off the
    wrapper rather than supplied the wrong arguments.

    Parameters
    ----------
    schema : type[BaseModel]
        The schema the arguments failed to validate against.
    value : dict
        The arguments which failed to validate.
    error : ValidationError
        The original failure, raised again whenever wrapping does not apply or does not help.
    name : str
        The function or tool the arguments were meant for, used to report the repair once.

    Returns
    -------
    BaseModel
        The schema holding the wrapped arguments.
    """
    wrappers = [
        field_name for field_name, field in schema.model_fields.items()
        if field.is_required() and isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
    ]

    # Nothing to wrap the arguments into, or a choice of fields to guess between: either way
    # the error the caller already has beats a guess.
    if (len(wrappers) != 1):
        raise error

    # Only a missing wrapper is repaired; any other complaint can come from a call which already
    # ran, and running that a second time would not be safe.
    if (not any(detail["type"] == "missing" and detail["loc"] == (wrappers[0], ) for detail in error.errors())):
        raise error

    try:
        wrapped = schema.model_validate({wrappers[0]: value})
    except ValidationError:
        raise error from None

    if (name not in _flat_input_logged):
        _flat_input_logged.add(name)
        logger.info("'%s' was called with the fields of '%s' supplied flat; wrapping them.", name, wrappers[0])

    return wrapped
