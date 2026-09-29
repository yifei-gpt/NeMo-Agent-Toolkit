# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class CalculatorToolConfig(FunctionGroupBaseConfig, name="calculator"):
    include: list[str] = Field(default_factory=lambda: ["add", "subtract", "multiply", "divide", "compare"],
                               description="The list of functions to include in the calculator function group.")


@register_function_group(config_type=CalculatorToolConfig)
async def calculator(_config: CalculatorToolConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """Create and register the calculator function group.

    Args:
        _config: Calculator function group configuration (unused).
        _builder: Workflow builder (unused).

    Yields:
        FunctionGroup: The configured calculator function group with add, subtract,
            multiply, divide, and compare operations.
    """
    import math

    group = FunctionGroup(config=_config)

    async def _add(numbers: list[float]) -> float:
        """Add two or more numbers together."""
        if len(numbers) < 2:
            raise ValueError("This tool only supports addition between two or more numbers.")
        return sum(numbers)

    async def _subtract(numbers: list[float]) -> float:
        """Subtract one number from another."""
        if len(numbers) != 2:
            raise ValueError("This tool only supports subtraction between two numbers.")
        a, b = numbers
        return a - b

    async def _multiply(numbers: list[float]) -> float:
        """Multiply two or more numbers together."""
        if len(numbers) < 2:
            raise ValueError("This tool only supports multiplication between two or more numbers.")
        return math.prod(numbers)

    async def _divide(numbers: list[float]) -> float:
        """Divide one number by another."""
        if len(numbers) != 2:
            raise ValueError("This tool only supports division between two numbers.")
        a, b = numbers
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    async def _compare(numbers: list[float]) -> str:
        """Compare two numbers."""
        if len(numbers) != 2:
            raise ValueError("This tool only supports comparison between two numbers.")
        a, b = numbers
        if a > b:
            return f"{a} is greater than {b}"
        if a < b:
            return f"{a} is less than {b}"
        return f"{a} is equal to {b}"

    async def _evaluate(expression: str) -> str:
        """Work out an arithmetic expression and return it with its result, so a caller can see
        which sum it got back. Takes + - * / ** %, the constants pi e tau inf, and sqrt cbrt exp
        log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh degrees radians hypot floor
        ceil factorial comb perm gcd lcm abs round min max, bare or as math.sqrt.
        Example: "143 * 12.50 + 87 * 3.20" or "log(2) * sqrt(pi)"."""
        import ast
        import operator
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
               ast.USub: operator.neg, ast.UAdd: operator.pos}
        consts = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
        fns = {n: getattr(math, n) for n in (
            "sqrt", "cbrt", "exp", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos", "atan",
            "atan2", "sinh", "cosh", "tanh", "degrees", "radians", "hypot", "floor", "ceil",
            "factorial", "comb", "perm", "gcd", "lcm")} | {"abs": abs, "round": round, "min": min, "max": max}

        def name_of(node):
            # `sqrt` or `math.sqrt`, as a model writes either.
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
                return node.attr
            return node.id if isinstance(node, ast.Name) else None

        def call(fn, *args):
            # In the agent's own process: an integer that keeps growing stalls every agent in the run.
            if (fn is operator.pow and isinstance(args[1], int) and abs(args[1]) > 10**4 and abs(args[0]) > 1) or (
                    fn in (math.factorial, math.comb, math.perm) and any(abs(a) > 10**4 for a in args)):
                raise ValueError("that number is too large to work out here; use run_code")
            return fn(*args)

        def walk(node):
            # Listed names only: any other name or call would make this an eval of whatever was sent.
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                return call(ops[type(node.op)], walk(node.left), walk(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](walk(node.operand))
            if name_of(node) in consts:
                return consts[name_of(node)]
            if isinstance(node, ast.Call) and name_of(node.func) in fns and not node.keywords:
                return call(fns[name_of(node.func)], *map(walk, node.args))
            raise ValueError(f"only numbers, + - * / ** %, and the listed constants and functions are allowed, "
                             f"not {ast.dump(node)[:40]}")

        try:
            value = walk(ast.parse(expression.strip(), mode="eval").body)
        except ZeroDivisionError:
            return f"{expression} = undefined (division by zero)"
        except Exception as exc:  # noqa: BLE001 -- the caller fixes the expression, not the tool
            return f"could not work out {expression!r}: {exc}"
        return f"{expression} = {value}"

    group.add_function(name="evaluate", fn=_evaluate, description=_evaluate.__doc__)
    group.add_function(name="add", fn=_add, description=_add.__doc__)
    group.add_function(name="subtract", fn=_subtract, description=_subtract.__doc__)
    group.add_function(name="multiply", fn=_multiply, description=_multiply.__doc__)
    group.add_function(name="divide", fn=_divide, description=_divide.__doc__)
    group.add_function(name="compare", fn=_compare, description=_compare.__doc__)

    yield group
