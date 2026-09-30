# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace file tools for benchmarks that hand an agent a directory and a brief."""

import asyncio
import hashlib
import json
import os
import re
import secrets
import shlex
import tempfile
import urllib.request
from collections.abc import AsyncGenerator
from collections import OrderedDict
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.tool import workspace_ops as ops


# Where a bridged session starts; the harness runs every container task set with this as its cwd.
_CONTAINER_ROOT = "/app"


def _root() -> Path:
    # No workspace was chosen, so scratch space -- the process cwd would hand over the whole checkout.
    if not os.environ.get("NAT_WORKSPACE_DIR"):
        os.environ["NAT_WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="nat_workspace_")
    return Path(os.environ["NAT_WORKSPACE_DIR"]).resolve()


def _staged() -> bool:
    """A world a run built, not one the user named. Both sides resolve: a directory reached through a link
    has two spellings, and comparing one with the other would read a staged world as the user's."""
    sweep = os.environ.get("MARKAGENTX_WORKSPACES")
    return bool(sweep and _root().is_relative_to(Path(sweep).resolve()))


def _bridge() -> str | None:
    """The task's own container, when the harness opened one. The same flag run_code reads."""
    return os.environ.get("NAT_BRIDGE_URL") if os.environ.get("NAT_BRIDGE_READY") else None


def _sh(command: str, timeout: float = 60.0) -> tuple[bool, str]:
    """One shell command where the agent's files are. -> (ran, output).

    Shell and not python: six of terminalbench's fourteen local images carry no python at all, and
    the bridged session routes a command line to bash either way. `cd` first so it routes there.
    """
    uri = _bridge()
    if not uri:
        return False, ""
    body = json.dumps({"generated_code": f"cd {_CONTAINER_ROOT} && " + command,
                        "language": "python", "timeout": timeout}).encode()
    try:
        with urllib.request.urlopen(
                urllib.request.Request(uri.rstrip("/") + "/execute", body,
                                       {"Content-Type": "application/json"}),
                timeout=timeout + 15) as answer:
            got = json.loads(answer.read())
    except Exception as exc:  # noqa: BLE001 -- unreachable and refusing mean the same thing here
        return True, f"the workspace is unreachable right now ({type(exc).__name__})."
    return True, (got.get("stdout") or "") + (got.get("stderr") or "")


def _where() -> str:
    """Container briefs name their files absolutely, and the root IS that directory: `x` and
    `/app/x` are one file. Unsaid, the model reads the mismatch as "these tools cannot reach it"
    and writes through bash heredocs instead."""
    return (f" This task's workspace root is {_CONTAINER_ROOT}, so `x` and {_CONTAINER_ROOT}/x name "
            f"the same file; either form works here." if _bridge() else "")


def _data() -> str:
    """The dataset a staged world's links reach into: reads may follow them, writes never."""
    owner, _, data = os.environ.get("MARKAGENTX_STAGED_DATA", "").partition(os.pathsep)
    return data if data and _staged() and owner == str(_root()) else ""


# Sent whole with every operation: never stale in a container started earlier, never in the agent's reach.
# Its cwd is the workspace, whose files must not shadow the standard library it imports.
_OPS_SOURCE = (f"import sys\nsys.path[:] = [p for p in sys.path if p not in ('', {ops.SANDBOX_ROOT!r})]\n"
               + Path(ops.__file__).read_text(encoding="utf-8"))


def _op_here(name: str, **args):
    """One file operation where the files are, as its tool's result; a ValueError is its refusal.

    In the sandbox whenever one can exist, so this host never opens a path the agent chose: a link its
    shell plants between a check and an open would lead here to anything this user can read. Only
    with no sandbox at all does the operation run in this process, and then no shell exists to plant one.
    """
    url = os.environ.get("NAT_SANDBOX_URL")
    call = json.dumps({"op": name, "root": ops.SANDBOX_ROOT if url else str(_root()), "data": _data(), **args})
    if not url:
        got = json.loads(ops.main(call))
    else:
        # Only the line carrying this nonce is the answer: the container is the agent's to write in.
        nonce = secrets.token_hex(8)
        code = _OPS_SOURCE + f"\nprint({nonce!r} + main({call!r}))\n"
        body = json.dumps({"generated_code": code, "language": "python", "timeout": 300}).encode()
        request = urllib.request.Request(url.rstrip("/") + "/execute", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=315) as answer:
                out = json.loads(answer.read())
        except Exception as exc:  # noqa: BLE001 -- unreachable and refusing mean the same thing here
            raise ValueError(f"the workspace is unreachable right now ({type(exc).__name__}); the next call "
                             "restarts the sandbox first") from None
        line = next((l for l in (out.get("stdout") or "").splitlines() if l.startswith(nonce)), None)
        if line is None:
            raise ValueError("the file operation died in the sandbox: " + (out.get("stderr") or "")[-300:].strip())
        got = json.loads(line[len(nonce):])
    if "error" in got:
        raise ValueError(got["error"])
    return got["result"]


async def _op(name: str, **args):
    # Off the event loop: a search may scan for a minute and a half, and other agents share this loop.
    return await asyncio.to_thread(_op_here, name, **args)


# What view_image showed, by key: the request that carries an image reads it here, never the path again.
_IMAGES: "OrderedDict[str, str]" = OrderedDict()


def image_url(key: str) -> str | None:
    return _IMAGES.get(key)


def _bare(step: str) -> str:
    """A step as text: agents send them bulleted, numbered, or already boxed."""
    return re.sub(r"^[-*\d.)\s]*(\[[ xX-]\])?\s*", "", step.strip())


_DONE = ("done", "completed", "complete", "finished", "checked", "status", "state")


def _step_text(raw: dict) -> str:
    """The step inside a wrapper: its first text, never its `done` flag."""
    return str(next((v for k, v in raw.items() if k.lower() not in _DONE and isinstance(v, str) and v.strip()),
                    next(iter(raw.values()), "")))


def _step_done(raw) -> bool:
    """Whether a wrapped step says it is already finished."""
    got = next((raw[k] for k in raw if k.lower() in _DONE), None) if isinstance(raw, dict) else None
    return str(got).strip().lower() in ("true", "1", "yes", "done", "completed", "finished")


def _steps_in(text) -> list[str]:
    """The steps an agent sent, whether as a list or a line per step.

    `steps` is `list[str] = []`, and the missing `| None` is the whole point. A tool call arrives
    as XML, where every parameter is text; the qwen3_xml parser turns the text back into a list by
    reading `properties["steps"]["type"]`, and an optional list is `anyOf: [array, null]`, which
    has no `type` at that level. The parser then falls back to string and hands the JSON array
    through as its own source text. That is how this went wrong the first time: splitlines() read
    the array as one step, the agent could not find its own plan in what came back, and one run
    rewrote it 245 times without ever writing the file it was asked for. Measured against the
    server: optional gives a string three times in three, plain `list[str]` a list three in three.
    `done` and `giving_up` are genuinely strings, and take the line-per-item path.
    """
    if isinstance(text, list):
        # Steps arrive wrapped ({"step": ...} / {"name": ..., "done": true}). Not the first VALUE:
        # one key ahead of the text and the step becomes "True".
        flat = [_step_text(x) if isinstance(x, dict) else str(x) for x in text]
        return [x.strip() for x in flat if x.strip()]
    return [line for line in (text or "").splitlines() if line.strip()]


# Counted where the cap middleware counts its own: a dropped match is a dropped match.
from nat.middleware.output_limit.output_limit_middleware import FIRED

class WorkspaceListConfig(FunctionBaseConfig, name="list_directory"):
    max_entries: int = Field(default=200, description="Cap on returned paths")


def _put(path: str, blob: bytes) -> tuple[bool, str]:
    """Place bytes at `path` in the container. Built here and carried over as base64, because the
    xlsx, docx and pptx writers need libraries the task's own image has no reason to hold."""
    import base64
    q = shlex.quote(path)
    encoded = base64.b64encode(blob).decode()
    return _sh(f"mkdir -p \"$(dirname {q})\" && printf %s {shlex.quote(encoded)} | base64 -d > {q} "
               f"&& echo __WROTE__")


def _from_find(out: str, where: str, contains: str, max_entries: int) -> str:
    """`find -printf '%P\\t%s'` output, formatted the way a local tree is."""
    pairs = []
    for line in out.splitlines():
        name, _, size = line.partition("\t")
        pairs.append((f"{where}/{name}".lstrip("./") if where not in (".", "") else name, size))
    return ops.formatted(_CONTAINER_ROOT, pairs, contains, max_entries)


@register_function(config_type=WorkspaceListConfig)
async def list_directory(config: WorkspaceListConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """List files in the workspace."""

    async def _run(subdir: str = "", contains: str = "") -> str:
        if _bridge():
            here = subdir.strip()
            parts = [x for x in Path(here).parts if x not in ("/", "", ".")]
            # `/app` IS the root: `find /app` looked for /app/app and returned only find's error,
            # 18 listings in one pass. Only a LEADING root is dropped.
            if here.startswith("/") and parts[:1] == [_CONTAINER_ROOT.strip("/")]:
                parts = parts[1:]
            where = "/".join(parts) or "."
            ran, out = _sh(f"find {shlex.quote(where)} -type f -printf '%P\\t%s\\n' 2>/dev/null "
                           f"|| find {shlex.quote(where)} -type f")
            if ran:
                # find's stderr arrives mixed in and reads as a row: a directory of one odd file.
                rows = "\n".join(l for l in out.splitlines() if not l.startswith("find:"))
                if not rows.strip():
                    return f"no such directory: {subdir or _CONTAINER_ROOT}"
                return _from_find(rows, where, contains, config.max_entries)
        return await _op("list", subdir=subdir, contains=contains, max_entries=config.max_entries, staged=_staged())

    yield FunctionInfo.from_fn(
        _run,
        description=("List workspace files. Args: `subdir` relative to the root, and `contains` "
                     "to keep only paths holding that substring -- use it, the tree is large."))


class WorkspaceReadConfig(FunctionBaseConfig, name="read_file"):
    max_chars: int = Field(default=20000, description="Cap on returned characters")


@register_function(config_type=WorkspaceReadConfig)
async def read_file(config: WorkspaceReadConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Read one workspace file as text."""

    async def _run(path: str, offset: int = 0) -> str:
        offset = max(0, int(offset))       # a negative one means the start, not a place past the end
        if _bridge():
            q = shlex.quote(path)
            ran, out = _sh(f"if [ -d {q} ]; then echo __DIR__; elif [ -f {q} ]; then "
                           f"wc -c < {q}; echo __CUT__; tail -c +{offset + 1} {q} | head -c "
                           f"{config.max_chars}; else echo __MISSING__; fi")
            if ran:
                if out.startswith("__DIR__"):
                    return f"{path} is a directory -- list it with list_directory, or name a file in it."
                if out.startswith("__MISSING__"):
                    return f"no such file: {path}"
                total, _, chunk = out.partition("__CUT__\n")
                try:
                    size = int(total.strip())
                except ValueError:
                    return out
                if not chunk:
                    return ops.nothing_read(path, size, offset)
                rest = size - offset - len(chunk)
                if rest <= 0:
                    return chunk
                return (f"{chunk}\n... {rest} more characters, call again with "
                        f"offset={offset + len(chunk)}")
        return await _op("read", path=path, offset=offset, max_chars=config.max_chars)

    yield FunctionInfo.from_fn(_run, description=(
        "Read a workspace file as text. Args: `path` relative to the root, and `offset` to continue "
        "a long file from where the last call stopped. `offset` counts characters, not lines: to "
        "read around a line number grep gave you, use bash `sed -n '990,1050p' path` instead." + _where()))


class WorkspaceViewImageConfig(FunctionBaseConfig, name="view_image"):
    pass


@register_function(config_type=WorkspaceViewImageConfig)
async def view_image(config: WorkspaceViewImageConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Show one workspace image to the model."""

    async def _run(path: str) -> str:
        if _bridge():
            return "view_image reads files on this host; this task's files live in its own container."
        got = await _op("image", path=path)
        if "data" not in got:
            return got["text"]
        key = hashlib.sha1(got["data"].encode()).hexdigest()[:16]
        _IMAGES[key] = f"data:{got['mime']};base64,{got['data']}"
        _IMAGES.move_to_end(key)
        while len(_IMAGES) > 64:
            _IMAGES.popitem(last=False)
        # The marker becomes the image itself on its way to the model (markagentx.adapters.base.show_images).
        return f"{got['text']}: [[markagentx-image:{key}]]"

    yield FunctionInfo.from_fn(_run, description=(
        "Look at a workspace image: a chart you plotted, a figure, a screenshot. Args: `path` relative "
        "to the root. For a PDF page, render it to PNG first (pymupdf in run_code). Only the newest few "
        "images stay visible, so view one again when you need it."))


class WorkspaceWriteConfig(FunctionBaseConfig, name="write_file"):
    pass


@register_function(config_type=WorkspaceWriteConfig)
async def write_file(config: WorkspaceWriteConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Write a deliverable into the workspace."""

    async def _run(path: str, content: str) -> str:
        if _bridge():
            with tempfile.TemporaryDirectory(prefix="ws-out-") as staging:
                local = Path(staging) / Path(path).name
                try:
                    detail = ops.built(local, content)
                except Exception as exc:  # noqa: BLE001 -- the builder's own words beat a traceback
                    return f"could not build {path}: {exc}"
                ran, out = _put(path, local.read_bytes())
            if ran:
                return (f"wrote {path} ({detail})" if "__WROTE__" in out
                        else out.strip() or f"could not write {path}")
        return await _op("write", path=path, content=content)

    yield FunctionInfo.from_fn(_run, description=(
        "Write a deliverable into the workspace, replacing whatever was there. To change part of a "
        "file that already exists, use edit_file instead -- everything you do not resend here "
        "is lost. Args: `path` relative to the root, and `content`. "
        "The extension picks the format that gets built, so send `content` in the shape it expects: "
        "`.xlsx` wants CSV text -- header row first, one line per row, fields holding a comma quoted; "
        "`.docx` wants markdown-ish prose -- a blank line ends a paragraph, a leading `#`, `##` or `###` "
        "makes a heading of that level, and a run of `| a | b |` rows becomes a real table; "
        "`.pptx` wants one blank-line-separated block per slide, first line the title and the rest bullets. "
        "Every other extension is stored as the exact text you send." + _where()))


class WorkspaceSearchConfig(FunctionBaseConfig, name="grep_files"):
    max_hits: int = Field(default=60, description="Cap on returned matching lines")
    # max_hits bounds the answer, not the work: a query that matches nothing still opened every
    # PDF in the tree. One workspace holds 1950 of them and the scan measured 81 minutes.
    max_seconds: float = Field(default=90.0, description="Wall clock one search may spend scanning")
    max_documents: int = Field(default=250, description="Documents whose text may be extracted per search")


@register_function(config_type=WorkspaceSearchConfig)
async def grep_files(config: WorkspaceSearchConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Find which workspace files hold a string, without reading each one whole."""

    async def _run(query: str, subdir: str = "", path_contains: str = "") -> str:
        needle = (query or "").strip().lower()
        if not needle:
            return "give a non-empty query"
        if _bridge():
            where = shlex.quote(subdir.strip("/ ") or ".")
            keep = f" | grep -F -- {shlex.quote(path_contains)}" if path_contains else ""
            ran, out = _sh(f"grep -rniI -F -- {shlex.quote(query)} {where} 2>/dev/null{keep} "
                           f"| head -n {config.max_hits + 1}")
            if ran:
                rows = [r for r in out.splitlines() if r.strip()]
                if not rows:
                    return f"(no line contains {query!r}){ops.literal(query)}"
                if len(rows) > config.max_hits:
                    return ("\n".join(rows[:config.max_hits])
                            + f"\n... more than {config.max_hits} matches; narrow the query.")
                return "\n".join(rows)
        got = await _op("grep", query=query, subdir=subdir, path_contains=path_contains, max_hits=config.max_hits,
                        max_seconds=config.max_seconds, max_documents=config.max_documents, staged=_staged())
        FIRED[config.type] += bool(got.get("cut"))
        return got["text"]

    yield FunctionInfo.from_fn(_run, description=(
        "Search workspace file contents and return matching lines with their paths. `query` is plain "
        "text matched case-insensitively, not a regular expression -- for a regex use bash with grep. "
        "Args: `query`, optional `subdir`, and `path_contains` to restrict which files are scanned." + _where()))


class WorkspaceEditConfig(FunctionBaseConfig, name="edit_file"):
    pass


@register_function(config_type=WorkspaceEditConfig)
async def edit_file(config: WorkspaceEditConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Replace one exact passage in a file, rather than rewriting the file around it."""

    async def _run(path: str, old: str, new: str) -> str:
        # "edited" for a no-op reads as success: one model resent the same one 415 times, a third
        # of its budget, waiting for the file to change.
        if old == new:
            return f"`old` and `new` are identical, so {path} is unchanged; put the replacement in `new`."
        if _bridge():
            q = shlex.quote(path)
            ran, body = _sh(f"if [ -f {q} ]; then cat {q}; else echo __MISSING__; fi")
            if ran:
                if body.startswith("__MISSING__"):
                    return f"{path} is not a file in the workspace; list_directory shows what is."
                seen = body.count(old)
                if seen == 0:
                    return f"that exact text is not in {path}; read it again and copy the passage.{ops.near(body, old)}"
                if seen > 1:
                    return f"that text appears {seen} times in {path}; include more of it."
                done, out = _put(path, body.replace(old, new, 1).encode())
                if done and "__WROTE__" in out:
                    return f"edited {path} ({len(old)} chars -> {len(new)})"
                return out.strip() or f"could not write {path}"
        return await _op("edit", path=path, old=old, new=new)

    yield FunctionInfo.from_fn(_run, description=(
        "Replace one exact passage inside a file, leaving the rest untouched. Use this to change an "
        "existing file; write_file replaces the whole file and loses anything you did not "
        "resend.\n\nArgs:\n    path (str): the file, relative to the workspace root.\n"
        "    old (str): the exact text to replace, unique in the file.\n"
        "    new (str): what to put there." + _where()))


class WorkspaceShellConfig(FunctionBaseConfig, name="workspace_shell"):
    uri: str = Field(default="http://127.0.0.1:6000", description="Sandbox base URL")
    timeout: float = Field(default=60.0, description="Seconds one command may run")
    max_output_characters: int = Field(default=16000, description="Truncate combined output here")


@register_function(config_type=WorkspaceShellConfig)
async def workspace_shell(config: WorkspaceShellConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """A shell in the sandbox, rooted at the workspace."""
    import httpx

    async def _run(command: str) -> str:
        # A bridged sandbox is the task's own container: its session already starts where the work
        # is and routes a command line to bash itself, so the host path baked in below would name a
        # directory that does not exist there. This is what run_code's own preamble check does.
        if os.environ.get("NAT_BRIDGE_READY"):
            wrapper = command
        else:
            # Run through the sandbox, never on this host: the tool exists because agents were
            # wrapping subprocess in run_code to get here anyway, and that path had no root and no cap.
            # The cut is caught here, not left to surface as a TimeoutExpired traceback: that reads
            # as a crash, and it throws away what the command had already printed.
            wrapper = (
                "import subprocess, os, tempfile\n"
                f"cwd = {ops.SANDBOX_ROOT!r}\n"
                "os.makedirs(cwd, exist_ok=True)\n"
                "out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()\n"
                "try:\n"
                f"    r = subprocess.run({command!r}, shell=True, cwd=cwd, stdout=out, stderr=err,\n"
                f"                       stdin=subprocess.DEVNULL, timeout={config.timeout})\n"
                "except subprocess.TimeoutExpired:\n"
                "    r = None\n"
                "for f in (out, err):\n"
                "    f.seek(0)\n"
                "    print(f.read().decode(errors='replace'), end='')\n"
                "if r is None:\n"
                f"    print('\\n[stopped at the {config.timeout:g}s limit -- anything after this "
                "was not run]', end='')\n"
                "elif r.returncode:\n"
                "    print(f'\\n[exit {r.returncode}]', end='')\n")
        try:
            async with httpx.AsyncClient(timeout=config.timeout + 15) as client:
                answer = await client.post(config.uri.rstrip("/") + "/execute",
                                           json={"generated_code": wrapper,
                                                 "timeout": config.timeout, "language": "python"})
                answer.raise_for_status()
                body = answer.json()
        except (httpx.RemoteProtocolError, httpx.ReadError):
            # Taken, then dropped mid-command: the command took the server down, and uwsgi respawns it.
            return ("the sandbox's server died while this command ran: a command that kills every process "
                    "(kill -1, pkill) or runs it out of memory takes it down too. It is back for the next "
                    "command; check what this one did before running it again.")
        except Exception as exc:  # noqa: BLE001 -- any failure to run means the same thing
            # Not "right now": that reads as a wait, and an agent told to wait re-sends the same
            # command until its budget is gone -- thirty-six times in one run measured here. What
            # it cannot work out for itself is that nothing it types will change this answer.
            return (f"the shell is not running ({type(exc).__name__}). This is the sandbox, not "
                    f"your command: the same command will get this same answer, so use the other "
                    f"tools and say so in your answer if the task needed a shell.")
        out = (body.get("stdout") or "") + (body.get("stderr") or "")
        if body.get("process_status") not in (None, "completed", "success"):
            out = f"[{body['process_status']}]\n{out}"
        # Silence reads as a tool that did nothing and gets re-sent: 95 times for a `#` comment,
        # 510 for a heredoc that wrote its file every time. Say what it means.
        out = out.strip() or ("exited 0 and printed nothing -- which is what a write, a move, or a "
                              "build with nothing to report does when it works. Re-running returns "
                              "this same line; if you meant to reason, use think.")
        cap = config.max_output_characters
        if len(out) <= cap:
            return out
        FIRED[config.type] += 1
        # The tail too: a build puts its first errors at the top and its verdict at the bottom.
        return out[:cap * 3 // 4] + (f"\n...[cut {len(out) - cap} characters; the end of the output "
                                     "follows. Narrow with grep -- piping to head would drop the "
                                     "exit status with it.]\n") + out[-(cap // 4):]

    # The image's own toolchain, named by its label: an agent never guesses $SKY130_LIB.
    tools = "" if _bridge() else os.environ.get("MARKAGENTX_SANDBOX_TOOLS", "")
    # Named, not just described: agents guessed /app, the image's own workdir, and lost 3 steps.
    yield FunctionInfo.from_fn(_run, description=(
        "Run one shell command in the workspace and return its output. Each call is a new shell, "
        "so a `cd` or an exported variable is gone by the next one -- chain them in one command "
        "line instead. The working directory is "
        # The bridged shell runs in the container: the host path does not exist there, and every
        # other tool says /app through _where().
        f"the workspace root, {_CONTAINER_ROOT if _bridge() else ops.SANDBOX_ROOT}, so paths are "
        "relative to it, and only files there are kept after the run. For anything on "
        "the web use search_web and fetch_url rather than curl or urllib here: those keep what "
        "they read where the rest of the run can see it. Long output is cut from the middle, never "
        "the end, so `| head` buys nothing and costs the exit status: a pipeline reports only its "
        "LAST command's, and `go build | head` reads as success however the build went."
        + (f" Installed here: {tools}" if tools else "") + "\n\n"
        "Args:\n    command (str): the command line, e.g. `ls -la` or `python -m pytest -q`."))


class ThinkConfig(FunctionBaseConfig, name="think"):
    pass


@register_function(config_type=ThinkConfig)
async def think(config: ThinkConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Somewhere to reason mid-task. Measured by Anthropic at +54% relative on tau-bench airline."""

    async def _run(thought: str) -> str:
        return "Noted. Continue."

    yield FunctionInfo.from_fn(_run, description=(
        "Think a step through without acting. Nothing happens and nothing is fetched or changed; "
        "use it to work out what a tool just told you, check a rule before you act on it, or plan "
        "the next few steps.\n\nArgs:\n    thought (str): the reasoning to work through."))


# Keyed by workspace and shared by every agent on one task. A process running tasks in turn reuses
# one workspace, so the key alone does not keep them apart -- reset_task_plan drops the plan at each
# task boundary rather than carrying one task's steps into the next.
_PLANS: dict[str, list[str]] = {}


def reset_task_plan(workspace: str = "") -> None:
    """Drop the task_list plan for a workspace, at a task boundary."""
    _PLANS.pop(str(Path(workspace).resolve()) if workspace else str(_root()), None)


def open_steps(workspace: str = "") -> list[str]:
    """The steps still open, for a caller that must show them without spending a turn reading the list."""
    key = str(Path(workspace).resolve()) if workspace else str(_root())
    return [l[6:].strip() for l in _PLANS.get(key, []) if l.startswith("- [ ]")]


class TaskListConfig(FunctionBaseConfig, name="task_list"):
    pass


@register_function(config_type=TaskListConfig)
async def task_list(config: TaskListConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """The plan as run state, shared by every agent on this task.

    State, not a workspace file: one there was searched, graded, and half-visible over the bridge.

    Three states, not two: closing a step by SOLVING it is the only way an agent had, so a step it
    could not solve stayed open and it kept trying. `giving_up` closes one and says why, and the
    reason is what stops the next agent -- or the next turn of this one -- repeating the attempt.
    """

    # `list`, not `list[str]`: wrapped steps had pydantic rejecting whole plans (Nemotron 10,
    # Qwen 5) before _steps_in saw them.
    async def _run(steps: list = [], done: str = "", giving_up: str = "",
                   because: str = "") -> str:
        if giving_up and not because.strip():
            return "Say why in `because`: the reason is what keeps the next attempt from repeating it."
        key = str(_root())
        lines = list(_PLANS.get(key, []))
        before = list(lines)
        asked = _steps_in(steps)
        if asked:
            # A closed step stays closed: each specialist rewrites this list, and a plain replace
            # reopened what the one before it had already finished.
            shut = {l[6:].split("  (")[0].strip().lower(): l for l in lines if not l.startswith("- [ ]")}
            # Some models close a step only as {"step": ..., "done": true}; dropping the flag left
            # the plan all-open and rewritten rather than advanced -- 8 calls, 19 plans re-sent.
            ticked = {_bare(t).strip().lower() for t, raw in zip(asked, steps if isinstance(steps, list) else [])
                      if _step_done(raw)}
            # Steps often arrive already bulleted, and "- [ ] - step" reads as a broken list.
            lines = [shut.get(_bare(s).strip().lower(),
                              f"- [{'x' if _bare(s).strip().lower() in ticked else ' '}] {_bare(s)}")
                     for s in asked]
        def hits(mark, line):
            """Either way round: a step returns as a prefix of itself as often as with a note
            appended, and neither is a different step."""
            sent, held = mark.strip().lower(), line[6:].split("  (")[0].strip().lower()
            return sent in held or (len(held) >= 12 and held in sent)

        missed, already = [], []
        for mark, box, why in ([(x, "- [x]", "") for x in _steps_in(done)]
                               + [(x, "- [-]", because) for x in _steps_in(giving_up)]):
            for i, line in enumerate(lines):
                if line.startswith("- [ ]") and hits(mark, line):
                    lines[i] = line.replace("- [ ]", box, 1) + (f"  ({why.strip()})" if why.strip() else "")
                    break
            else:
                # Which of the two: told "closed, or never listed" an agent cannot tell it
                # succeeded -- 15 in 100 sent the same close again.
                (already if any(not l.startswith("- [ ]") and hits(mark, l) for l in lines)
                 else missed).append(mark.strip())
        if lines:
            _PLANS[key] = lines
        if not lines:
            return "The list is empty; send `steps` to start one."
        left = sum(l.startswith("- [ ]") for l in lines)
        gave = sum(l.startswith("- [-]") for l in lines)
        tail = f"({left} of {len(lines)} still open" + (f", {gave} given up on)" if gave else ")")
        # Saying nothing changed is the whole point: a silent no-op reads as success and gets
        # retried -- one run sent the same plan 19 times and read back the same words every time.
        if lines == before and (asked or done.strip() or giving_up.strip()):
            tail += "\nNothing changed: this is the list as it already stood."
        if already:
            tail += "\n" + ", ".join(repr(m) for m in already) + " was closed already; nothing to do."
        if missed:
            tail += "\nNo step matches " + ", ".join(repr(m) for m in missed) + " -- it is not on this list."
        return "\n".join(lines) + "\n\n" + tail

    yield FunctionInfo.from_fn(_run, description=(
        "Keep the plan for this task as a checklist. Call it once with `steps` to write the plan, "
        "then with `done` after finishing one, or with `giving_up` and `because` for one you tried "
        "and could not settle -- writing that down is what keeps you, and anyone after you, from "
        "trying it again. With no arguments it shows where you are. Every agent here shares it."
        "\n\nArgs:\n    steps (list[str]): the plan, one step per item -- replaces any existing "
        "list.\n"
        "    done (str): text identifying a step to tick off, one per line.\n"
        "    giving_up (str): text identifying a step to close unsolved.\n"
        "    because (str): why that step could not be settled."))


class FinishConfig(FunctionBaseConfig, name="finish"):
    pass


@register_function(config_type=FinishConfig)
async def finish(config: FinishConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Stopping, as an action. Without one an agent keeps picking the best tool it has."""
    from nat.middleware.agent_finish import AgentFinished

    async def _run(answer: str) -> str:
        raise AgentFinished(answer)

    yield FunctionInfo.from_fn(_run, description=(
        "Finish, giving your answer. Call this when the task is done -- checking work you have "
        "already checked cannot change it.\n\n"
        "Args:\n    answer (str): the complete answer, in the form the task asked for."))
