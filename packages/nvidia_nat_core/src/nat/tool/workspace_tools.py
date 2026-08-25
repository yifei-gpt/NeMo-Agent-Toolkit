# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace file tools for benchmarks that hand an agent a directory and a brief."""

import logging
import os
import re
import tempfile
import subprocess
import time
import sys
from collections.abc import AsyncGenerator
from collections import OrderedDict
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

MAX_READ_CHARS = 20000
CENSUS_ROWS = 40


def _root() -> Path:
    # No workspace was chosen, so scratch space -- the process cwd would hand over the whole checkout.
    if not os.environ.get("NAT_WORKSPACE_DIR"):
        os.environ["NAT_WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="nat_workspace_")
    return Path(os.environ["NAT_WORKSPACE_DIR"]).resolve()


def _bare(step: str) -> str:
    """A step as text: agents send them bulleted, numbered, or already boxed."""
    return re.sub(r"^[-*\d.)\s]*(\[[ xX-]\])?\s*", "", step.strip())


def _resolve(rel: str) -> Path:
    # Briefs echo the root path whole, partial or not at all; each form folds under the root.
    root = _root()
    parts = Path(rel.strip()).parts
    parts = parts[parts.index(root.name) + 1:] if root.name in parts else tuple(
        x for x in parts if x not in ("/", "", "."))
    # Lexical containment: resolve() would follow the links the workspace itself placed.
    p = Path(os.path.normpath(root.joinpath(*parts)))
    if p != root and root not in p.parents:
        raise ValueError(f"path escapes workspace: {rel}")
    return p


logger = logging.getLogger(__name__)

# Out of process: one malformed PDF can hang pdfminer or crash the interpreter, uncatchably.
_PDF_CHILD = ("import sys, pdfplumber\n"
              "with pdfplumber.open(sys.argv[1]) as pdf:\n"
              "    sys.stdout.write('\\n'.join((p.extract_text() or '') for p in pdf.pages[:40]))\n")


# A PDF that could not be parsed cannot be parsed the next time either, and each attempt costs
# the full timeout. Two workspace runs spent their wall clock re-parsing the same broken file.
_PDF_REFUSED: dict[tuple[str, int, float], None] = {}
# Successful extractions too: a second search over the same tree re-parsed 1950 PDFs from scratch.
_EXTRACTED: "OrderedDict[tuple[str, int, float], str | None]" = OrderedDict()
_EXTRACT_CACHE_MAX = 4000


def _pdf_text(p: Path) -> str | None:
    try:
        stamp = (str(p), p.stat().st_size, p.stat().st_mtime)
    except OSError:
        stamp = (str(p), -1, -1.0)
    if stamp in _PDF_REFUSED:
        return None
    try:
        done = subprocess.run([sys.executable, "-c", _PDF_CHILD, str(p)],
                              capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("PDF %s took over 60s to parse and was skipped", p.name)
        _PDF_REFUSED[stamp] = None
        return None
    if done.returncode != 0:
        logger.warning("PDF %s could not be parsed (exit %s)", p.name, done.returncode)
        _PDF_REFUSED[stamp] = None
        return None
    return done.stdout.decode("utf-8", "ignore").strip() or None


def _extract(p: Path) -> str | None:
    """Text from a workspace file, or None: Office formats are zipped XML read via the stdlib;
    reading them as UTF-8 yields mojibake that floods the context window."""
    try:
        stamp = (str(p), p.stat().st_size, p.stat().st_mtime)
    except OSError:
        stamp = None
    if stamp is not None and stamp in _EXTRACTED:
        _EXTRACTED.move_to_end(stamp)
        return _EXTRACTED[stamp]
    text = _extract_uncached(p)
    if stamp is not None:
        _EXTRACTED[stamp] = text
        while len(_EXTRACTED) > _EXTRACT_CACHE_MAX:
            _EXTRACTED.popitem(last=False)
    return text


def _extract_uncached(p: Path) -> str | None:
    import re
    import zipfile

    suffix = p.suffix.lower()
    if suffix in {".docx", ".xlsx", ".pptx"}:
        try:
            with zipfile.ZipFile(p) as z:
                parts = [n for n in z.namelist() if n.endswith(".xml") and "rels" not in n]
                chunks = []
                for name in parts[:40]:
                    raw = z.read(name).decode("utf-8", errors="ignore")
                    chunks.append(re.sub(r"<[^>]+>", " ", raw))
                return re.sub(r"\s+", " ", " ".join(chunks)).strip() or None
        except Exception:
            return None
    if suffix == ".pdf":
        text = _pdf_text(p)
        # An unparseable .pdf that is really plain text is an agent-written deliverable: let it read back.
        if text is not None:
            return text
    # ASCII-leading binaries still read as mojibake, so the extension decides, not a byte probe.
    if suffix in {".doc", ".xls", ".ppt", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".bin", ".so"}:
        return None
    try:
        data = p.read_bytes()
    except Exception:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


class WorkspaceListConfig(FunctionBaseConfig, name="workspace_list"):
    max_entries: int = Field(default=200, description="Cap on returned paths")


def _listing(subdir: str = "", contains: str = "", max_entries: int = 200) -> str:
    """Module level so a test can reach it: the census branch shipped broken with nothing covering it."""
    base = _resolve(subdir) if subdir else _root()
    if not base.is_dir():
        return f"not a directory: {subdir}"
    needle = (contains or "").strip().lower()
    hits: list[str] = []
    census: dict[str, int] = {}
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(_root()))
        if needle and needle not in rel.lower():
            continue
        census[str(Path(rel).parent)] = census.get(str(Path(rel).parent), 0) + 1
        hits.append(f"{rel}  ({p.stat().st_size} bytes)")
    # The absolute root, so code run in a sandbox can open these files by path.
    head = f"workspace root: {_root()}"
    if not hits:
        return head + ("\n(no file matches %r)" % contains if needle else "\n(empty)")
    if len(hits) <= max_entries:
        return "\n".join([head, *hits])
    # Past the cap an arbitrary slice tells nothing: the listing becomes a folder census.
    rows = sorted(census.items(), key=lambda kv: -kv[1])[:CENSUS_ROWS]
    return (f"{head}\n{len(hits)} files match -- too many to list. Folders, largest first; "
            f"open one with `subdir`, or filter with `contains`:\n" +
            "\n".join(f"{d}/  ({n} files)" for d, n in rows))


@register_function(config_type=WorkspaceListConfig)
async def workspace_list(config: WorkspaceListConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """List files in the workspace."""

    async def _run(subdir: str = "", contains: str = "") -> str:
        return _listing(subdir, contains, config.max_entries)

    yield FunctionInfo.from_fn(
        _run,
        description=("List workspace files. Args: `subdir` relative to the root, and `contains` "
                     "to keep only paths holding that substring -- use it, the tree is large."))


class WorkspaceReadConfig(FunctionBaseConfig, name="workspace_read"):
    max_chars: int = Field(default=MAX_READ_CHARS, description="Cap on returned characters")


@register_function(config_type=WorkspaceReadConfig)
async def workspace_read(config: WorkspaceReadConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Read one workspace file as text."""

    async def _run(path: str, offset: int = 0) -> str:
        p = _resolve(path)
        if p.is_dir():
            return f"{path} is a directory -- list it with workspace_list, or name a file in it."
        if not p.is_file():
            return f"no such file: {path}"
        text = _extract(p)
        if text is None:
            return f"{path} is a binary file ({p.stat().st_size} bytes) with no text extractor."
        chunk = text[offset:offset + config.max_chars]
        rest = len(text) - offset - len(chunk)
        if rest <= 0:
            return chunk
        # Say where to continue, or the model re-reads the same head and looks like a repeat loop.
        return f"{chunk}\n... {rest} more characters, call again with offset={offset + len(chunk)}"

    yield FunctionInfo.from_fn(_run, description=(
        "Read a workspace file as text. Args: `path` relative to the root, and `offset` to continue "
        "a long file from where the last call stopped."))


def _add_table(doc, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    # The `|---|:--:|` rule under a markdown header carries no data, so it must not become a row.
    cells = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    if not cells:
        return
    table = doc.add_table(rows=len(cells), cols=max(len(r) for r in cells))
    table.style = "Table Grid"
    for i, row in enumerate(cells):
        for j, value in enumerate(row):
            table.cell(i, j).text = value


def _write_docx(p: Path, text: str) -> str:
    """Build a real Word document out of markdown-ish text."""
    import re

    from docx import Document

    doc = Document()
    para: list[str] = []
    rows: list[str] = []

    def flush_para() -> None:
        if para:
            doc.add_paragraph(" ".join(para))
            para.clear()

    def flush_rows() -> None:
        if rows:
            _add_table(doc, rows)
            rows.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("|") and line.endswith("|"):
            flush_para()
            rows.append(line)
            continue
        flush_rows()
        head = re.match(r"(#{1,3})\s+(.+)", line)
        bullet = re.match(r"([-*+\u2022]|\d+[.)])\s+", line)
        if head or bullet or not line:
            flush_para()
        if head:
            doc.add_heading(head[2].strip(" #"), level=len(head[1]))
        elif bullet:
            doc.add_paragraph(line)
        elif line:
            para.append(line)
    flush_rows()
    flush_para()
    doc.save(p)
    return f"{len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables"


def _write_xlsx(p: Path, text: str) -> str:
    """Build a real Excel workbook out of CSV text."""
    import csv
    import io

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    # StringIO rather than splitlines(): only a real stream keeps a newline inside a quoted field.
    for row in csv.reader(io.StringIO(text)):
        ws.append(row)
    wb.save(p)
    return f"{ws.max_row} rows x {ws.max_column} columns"


def _write_pptx(p: Path, text: str) -> str:
    """Build a real PowerPoint deck, one slide per blank-line-separated block."""
    import re

    from pptx import Presentation

    prs = Presentation()
    layout = prs.slide_layouts[1]
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = [ln.strip().lstrip("#-*\u2022 ").strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = lines[0]
        slide.placeholders[1].text = "\n".join(lines[1:])
    prs.save(p)
    return f"{len(prs.slides)} slides"


class WorkspaceWriteConfig(FunctionBaseConfig, name="workspace_write"):
    pass


@register_function(config_type=WorkspaceWriteConfig)
async def workspace_write(config: WorkspaceWriteConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Write a deliverable into the workspace."""

    async def _run(path: str, content: str) -> str:
        p = _resolve(path)
        # An empty or directory `path` writes onto the directory itself, and that OS error carries
        # an absolute host path the caller can do nothing with.
        if p == _root() or p.is_dir():
            return f"`path` must name a file inside the workspace; {path!r} names a directory."
        p.parent.mkdir(parents=True, exist_ok=True)
        # Writing to a symlink never means editing its target; linked worlds would reject or corrupt.
        if p.is_symlink():
            p.unlink()
        build = {".docx": _write_docx, ".xlsx": _write_xlsx, ".pptx": _write_pptx}.get(p.suffix.lower())
        if build is None:
            # Said at the moment the lines go: a description telling the agent to prefer
            # workspace_edit moved 1 framework in 4, and the loss is silent otherwise.
            before = p.read_text(encoding="utf-8", errors="ignore").count("\n") + 1 if p.is_file() else 0
            try:
                p.write_text(content, encoding="utf-8")
            except OSError as exc:
                return f"could not write {path}: {exc.strerror or exc}."
            after = content.count("\n") + 1
            note = (f" It held {before} lines and now holds {after}; the other {before - after} are "
                    "gone. If that was not intended, workspace_edit changes one passage and leaves "
                    "the rest." if before > after + max(5, before // 10) else "")
            return f"wrote {p.relative_to(_root())} ({len(content)} chars){note}"
        try:
            detail = build(p, content)
        except Exception as exc:
            # Text saved under an Office name looks delivered and grades zero; the failure must be heard.
            return f"failed to build {p.name}: {exc}. Resend `content` in the shape that extension expects."
        return f"wrote {p.relative_to(_root())} ({detail})"

    yield FunctionInfo.from_fn(_run, description=(
        "Write a deliverable into the workspace, replacing whatever was there. To change part of a "
        "file that already exists, use workspace_edit instead -- everything you do not resend here "
        "is lost. Args: `path` relative to the root, and `content`. "
        "The extension picks the format that gets built, so send `content` in the shape it expects: "
        "`.xlsx` wants CSV text -- header row first, one line per row, fields holding a comma quoted; "
        "`.docx` wants markdown-ish prose -- a blank line ends a paragraph, a leading `#`, `##` or `###` "
        "makes a heading of that level, and a run of `| a | b |` rows becomes a real table; "
        "`.pptx` wants one blank-line-separated block per slide, first line the title and the rest bullets. "
        "Every other extension is stored as the exact text you send."))


class WorkspaceSearchConfig(FunctionBaseConfig, name="workspace_search"):
    max_hits: int = Field(default=60, description="Cap on returned matching lines")
    # max_hits bounds the answer, not the work: a query that matches nothing still opened every
    # PDF in the tree. One workspace holds 1950 of them and the scan measured 81 minutes.
    max_seconds: float = Field(default=90.0, description="Wall clock one search may spend scanning")
    max_documents: int = Field(default=250, description="Documents whose text may be extracted per search")


@register_function(config_type=WorkspaceSearchConfig)
async def workspace_search(config: WorkspaceSearchConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Find which workspace files hold a string, without reading each one whole."""

    async def _run(query: str, subdir: str = "", path_contains: str = "") -> str:
        needle = (query or "").strip().lower()
        if not needle:
            return "give a non-empty query"
        base = _resolve(subdir) if subdir else _root()
        if not base.is_dir():
            return f"not a directory: {subdir}"
        hits: list[str] = []
        started, opened, seen, skipped = time.time(), 0, 0, 0
        for p in sorted(base.rglob("*")):
            if not p.is_file() or (path_contains and path_contains.lower() not in str(p).lower()):
                continue
            seen += 1
            costly = p.suffix.lower() in {".pdf", ".docx", ".xlsx", ".pptx"}
            if costly and (opened >= config.max_documents
                           or time.time() - started > config.max_seconds):
                skipped += 1
                continue
            opened += costly
            text = _extract(p)
            if text is None:
                continue
            rel = str(p.relative_to(_root()))
            for i, line in enumerate(text.splitlines(), 1):
                if needle in line.lower():
                    hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(hits) > config.max_hits:
                        # Past the cap the agent needs a narrower query, not an arbitrary prefix.
                        return ("\n".join(hits[:config.max_hits]) +
                                f"\n... more than {config.max_hits} matches; narrow the query.")
        # Said, not hidden: a truncated scan that reads as "no matches" sends the agent away
        # from the file it was looking for.
        note = (f"\n[scanned {seen - skipped} of {seen} files in {time.time() - started:.0f}s; "
                f"{skipped} documents were left unopened -- narrow with subdir= or path_contains=]"
                if skipped else "")
        return ("\n".join(hits) + note) if hits else (f"(no line contains {query!r})" + note)

    yield FunctionInfo.from_fn(_run, description=(
        "Search workspace file contents and return matching lines with their paths. `query` is plain "
        "text matched case-insensitively, not a regular expression -- for a regex use bash with grep. "
        "Args: `query`, optional `subdir`, and `path_contains` to restrict which files are scanned."))


class WorkspaceEditConfig(FunctionBaseConfig, name="workspace_edit"):
    pass


@register_function(config_type=WorkspaceEditConfig)
async def workspace_edit(config: WorkspaceEditConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Replace one exact passage in a file, rather than rewriting the file around it."""

    async def _run(path: str, old: str, new: str) -> str:
        p = _resolve(path)
        if not p.is_file():
            return f"{path} is not a file in the workspace; workspace_list shows what is."
        try:
            body = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return f"could not read {path}: {getattr(exc, 'strerror', None) or exc}."
        hits = body.count(old)
        if hits == 0:
            return (f"that passage does not appear in {path}; read it first and copy the text "
                    "exactly, whitespace included.")
        if hits > 1:
            # Editing the first of several is how a file quietly gets the wrong one changed.
            return f"that passage appears {hits} times in {path}; extend `old` until it is unique."
        p.write_text(body.replace(old, new), encoding="utf-8")
        return f"edited {p.relative_to(_root())} ({len(old)} chars -> {len(new)})"

    yield FunctionInfo.from_fn(_run, description=(
        "Replace one exact passage inside a file, leaving the rest untouched. Use this to change an "
        "existing file; workspace_write replaces the whole file and loses anything you did not "
        "resend.\n\nArgs:\n    path (str): the file, relative to the workspace root.\n"
        "    old (str): the exact text to replace, unique in the file.\n"
        "    new (str): what to put there."))


class WorkspaceShellConfig(FunctionBaseConfig, name="workspace_shell"):
    uri: str = Field(default="http://127.0.0.1:6000", description="Sandbox base URL")
    timeout: float = Field(default=60.0, description="Seconds one command may run")
    max_output_characters: int = Field(default=4000, description="Truncate combined output here")


@register_function(config_type=WorkspaceShellConfig)
async def workspace_shell(config: WorkspaceShellConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """A shell in the sandbox, rooted at the workspace."""
    import json as _json

    import httpx

    async def _run(command: str) -> str:
        # Run through the sandbox, never on this host: the tool exists because agents were wrapping
        # subprocess in run_code to get here anyway, and that path had no root and no cap.
        wrapper = (
            "import subprocess, os\n"
            f"cwd = {_root().as_posix()!r}\n"
            "os.makedirs(cwd, exist_ok=True)\n"
            f"r = subprocess.run({command!r}, shell=True, cwd=cwd, capture_output=True,\n"
            f"                   text=True, timeout={config.timeout})\n"
            "print(r.stdout, end='')\n"
            "print(r.stderr, end='')\n"
            "print(f'\\n[exit {r.returncode}]' if r.returncode else '', end='')\n")
        try:
            async with httpx.AsyncClient(timeout=config.timeout + 15) as client:
                answer = await client.post(config.uri.rstrip("/") + "/execute",
                                           json={"generated_code": wrapper,
                                                 "timeout": config.timeout, "language": "python"})
                answer.raise_for_status()
                body = answer.json()
        except Exception as exc:  # noqa: BLE001 -- any failure to run means the same thing
            return f"the shell is unavailable right now ({type(exc).__name__})."
        out = (body.get("stdout") or "") + (body.get("stderr") or "")
        if body.get("process_status") not in (None, "completed", "success"):
            out = f"[{body['process_status']}]\n{out}"
        out = out.strip() or "(no output)"
        cap = config.max_output_characters
        return out[:cap] + f"\n...[cut at {cap} characters]" if len(out) > cap else out

    yield FunctionInfo.from_fn(_run, description=(
        "Run one shell command in the workspace and return its output. The working directory is "
        "the workspace root, so paths are relative to it. For anything on the web use web_search "
        "and web_fetch rather than curl or urllib here: those keep what they read where the rest "
        "of the run can see it.\n\n"
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


class TaskListConfig(FunctionBaseConfig, name="task_list"):
    path: str = Field(default="_notes/tasks.md", description="Where the list is kept")


@register_function(config_type=TaskListConfig)
async def task_list(config: TaskListConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """The plan as a file, so a long run can be resumed and audited rather than remembered.

    Three states, not two: closing a step by SOLVING it is the only way an agent had, so a step it
    could not solve stayed open and it kept trying. `giving_up` closes one and says why, and the
    reason is what stops the next agent -- or the next turn of this one -- repeating the attempt.
    """

    async def _run(steps: str = "", done: str = "", giving_up: str = "", because: str = "") -> str:
        p = _resolve(config.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
        if steps.strip():
            # Steps often arrive already bulleted, and "- [ ] - step" reads as a broken list.
            lines = [f"- [ ] {_bare(s)}"
                     for s in steps.splitlines() if s.strip()]
        for mark, box, why in ([(x, "- [x]", "") for x in done.splitlines() if x.strip()]
                               + [(x, "- [-]", because) for x in giving_up.splitlines() if x.strip()]):
            for i, line in enumerate(lines):
                if line.startswith("- [ ]") and mark.strip().lower() in line.lower():
                    lines[i] = line.replace("- [ ]", box, 1) + (f"  ({why.strip()})" if why.strip() else "")
                    break
        if lines:
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not lines:
            return "The list is empty; send `steps` to start one."
        left = sum(l.startswith("- [ ]") for l in lines)
        gave = sum(l.startswith("- [-]") for l in lines)
        tail = f"({left} of {len(lines)} still open" + (f", {gave} given up on)" if gave else ")")
        return "\n".join(lines) + "\n\n" + tail

    yield FunctionInfo.from_fn(_run, description=(
        "Keep the plan for this task as a checklist. Call it once with `steps` to write the plan, "
        "then with `done` after finishing one, or with `giving_up` and `because` for one you tried "
        "and could not settle -- writing that down is what keeps you, and anyone after you, from "
        "trying it again. With no arguments it shows where you are. It outlives this conversation."
        "\n\nArgs:\n    steps (str): the plan, one step per line -- replaces any existing list.\n"
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
