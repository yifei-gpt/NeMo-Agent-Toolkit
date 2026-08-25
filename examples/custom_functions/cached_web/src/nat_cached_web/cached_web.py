# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Real web search and page reading, memoized on disk.

Search goes through a self-hosted SearxNG, so no vendor key is needed and no query leaves as a
billable call. Every answer is cached by its query: a marked run and its unmarked control replay
the same web when they ask the same thing, which is what makes a utility comparison a comparison.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

_USER_AGENT = "markagentx-benchmarks (local research agent)"
# A miss is usually throttling, so it expires instead of poisoning the query for every later run.
_FAILURE_TTL_S = 86400.0
_NO_RESULTS = "No web results for: "
_STRIP = [
    (re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.DOTALL | re.I), " "),
    (re.compile(r"<!--.*?-->", re.DOTALL), " "),
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"&nbsp;?"), " "),
    (re.compile(r"[ \t\r\f\v]+"), " "),
    (re.compile(r"\n\s*\n\s*\n+"), "\n\n"),
]


def _cache(directory: Path, key: str) -> Path:
    return directory / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


def _read(path: Path) -> str | None:
    try:
        entry = json.loads(path.read_text())
        age = time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        return None
    result = entry.get("result", "")
    return result if entry.get("ok") or age < _FAILURE_TTL_S else None


def _write(path: Path, question: str, ok: bool, result: str) -> None:
    # Campaign processes share one cache dir, so a reader must never catch a half-written file.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"question": question, "ok": ok, "result": result}))
    os.replace(tmp, path)


def _as_text(body: str, limit: int) -> str:
    for pattern, repl in _STRIP:
        body = pattern.sub(repl, body)
    body = "\n".join(line.strip() for line in body.splitlines())
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body[:limit] + ("\n...[truncated]" if len(body) > limit else "")


def _page_key(url: str, store: int) -> str:
    return f"fetch|v2|{url}|{store}"


def _window(page: str, offset: int, width: int) -> str:
    """One slice of a page, told plainly enough that the next slice is asked for, not re-fetched."""
    offset = max(0, int(offset or 0))
    if offset >= len(page):
        return f"Nothing at offset {offset}; this page is {len(page)} characters long."
    end = min(offset + width, len(page))
    tail = (f"\n\n[characters {offset}-{end} of {len(page)}; call web_fetch again with "
            f"offset={end} to read on]" if end < len(page) else "")
    return page[offset:end] + tail


def _from_pdf(raw: bytes, limit: int) -> str:
    """A PDF's text, when the reader is installed; its absence is said rather than hidden."""
    try:
        import pymupdf
    except ImportError:
        return "This source is a PDF and no PDF reader is installed here."
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            text = "\n\n".join(page.get_text() for page in doc)
    except Exception as exc:  # noqa: BLE001
        return f"This source is a PDF that could not be read ({type(exc).__name__})."
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:limit] + ("\n...[truncated]" if len(text) > limit else "")


class WebSearchConfig(FunctionBaseConfig, name="web_search_cached"):
    """Web search through a self-hosted SearxNG, memoized on disk."""

    base_url: str = Field(default="http://127.0.0.1:8888", description="SearxNG base URL")
    cache_dir: str = Field(description="Directory holding cached query results")
    max_results: int = Field(default=8, description="Results to return per query")
    max_chars_per_result: int = Field(default=400, description="Truncate each snippet here")
    min_request_interval_s: float = Field(default=0.3, description="Throttle between live calls")


@register_function(config_type=WebSearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def web_search_cached(tool_config: WebSearchConfig, builder: Builder):
    directory = Path(tool_config.cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()

    async def _fetch(question: str) -> tuple[bool, str]:
        params = {"q": question, "format": "json", "safesearch": 0}
        try:
            async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=45.0) as client:
                answer = await client.get(tool_config.base_url.rstrip("/") + "/search", params=params)
                answer.raise_for_status()
                found = answer.json().get("results", [])
        except Exception as exc:  # noqa: BLE001 -- any failure to answer means the same thing
            logger.warning("web search failed for %r: %s", question[:60], exc)
            return False, f"Web search is unavailable right now ({type(exc).__name__})."
        if not found:
            return False, _NO_RESULTS + question
        lines = []
        for index, row in enumerate(found[: tool_config.max_results], 1):
            snippet = " ".join((row.get("content") or "").split())[: tool_config.max_chars_per_result]
            lines.append(f"[{index}] {row.get('title', '')}\n    {row.get('url', '')}\n    {snippet}")
        return True, "\n".join(lines)

    async def _search(question: str) -> str:
        question = " ".join(question.split())[:400]
        path = _cache(directory, f"search|v1|{question}|{tool_config.max_results}")
        cached = _read(path)
        if cached is not None:
            return cached
        # Serialised: a fan-out of agents would otherwise look like a scraper to the upstreams.
        async with lock:
            cached = _read(path)
            if cached is not None:
                return cached
            ok, result = await _fetch(question)
            _write(path, question, ok, result)
            await asyncio.sleep(tool_config.min_request_interval_s)
        return result

    yield FunctionInfo.from_fn(
        _search,
        description=("Search the web and return the top results as numbered title/URL/snippet "
                     "entries. Use web_fetch on a URL to read the page itself.\n\n"
                     "Args:\n    question (str): what to search for."),
    )


class WebFindConfig(FunctionBaseConfig, name="web_find_cached"):
    """Locate a phrase inside a page already opened, without opening it again."""

    cache_dir: str = Field(description="Directory holding cached pages")
    store_chars: int = Field(default=240000, description="The length web_fetch was told to keep")
    context_chars: int = Field(default=600, description="Text to return around each hit")
    max_hits: int = Field(default=5, description="Hits to return per search")


@register_function(config_type=WebFindConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def web_find_cached(tool_config: WebFindConfig, builder: Builder):
    """Finding a document and reading the part that matters are different acts, and the second is
    where the answer is: search alone answers 43.9% of BrowseComp-Plus, opening pages takes it to
    56.4%, and being able to find within them to 62.2% (OpenResearcher RQ4).
    """
    directory = Path(tool_config.cache_dir)

    async def _find(url: str, phrase: str) -> str:
        page = _read(_cache(directory, _page_key(url.strip(), tool_config.store_chars)))
        if page is None:
            return f"{url} has not been read yet -- call web_fetch on it first."
        hits, start, width = [], 0, tool_config.context_chars
        lowered, needle = page.lower(), " ".join(phrase.split()).lower()
        if not needle:
            return "Give a phrase to look for."
        while len(hits) < tool_config.max_hits:
            at = lowered.find(needle, start)
            if at < 0:
                break
            line = page.count("\n", 0, at) + 1
            piece = page[max(0, at - width // 2): at + len(needle) + width // 2]
            hits.append(f"[line {line}] ...{' '.join(piece.split())}...")
            start = at + len(needle)
        if not hits:
            return f"{phrase!r} does not appear in {url}."
        return f"{len(hits)} place(s) in {url}:\n" + "\n\n".join(hits)

    yield FunctionInfo.from_fn(
        _find,
        description=("Find a phrase inside a page you have already read with web_fetch, and get "
                     "the text around each place it appears.\n\n"
                     "Args:\n    url (str): the page, as passed to web_fetch.\n"
                     "    phrase (str): the words to look for."),
    )


class WebFetchConfig(FunctionBaseConfig, name="web_fetch_cached"):
    """One page, as readable text, memoized on disk and read a window at a time."""

    cache_dir: str = Field(description="Directory holding cached pages")
    max_chars: int = Field(default=12000, description="Characters returned per call")
    store_chars: int = Field(default=240000, description="How much of the page is kept on disk")
    min_request_interval_s: float = Field(default=0.3, description="Throttle between live calls")


@register_function(config_type=WebFetchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def web_fetch_cached(tool_config: WebFetchConfig, builder: Builder):
    directory = Path(tool_config.cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()

    async def _fetch(url: str) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=45.0,
                                         follow_redirects=True) as client:
                answer = await client.get(url)
                answer.raise_for_status()
                kind = answer.headers.get("content-type", "")
                if "pdf" in kind or url.lower().endswith(".pdf"):
                    # Research sources are PDFs as often as pages; refusing them loses the source.
                    return True, _from_pdf(answer.content, tool_config.store_chars)
                if "html" not in kind and "text" not in kind and "json" not in kind:
                    return False, f"{url} is {kind or 'not text'}; nothing to read."
                return True, _as_text(answer.text, tool_config.store_chars)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web fetch failed for %s: %s", url[:80], exc)
            # Said plainly, because the common cause is a guessed URL and the agent otherwise
            # guesses a second one instead of searching for the real address.
            return False, (f"Could not read {url} ({type(exc).__name__}). If you guessed this "
                           "address, call web_search for the page instead of guessing again.")

    async def _read_page(url: str, offset: int = 0) -> str:
        url = url.strip()
        if not url.lower().startswith(("http://", "https://")):
            return "Give a full http(s) URL, as web_search returns."
        path = _cache(directory, _page_key(url, tool_config.store_chars))
        page = _read(path)
        if page is None:
            async with lock:
                page = _read(path)
                if page is None:
                    ok, page = await _fetch(url)
                    _write(path, url, ok, page)
                    await asyncio.sleep(tool_config.min_request_interval_s)
        return _window(page, offset, tool_config.max_chars)

    yield FunctionInfo.from_fn(
        _read_page,
        description=("Read one web page and return its text. Long pages come back one window at "
                     "a time; the reply says whether more is left and what offset reads it.\n\n"
                     "Args:\n    url (str): a full http(s) URL, as web_search returns.\n"
                     "    offset (int): character to start at, 0 for the beginning."),
    )
