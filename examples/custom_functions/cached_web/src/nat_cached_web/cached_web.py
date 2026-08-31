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
import fcntl
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

_USER_AGENT = "nat-agent (local benchmark run)"
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


async def _ask(question: str, key: str, want: int, url: str) -> list[dict] | None:
    """The provider's rows as (title, url, text), or None when it could not be reached at all."""
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            answer = await client.post(url, headers={"X-API-KEY": key},
                                       json={"q": question, "num": want})
            answer.raise_for_status()
            body = answer.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("web search failed for %r: %s", question[:60], exc)
        return None
    rows = []
    # The direct answer when the provider has one: the whole failure this replaced was results
    # that were about the subject without carrying what was asked.
    if box := body.get("answerBox"):
        text = box.get("answer") or box.get("snippet") or ""
        if text:
            rows.append({"title": box.get("title") or "answer", "url": box.get("link", ""),
                         "text": text})
    rows += [{"title": r.get("title", ""), "url": r.get("link", ""), "text": r.get("snippet", "")}
             for r in (body.get("organic") or [])]
    return rows[:want]


# Every reply the metasearch makes carries this. It was taken out once for returning landing pages
# silently, and a fallback nobody can see is the loop it caused: over six questions with known
# answers it carried the answer none of six times, against six of six for the keyed API.
_DEGRADED = ("\n\n[searched without a key, through the local metasearch. Most engines refuse this "
             "host, and what answers tends to return pages ABOUT the subject rather than pages "
             "holding the answer -- so treat a missing answer as a search that failed, not as a "
             "fact that does not exist, and do not reword the same question against it. Set a "
             "Serper key in ~/.config/nat/serper.key for real results.]")


async def _searx(question: str, base_url: str, want: int, chars: int) -> tuple[bool, str] | None:
    """The unkeyed path. -> (found anything, text), or None when the metasearch is not there."""
    try:
        async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=45.0) as client:
            answer = await client.get(base_url.rstrip("/") + "/search",
                                      params={"q": question, "format": "json", "safesearch": 0})
            answer.raise_for_status()
            body = answer.json()
    except Exception as exc:  # noqa: BLE001 -- not reachable and not answering mean the same here
        logger.warning("metasearch at %s did not answer: %s", base_url, exc)
        return None
    # Which engines refused: dropping this once turned a CAPTCHA'd Google and a rate-limited Brave
    # into ten sign-in pages served as ordinary results, and agents reworded the same question
    # forty times against them.
    down = [str(e[0]) for e in (body.get("unresponsive_engines") or []) if e]
    refused = ("\n[these engines refused this search: " + ", ".join(sorted(set(down))) + "]") if down else ""
    found = body.get("results") or []
    if not found:
        return False, _NO_RESULTS + question + _DEGRADED + refused
    lines = [f"[{i}] {r.get('title', '')}\n    {r.get('url', '')}\n    "
             + " ".join((r.get("content") or "").split())[:chars]
             for i, r in enumerate(found[:want], 1)]
    return True, "\n".join(lines) + _DEGRADED + refused


def _key_from(path: str) -> str:
    try:
        return Path(path).expanduser().read_text().strip()
    except OSError:
        return ""


async def _pace(directory: Path, interval: float) -> None:
    """Hold the whole machine to one live call per `interval`.

    The asyncio lock beside this one covers a single process, and a sweep runs dozens: the rate
    upstream saw was that many times what any one of them intended, and four of five engines
    answered with a CAPTCHA or a 429 for the rest of the day.
    """
    if interval <= 0:
        return
    gate = directory / ".pace"
    for _ in range(600):
        with open(gate, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                last = float(fh.read().strip() or 0)
                wait = last + interval - time.time()
                if wait <= 0:
                    fh.seek(0), fh.truncate()
                    fh.write(str(time.time()))
                    return
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        await asyncio.sleep(min(wait, interval))


def _cache(directory: Path, key: str) -> Path:
    return directory / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"


def _entry(path: Path) -> tuple[bool, str] | None:
    """(the fetch succeeded, what was stored), or None when nothing usable is cached."""
    try:
        entry = json.loads(path.read_text())
        age = time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        return None
    ok = bool(entry.get("ok"))
    return (ok, entry.get("result", "")) if ok or age < _FAILURE_TTL_S else None


def _read(path: Path) -> str | None:
    got = _entry(path)
    return got[1] if got else None


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


def _window(page: str, offset: int, width: int, store: int = 0) -> str:
    """One slice of a page, told plainly enough that the next slice is asked for, not re-fetched."""
    offset = max(0, int(offset or 0))
    # A document that filled the store was itself cut, and paging to its end still never reaches
    # what was dropped: agents walked 22 windows of a 240k JSON array for a date that was not in it.
    cut = ("\n[this document was longer than this tool keeps and was cut at "
           f"{store} characters, so what you are looking for may not be in it at all]"
           if store and len(page) >= store else "")
    if offset >= len(page):
        return f"Nothing at offset {offset}; this page is {len(page)} characters long.{cut}"
    end = min(offset + width, len(page))
    tail = (f"\n\n[characters {offset}-{end} of {len(page)}; call web_fetch again with "
            f"offset={end} to read on]" if end < len(page) else "")
    return page[offset:end] + tail + cut


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
    """Web search through a keyed API, memoized on disk.

    A keyed API when there is a key, and a self-hosted metasearch when there is not, so the tool
    works on a machine nobody has configured. They are not equals: seven of the metasearch's eight
    engines refuse this host -- four on the first request we ever sent, so it is the address and
    not our rate -- and the one that answers returns landing pages. Over six questions with known
    answers it carried the answer none of six times, against six of six for the keyed API. An agent
    handed python.org for "Python 3.0 release year" rewords the question, and that is the loop, so
    every reply from the unkeyed path says which path it came from and what follows from it.
    """

    key_file: str = Field(default="~/.config/nat/serper.key", description="Serper API key file")
    # Without a key the tool would be dead on a fresh machine, so the metasearch stands behind it --
    # loudly, because it is the weaker search and a reply that does not say so is the loop.
    searx_url: str = Field(default="http://127.0.0.1:8888",
                           description="Local SearxNG, used only when there is no key")
    # Serper's, and only Serper's: the request carries its header and its parameter names, and the
    # reply is read for `answerBox` and `organic`. Another provider answers 200 with nothing this
    # can parse, which arrives as a search that found no results rather than as an error.
    api_url: str = Field(default="https://google.serper.dev/search",
                         description="Serper search endpoint")
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
        key = _key_from(tool_config.key_file)
        if not key:
            thin = await _searx(question, tool_config.searx_url, tool_config.max_results,
                                tool_config.max_chars_per_result)
            if thin is not None:
                return thin
            return False, ("Web search is not configured on this machine: no key in "
                           f"{tool_config.key_file} and no metasearch at {tool_config.searx_url}. "
                           "Work from what you have and say which parts you could not look up.")
        rows = await _ask(question, key, tool_config.max_results, tool_config.api_url)
        if rows is None:
            return False, "Web search is unavailable right now; try again later or work without it."
        if not rows:
            return False, _NO_RESULTS + question
        return True, "\n".join(
            f"[{i}] {r['title']}\n    {r['url']}\n    "
            + " ".join(r["text"].split())[: tool_config.max_chars_per_result]
            for i, r in enumerate(rows, 1))

    # Per run, since the closure is built per workflow: the cache is shared with other runs on
    # purpose, so a hit says nothing, but the same question asked twice HERE does. One run asked
    # the same search 132 times, each answer byte-identical, until the context window gave out.
    asked: dict[str, int] = {}

    def _said_before(question: str, result: str) -> str:
        asked[question] = n = asked.get(question, 0) + 1
        if n == 1:
            return result
        return f"{result}\n\n[you have asked this {n} times in this run; the answer is the same]"

    async def _search(question: str) -> str:
        question = " ".join(question.split())[:400]  # noqa: E501
        path = _cache(directory, f"search|v1|{question}|{tool_config.max_results}")
        cached = _read(path)
        if cached is not None:
            return _said_before(question, cached)
        # Serialised: a fan-out of agents would otherwise look like a scraper to the upstreams.
        async with lock:
            cached = _read(path)
            if cached is not None:
                return _said_before(question, cached)
            await _pace(directory, tool_config.min_request_interval_s)
            ok, result = await _fetch(question)
            _write(path, question, ok, result)
        return _said_before(question, result)

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
        got = _entry(_cache(directory, _page_key(url.strip(), tool_config.store_chars)))
        # A cached failure is not a page: searched as one, every phrase "does not appear" in it.
        if got is None or not got[0]:
            return f"{url} has not been read yet -- call web_fetch on it first."
        page = got[1]
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
            code = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("web fetch failed for %s: %s", url[:80], exc or type(exc).__name__)
            # Which advice is right turns on the code: re-searching a 403 returns the same blocked
            # address, and the agent re-searched it until its step budget ran out.
            if code in (401, 403):
                return False, (f"Could not read {url} ({code}): this site refuses automated "
                               "readers. The address is fine -- find the same fact somewhere else "
                               "rather than searching for this page again.")
            if code == 429:
                return False, f"Could not read {url} (429): asked too often. Come back to it later."
            if code == 404:
                return False, (f"Could not read {url} (404): nothing is there. If you guessed this "
                               "address, call web_search for the page instead of guessing again.")
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
                    await _pace(directory, tool_config.min_request_interval_s)
                    ok, page = await _fetch(url)
                    _write(path, url, ok, page)
        return _window(page, offset, tool_config.max_chars, tool_config.store_chars)

    yield FunctionInfo.from_fn(
        _read_page,
        description=("Read one web page and return its text. Long pages come back one window at "
                     "a time; the reply says whether more is left and what offset reads it.\n\n"
                     "Args:\n    url (str): a full http(s) URL, as web_search returns.\n"
                     "    offset (int): character to start at, 0 for the beginning."),
    )
