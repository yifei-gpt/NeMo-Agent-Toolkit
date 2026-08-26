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
"""Wikipedia search in two batched API calls per query, cached on disk: the `wikipedia`
package spends ~7 requests per query and trips the 429 limit within a handful."""

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "nvidia-nat-benchmarks (https://github.com/NVIDIA/NeMo-Agent-Toolkit)"
_MAX_RETRIES = 4
_NO_RESULTS = "No Wikipedia results for: "
# A miss is often throttling: it expires instead of poisoning the query for every later run.
_FAILURE_TTL_S = 86400.0

# Wikitext carries infoboxes and tables that plaintext extracts drop, at the cost of markup.
_NOISE = [
    (re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL), ""),
    (re.compile(r"<ref[^>]*/>"), ""),
    (re.compile(r"\[\[(?:File|Image):[^\]]*\]\]"), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]


class CachedWikiSearchConfig(FunctionBaseConfig, name="wiki_search_cached"):
    """Wikipedia search whose results are memoized on disk."""

    max_results: int = Field(default=3, description="Articles to return per query")
    cache_dir: str = Field(description="Directory holding cached query results")
    max_chars_per_article: int = Field(default=12000, description="Truncate each article beyond this length")
    min_request_interval_s: float = Field(default=0.5, description="Client-side throttle between live calls")


@register_function(config_type=CachedWikiSearchConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def wiki_search_cached(tool_config: CachedWikiSearchConfig, builder: Builder):
    cache_dir = Path(tool_config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()

    def _clean(text: str) -> str:
        for pattern, repl in _NOISE:
            text = pattern.sub(repl, text)
        text = text.strip()
        if len(text) > tool_config.max_chars_per_article:
            text = text[:tool_config.max_chars_per_article] + (
                "\n...[the article is longer than this; web_fetch on the url in this "
                "Document tag reads the whole of it a window at a time]")
        return text

    async def _get(client: httpx.AsyncClient, params: dict) -> dict:
        for attempt in range(_MAX_RETRIES):
            response = await client.get(_API,
                                        params={
                                            "action": "query", "format": "json", "formatversion": 2, **params
                                        })
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            delay = float(response.headers.get("retry-after", 5)) + attempt
            logger.warning("Wikipedia rate limited, sleeping %.0fs", delay)
            await asyncio.sleep(delay)
        raise RuntimeError("Wikipedia kept returning 429")

    async def _fetch(question: str) -> tuple[bool, str]:
        async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}, timeout=30.0) as client:
            found = await _get(client, {
                "list": "search", "srsearch": question, "srlimit": tool_config.max_results
            })
            titles = [hit["title"] for hit in found.get("query", {}).get("search", [])]
            if not titles:
                return False, _NO_RESULTS + question

            content = await _get(client, {
                "titles": "|".join(titles), "prop": "revisions", "rvprop": "content", "rvslots": "main"
            })

        docs = []
        for page in content.get("query", {}).get("pages", []):
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            text = revisions[0].get("slots", {}).get("main", {}).get("content", "")
            # The url is here so the truncation notice has something real to point at.
            url = "https://en.wikipedia.org/wiki/" + quote(page["title"].replace(" ", "_"))
            # Titles carry quotes and ampersands ('"Weird Al" Yankovic'), which would end the tag early.
            docs.append(f'<Document title="{html.escape(page["title"])}" url="{html.escape(url)}">\n'
                        f'{_clean(text)}\n</Document>')
        return bool(docs), "\n\n---\n\n".join(docs)

    def _cache_path(question: str) -> Path:
        key = f"v4|{question}|{tool_config.max_results}|{tool_config.max_chars_per_article}"
        return cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.json"

    def _read(path: Path) -> str | None:
        try:
            entry = json.loads(path.read_text())
            age = time.time() - path.stat().st_mtime
        except (OSError, ValueError):
            return None
        result = entry.get("result", "")
        # Entries predating the marker are judged by their shape.
        ok = entry.get("ok", bool(result) and not result.startswith(_NO_RESULTS))
        return result if ok or age < _FAILURE_TTL_S else None

    def _write(path: Path, question: str, ok: bool, result: str) -> None:
        # Campaign processes share one cache dir, so a reader must never catch a half-written file.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"question": question, "ok": ok, "result": result}))
        os.replace(tmp, path)

    async def _search(question: str) -> str:
        # An article pasted back as the query 414s; normalising also merges whitespace variants.
        question = " ".join(question.split())[:300]
        path = _cache_path(question)
        cached = _read(path)
        if cached is not None:
            return cached

        # Serializing live calls keeps a fan-out of agents inside Wikipedia's rate limit.
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
        description=("Look a subject up on Wikipedia and get the opening of the matching articles. Use it "
                     "for a definition, a date or a name you need settled; web_search covers everything "
                     "else, and anything recent or contested belongs there instead.\n\n"
                     "Args:\n    question (str): the subject to look up."),
    )
