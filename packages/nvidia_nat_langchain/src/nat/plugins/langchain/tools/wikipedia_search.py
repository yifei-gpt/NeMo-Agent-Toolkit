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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# The `wikipedia` package's hardcoded User-Agent now gets HTTP 403 from Wikipedia, which
# surfaces as a JSON decode error. See https://meta.wikimedia.org/wiki/User-Agent_policy
DEFAULT_WIKIPEDIA_USER_AGENT = "nvidia-nat (https://github.com/NVIDIA/NeMo-Agent-Toolkit)"


# Wikipedia Search tool
class WikiSearchToolConfig(FunctionBaseConfig, name="wiki_search"):
    """
    Tool that retrieves relevant contexts from wikipedia search for the given question.
    """
    max_results: int = 2
    user_agent: str = DEFAULT_WIKIPEDIA_USER_AGENT


# Wiki search
@register_function(config_type=WikiSearchToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def wiki_search(tool_config: WikiSearchToolConfig, builder: Builder):
    import wikipedia.wikipedia as wikipedia_module
    from langchain_community.document_loaders import WikipediaLoader

    # The `wikipedia` package reads this module global on every API request.
    wikipedia_module.USER_AGENT = tool_config.user_agent

    async def _wiki_search(question: str) -> str:
        # Search the web and get the requested amount of results
        search_docs = await WikipediaLoader(query=question, load_max_docs=tool_config.max_results).aload()
        wiki_search_results = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" '
            f'page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>' for doc in search_docs
        ])
        return wiki_search_results

    # Create a NAT wiki search tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _wiki_search,
        description=("""This tool retrieves relevant contexts from wikipedia search for the given question.

                        Args:
                            question (str): The question to be answered.
                    """),
    )
