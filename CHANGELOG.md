## What's Changed
### 🚨 Breaking Changes
* Add NeMo Guardrails policy middleware by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2006
* Remove integration package for RagaAI Catalyst by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2013
* docs: Migrate Tavily search to provider-managed third-party plugin by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2015
* fix(middleware): wire is_final enforcement with build-time validation and runtime call_next suppression by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2070
### ✨ New Features
* feat(a365): add Microsoft Agent 365 integration plugins by @afourniernv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1571
* experimental(agent): Add experimental coding-agent adapters with NeMo-Relay telemetry by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1995
### 🔧 Improvements
* Generate a list of NIM models used in workflow YAML files in `pre-commit` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1982
* Fix running `pytest` from the root of the repo by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2000
* Bump langchain NVIDIA endpoints dependency by @freshyjmp in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2002
* chore: Update dependencies by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2022
* chore: bump nat-ui submodule to f4926a2 by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2031
* fix(hitl): improve por_to_jiratickets example by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2032
* Update ATIF Eval notebooks to install NAT from pypi by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2035
* chore: Remove DBNL appears to have gone away by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2043
* chore: Update to latest commit in NAT-UI main by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2053
* chore: Update ADK version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2056
### 🐛 Bug Fixes
* fix(observability): Emit `OpenInference` LLM cost lookup attributes for span exports by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1988
* fix(tool_wrapper): Fix `LangChain` tool input normalization for chat agent branches by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1990
* fix(agent): Fix thinking metadata handling in agents by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1992
* Fix and improve integration tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1996
* fix(packaging): ship nat.plugin_api in nvidia-nat-core wheel by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1997
* Fix HTTP 500 on human-in-the-loop pause for non-streaming FastAPI endpoints by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1999
* Fix LangSmith integration tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1998
* fix(agent): Fix ReWOO nested evidence placeholder substitution by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2003
* docs: Fix vLLM serve command for Nemotron model in local LLMs guide by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2007
* Subclass EmptyFunctionConfig, avoids name clash with other registered functions by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2012
* fix(example): Fix `Agno` personal finance workflow for Agno 2.x and switch to NIM by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2016
* fix(auth): Fix protected `FastMCP` OAuth token validation by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2019
* fix: handle empty web ingest responses by @mnajafian-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2021
* fix: update Dynamo latency demo compose path by @mnajafian-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2020
* fix(fastapi): fix Generate endpoint for WebSocket non-streaming and HTTP streaming by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2009
* fix(examples): Keep Haystack indexing chunks within NIM embedder token limit by @zhongxuanwang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2008
* fix: memmachine notebook by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2041
* fix: improve Dynamo latency demo startup reliability by @mnajafian-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2050
* fix: Delay deleting of environment directories when generating coverage files by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2061
* fix: harden Dynamo latency demo against SGlang version drift and degenerate trie predictions by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2067
* fix(dynamo): correct request-priority polarity for Dynamo >= 1.1.0 and mark integration as experimental by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2076
### 📝 Documentation Updates
* Move Kaggle MCP example out of toolkit by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1972
* plugin docs: Stabilize the public plugin authoring API by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/1959
* docs: update Tavily examples to use function group by @lakshyaag-tavily in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2028
* docs: highlight third-party plugin contributors by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2039
* Docs: add Redis third party package reference to release notes by @bbednarski9 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2051
* docs: add Windows WSL2 setup section by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2065

## New Contributors
* @freshyjmp made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2002
* @lakshyaag-tavily made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2028
* @hemachandra666 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2065
