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
"""
Customer Support Triage — each LangGraph node is a registered NAT function.

Topology:
                      ┌─── research_context (Function) ──┐
    classify (Fn) ──► │                                    ├──► draft_response (Fn) ──► review (Fn)
                      └─── lookup_policy (Function) ──────┘

Each node is a separately registered NAT function so the profiler records
individual spans per node.  This lets the prediction trie's auto-sensitivity
algorithm differentiate nodes by their position, fan-out, critical-path
contribution, and parallel slack.
"""

import logging
from typing import TypedDict

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Node functions — each is a NAT Function with its own profiler span
# ──────────────────────────────────────────────────────────────────────────────


class ClassifyConfig(FunctionBaseConfig, name="classify_query"):
    llm: LLMRef


@register_function(config_type=ClassifyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def classify_query_function(config: ClassifyConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support classifier. Categorize the query into exactly one of: "
         "billing, account, technical, general. Reply with only the category name."),
        ("human", "{query}"),
    ]) | llm | StrOutputParser())

    async def _classify(query: str) -> str:
        """Classify a customer query into a support category."""
        result = await chain.ainvoke({"query": query})
        return result.strip().lower()

    yield FunctionInfo.from_fn(_classify, description=_classify.__doc__)


class ResearchContextConfig(FunctionBaseConfig, name="research_context"):
    llm: LLMRef


@register_function(config_type=ResearchContextConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def research_context_function(config: ResearchContextConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support researcher. Given the query and its category, "
         "gather relevant context from the knowledge base. Provide a brief summary "
         "of relevant information that would help draft a response."),
        ("human", "Category: {category}\nQuery: {query}"),
    ]) | llm | StrOutputParser())

    async def _research(input_text: str) -> str:
        """Research relevant context for a customer query."""
        # Parse "category|query" format
        parts = input_text.split("|", 1)
        category = parts[0].strip() if len(parts) > 1 else ""
        query = parts[-1].strip()
        return await chain.ainvoke({"category": category, "query": query})

    yield FunctionInfo.from_fn(_research, description=_research.__doc__)


class LookupPolicyConfig(FunctionBaseConfig, name="lookup_policy"):
    llm: LLMRef


@register_function(config_type=LookupPolicyConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def lookup_policy_function(config: LookupPolicyConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a policy lookup specialist. Given the query category, retrieve the "
         "applicable company policies and guidelines. Be specific about what actions "
         "are allowed, required timelines, and any escalation procedures."),
        ("human", "Category: {category}\nQuery: {query}"),
    ]) | llm | StrOutputParser())

    async def _lookup(input_text: str) -> str:
        """Look up company policy for a customer query category."""
        parts = input_text.split("|", 1)
        category = parts[0].strip() if len(parts) > 1 else ""
        query = parts[-1].strip()
        return await chain.ainvoke({"category": category, "query": query})

    yield FunctionInfo.from_fn(_lookup, description=_lookup.__doc__)


class DraftResponseConfig(FunctionBaseConfig, name="draft_response"):
    llm: LLMRef


@register_function(config_type=DraftResponseConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def draft_response_function(config: DraftResponseConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support agent. Using the research context and company "
         "policy provided, draft a helpful response to the customer query. Be "
         "professional, empathetic, and actionable."),
        ("human", "Query: {query}\nCategory: {category}\nContext: {context}\nPolicy: {policy}"),
    ]) | llm | StrOutputParser())

    async def _draft(input_text: str) -> str:
        """Draft a support response using context and policy."""
        # Parse "query|category|context|policy" format
        parts = input_text.split("|")
        query = parts[0].strip() if len(parts) > 0 else ""
        category = parts[1].strip() if len(parts) > 1 else ""
        context = parts[2].strip() if len(parts) > 2 else ""
        policy = parts[3].strip() if len(parts) > 3 else ""
        return await chain.ainvoke({"query": query, "category": category, "context": context, "policy": policy})

    yield FunctionInfo.from_fn(_draft, description=_draft.__doc__)


class ReviewConfig(FunctionBaseConfig, name="review_response"):
    llm: LLMRef


@register_function(config_type=ReviewConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def review_response_function(config: ReviewConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior support reviewer. Review the draft response for accuracy, "
         "tone, and completeness. Output the final polished response ready to send "
         "to the customer. If the draft is good, return it as-is."),
        ("human", "Original query: {query}\nDraft response: {draft}"),
    ]) | llm | StrOutputParser())

    async def _review(input_text: str) -> str:
        """Review and finalize a draft support response."""
        parts = input_text.split("|", 1)
        query = parts[0].strip() if len(parts) > 1 else ""
        draft = parts[-1].strip()
        return await chain.ainvoke({"query": query, "draft": draft})

    yield FunctionInfo.from_fn(_review, description=_review.__doc__)


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator workflow — builds the LangGraph and delegates to NAT functions
# ──────────────────────────────────────────────────────────────────────────────


class SupportState(TypedDict):
    """State passed through the customer support triage graph."""

    query: str
    category: str
    context: str
    policy: str
    draft: str
    final_response: str


class LatencySensitivityDemoConfig(FunctionBaseConfig, name="latency_sensitivity_demo"):
    """Configuration for the latency sensitivity demo workflow."""

    classify_fn: FunctionRef = Field(default=FunctionRef("classify_query"), description="Function to classify queries")
    research_fn: FunctionRef = Field(default=FunctionRef("research_context"),
                                     description="Function to research context")
    policy_fn: FunctionRef = Field(default=FunctionRef("lookup_policy"), description="Function to look up policy")
    draft_fn: FunctionRef = Field(default=FunctionRef("draft_response"), description="Function to draft response")
    review_fn: FunctionRef = Field(default=FunctionRef("review_response"), description="Function to review response")


@register_function(config_type=LatencySensitivityDemoConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def latency_sensitivity_demo_function(config: LatencySensitivityDemoConfig, builder: Builder):
    """Orchestrate the customer support triage workflow with parallel fan-out."""

    from langgraph.graph import END
    from langgraph.graph import StateGraph

    # Get each node as a NAT Function — each .ainvoke() creates its own profiler span
    classify_fn = await builder.get_function(config.classify_fn)
    research_fn = await builder.get_function(config.research_fn)
    policy_fn = await builder.get_function(config.policy_fn)
    draft_fn = await builder.get_function(config.draft_fn)
    review_fn = await builder.get_function(config.review_fn)

    # ── LangGraph node wrappers ──────────────────────────────────────────
    async def classify(state: SupportState) -> dict:
        category = await classify_fn.ainvoke(state["query"])
        return {"category": str(category).strip().lower()}

    async def research_context(state: SupportState) -> dict:
        context = await research_fn.ainvoke(f"{state['category']}|{state['query']}")
        return {"context": str(context)}

    async def lookup_policy(state: SupportState) -> dict:
        policy = await policy_fn.ainvoke(f"{state['category']}|{state['query']}")
        return {"policy": str(policy)}

    async def draft_response(state: SupportState) -> dict:
        draft = await draft_fn.ainvoke(f"{state['query']}|{state['category']}|{state['context']}|{state['policy']}")
        return {"draft": str(draft)}

    async def review(state: SupportState) -> dict:
        final = await review_fn.ainvoke(f"{state['query']}|{state['draft']}")
        return {"final_response": str(final)}

    # ── Build the graph ──────────────────────────────────────────────────

    graph = StateGraph(SupportState)

    graph.add_node("classify", classify)
    graph.add_node("research_context", research_context)
    graph.add_node("lookup_policy", lookup_policy)
    graph.add_node("draft_response", draft_response)
    graph.add_node("review", review)

    graph.set_entry_point("classify")

    # Parallel fan-out
    graph.add_edge("classify", "research_context")
    graph.add_edge("classify", "lookup_policy")

    # Converge
    graph.add_edge("research_context", "draft_response")
    graph.add_edge("lookup_policy", "draft_response")

    # Sequential tail
    graph.add_edge("draft_response", "review")
    graph.add_edge("review", END)

    app = graph.compile()

    async def _run(query: str) -> str:
        """Customer support triage workflow with parallel context and policy lookup."""
        result = await app.ainvoke({
            "query": query,
            "category": "",
            "context": "",
            "policy": "",
            "draft": "",
            "final_response": "",
        })
        return result["final_response"]

    try:
        yield FunctionInfo.from_fn(_run, description=_run.__doc__)
    except GeneratorExit:
        logger.exception("Exited early!")
    finally:
        logger.debug("Cleaning up latency_sensitivity_demo workflow.")
