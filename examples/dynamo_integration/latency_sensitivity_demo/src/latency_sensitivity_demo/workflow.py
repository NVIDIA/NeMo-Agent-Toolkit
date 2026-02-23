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

Topology (designed for demonstrating priority-based scheduling):

                      ┌─── research_context   (LOW, ~500 tok) ──────┐
                      ├─── lookup_policy       (LOW, ~500 tok) ──────┤
    classify (HIGH) ─►├─── check_compliance    (LOW, ~500 tok) ──────├─► draft_response (MED) ─► review (HIGH)
                      └─── analyze_sentiment   (LOW, ~500 tok) ──────┘
       (~5 tok)                                                            (~500 tok)           (~20 tok)

The 4 parallel LOW-priority branches produce long outputs, saturating GPU
decode capacity at high concurrency.  The HIGH-priority ``classify`` and
``review`` nodes produce short outputs that can be served quickly when the
router prioritizes them.  This creates a measurable latency gap between HIGH
and LOW calls when priority-based scheduling is active.

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

# --- HIGH priority: short output ---


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
         "billing, account, technical, general. Respond with ONLY the single category word, "
         "nothing else."),
        ("human", "{query}"),
    ]) | llm | StrOutputParser())

    async def _classify(query: str) -> str:
        """Classify a customer query into a support category."""
        result = await chain.ainvoke({"query": query})
        return result.strip().lower()

    yield FunctionInfo.from_fn(_classify, description=_classify.__doc__)


# --- LOW priority: long output (parallel siblings) ---


class ResearchContextConfig(FunctionBaseConfig, name="research_context"):
    llm: LLMRef


@register_function(config_type=ResearchContextConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def research_context_function(config: ResearchContextConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support knowledge-base researcher. Given the query and its "
         "category, write a COMPREHENSIVE and EXTREMELY DETAILED research summary. "
         "You MUST cover ALL of the following sections in full, with multiple paragraphs each:\n\n"
         "1. KNOWLEDGE BASE ARTICLES: List every relevant article with title, ID, and a full "
         "paragraph summarizing each article's content and how it applies.\n"
         "2. TROUBLESHOOTING GUIDES: Provide complete step-by-step troubleshooting procedures "
         "with at least 10 steps each, including expected outcomes at each step.\n"
         "3. PRIOR CASE RESOLUTIONS: Describe at least 5 similar past cases with full details "
         "of the problem, resolution steps taken, timeline, and customer outcome.\n"
         "4. ROOT CAUSE ANALYSIS: Enumerate all common root causes with technical explanations, "
         "frequency statistics, and diagnostic procedures for each.\n"
         "5. EDGE CASES AND EXCEPTIONS: Document unusual scenarios, workarounds, known bugs, "
         "and special handling procedures.\n"
         "6. ESCALATION PATHS: Map out the full escalation tree with response time SLAs.\n\n"
         "Be EXTREMELY verbose. Write at least 800 words. The support agent depends on this."),
        ("human", "Category: {category}\nQuery: {query}"),
    ]) | llm | StrOutputParser())

    async def _research(input_text: str) -> str:
        """Research relevant context for a customer query."""
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
         "You are a company policy specialist. Given the query category, write an EXHAUSTIVE "
         "policy reference document. You MUST cover ALL of the following sections in full, "
         "with multiple paragraphs each:\n\n"
         "1. TERMS OF SERVICE: Quote all relevant ToS clauses verbatim with section numbers, "
         "effective dates, and full legal interpretation for this scenario.\n"
         "2. SLA COMMITMENTS: List every applicable SLA with metric definitions, measurement "
         "windows, penalty calculations, credit procedures, and exclusion criteria.\n"
         "3. REFUND AND CANCELLATION POLICIES: Document the complete refund matrix including "
         "eligibility windows, proration rules, restocking fees, and exception approval flows.\n"
         "4. ESCALATION PROCEDURES: Map the entire escalation hierarchy with names, roles, "
         "contact methods, response time targets, and authority limits at each level.\n"
         "5. REGULATORY REQUIREMENTS: Cover GDPR, CCPA, PCI-DSS, SOX, and industry-specific "
         "regulations with specific article references and compliance obligations.\n"
         "6. APPROVAL HIERARCHIES: Document the full approval chain for refunds, credits, "
         "exceptions, and policy overrides with dollar thresholds at each level.\n"
         "7. PRECEDENT DECISIONS: Reference at least 5 prior policy decisions in similar cases.\n\n"
         "Be EXTREMELY verbose. Write at least 800 words. The agent needs every detail."),
        ("human", "Category: {category}\nQuery: {query}"),
    ]) | llm | StrOutputParser())

    async def _lookup(input_text: str) -> str:
        """Look up company policy for a customer query category."""
        parts = input_text.split("|", 1)
        category = parts[0].strip() if len(parts) > 1 else ""
        query = parts[-1].strip()
        return await chain.ainvoke({"category": category, "query": query})

    yield FunctionInfo.from_fn(_lookup, description=_lookup.__doc__)


class CheckComplianceConfig(FunctionBaseConfig, name="check_compliance"):
    llm: LLMRef


@register_function(config_type=CheckComplianceConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def check_compliance_function(config: CheckComplianceConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a regulatory compliance auditor. Given the customer query and its category, "
         "write an EXTREMELY THOROUGH compliance assessment. You MUST cover ALL of the "
         "following sections in full, with multiple paragraphs each:\n\n"
         "1. GDPR ASSESSMENT: Full analysis of data subject rights (Articles 12-23), lawful "
         "basis for processing (Article 6), data protection impact assessment requirements, "
         "cross-border transfer implications, and DPO notification obligations.\n"
         "2. CCPA ASSESSMENT: Consumer rights analysis, opt-out requirements, data sale "
         "implications, service provider obligations, and financial incentive disclosures.\n"
         "3. PCI-DSS ASSESSMENT: Cardholder data environment scope, requirement applicability "
         "matrix, compensating controls, and SAQ determination.\n"
         "4. SOX ASSESSMENT: Internal control implications, audit trail requirements, "
         "segregation of duties analysis, and material weakness considerations.\n"
         "5. INDUSTRY-SPECIFIC: Identify and analyze all sector-specific regulations with "
         "full citation of relevant statutes, enforcement actions, and safe harbor provisions.\n"
         "6. RISK MATRIX: Rate each identified risk by likelihood and impact with specific "
         "mitigation strategies and residual risk acceptance criteria.\n"
         "7. MANDATORY REPORTING: List all notification obligations with deadlines, "
         "responsible parties, template references, and regulatory contact information.\n\n"
         "Be EXTREMELY verbose. Write at least 800 words. Compliance failures are costly."),
        ("human", "Category: {category}\nQuery: {query}"),
    ]) | llm | StrOutputParser())

    async def _check(input_text: str) -> str:
        """Check regulatory compliance requirements for a customer query."""
        parts = input_text.split("|", 1)
        category = parts[0].strip() if len(parts) > 1 else ""
        query = parts[-1].strip()
        return await chain.ainvoke({"category": category, "query": query})

    yield FunctionInfo.from_fn(_check, description=_check.__doc__)


class AnalyzeSentimentConfig(FunctionBaseConfig, name="analyze_sentiment"):
    llm: LLMRef


@register_function(config_type=AnalyzeSentimentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def analyze_sentiment_function(config: AnalyzeSentimentConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer experience analyst. Given the customer query, write an EXTREMELY "
         "DETAILED sentiment and intent analysis. You MUST cover ALL of the following sections "
         "in full, with multiple paragraphs each:\n\n"
         "1. EMOTIONAL TONE ANALYSIS: Identify every emotional indicator in the query text "
         "with direct quotes as evidence, classify primary and secondary emotions, rate "
         "intensity on a 1-10 scale with justification for the rating.\n"
         "2. URGENCY ASSESSMENT: Evaluate time-sensitivity signals, business impact indicators, "
         "and customer dependency factors with a detailed urgency score breakdown.\n"
         "3. FRUSTRATION INDICATORS: Catalog all frustration signals (repeated contacts, "
         "strong language, escalation threats, social media mentions) with severity ratings.\n"
         "4. CHURN RISK EVALUATION: Calculate churn probability based on sentiment signals, "
         "account tenure, usage patterns, and competitive landscape factors.\n"
         "5. CUSTOMER LIFETIME VALUE: Estimate CLV impact of this interaction with revenue "
         "projections, retention cost analysis, and referral network implications.\n"
         "6. RESPONSE STRATEGY: Prescribe detailed tone guidelines, empathy phrases to use, "
         "topics to avoid, and specific language patterns matched to this customer's style.\n"
         "7. HISTORICAL PATTERN ANALYSIS: Compare to at least 5 similar past interactions "
         "with outcomes, satisfaction scores, and lessons learned.\n"
         "8. DE-ESCALATION PLAYBOOK: Provide a step-by-step de-escalation plan with scripts, "
         "fallback positions, and executive escalation triggers.\n\n"
         "Be EXTREMELY verbose. Write at least 800 words. This shapes response quality."),
        ("human", "Category: {category}\nQuery: {query}"),
    ]) | llm | StrOutputParser())

    async def _analyze(input_text: str) -> str:
        """Analyze customer sentiment and intent for a support query."""
        parts = input_text.split("|", 1)
        category = parts[0].strip() if len(parts) > 1 else ""
        query = parts[-1].strip()
        return await chain.ainvoke({"category": category, "query": query})

    yield FunctionInfo.from_fn(_analyze, description=_analyze.__doc__)


# --- MEDIUM priority: moderate output ---


class DraftResponseConfig(FunctionBaseConfig, name="draft_response"):
    llm: LLMRef


@register_function(config_type=DraftResponseConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def draft_response_function(config: DraftResponseConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support agent. Using ALL of the research context, company "
         "policy, compliance notes, and sentiment analysis provided, draft a helpful "
         "response to the customer query. Be professional, empathetic, and actionable. "
         "Address all aspects of the customer's concern."),
        ("human",
         "Query: {query}\nCategory: {category}\nContext: {context}\nPolicy: {policy}\n"
         "Compliance: {compliance}\nSentiment: {sentiment}"),
    ]) | llm | StrOutputParser())

    async def _draft(input_text: str) -> str:
        """Draft a support response using context, policy, compliance, and sentiment."""
        parts = input_text.split("|")
        query = parts[0].strip() if len(parts) > 0 else ""
        category = parts[1].strip() if len(parts) > 1 else ""
        context = parts[2].strip() if len(parts) > 2 else ""
        policy = parts[3].strip() if len(parts) > 3 else ""
        compliance = parts[4].strip() if len(parts) > 4 else ""
        sentiment = parts[5].strip() if len(parts) > 5 else ""
        return await chain.ainvoke({
            "query": query,
            "category": category,
            "context": context,
            "policy": policy,
            "compliance": compliance,
            "sentiment": sentiment,
        })

    yield FunctionInfo.from_fn(_draft, description=_draft.__doc__)


# --- HIGH priority: short output ---


class ReviewConfig(FunctionBaseConfig, name="review_response"):
    llm: LLMRef


@register_function(config_type=ReviewConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def review_response_function(config: ReviewConfig, builder: Builder):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    chain = (ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior support QA reviewer. Review the draft response for accuracy, "
         "policy compliance, and appropriate tone. Respond with ONLY one of:\n"
         "  APPROVED - <one sentence summary of why>\n"
         "  REJECTED - <one sentence explanation of the issue>\n"
         "Do not rewrite the response. Just approve or reject with a brief reason."),
        ("human", "Original query: {query}\nDraft response: {draft}"),
    ]) | llm | StrOutputParser())

    async def _review(input_text: str) -> str:
        """Review and approve/reject a draft support response."""
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
    compliance: str
    sentiment: str
    draft: str
    final_response: str


class LatencySensitivityDemoConfig(FunctionBaseConfig, name="latency_sensitivity_demo"):
    """Configuration for the latency sensitivity demo workflow."""

    classify_fn: FunctionRef = Field(default=FunctionRef("classify_query"), description="Function to classify queries")
    research_fn: FunctionRef = Field(default=FunctionRef("research_context"),
                                     description="Function to research context")
    policy_fn: FunctionRef = Field(default=FunctionRef("lookup_policy"), description="Function to look up policy")
    compliance_fn: FunctionRef = Field(default=FunctionRef("check_compliance"),
                                       description="Function to check compliance")
    sentiment_fn: FunctionRef = Field(default=FunctionRef("analyze_sentiment"),
                                      description="Function to analyze sentiment")
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
    compliance_fn = await builder.get_function(config.compliance_fn)
    sentiment_fn = await builder.get_function(config.sentiment_fn)
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

    async def check_compliance(state: SupportState) -> dict:
        compliance = await compliance_fn.ainvoke(f"{state['category']}|{state['query']}")
        return {"compliance": str(compliance)}

    async def analyze_sentiment(state: SupportState) -> dict:
        sentiment = await sentiment_fn.ainvoke(f"{state['category']}|{state['query']}")
        return {"sentiment": str(sentiment)}

    async def draft_response(state: SupportState) -> dict:
        draft = await draft_fn.ainvoke(f"{state['query']}|{state['category']}|{state['context']}|"
                                       f"{state['policy']}|{state['compliance']}|{state['sentiment']}")
        return {"draft": str(draft)}

    async def review(state: SupportState) -> dict:
        final = await review_fn.ainvoke(f"{state['query']}|{state['draft']}")
        return {"final_response": str(final)}

    # ── Build the graph ──────────────────────────────────────────────────

    graph = StateGraph(SupportState)

    graph.add_node("classify", classify)
    graph.add_node("research_context", research_context)
    graph.add_node("lookup_policy", lookup_policy)
    graph.add_node("check_compliance", check_compliance)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("draft_response", draft_response)
    graph.add_node("review", review)

    graph.set_entry_point("classify")

    # Parallel fan-out: 4 LOW-priority branches
    graph.add_edge("classify", "research_context")
    graph.add_edge("classify", "lookup_policy")
    graph.add_edge("classify", "check_compliance")
    graph.add_edge("classify", "analyze_sentiment")

    # Converge: all 4 branches feed into draft
    graph.add_edge("research_context", "draft_response")
    graph.add_edge("lookup_policy", "draft_response")
    graph.add_edge("check_compliance", "draft_response")
    graph.add_edge("analyze_sentiment", "draft_response")

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
            "compliance": "",
            "sentiment": "",
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
