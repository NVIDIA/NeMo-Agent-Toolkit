# prompts.py
from typing import Dict, List
from langchain.prompts import PromptTemplate

# Default config-like dictionaries (can be overridden by your workflow YAML/loader)
DEFAULT_TOOL_INTENTS: Dict[str, Dict[str, List[str]]] = {
    "Fee Inquiry": {
        "description": "Customer asks about, disputes, or shows curiosity about fees charged to their account, or future fees.",
        "data_intents": ["account_id", "transaction_date"],
    },
    "Lending Offer Inquiry": {
        "description": "Customer asks about Credit Cards or Personal Loans, or shows interest in offers available.",
        "data_intents": ["customer_name", "account_id", "product_type"],
    },
    "Knowledge Base Inquiry": {
        "description": "Customer has a general or specialized question about the Royal Bank of Canada, not related to other intents.",
        "data_intents": ["customer_name", "account_id"],
    },
}

DEFAULT_DATA_INTENTS: Dict[str, str] = {
    "account_id": "The customer's unique bank account ID or number.",
    "transaction_date": "Date of the transaction in question.",
    "customer_name": "The customer's full legal name.",
    "product_type": "The type of lending product (Credit Card, Personal Loan, etc.).",
}


def construct_prompt(
    tool_intents: Dict[str, Dict[str, List[str]]],
    data_intents: Dict[str, str],
    transcript: List[str]
) -> str:
    # Tool intents section
    tool_lines: List[str] = []
    for tool_name, meta in tool_intents.items():
        tool_lines.append(f"{tool_name}\n  Trigger: {meta['description']}")

    # Data intents section
    data_lines: List[str] = []
    for di, desc in data_intents.items():
        data_lines.append(f"{di}: {desc}")

    # Tool -> Data links
    links_lines: List[str] = []
    for tool_name, meta in tool_intents.items():
        required = ", ".join(meta.get("data_intents", [])) or "(none)"
        links_lines.append(f"{tool_name} -> requires: {required}")

    # Core instruction text aligned with NeMo Agent Toolkit’s workflow-driven consumption:
    # - JSON-only
    # - Trigger-once semantics
    # - Multi-tool to multi-data aggregation in the same step
    instructions = f"""
You are an Intent Classification AI for a conversation transcript that arrives in small streamed chunks.

GOALS:
1) Detect any NEW tool intents from the latest conversation (based on the definitions below).
2) Consider ALL ACTIVE tool intents (previously triggered OR newly triggered this step).
3) Aggregate all REQUIRED data intents for all ACTIVE tools, deduplicate them, and include ALL not-yet-triggered data intents in the SAME response.
4) One data intent (e.g., "account_id") may be required by multiple tools; include it ONCE if it is newly required.
5) Do not re-trigger any tool or data intent that has already been triggered earlier.
6) Use the entire transcript for context, even if specific content arrived in earlier chunks.

---
TOOL INTENTS:
{"\n\n".join(tool_lines)}

---
DATA INTENTS:
{"\n\n".join(data_lines)}

---
TOOL → DATA LINKS:
{"\n\n".join(links_lines)}

---
Current full transcript:
f{"\n".join(transcript)}

Already triggered tool intents: {{triggered_tool_intents}}
Already triggered data intents: {{triggered_data_intents}}

RULES:
- Always return JSON ONLY with exactly these keys:
  {{
    "tool_intents_triggered": [],
    "data_intents_triggered": []
  }}
- "tool_intents_triggered": list all NEW tool intents detected in this step.
- "data_intents_triggered": list ALL newly required data intents for ANY active tools that are NOT already triggered.
- If nothing new, return both lists as [].
- No prose, no explanations, JSON only.
""".strip()
    return instructions


def get_intent_prompt_template(
    tool_intents: Dict[str, Dict[str, List[str]]] = None,
    data_intents: Dict[str, str] = None,
) -> PromptTemplate:
    # Allow runtime override from workflow config (alert_triage_agent-style)
    tool_intents = tool_intents or DEFAULT_TOOL_INTENTS
    data_intents = data_intents or DEFAULT_DATA_INTENTS

    template_text = construct_prompt(tool_intents, data_intents)

    # Use LangChain PromptTemplate API expected by many frameworks
    return PromptTemplate(
        input_variables=[
            "transcript",
            "triggered_tool_intents",
            "triggered_data_intents",
        ],
        template=template_text,
    )
