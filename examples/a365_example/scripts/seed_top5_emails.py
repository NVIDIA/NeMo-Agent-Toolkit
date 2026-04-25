# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Seeds synthetic Top 5 emails into a Microsoft 365 mailbox via Graph API.
# Uses real NVIDIA project names and people names, but fake content.
#
# Usage:
#   export GRAPH_MAIL_TOKEN="$(az account get-access-token --resource https://graph.microsoft.com --query accessToken -o tsv)"
#   uv run python scripts/seed_top5_emails.py

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import urllib.request
import urllib.error

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

EMAILS = [
    {
        "subject": "Top 5 Things — NemoClaw Engineering",
        "from_name": "Aaron Erickson",
        "from_address": "aerickson@nvidia.com",
        "to": ["agent-lab@nvidia.com", "nemoclaw@nvidia.com", "openshell@nvidia.com"],
        "cc": ["Kris Murphy <krism@nvidia.com>", "Bartley Richardson <brichardson@nvidia.com>"],
        "received_days_ago": 1,
        "body": """Mission: Ship a safe, reliable reference stack for running OpenClaw agents in OpenShell sandboxes.

Top 5 Things

1. NemoClaw 0.9.1 released — 12 security fixes, improved plugin isolation, MCP tool calling stability improved 40%.
2. OpenShell sandbox hardening complete — gateway TLS issue (#888) resolved, all P0 security items closed.
3. NAT integration deepened — NemoClaw now ships with a default NAT ReAct agent config out of the box.
4. Sprint 3 velocity strong — 280 PRs merged, 190 issues closed, 62 contributors with merged code.
5. Partner onboarding — 3 new enterprise partners evaluating NemoClaw for internal agentic workflows.

Details

NemoClaw 0.9.1
Fixed shell injection vector in plugin execution path. MCP streamable-http client now retries on 429.
NAT integration allows customers to swap the LLM backend without changing agent logic.

OpenShell Security
All 7 inference providers validated against hardened sandbox. Gateway TLS now regenerates correctly on restart.
Remaining P1: rate limiting on tool call endpoints — sprint 4.

Future Plans
GA release targeting May 29. Focus: documentation, enterprise onboarding guide, NemoClaw x NAT blueprint.

Aaron Erickson
Engineering Lead, NemoClaw
aerickson@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — Omniverse x AI Product Management",
        "from_name": "Alex Qi",
        "from_address": "lingq@nvidia.com",
        "to": ["omniverse-product-management@nvidia.com", "agent-lab@nvidia.com", "omniverse-interest@nvidia.com"],
        "cc": ["Damien Fagnou <dfagnou@nvidia.com>", "Bartley Richardson <brichardson@nvidia.com>", "Rev Lebaredian <revl@nvidia.com>"],
        "received_days_ago": 2,
        "body": """Mission: Bring Agentic AI to Omniverse to advance Physical AI workflows.

Top 5 Things

1. Isaac Sim MCP server shipped — developers can now control Isaac Sim via natural language through Claude Code or Cursor.
2. Material Assignment Agent 0.4 — 15% accuracy improvement on GB200 assembly guide assets, Siemens evaluation started.
3. NemoClaw x Omniverse robotics workflow — task-driven environment generation for policy training, GTC demo recorded.
4. Vibe Code Club — 110 members, weekly sessions driving internal adoption of OV MCPs and agent-ready software.
5. USD Search 1.4 — SigLIP2 model upgrade, 3x search quality improvement, Lightwheel AI integration live.

Details

Isaac Sim MCP
Partners can now issue natural language commands to Isaac Sim via MCP. Early feedback: "reduced onboarding time from 2 days to 2 hours."
Next: Isaac Lab skills shipping alongside Isaac Sim 6.1.

Material Assignment Agent
Large-scene pipeline now handles scenes with 10k+ components. Siemens reviewing for factory digital twin use case.
Accuracy on interior components still needs work — adding more ground truth data this sprint.

Future Plans
Agent-ready software day-0 for all OV libraries at next major release.
MAA public beta — targeting end of month.

Alex Qi
Product Manager, Omniverse x AI
lingq@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — Nemotron | NALA | SA Manager",
        "from_name": "Seth Henneman",
        "from_address": "shenneman@nvidia.com",
        "to": ["Sales T5T-Core@nvidia.com", "nemotron-ecosystem-core@nvidia.com"],
        "cc": ["Jane Polak Scowcroft <jpolak@nvidia.com>", "Robert Clark <rclark@nvidia.com>", "Jiantao Jiao <jjiao@nvidia.com>"],
        "received_days_ago": 3,
        "body": """Mission: Accelerate Nemotron ecosystem integration.

Industry Business Development / Account Updates

CustomerEval Bench now tracking 18 active LHA customers. Eval automation running on Slurm via NEL — fully automated, no manual overhead.
BLADE benchmark results: LLMs automating 34% of eval failure analyses that were previously manual. Team targeting 50% by GA.
Nemotron Ultra post-training complete — 20T tokens, 1M context, Hybrid-MoE. SFT eval mobilized across 22 benchmark categories on 512 GB200.
Nemotron Nano Omni EA — 28 partners now evaluating. H-company shared first post-training results: strong gains on tool calling.
Nemotron Super 49B v1.5 deployed at 4 CSPs — performance benchmarks ahead of schedule.

Future Plans

Nemotron Ultra GA — continue mobilizing SA eval support. (@Robert Clark)
Nano Omni EA/GA — incorporate partner feedback, target April 28 GTM launch. (@Aastha Jhunjhunwala)
Support "Build as LHA" initiative for Eval Bench internal data. (@Tom Balough)

Team Mission: Accelerate Nemotron ecosystem integration

Seth Henneman
Solutions Architect Manager
shenneman@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — Retail/CPG/QSR | Global | TME",
        "from_name": "Antonio Martinez Torres",
        "from_address": "anmartinez@nvidia.com",
        "to": ["Sales T5T-Core@nvidia.com", "agent-lab@nvidia.com", "agent-skills@nvidia.com"],
        "cc": ["Logan Vadivelu <lvadivelu@nvidia.com>", "Azita Martin <azitam@nvidia.com>"],
        "received_days_ago": 4,
        "body": """Mission: Drive NVIDIA AI adoption in Retail, CPG, and QSR verticals.

Insights, Market & Competition

Gemma 4 31B benchmarks strong — parity with Claude Sonnet on several retail task benchmarks. Testing on NIM now.
Retail Agentic Commerce Blueprint gaining traction — 3 new GSI partners requesting integration support this week.
Perplexity Computer adoption signal: enterprise retail teams asking about similar "skills" architecture for internal agents.

Industry / Account Updates

Lowe's — deep-dive session on NemoClaw for store operations automation. Pilot scoped for Q3.
Grid Dynamics — live demo at GTC using NAT + NIM for promotion optimization. Customer impressed, expanding engagement.
Walmart — GB200 GPU efficiency questions resolved. Agentic commerce blueprint demo scheduled for April 22.
NVSimGym — synthetic buyer agent for A/B testing now running on Nemotron. Conversion rate prediction improving week over week.

Future Plans

Define GTM strategy with Google for Agentic Commerce blueprint.
Present NVSimGym findings to broader NVIDIA simulation team.
Lowe's NemoClaw production pilot — kick off Q3 planning.

Antonio Martinez Torres
Technical Marketing Engineer, Retail/CPG/QSR
anmartinez@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — DeepStream + Accelerated Microservices CI Agent",
        "from_name": "Unnikrishnan Kizhakkemadam",
        "from_address": "ukizhakkemadam@nvidia.com",
        "to": ["agent-lab@nvidia.com", "openshell@nvidia.com", "brichardson-org@nvidia.com"],
        "cc": ["Meryl Mathew <mmathew@nvidia.com>", "Camille Huang <cahuang@nvidia.com>"],
        "received_days_ago": 5,
        "body": """Mission: Automate build, CI, and dev productivity workflows for DeepStreamSDK and Accelerated Microservices using NAT.

Executive Summary

CI Agent v2 shipped — now monitoring 14 projects (up from 10), ARB reports now include week-over-week regression trend charts.
Auto-triage in beta — agent identifies root cause for 60% of CI failures without human intervention.
DeepStreamSDK Build Agent — release pipeline automation complete, saving ~6 hours per release cycle.
NOVA Guardrails — now active across all 22 team repos, blocking secrets and license violations pre-commit.
Skills Marketplace — 8 skills upstreamed to NVCARPS, available org-wide.

Details

CI Agent v2
New projects added: rtvi-audio, rtvi-3d, orin-l4t-mm-v2.
ARB emails now include Jenkins job links, per-project owner assignments, and trend sparklines.
Agent interface: http://10.111.53.164:8010/

Auto-Triage (Beta)
Root cause analysis using log embeddings + NAT ReAct agent. Patches suggested for 23% of failures.
Next: integrate Coding Agent for auto-patch submission.

Future Plans

Deploy agents in OpenShell for broader org access.
Integrate auto-patch workflow with gerrit for seamless CI fix loop.

Unnikrishnan Kizhakkemadam
Senior Engineer, DeepStream CI Automation
ukizhakkemadam@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — NVIDIA Blueprint Architecture and Benchmarking",
        "from_name": "Juana Nakfour",
        "from_address": "jnakfour@nvidia.com",
        "to": ["agent-lab@nvidia.com", "omniverse-interest@nvidia.com", "DevMktg-Top5s@nvidia.com"],
        "cc": ["Bartley Richardson <brichardson@nvidia.com>", "Kris Murphy <krism@nvidia.com>"],
        "received_days_ago": 6,
        "body": """Mission: Drive NVIDIA Blueprints Architecture and Benchmarking.

Top 5

1. Blueprints + Agentic AI playbook — new section published on integrating Blueprints with NAT, NemoClaw, and MCP. First draft reviewed and approved.
2. Autonomous benchmarking pipeline — PoC complete using NemoClaw + Claude Code + GitLab CI. Next: integrate with NemoClaw.
3. VSS Alert Verification Benchmark — Phase 1 design complete, initial latency benchmarks running on K8s.
4. RAG 2.5.0 benchmarks — memory leak fix unblocked P1+ runs. First full benchmark suite complete.
5. NIM-d integration — TTFT improvements validated on Blueprint RAG workload, results shared with Andy Henroid.

Details

Blueprint Playbook
SKILL.md and MCP tool integration patterns now documented. Partners can follow the playbook to make any Blueprint agent-ready in under a day.

Autonomous Benchmarking
OpenClaw triggers GitLab runner, Claude Code interprets results, files issues automatically. Saved 4 hours per benchmark cycle in PoC.

Future Plans
Full autonomous benchmarking pipeline with NemoClaw integration — targeting next sprint.
VSS Phase 2: Alert Bridge throughput testing.

Juana Nakfour
Solutions Architect, Blueprint Architecture
jnakfour@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — NeMo Agent Toolkit (NAT)",
        "from_name": "Bartley Richardson",
        "from_address": "brichardson@nvidia.com",
        "to": ["agent-lab@nvidia.com", "brichardson-org@nvidia.com", "nemotron-ecosystem-core@nvidia.com"],
        "cc": ["Kris Murphy <krism@nvidia.com>", "Michael Demoret <mdemoret@nvidia.com>", "Alex Fournier <afournier@nvidia.com>"],
        "received_days_ago": 7,
        "body": """Mission: Make NAT the standard runtime for NVIDIA agentic AI across internal and external deployments.

Top 5

1. NAT + Microsoft Agent 365 integration — e2e demo working: Teams front-end → NAT ReAct agent → Microsoft-hosted MCP servers (Planner, Task Personalization). Strong signal for enterprise agentic workflows.
2. NAT 2.1 released — new MCP client supports streamable-http with session-aware tools, reconnect logic, and per-server auth.
3. NemoClaw x NAT — NemoClaw now ships a default NAT config. Omniverse DSX blueprint migrated to NAT 2.0 sandboxing.
4. Observability — A365 telemetry exporter shipping in 2.1, traces visible in Microsoft's agent monitoring dashboard.
5. Partner momentum — 6 external teams now using NAT in production or active evaluation (DeepStream CI, Omniverse DSX, Retail blueprint, NemoClaw, Project Nemo Synapse, Blueprint benchmarking).

Details

A365 Integration
NAT connects to Microsoft's Agent 365 ecosystem, authenticates via Entra, auto-discovers Microsoft-hosted MCP servers at startup.
Planner tools (CreatePlan, CreateTask, QueryPlans, etc.) called successfully from Teams conversation.
Next: Graph API MCP server for email/SharePoint access.

NAT 2.1
Breaking change: MCP client config schema updated — see migration guide.
New: `a365_mcp_tooling` function group type for zero-config A365 tool discovery.

Future Plans
Graph Mail MCP server — query org emails from Teams agent.
POR → JIRA ticket extraction agent demo.
NAT on Azure Marketplace — packaging in progress.

Bartley Richardson
Engineering Director, NeMo Agent Toolkit
brichardson@nvidia.com | NVIDIA""",
    },
    {
        "subject": "Top 5 Things — Strategic Research Collaboration | NVAITC",
        "from_name": "Chi-Cheng Fu",
        "from_address": "ccfu@nvidia.com",
        "to": ["agent-lab@nvidia.com", "omniverse-interest@nvidia.com"],
        "cc": ["Johnson Sun <johnsons@nvidia.com>", "Tomasz Bednarz <tbednarz@nvidia.com>"],
        "received_days_ago": 8,
        "body": """Mission: Influence top researchers to use NVIDIA software technologies for scientific breakthroughs.

Top Researchers Engaged

Wei-Chang Yeh, NTHU — AMR + simulation. Isaac Sim and Isaac Lab deployments progressing. Co-authored paper on reliability-focused robot navigation submitted to IEEE.
Chi-Kuang Sun, NTU — AI in cancer pathology. MONAI deployment live, cuCIM in evaluation. GTC poster finalist.
Hung-yi Lee, NTU — LLM/speech. NeMo and Megatron adoption accelerating. AGP awarded for spoken language model research.
Ching-Ray Chang, NTU — Quantum computing. CUDA-Q deployed, joint social post on GPU-accelerated quantum stochastic walks live.
Albert Yang, NYCU — Robotic surgery. Isaac Sim and MONAI in development. 4x RTX PRO 6000 setup complete.

Priorities

TRDC Phase 8 — 18 AGP projects across top Taiwan institutions, 31 GPUs and 95K GPU hours allocated.
Academia Sinica MOU — advanced collaboration on quantum AI agents, joint publications planned.
NTU School of Pharmacy — AI-driven traditional medicine identification system, NVIDIA SDKs deployed.

Future Plans

Publish 3 co-authored research papers Q2.
Expand CUDA-Q education programs to 5 additional institutions.
Support NeMo Evaluator adoption for automated benchmark runs at partner labs.

Chi-Cheng Fu, PhD
Strategic Research Collaboration, NVAITC
ccfu@nvidia.com | NVIDIA""",
    },
]


def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _graph_post(path: str, body: dict, token: str) -> dict:
    url = f"{GRAPH_BASE}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=_auth_headers(token), method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        raise RuntimeError(f"Graph API error {e.code}: {error_body}") from e


def build_message(email: dict) -> dict:
    received = datetime.now(timezone.utc) - timedelta(days=email["received_days_ago"])

    to_recipients = [
        {"emailAddress": {"address": addr if "@" in addr else f"{addr}@nvidia.com"}}
        for addr in email["to"]
    ]

    cc_recipients = []
    for cc in email.get("cc", []):
        if "<" in cc:
            name, addr = cc.split("<")
            addr = addr.strip().rstrip(">")
            cc_recipients.append({"emailAddress": {"name": name.strip(), "address": addr}})
        else:
            cc_recipients.append({"emailAddress": {"address": cc}})

    return {
        "subject": email["subject"],
        "body": {
            "contentType": "Text",
            "content": email["body"],
        },
        "toRecipients": to_recipients,
        "ccRecipients": cc_recipients,
        "from": {
            "emailAddress": {
                "name": email["from_name"],
                "address": email["from_address"],
            }
        },
        "receivedDateTime": received.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "isRead": True,
    }


def main():
    token = os.environ.get("GRAPH_MAIL_TOKEN") or os.environ.get("GRAPH_TOKEN", "")
    if not token:
        print("Set GRAPH_MAIL_TOKEN or GRAPH_TOKEN to a Graph API bearer token.")
        print("  export GRAPH_MAIL_TOKEN=\"$(az account get-access-token --resource https://graph.microsoft.com --query accessToken -o tsv)\"")
        sys.exit(1)

    print(f"Seeding {len(EMAILS)} Top 5 emails into mailbox...")
    for i, email_def in enumerate(EMAILS, 1):
        msg = build_message(email_def)
        try:
            result = _graph_post("/me/messages", msg, token)
            print(f"  [{i}/{len(EMAILS)}] Created: {email_def['subject'][:60]} (id: {result.get('id', '?')[:20]}...)")
        except Exception as e:
            print(f"  [{i}/{len(EMAILS)}] FAILED: {email_def['subject'][:60]}: {e}")

    print("\nDone. Check your mailbox or ask the agent: 'list my Top 5 emails from the last 14 days'")


if __name__ == "__main__":
    main()
