<!--
Copyright (c) 2025 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# System Architecture

This document provides detailed architecture diagrams for the React Benchmark Agent evaluation system,
specifically details around integration of the agent with LLM inference on a NVIDIA Dynamo server.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Interaction Flow](#component-interaction-flow)
3. [Self-Evaluation Loop](#self-evaluation-loop)
4. [Dynamo Backend Architecture](#dynamo-backend-architecture)
5. [Metrics Calculation](#metrics-calculation)
6. [File Structure](#file-structure)

---

## System Overview

```text
╔═══════════════════════════════════════════════════════════════════════════════╗
║           AGENT LEADERBOARD V2 DECISION-ONLY EVALUATION SYSTEM                ║
║                                                                               ║
║  "Evaluate tool-selection decisions without executing banking operations"     ║
╚═══════════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA INGESTION LAYER                             │
└─────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────┐         ┌──────────────────────────┐
        │  Hugging Face Dataset   │         │   Preprocessed JSON      │
        │  ─────────────────────  │         │  ──────────────────────  │
        │  galileo-ai/            │         │  agent_leaderboard_v2_   │
        │  agent-leaderboard-v2   │         │  banking.json            │
        │                         │         │                          │
        │  • Raw HF dataset       │         │  • 100 banking scenarios │
        │  • download script      │         │  • expected_tool_calls   │
        │  • preprocessing        │         │  • Full tool schemas     │
        └────────────┬────────────┘         └─────────────┬────────────┘
                     │                                    │
                     └────────────────┬───────────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │   Dataset Entry         │
                        │  ─────────────────────  │
                        │  • question             │
                        │  • user_goals           │
                        │  • available_tools      │
                        │  • expected_tool_calls  │
                        │  • metadata             │
                        └────────────┬────────────┘
                                     │
                                     ▼


┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOOL STUB REGISTRATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────────┐
                        │  BankingToolsGroup     │
                        │  ───────────────────── │
                        │  • Reads tools.json    │
                        │  • Creates 20 stubs    │
                        │  • decision_only: true │
                        └──────────┬─────────────┘
                                   │
                                   ├──────────────────────────────┐
                                   │                              │
                                   ▼                              ▼
            ┌──────────────────────────────┐      ┌──────────────────────────┐
            │  create_tool_stub_function() │      │   ToolIntentBuffer       │
            │  ──────────────────────────  │      │  ──────────────────────  │
            │  • Reads tool schema         │◄─────┤  • Shared buffer         │
            │  • Creates async stub        │      │  • Records intents       │
            │  • Returns mock response     │      │  • No real execution     │
            └──────────────┬───────────────┘      └──────────────────────────┘
                           │
                           ▼
                    ┌──────────────────────────────────────────────────────┐
                    │                  20 Banking Tool Stubs               │
                    │  ─────────────────────────────────────────────────   │
                    │  get_account_balance()     → "Mock: Balance $1000"   │
                    │  transfer_funds()          → "Mock: Transfer OK"     │
                    │  get_transaction_history() → "Mock: 5 transactions"  │
                    │  get_loan_information()    → "Mock: Loan details"    │
                    │  report_lost_stolen_card() → "Mock: Card blocked"    │
                    │  ... 15 more banking tools                           │
                    └──────────────────────────┬───────────────────────────┘
                                               │
                                               ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENT EXECUTION LAYER                              │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────────────────────────┐
                        │        Self-Evaluating Agent           │
                        │  ────────────────────────────────────  │
                        │  • Wraps ReAct workflow                │
                        │  • Evaluates tool sequence             │
                        │  • Retries if insufficient             │
                        │  • Passes feedback on retry            │
                        └──────────────────┬─────────────────────┘
                                           │
                                           ▼
                        ┌────────────────────────────────────────┐
                        │           ReAct Agent                  │
                        │  ────────────────────────────────────  │
                        │  • LLM: Llama-3.3-70b (Dynamo)         │
                        │  • 20 banking tools available          │
                        │  • Thought → Action → Observation      │
                        └──────────────────┬─────────────────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
                 ▼                                                   ▼
    ┌───────────────────────┐                         ┌───────────────────────┐
    │  Thought: I need to   │                         │  Thought: Now I'll    │
    │  check the balance... │                         │  transfer the funds...│
    └───────────┬───────────┘                         └───────────┬───────────┘
                │                                                 │
                ▼                                                 ▼
    ┌───────────────────────┐                         ┌───────────────────────┐
    │  Action:              │                         │  Action:              │
    │  get_account_balance  │                         │  transfer_funds       │
    │  ─────────────────    │                         │  ────────────────     │
    │  {                    │                         │  {                    │
    │    account: "12345"   │                         │    from: "12345",     │
    │  }                    │                         │    to: "67890",       │
    └───────────┬───────────┘                         │    amount: 500        │
                │                                     │  }                    │
                │ CAPTURED!                           └───────────┬───────────┘
                ▼                                                 │ CAPTURED!
    ┌───────────────────────┐                                     ▼
    │  ToolIntentBuffer     │                         ┌───────────────────────┐
    │  ───────────────────  │                         │  ToolIntentBuffer     │
    │  intents = [          │                         │  intents = [          │
    │    {                  │                         │    {...},             │
    │      tool: "get_acc.."│                         │    {                  │
    │      params: {...}    │                         │      tool: "transfer" │
    │    }                  │                         │      params: {...}    │
    │  ]                    │                         │    }                  │
    └───────────┬───────────┘                         │  ]                    │
                │                                     └───────────┬───────────┘
                ▼                                                 │
    ┌───────────────────────┐                                     ▼
    │  Observation:         │                         ┌───────────────────────┐
    │  "Mock: Balance is    │                         │  Observation:         │
    │   $1000"              │                         │  "Mock: Transfer      │
    └───────────────────────┘                         │   successful"         │
                                                      └───────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Final Response      │
                        │  ─────────────────── │
                        │  "I've checked your  │
                        │  balance ($1000) and │
                        │  transferred $500"   │
                        └──────────┬───────────┘
                                   │
                                   ▼


┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION LAYER                                   │
└─────────────────────────────────────────────────────────────────────────────┘

        ┌───────────────────────────────────────────────────────┐
        │                   TSQ EVALUATOR                       │
        │  ─────────────────────────────────────────────────────│
        │  1. Extract actual tool calls from intent buffer      │
        │  2. Get expected tool calls from dataset              │
        │  3. Normalize tool names (strip prefixes)             │
        │  4. Calculate F1 score (precision × recall)           │
        └───────────────────────────────────────────────────────┘

    ┌─────────────────────┐              ┌──────────────────────┐
    │  Actual Tool Calls  │              │  Expected Tool Calls │
    │  ─────────────────  │              │  ──────────────────  │
    │  From intent buffer:│              │  From dataset:       │
    │  • get_account_bal  │              │  • get_account_bal   │
    │  • transfer_funds   │              │  • transfer_funds    │
    │  • get_transaction  │              │  • verify_transfer   │
    │                     │              │  • get_transaction   │
    └──────────┬──────────┘              └──────────┬───────────┘
               │                                    │
               └────────────────┬───────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │  TSQ Calculation    │
                    │  ────────────────   │
                    │  Intersection: 3    │
                    │  Actual: 3          │
                    │  Expected: 4        │
                    │                     │
                    │  Precision: 3/3=1.0 │
                    │  Recall: 3/4=0.75   │
                    │  F1: 0.857          │
                    └──────────┬──────────┘
                               │
                               ▼


┌─────────────────────────────────────────────────────────────────────────────┐
│                            RESULTS LAYER                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────────────────────────────┐
                        │            Output Directory                │
                        │  outputs/dynamo_evals/<job_id>/            │
                        │  ────────────────────────────────────────  │
                        │                                            │
                        │  tool_selection_quality_output.json        │
                        │  ──────────────────────────────────        │
                        │  • Average TSQ: 0.XYZ                      │
                        │  • Per-scenario scores                     │
                        │  • Actual vs expected tools                │
                        │                                            │
                        │  standardized_data_all.csv                 │
                        │  ──────────────────────────────────        │
                        │  • Token counts                            │
                        │  • Timestamps                              │
                        │  • LLM call metadata                       │
                        │                                            │
                        │  workflow_profiling_report.txt             │
                        │  ──────────────────────────────────        │
                        │  • Bottleneck analysis                     │
                        │  • Concurrency statistics                  │
                        └────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                              KEY FEATURES                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ✓ DECISION-ONLY MODE: Tools captured, not executed                           ║
║  ✓ SELF-EVALUATION: Agent can retry with feedback                             ║
║  ✓ DYNAMIC TOOLS: 20 banking tools from JSON schema                           ║
║  ✓ TSQ METRICS: F1 score for tool selection quality                           ║
║  ✓ DYNAMO BACKEND: High-performance LLM inference                             ║
║  ✓ PREFIX HEADERS: KV cache optimization with Thompson Sampling               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Component Interaction Flow

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INTERACTIONS                               │
└──────────────────────────────────────────────────────────────────────────────┘

  NAT Eval        Dataset        Tool Stubs       Agent         Evaluator
  ────────        ───────        ──────────       ─────         ─────────
     │               │               │              │               │
     │  1. Load      │               │              │               │
     │    Config     │               │              │               │
     ├───────────────►               │              │               │
     │               │               │              │               │
     │  2. Load      │               │              │               │
     │    Dataset    │               │              │               │
     ├───────────────►               │              │               │
     │               │               │              │               │
     │  3. Register  │               │              │               │
     │    Tools      │               │              │               │
     ├───────────────────────────────►              │               │
     │               │               │              │               │
     │  4. For each scenario:        │              │               │
     │  ──────────────────────────────────────────────────────────  │
     │  │            │               │              │               │
     │  │ 5. Start   │               │              │               │
     │  │    Eval    │               │              │               │
     │  ├────────────────────────────────────────────►              │
     │  │            │               │              │               │
     │  │            │               │ 6. Reason    │               │
     │  │            │               │◄─────────────┤               │
     │  │            │               │              │               │
     │  │            │               │ 7. Call Tool │               │
     │  │            │               │◄─────────────┤               │
     │  │            │               │              │               │
     │  │            │               │ 8. Record    │               │
     │  │            │               │    Intent    │               │
     │  │            │               ├──────┐       │               │
     │  │            │               │      │       │               │
     │  │            │               │◄─────┘       │               │
     │  │            │               │              │               │
     │  │            │               │ 9. Return    │               │
     │  │            │               │    Mock      │               │
     │  │            │               ├─────────────►│               │
     │  │            │               │              │               │
     │  │            │               │   (repeat 6-9 for each tool) │
     │  │            │               │              │               │
     │  │            │               │ 10. Final    │               │
     │  │            │               │     Response │               │
     │  │            │               │              ├───────────────►
     │  │            │               │              │               │
     │  │            │  11. Get      │              │               │
     │  │            │      Intents  │              │               │
     │  │            │               ◄──────────────────────────────┤
     │  │            │               │              │               │
     │  │ 12. Get Expected           │              │               │
     │  │     Tool Calls             │              │               │
     │  │◄───────────┤               │              │               │
     │  │            │               │              │               │
     │  │            │               │              │ 13. Calculate │
     │  │            │               │              │     TSQ       │
     │  │            │               │              │◄──────────────┤
     │  │            │               │              │               │
     │  │ 14. Store  │               │              │               │
     │  │     Result │               │              │               │
     │  ◄────────────────────────────────────────────────────────────
     │  │            │               │              │               │
     │  └────────────────────────────────────────────────────────── │
     │               │               │              │               │
     │ 15. Write     │               │              │               │
     │    Results    │               │              │               │
     ├───────────────►               │              │               │
     │               │               │              │               │
     └───────────────┴───────────────┴──────────────┴───────────────┘
```

---

## Self-Evaluation Loop

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                      SELF-EVALUATION LOOP ARCHITECTURE                       │
└──────────────────────────────────────────────────────────────────────────────┘

                              User Question
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    Self-Evaluating Agent      │
                    │    ───────────────────────    │
                    │    max_retries: 5             │
                    │    confidence_threshold: 0.85 │
                    │    pass_feedback: true        │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │       Clear Intent Buffer     │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │                                                       │
        │   ┌────────────────────────────────────────────────┐  │
        │   │              ATTEMPT 1                         │  │
        │   └────────────────────────────────────────────────┘  │
        │                          │                            │
        │                          ▼                            │
        │   ┌────────────────────────────────────────────────┐  │
        │   │           ReAct Agent Execution                │  │
        │   │   ────────────────────────────────────────     │  │
        │   │   Thought → Action → Observation (loop)        │  │
        │   │   Tools called: [get_balance, transfer]        │  │
        │   └─────────────────────┬──────────────────────────┘  │
        │                         │                             │
        │                         ▼                             │
        │   ┌────────────────────────────────────────────────┐  │
        │   │         Self-Evaluation (eval_llm)             │  │
        │   │   ────────────────────────────────────────     │  │
        │   │   Question: "Transfer $500 and verify"         │  │
        │   │   Tool calls: [get_balance, transfer]          │  │
        │   │                                                │  │
        │   │   Evaluation:                                  │  │
        │   │     is_sufficient: FALSE                       │  │
        │   │     confidence: 0.60                           │  │
        │   │     reasoning: "Missing verification"          │  │
        │   │     missing_steps: ["get_transaction_history"] │  │
        │   └─────────────────────┬──────────────────────────┘  │
        │                         │                             │
        │                         ▼                             │
        │            ┌────────────────────────────┐             │
        │            │  confidence < threshold?   │             │
        │            │      0.60 < 0.85 = YES     │             │
        │            └──────────────┬─────────────┘             │
        │                           │                           │
        │              ┌────────────▼────────────┐              │
        │              │   retries remaining?    │              │
        │              │      5 > 0 = YES        │              │
        │              └────────────┬────────────┘              │
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │   Generate Feedback Message   │
                    │   ─────────────────────────   │
                    │   "PREVIOUS ATTEMPT FEEDBACK: │
                    │    Missing verification step  │
                    │    Add: get_transaction..."   │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │       Clear Intent Buffer     │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────▼───────────────────────────┐
        │                                                       │
        │   ┌────────────────────────────────────────────────┐  │
        │   │              ATTEMPT 2 (with feedback)         │  │
        │   └────────────────────────────────────────────────┘  │
        │                          │                            │
        │                          ▼                            │
        │   ┌────────────────────────────────────────────────┐  │
        │   │           ReAct Agent Execution                │  │
        │   │   ────────────────────────────────────────     │  │
        │   │   Input: Question + Feedback message           │  │
        │   │   Tools called: [get_balance, transfer,        │  │
        │   │                  get_transaction_history]      │  │
        │   └─────────────────────┬──────────────────────────┘  │
        │                         │                             │
        │                         ▼                             │
        │   ┌────────────────────────────────────────────────┐  │
        │   │         Self-Evaluation (eval_llm)             │  │
        │   │   ────────────────────────────────────────     │  │
        │   │   Evaluation:                                  │  │
        │   │     is_sufficient: TRUE                        │  │
        │   │     confidence: 0.92                           │  │
        │   │     reasoning: "All steps complete"            │  │
        │   └─────────────────────┬──────────────────────────┘  │
        │                         │                             │
        │                         ▼                             │
        │            ┌────────────────────────────┐             │
        │            │  sufficient && confident?  │             │
        │            │  TRUE && 0.92 >= 0.85 = YES│             │
        │            └──────────────┬─────────────┘             │
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     ✓ ACCEPT RESULT           │
                    │     Return final response     │
                    │     TSQ: 0.857                │
                    └───────────────────────────────┘
```

---

## Dynamo Backend Architecture

For detailed Dynamo backend architecture including:
- Frontend, Processor, and Router components
- Unified vs Disaggregated worker modes
- Thompson Sampling router configuration
- Infrastructure services (ETCD, NATS)
- Dynamic prefix headers for KV cache optimization

**See: [Dynamo Setup Guide](../../external/dynamo/README.md#architecture-overview)**

---

## Metrics Calculation

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         TSQ CALCULATION                             │
└─────────────────────────────────────────────────────────────────────┘

    Actual Tools:    {get_account_balance, transfer_funds, get_transaction}
    Expected Tools:  {get_account_balance, transfer_funds, verify_transfer, notify}
                            │                    │
                            ▼                    ▼
                    ┌──────────────┐    ┌───────────────┐
                    │ Normalize    │    │ Normalize     │
                    │ Names        │    │ Names         │
                    └──────┬───────┘    └───────┬───────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────────────────────────────┐
                    │           Set Comparison             │
                    │  ─────────────────────────────────   │
                    │  Intersection: {get_account_balance, │
                    │                 transfer_funds}      │
                    │  Count: 2                            │
                    └──────────────────┬───────────────────┘
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                           ▼                       ▼
                  ┌─────────────────┐    ┌─────────────────┐
                  │   PRECISION     │    │   RECALL        │
                  │   ───────────   │    │   ────────────  │
                  │   Correct / Act │    │   Correct / Exp │
                  │   2 / 3 = 0.667 │    │   2 / 4 = 0.500 │
                  └────────┬────────┘    └────────┬────────┘
                           │                      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │      F1 SCORE        │
                           │  ────────────────    │
                           │  2 × (P × R)/(P + R) │
                           │  2 × (0.667 × 0.500) │
                           │  ─────────────────── │
                           │  (0.667 + 0.500)     │
                           │                      │
                           │  = 0.571             │
                           └──────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    THROUGHPUT METRICS                               │
└─────────────────────────────────────────────────────────────────────┘

    LLM Call Start ──┬──► First Token ──┬──► Token 2 ──┬──► ... ──► Last Token
                     │                  │              │                   │
                     │◄──── TTFT ──────►│              │                   │
                     │                  │◄─ ITL ──────►│                   │
                     │◄────────────── Total Latency ──────────────────────►│
                     │                                                     │
                     │                    Total Tokens                     │
                     │  Throughput = ─────────────────────────             │
                     │                   Total Latency                     │


    ┌────────────────────────────────────────────────────────────────────┐
    │                    METRIC DEFINITIONS                              │
    ├────────────────────────────────────────────────────────────────────┤
    │  TTFT (Time To First Token)                                        │
    │    Time from request start to first token received                 │
    │    Lower is better. Measures prompt processing time.               │
    │                                                                    │
    │  ITL (Inter-Token Latency) / TPOT (Time Per Output Token)          │
    │    Time between consecutive tokens                                 │
    │    Lower is better. Measures decode speed.                         │
    │                                                                    │
    │  Per-Request Throughput                                            │
    │    tokens_in_request / request_duration                            │
    │    Higher is better. Per-call efficiency.                          │
    │                                                                    │
    │  Aggregate Throughput                                              │
    │    total_tokens / wall_clock_time                                  │
    │    Higher is better. Accounts for concurrency.                     │
    └────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```text
examples/dynamo_integration/                    # Main example directory
│
├── 📄 README.md                                # Overview and quick start
├── 📄 ARCHITECTURE.md                          # This file - system diagrams
│
├── 📁 data/                                    # Datasets
│   ├── agent_leaderboard_v2_banking.json       # 100 banking scenarios
│   ├── agent_leaderboard_v2_test_subset.json   # 3-scenario test subset, generated with ./scripts/create_test_subset
│   └── raw/banking/
│       └── tools.json                          # 20 tool schemas
│
├── 📁 scripts/                                 # Utility scripts
│   ├── download_agent_leaderboard_v2.py        # Dataset download
│   ├── create_test_subset.py                   # Create test subsets
│   ├── throughput_analysis.py                  # Analyze profiler CSV
│   ├── plot_throughput_vs_tsq_per_request.py   # Generate plots
│   └── run_concurrency_benchmark.sh            # Throughput benchmarking
│
└── 📁 react_benchmark_agent/                   # Workflow package
    │
    ├── 📄 README.md                            # Complete evaluation guide
    ├── 📄 pyproject.toml                       # Package definition
    │
    ├── 📁 configs/                             # Configuration files (symlink)
    │   ├── eval_config_no_rethinking_full_test.yml   # Full 100-scenario eval
    │   ├── eval_config_no_rethinking_minimal_test.yml # Quick 3-scenario test
    │   ├── eval_config_rethinking_full_test.yml       # Self-evaluation enabled
    │   ├── profile_rethinking_full_test.yml         # Profiler + self-eval
    │   ├── optimize_rethinking_full_test.yml        # Prefix header optimization
    │   ├── config_dynamo_e2e_test.yml           # Basic Dynamo workflow
    │   └── config_dynamo_prefix_e2e_test.yml   # With prefix headers
    │
    ├── 📁 src/react_benchmark_agent/           # Source code
    │   ├── register.py                         # Component registration
    │   ├── banking_tools.py                    # Tool stub group
    │   ├── tool_intent_stubs.py                # Intent capture
    │   ├── react_benchmark_agent.py            # Main agent
    │   ├── self_evaluating_agent.py            # Basic self-eval wrapper
    │   ├── self_evaluating_agent_with_feedback.py  # With feedback
    │   └── evaluators/
    │       └── tsq_evaluator.py                # TSQ evaluation
    │
    ├── 📁 tests/                               # Unit tests
    │   ├── test_tsq_formula.py                 # TSQ calculation tests
    │   └── test_self_evaluation.py             # Self-evaluation tests
    │
    └── 📁 outputs/                             # Evaluation results
        ├── dynamo_evals/
        │   └── <experiment_name>/
        │       └── jobs/<job_id>/
        │           ├── tool_selection_quality_output.json
        │           ├── standardized_data_all.csv
        │           └── workflow_profiling_report.txt
        └── benchmarks/
            └── <benchmark_name>/
                ├── benchmark_results.csv
                ├── benchmark_report.md
                └── analysis_*.txt
```

For Dynamo backend file structure, see: [Dynamo Setup Guide](../../external/dynamo/README.md#file-structure)

---

## Quick Reference Commands

All commands assume you are in the NeMo-Agent-Toolkit root directory.

```bash
# Start Dynamo (unified mode)
cd external/dynamo
bash start_dynamo_unified.sh

# Start Dynamo (Thompson Sampling router)
cd external/dynamo
bash start_dynamo_unified_thompson_hints.sh

# Stop Dynamo
cd external/dynamo
bash stop_dynamo.sh

# Run full evaluation (100 scenarios)
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_full_test.yml

# Run minimal test (3 scenarios)
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_no_rethinking_minimal_test.yml

# Run with self-evaluation loop
nat eval --config_file examples/dynamo_integration/react_benchmark_agent/configs/eval_config_rethinking_full_test.yml

# Analyze throughput from profiler output
cd examples/dynamo_integration
python scripts/throughput_analysis.py \
  ./react_benchmark_agent/outputs/dynamo_evals/<experiment_name>/jobs/<job_id>/standardized_data_all.csv

# Generate throughput vs TSQ scatter plots for an experiment
cd examples/dynamo_integration
python scripts/plot_throughput_vs_tsq_per_request.py \
  ./react_benchmark_agent/outputs/dynamo_evals/<experiment_name>/

# Run concurrency benchmark
cd examples/dynamo_integration
bash scripts/run_concurrency_benchmark.sh
```

