# React Benchmark Agent - Implementation Guide

This document details the source code implementation of the React Benchmark Agent, explaining how the different configuration files map to the underlying components, evaluators, and workflows.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Registry](#component-registry)
3. [Deployment Patterns](#deployment-patterns)
   - [Standard Deployment](#1-standard-deployment-no-rethinking)
   - [Self-Evaluation with Feedback](#2-self-evaluation-with-feedback-rethinking)
   - [Optimization Configuration](#3-optimization-configuration)
   - [Profiling Configuration](#4-profiling-configuration)
4. [Source Code Reference](#source-code-reference)
5. [Evaluators](#evaluators)
6. [Deprecation Analysis & Recommendations](#deprecation-analysis--recommendations)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REACT BENCHMARK AGENT ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────────┘

                              Configuration File (.yml)
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              register.py                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Entry point that imports and registers all components:                      │
│  • react_benchmark_agent_function  (from react_benchmark_agent.py)           │
│  • banking_tools_group_function    (from banking_tools.py)                   │
│  • self_evaluating_agent_function  (from self_evaluating_agent.py)           │
│  • self_evaluating_agent_with_feedback_function                              │
│  • tsq_evaluator_function          (from evaluators/)                        │
│  • action_completion_evaluator_function                                      │
│  • dynamo_openai_llm               (from dynamo_llm_config.py)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
           ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
           │  LLM Configs  │   │   Functions   │   │  Evaluators   │
           │───────────────│   │───────────────│   │───────────────│
           │ dynamo_llm    │   │ react_agent   │   │ tsq_evaluator │
           │ eval_llm      │   │ banking_tools │   │ ac_evaluator  │
           └───────────────┘   │ self_eval     │   └───────────────┘
                               └───────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │   Workflow        │
                              │ ─────────────────-│
                              │ react_agent  OR   │
                              │ self_evaluating_  │
                              │ agent_with_feedback│
                              └───────────────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │  Tool Intent      │
                              │  Capture System   │
                              │ ─────────────────-│
                              │ tool_intent_stubs │
                              │ ToolIntentBuffer  │
                              │ Global Registry   │
                              └───────────────────┘
```

---

## Component Registry

All components are registered in `register.py`:

```python
# register.py - Entry point for all custom components

# Core agent function
from .react_benchmark_agent import react_benchmark_agent_function

# Banking tools function group
from .banking_tools import banking_tools_group_function

# Self-evaluation wrappers
from .self_evaluating_agent import self_evaluating_agent_function
from .self_evaluating_agent_with_feedback import self_evaluating_agent_with_feedback_function

# Custom evaluators
from .evaluators import tsq_evaluator_function, action_completion_evaluator_function

# Custom Dynamo-aware LLM config with optimizable prefix parameters
from .dynamo_llm_config import dynamo_openai_llm
```

---

## Deployment Patterns

### 1. Standard Deployment (No Rethinking)

**Configuration:** `eval_config_no_rethinking_full_test.yml`

This is the baseline deployment that runs a ReAct agent directly without self-evaluation.

#### Configuration → Code Mapping

| Config Section | Source File | Component |
|----------------|-------------|-----------|
| `workflow._type: react_agent` | NAT core | Built-in ReAct agent |
| `function_groups.banking_tools._type: banking_tools_group` | `banking_tools.py` | `BankingToolsGroupConfig` |
| `evaluators.tool_selection_quality._type: tsq_evaluator` | `evaluators/tsq_evaluator.py` | `TSQEvaluatorConfig` |
| `llms.dynamo_llm._type: openai` | NAT core | Standard OpenAI LLM |

#### Data Flow

```
User Question
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│                         ReAct Agent Loop                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Thought    │ →  │    Action    │ →  │ Action Input │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   ▼                   │                   │
│         │           ┌──────────────┐            │                   │
│         │           │ Tool Stub    │◄───────────┘                   │
│         │           │ Execution    │                                │
│         │           │ (banking_    │                                │
│         │           │  tools.py)   │                                │
│         │           └──────────────┘                                │
│         │                   │                                       │
│         │                   ▼                                       │
│         │           ┌──────────────┐                                │
│         │           │ToolIntent   │                                 │
│         │           │Buffer.record│                                 │
│         │           │(tool_intent_│                                 │
│         │           │ stubs.py)   │                                 │
│         │           └──────────────┘                                │
│         │                   │                                       │
│         ▼                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Observation                             │   │
│  │           (Canned response from tool stub)                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         └──────────────────────┐                                    │
│                                ▼                                    │
│                    ┌──────────────────┐                             │
│                    │   Continue or    │                             │
│                    │   Final Answer   │                             │
│                    └──────────────────┘                             │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────┐
                    │  TSQ Evaluator   │
                    │ (tsq_evaluator.  │
                    │     py)          │
                    └──────────────────┘
```

#### Key Source Files

**`react_benchmark_agent.py`** (lines 15-94)
```python
class ReactBenchmarkAgentFunctionConfig(FunctionBaseConfig, name="react_benchmark_agent"):
    """
    React Benchmark Agent for Agent Leaderboard evaluation.
    
    This function supports two modes:
    1. Standard mode: Acts as a regular tool in the workflow
    2. Decision-only mode: Dynamically registers tool stubs from dataset
    """
    prefix: str = Field(default="Agent:")
    decision_only: bool = Field(default=False)
    canned_response_template: str = Field(default="Successfully executed {tool_name}...")
```

**`banking_tools.py`** (lines 30-138)
- Loads tool schemas from `data/raw/banking/tools.json`
- Creates stub functions for each tool via `create_tool_stub_function()`
- Registers them as a function group accessible by `banking_tools.<tool_name>`

**`tool_intent_stubs.py`** (lines 79-136)
- `ToolIntentBuffer` class stores captured tool intents
- `create_tool_stub_function()` creates async stubs that record to the buffer
- Global registry `_GLOBAL_INTENT_REGISTRY` enables cross-module intent access

---

### 2. Self-Evaluation with Feedback (Rethinking)

**Configuration:** `eval_config_rethinking_full_test.yml`

This advanced deployment wraps the ReAct agent with a self-evaluation loop that:
- Evaluates tool selection after each attempt
- Provides structured feedback on retry
- Continues until confidence threshold is met

#### Configuration → Code Mapping

| Config Section | Source File | Component |
|----------------|-------------|-----------|
| `functions.react_workflow._type: react_agent` | NAT core | Inner ReAct agent |
| `workflow._type: self_evaluating_agent_with_feedback` | `self_evaluating_agent_with_feedback.py` | Self-eval wrapper |
| `workflow.wrapped_agent: react_workflow` | N/A | Reference to inner agent |
| `workflow.evaluator_llm: eval_llm` | N/A | LLM for self-evaluation |
| `workflow.pass_feedback_to_agent: true` | `self_evaluating_agent_with_feedback.py` | Feedback loop enabled |

#### Data Flow

```
User Question
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│              Self-Evaluating Agent with Feedback                    │
│              (self_evaluating_agent_with_feedback.py)               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ATTEMPT 1                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │  Clear Intent │ ← clear_global_intents(scenario_id)       │   │
│  │  │     Buffer    │                                           │   │
│  │  └───────────────┘                                           │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │  Execute      │ ← wrapped_agent.ainvoke(question)         │   │
│  │  │  ReAct Agent  │                                           │   │
│  │  └───────────────┘                                           │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │  Get Intents  │ ← get_global_intents(scenario_id)         │   │
│  │  │  [Tool A, B]  │                                           │   │
│  │  └───────────────┘                                           │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │ Self-Evaluate │ ← _evaluate_tool_sequence()               │   │
│  │  │ via eval_llm  │                                           │   │
│  │  └───────────────┘                                           │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │ is_sufficient:│  confidence < threshold?                  │   │
│  │  │   false       │  → RETRY                                  │   │
│  │  │ confidence:   │                                           │   │
│  │  │   0.60        │                                           │   │
│  │  └───────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         │ Format feedback from evaluation                           │
│         │ using feedback_template                                   │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ATTEMPT 2 (with feedback)                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │ query =       │                                           │   │
│  │  │ question +    │ ← Feedback appended to original question  │   │
│  │  │ feedback      │                                           │   │
│  │  └───────────────┘                                           │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │  Execute      │ ← Agent sees previous mistakes            │   │
│  │  │  ReAct Agent  │                                           │   │
│  │  └───────────────┘                                           │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  ┌───────────────┐                                           │   │
│  │  │ is_sufficient:│  confidence >= threshold?                 │   │
│  │  │   true        │  → ACCEPT                                 │   │
│  │  │ confidence:   │                                           │   │
│  │  │   0.85        │                                           │   │
│  │  └───────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

#### Key Source Files

**`self_evaluating_agent_with_feedback.py`** (lines 41-109)
```python
class SelfEvaluatingAgentWithFeedbackConfig(FunctionBaseConfig, name="self_evaluating_agent_with_feedback"):
    """Configuration for Self-Evaluating Agent with Feedback Loop."""
    
    wrapped_agent: FunctionRef      # Reference to inner ReAct agent
    evaluator_llm: LLMRef           # LLM for self-evaluation
    max_retries: int = 3            # Maximum retry attempts
    min_confidence_threshold: float = 0.85  # Minimum confidence to accept
    pass_feedback_to_agent: bool = True     # Pass evaluation feedback on retry
    feedback_template: str = "..."   # Template for constructing feedback
    evaluation_prompt_template: str = "..."  # Template for self-evaluation prompt
```

**Intent Isolation for Concurrent Execution** (`tool_intent_stubs.py`, lines 33-76)
```python
# Context variable for async-safe scenario isolation
_current_scenario_id: contextvars.ContextVar[str] = contextvars.ContextVar("scenario_id", default="current")

def set_current_scenario_id(scenario_id: str) -> contextvars.Token:
    """Set the current scenario ID for this async context."""
    
def get_global_intents(scenario_id: str = "current") -> list[dict[str, Any]]:
    """Retrieve tool intents from the global registry."""
```

---

### 3. Optimization Configuration

**Configuration:** `optimize_predictive_prefix_headers.yml`

This configuration enables NAT's optimizer to tune Dynamo router parameters for latency/throughput.

#### Configuration → Code Mapping

| Config Section | Source File | Component |
|----------------|-------------|-----------|
| `llms.dynamo_llm._type: dynamo_openai` | `dynamo_llm_config.py` | Custom LLM with optimizable fields |
| `llms.dynamo_llm.optimizable_params` | `dynamo_llm_config.py` | Fields to optimize |
| `llms.dynamo_llm.search_space` | `dynamo_llm_config.py` | Parameter search ranges |
| `evaluators.avg_llm_latency._type: avg_llm_latency` | NAT core | Runtime performance metric |
| `optimizer.eval_metrics` | NAT core | Metrics to minimize |

#### Optimizable Parameters

**`dynamo_llm_config.py`** (lines 54-100)
```python
class DynamoLLMConfig(OpenAIModelConfig, name="dynamo_openai"):
    """OpenAI-compatible LLM config with Dynamo prefix optimization support."""
    
    # OPTIMIZABLE: Total expected requests per conversation/prefix
    prefix_total_requests: int = OptimizableField(
        default=10,
        description="Expected requests for this prefix. Higher = more stickiness.",
        space=SearchSpace(low=1, high=20, step=5)
    )
    
    # OPTIMIZABLE: Output Sequence Length hint
    prefix_osl: PrefixLevel = OptimizableField(
        default="MEDIUM",
        description="LOW=short, MEDIUM=typical, HIGH=long responses",
        space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"])
    )
    
    # OPTIMIZABLE: Inter-Arrival Time hint
    prefix_iat: PrefixLevel = OptimizableField(
        default="MEDIUM",
        description="LOW=rapid bursts, MEDIUM=normal, HIGH=slow requests",
        space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"])
    )
```

#### Optimization Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NAT Optimizer                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ prefix_total_ │      │  prefix_osl   │      │  prefix_iat   │
│ requests: 1   │      │     LOW       │      │     LOW       │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │  Run Evaluation   │
                    │  (100 scenarios)  │
                    └───────────────────┘
                                │
                                ▼
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ avg_llm_      │      │ avg_workflow_ │      │ avg_num_      │
│ latency       │      │ runtime       │      │ llm_calls     │
│ weight: 0.7   │      │ weight: 0.2   │      │ weight: 0.1   │
└───────────────┘      └───────────────┘      └───────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │  Optuna Grid      │
                    │  Search / Bayesian│
                    │  Optimization     │
                    └───────────────────┘
                                │
                                ▼
                    ┌───────────────────┐
                    │  Best Parameters  │
                    │  Found            │
                    └───────────────────┘
```

---

### 4. Profiling Configuration

**Configuration:** `profile_predictive_prefix_headers.yml`

This configuration enables comprehensive profiling for performance analysis.

#### Configuration → Code Mapping

| Config Section | Source File | Component |
|----------------|-------------|-----------|
| `eval.general.profiler.compute_llm_metrics: true` | NAT core | TTFT, ITL, throughput metrics |
| `eval.general.profiler.token_uniqueness_forecast: true` | NAT core | Token pattern analysis |
| `eval.general.profiler.bottleneck_analysis.enable_nested_stack: true` | NAT core | Call stack analysis |
| `eval.general.profiler.prompt_caching_prefixes.enable: true` | NAT core | KV cache prefix detection |

#### Profiler Output Files

```
outputs/dynamo_evals/<job_id>/
├── standardized_data_all.csv      # Per-LLM-call metrics (TTFT, tokens, etc.)
├── workflow_profiling_report.txt  # Human-readable summary
├── all_requests_profiler_traces.json  # Raw trace data
└── tool_selection_quality_output.json # TSQ scores per scenario
```

---

## Source Code Reference

### Core Components

| File | Purpose | Config Type Name |
|------|---------|------------------|
| `react_benchmark_agent.py` | Main agent function | `react_benchmark_agent` |
| `banking_tools.py` | Banking tool stubs | `banking_tools_group` |
| `tool_intent_stubs.py` | Intent capture system | N/A (infrastructure) |
| `self_evaluating_agent_with_feedback.py` | Self-eval wrapper (unified) | `self_evaluating_agent`, `self_evaluating_agent_with_feedback` |
| `dynamo_llm_config.py` | Optimizable LLM config | `dynamo_openai` |

### Evaluators

| File | Purpose | Config Type Name |
|------|---------|------------------|
| `evaluators/tsq_evaluator.py` | Tool Selection Quality | `tsq_evaluator` |
| `evaluators/action_completion_evaluator.py` | Action Completion | `action_completion_evaluator` |

### Utilities

| File | Purpose | Status |
|------|---------|--------|
| `dynamic_tool_loader.py` | Dynamic tool registration | See deprecation notes |
| `prefix_utils.py` | Prefix ID management examples | See deprecation notes |

---

## Evaluators

### Tool Selection Quality (TSQ) Evaluator

**File:** `evaluators/tsq_evaluator.py`

The TSQ evaluator measures how accurately the agent selects tools compared to expected tool calls.

#### Key Functions

```python
def extract_tool_calls_from_trajectory(trajectory):
    """
    Extract tool calls from agent trajectory.
    Handles multiple formats:
    - Nested payload structure (profiler format)
    - Flat structure with event_type (legacy)
    - LangChain action/action_input format
    - IntermediateStep Pydantic objects
    """

def calculate_tool_accuracy(actual, expected):
    """
    Calculate F1 score:
    precision = correct / actual_called
    recall = correct / expected
    F1 = 2 * (precision * recall) / (precision + recall)
    """
```

#### Configuration Options

```yaml
evaluators:
  tool_selection_quality:
    _type: tsq_evaluator
    llm_name: eval_llm      # Optional: for semantic comparison
    strict_mode: false      # Allow fuzzy matching
    tool_weight: 1.0        # Weight for tool selection (0-1)
    parameter_weight: 0.0   # Weight for parameter accuracy (0-1)
```

### Action Completion (AC) Evaluator

**File:** `evaluators/action_completion_evaluator.py`

The AC evaluator measures whether the agent addressed all user goals.

```yaml
evaluators:
  action_completion:
    _type: action_completion_evaluator
    llm_name: eval_llm      # Optional: for semantic goal matching
    strict_mode: false      # Allow semantic matching
```

---

## Deprecation Analysis & Recommendations

### Files to Consider Consolidating

#### `dynamic_tool_loader.py` - Likely Unused

**Current State:**
- Provides `DynamicToolLoader` class for registering tools from dataset entries
- Not imported in `register.py`
- Not used in any configuration file

**Issue:** 
- `banking_tools.py` already handles tool registration via function groups
- `DynamicToolLoader` appears to be an alternative approach that was superseded

**Recommendation:** 
- Verify no external usage
- If unused, remove or move to `examples/` directory

---

#### `prefix_utils.py` - Example/Documentation Code

**Current State:**
- Contains example functions for managing Dynamo prefix IDs
- Re-exports functions from `nat.plugins.langchain.dynamo_prefix_headers`

**Issue:**
- Functions like `run_question_with_prefix()` are illustrative examples
- `generate_benchmark_prefix()` duplicates logic in `dynamo_llm_config.py`

**Recommendation:**
- Move to `examples/` subdirectory
- Or convert to unit tests demonstrating usage patterns

---

#### ✅ Type Hints Consistency - RESOLVED

**Status:** COMPLETED

Standardized all type hints to Python 3.9+ style (`list[dict]`, `dict[str, Any]`).
Removed `from typing import Dict` import from `tool_intent_stubs.py`.

---

### Summary of Recommended Changes

| Priority | File | Action | Status |
|----------|------|--------|--------|
| Low | `dynamic_tool_loader.py` | Remove if unused | Pending |
| Low | `prefix_utils.py` | Move to examples/ | Pending |

---

## Quick Reference

### Running Standard Evaluation
```bash
nat eval --config_file configs/eval_config_no_rethinking_full_test.yml
```

### Running with Self-Evaluation
```bash
nat eval --config_file configs/eval_config_rethinking_full_test.yml
```

### Running Optimization
```bash
nat optimize --config_file configs/optimize_predictive_prefix_headers.yml
```

### Running with Profiling
```bash
nat eval --config_file configs/profile_predictive_prefix_headers.yml
```

