<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

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

# Evaluating AI Agents and Multi-Agentic Workflows: A Comprehensive Playbook

---

Table of Contents

[Executive Summary](#executive-summary)

[1\. The Paradigm Shift: From Traditional Testing to Agent Evaluation](#1-the-paradigm-shift-from-traditional-testing-to-agent-evaluation)

[2\. The Agent Evaluation Test Pyramid](#2-the-agent-evaluation-test-pyramid)

[3\. Step-by-Step Playbook for Testing Complex Agents](#3-step-by-step-playbook-for-testing-complex-agents)

[4\. Practical Implementation Guide](#4-practical-implementation-guide)

[5\. Conclusion and Future Directions](#5-conclusion-and-future-directions)

---

## Executive Summary

As AI agents transition from experimental prototypes to production systems handling critical system softwares, the need for robust evaluation frameworks has never been more pressing. Traditional software testing methodologies \- built on assumptions of deterministic behavior and predictable outputs \- fall short when applied to autonomous agents that reason, plan, and adapt in real-time.

This document provides a comprehensive framework for evaluating AI agents and multi-agentic workflows. We introduce the **Agent Evaluation Test Pyramid**, a mental model for structuring evaluations at different levels of abstraction, and provide a step-by-step playbook for implementing rigorous testing strategies.

## 1\. The Paradigm Shift: From Traditional Testing to Agent Evaluation

### 1.1 The Deterministic Fallacy

Traditional software testing operates on a fundamental assumption: *given the same input, the system will always produce the same output*. This deterministic worldview has shaped decades of quality assurance practices \- unit tests assert exact return values, integration tests verify precise API responses, and end-to-end tests follow scripted user journeys with predictable outcomes.

AI agents shatter this assumption entirely.

A code-writing agent might solve the same problem with different but equally valid implementations. **This non-deterministic nature requires us to fundamentally rethink what "correct" means.**

**Example - The Summarization Dilemma:**

```py
# Traditional test - deterministic, exact match expected
def test_add_numbers():
    assert add(2, 3) == 5  # Always produces 5

# Agent test - same input, multiple valid outputs
def test_summarize_article():
    result = agent.summarize(article)
    # Run 1: "The article discusses climate change impacts..."
    # Run 2: "Climate change effects are the main topic..."
    # Run 3: "This piece examines how climate change..."
    # All are valid! Traditional assertions don't work.
```

### 1.2 The Five Dimensions of Agent Behavior

Unlike traditional applications that transform inputs to outputs through fixed logic, agents exhibit complex behaviors across five distinct dimensions:

| Dimension | Traditional App | AI Agent |
| :---- | :---- | :---- |
| **Reasoning** | Hardcoded logic | Dynamic chain-of-thought |
| **Planning** | Predefined workflows | Emergent multi-step strategies |
| **Tool Use** | Fixed function calls | Dynamic tool selection and sequencing |
| **Memory** | Stateless or simple state | Context accumulation and retrieval |
| **Adaptation** | Version-based updates | In-context learning |

Each dimension introduces new failure modes that traditional testing cannot capture:

- **Reasoning failures**: Logical errors in chain-of-thought, incorrect conclusions from valid premises
- **Planning failures**: Inefficient action sequences, getting stuck in loops, abandoning viable paths
- **Tool use failures**: Calling wrong tools, passing incorrect parameters, misinterpreting results
- **Memory failures**: Context window overflow, retrieval of irrelevant information, forgetting critical details
- **Adaptation failures**: Overfitting to conversation history, inconsistent persona maintenance

### 1.3 From Output Testing to Trajectory Evaluation

The most profound shift in agent evaluation is moving from testing outputs to evaluating **trajectories** \- the complete sequence of decisions, actions, and intermediate states an agent traverses to reach its goal.

Two agents might reach the same correct answer through vastly different paths \- one efficient and safe, the other wasteful and risky. Trajectory evaluation captures these differences.

### 1.4 The Evaluation Mindset Transformation

Moving from traditional testing to agent evaluation requires a fundamental mindset shift:

| Traditional Mindset | Agent Evaluation Mindset |
| :---- | :---- |
| Binary pass/fail | Continuous quality scores |
| Exact match assertions | Semantic equivalence checks |
| Edge case coverage | Distribution coverage |
| Bug-free goal | Acceptable error rate goal |
| Test once, deploy | Continuous monitoring |
| Developer-written tests | Human \+ LLM evaluators |
| "This test failed" | "This run scored 0.7 on quality" |
| "Add more unit tests" | "Improve evaluator coverage" |
| "100% code coverage" | "Comprehensive evaluator coverage" |

**This transformation isn't just technical \- it's philosophical**. We must accept that agents will sometimes fail, and our goal shifts from preventing all failures to:

1. **Understanding** failure modes and their frequencies
2. **Bounding** the severity of potential failures
3. **Detecting** failures quickly in production
4. **Recovering** gracefully when failures occur

---

## 2\. The Agent Evaluation Test Pyramid

Just as the traditional test pyramid guides software testing strategy (unit → integration → end-to-end), we propose the **Agent Evaluation Test Pyramid** to structure AI agent testing at appropriate levels of abstraction.

### 2.1 Level 1: Foundation \- LLM Response Quality Tests

**Purpose**: Establish baseline quality of the underlying language model's outputs independent of agent behavior.

**What to Test**:

- Response coherence and fluency
- Instruction following accuracy
- Factual accuracy (where verifiable)
- Safety and content policy compliance
- Output format adherence (JSON, markdown, etc.)

**Characteristics**:

- Deterministic where possible
- Run on every commit

**Evaluation Methods**:

- Schema validation (static)
- Type checking (static)
- Regex/pattern matching (static)
- LLM-as-Judge scoring for subjective quality
- Embedding similarity for semantic consistency

**Coverage Target**: Run on every LLM call in development; sample in production.

### 2.2 Level 2: Component & Tool Evaluation

**Purpose**: Verify that individual agent capabilities work correctly in isolation before testing their orchestration.

**What to Test**:

- Tool invocation accuracy (right tool for the task)
- Tool parameter extraction (correct arguments)
- Tool result interpretation (proper use of returned data)
- Retrieval quality (relevant context fetched)
- Individual reasoning steps (single-hop conclusions)

**Characteristics**:

- Tests single prompts/chains
- Can use LLM-as-Judge

**Evaluation Methods**:

- Exact match for tool selection
- Parameter schema validation
- Retrieval relevance scoring (precision, recall, MRR)
- Step-wise correctness evaluation

**Coverage Target**: Comprehensive coverage of all tools and retrieval paths.

### 2.3 Level 3: Trajectory Evaluation

**Purpose**: Assess the quality of multi-step action sequences and decision-making over time.

**What to Test**:

- Decision point correctness (right choice at each branch)
- Error recovery behavior (graceful handling of tool failures)
- Loop detection (avoiding infinite cycles)
- Resource efficiency (token usage, API calls, time)

**Characteristics**:

- Tests agent with tool use across multiple steps
- Measures agent capabilities holistically

**Evaluation Methods**:

- Trajectory comparison against reference paths
- Step-by-step LLM-as-Judge scoring

**Coverage Target**: Representative scenarios covering major workflow patterns.

### 2.4 Level 4: End-to-End Agentic Scenarios

**Purpose**: Validate complete agent behavior on realistic, complex tasks that exercise the full system.

**What to Test**:

- Task completion success rate
- Output quality on complex goals
- Behavior under ambiguity
- Cross-capability integration

**Characteristics**:

- Most realistic conditions
- Run periodically (nightly/weekly) or pre-release

**Evaluation Methods**:

- Human evaluation panels
- LLM-as-Judge with detailed rubrics
- Task-specific success criteria
- User simulation testing

**Coverage Target**: Curated set of high-value scenarios (typically 50-200 test cases).

### 2.5 Multi-Turn Conversation Evaluation (if applicable)

Production agents rarely operate in single-turn interactions. Users engage in extended conversations, refining requests, providing clarifications, and building on previous exchanges. **Multi-turn evaluation** assesses whether agents accomplish user goals across entire interactions, not just individual responses.

**Why Multi-Turn Evaluation Matters:**

Traditional evaluation approaches focus on isolated traces or individual steps, creating critical visibility gaps:

| Single-Turn Limitation | Multi-Turn Advantage |
| :---- | :---- |
| Evaluates responses in isolation | Measures goal achievement across full conversation |
| Misses context accumulation errors | Captures memory and state management issues |
| Cannot detect conversation derailment | Identifies when agents lose track of objectives |
| Ignores user clarification handling | Tests adaptation to refined requirements |

**The Three Dimensions of Multi-Turn Evaluation:**

1. **Semantic Intent**

   - What did the user actually intend to accomplish?
   - Did the agent correctly interpret evolving requirements?
   - How well did the agent handle ambiguous or changing goals?

2. **Semantic Outcomes**

   - Was the user's goal ultimately achieved?
   - If not, what caused the failure?
   - At what point in the conversation did success/failure occur?

3. **Conversation Trajectory**

   - How did the interaction unfold over multiple turns?
   - Were tool usages appropriate throughout the conversation?
   - Did the agent maintain coherent reasoning across turns?

**Multi-Turn Evaluation Characteristics:**

| Aspect | Details |
| :---- | :---- |
| **Unit of Evaluation** | Complete conversation thread, not individual messages |
| **Timing** | Evaluated upon conversation completion |
| **Primary Method** | LLM-as-Judge with conversation-aware prompts |
| **Key Metrics** | Goal completion rate, turns to resolution, context retention accuracy |
| **Complexity** | Higher than single-turn due to state dependencies |

**What to Test in Multi-Turn Scenarios:**

- **Context retention**: Does the agent remember and correctly apply information from earlier turns?
- **Goal tracking**: Does the agent maintain focus on the user's original objective through clarifications?
- **Graceful recovery**: When misunderstandings occur, does the agent recover effectively?
- **Conversation efficiency**: Does the agent resolve goals in a reasonable number of turns?
- **State consistency**: Does the agent maintain a consistent internal state across the conversation?

**Implementation Approach:**

Multi-turn evaluations operate as online evaluations that measure conversation-level outcomes:

1. **Organize as Threads**: Structure multi-turn exchanges as conversation threads with clear boundaries
2. **Define Completion Triggers**: Specify when a conversation is considered "complete" for evaluation
3. **Configure Thread-Level Evaluators**: Use LLM-as-Judge prompts designed for full conversation assessment
4. **Capture Conversation Metadata**: Track turn count, tool usage patterns, and user satisfaction signals

**Example Multi-Turn Evaluation Criteria:**

```text
Evaluator Prompt (Goal Achievement):
Given the complete conversation between user and agent:
1. What was the user's primary goal?
2. Was this goal fully achieved, partially achieved, or not achieved?
3. If not fully achieved, identify the turn where the conversation went off track.
4. Rate the conversation efficiency (1-5) based on turns required vs. optimal path.

Score: [achieved/partial/failed]
Reasoning: [explanation]
```

**Coverage Target**: Include multi-turn scenarios in your evaluation dataset covering:

- Simple clarification flows (2-3 turns)
- Complex multi-step tasks (5-10 turns)
- Error recovery scenarios (agent misunderstands, then corrects)
- Goal refinement conversations (user iteratively specifies requirements)

---

## 3\. Step-by-Step Playbook for Testing Complex Agents

### 3.1 Step 1: Define Success Criteria and Quality Dimensions

Before building evaluations, establish clear criteria for what constitutes success for your agent. This foundational step shapes all subsequent evaluation decisions.

**Key Questions to Answer:**

1. What does "success" look like for your agent?
2. Which behaviors are critical vs. nice-to-have?
3. What failure modes are you most concerned about?
4. Do you have reference outputs (ground truth) available?

**Define Your Quality Dimensions:**

Identify the 3-5 most critical quality dimensions for your specific agent. Common dimensions include:

| Quality Dimension | Description | Example Metrics |
| :---- | :---- | :---- |
| **Correctness** | Does the agent produce accurate, factually correct outputs? | Accuracy rate, factual error count |
| **Helpfulness** | Does the output actually address the user's need? | Task completion rate, user satisfaction |
| **Safety** | Does the agent avoid harmful or inappropriate outputs? | Policy violation rate, safety score |
| **Efficiency** | Does the agent complete tasks with minimal steps/resources? | Steps to completion, token usage, latency |
| **Consistency** | Does the agent produce stable results across similar inputs? | Variance across runs, behavior drift |
| **Tool Accuracy** | Does the agent select and use tools correctly? | Tool selection precision, parameter accuracy |

**Establish Success Thresholds:**

For each quality dimension, define concrete thresholds (subjective to each agent):

| Threshold Level | Purpose | Example |
| :---- | :---- | :---- |
| **Minimum Acceptable** | Below this, the agent should not deploy | Correctness \> 85% |
| **Target** | The goal for production readiness | Correctness \> 95% |
| **Stretch** | Aspirational quality level | Correctness \> 99% |

**Identify Critical Failure Modes:**

Document the failure modes that matter most for your use case:

- **High Severity**: Failures that cause harm, data loss, or major user impact
- **Medium Severity**: Failures that degrade experience but are recoverable
- **Low Severity**: Minor issues that don't significantly impact outcomes

**Determine Reference Availability:**

Assess what ground truth data you have access to:

- **Full Reference**: You have correct answers for all test cases
- **Partial Reference**: Reference outputs available for some scenarios
- **No Reference**: Must rely on reference-free evaluation methods

This assessment determines which evaluator types you can use effectively.

### 3.2 Step 2: Build Your Evaluation Golden Dataset

Datasets form the foundation of systematic agent evaluation. A well-constructed dataset enables reproducible testing and meaningful comparisons across agent versions.

**Dataset Construction Strategy:**

1. **Start Small, Iterate Often**
   - Begin with 10-20 manually curated examples
   - Cover core scenarios and known edge cases
   - Prioritize quality over quantity initially

2. **Structure Your Examples**
   - **Inputs**: The user query or task for the agent
   - **Reference Outputs** (optional): Expected correct responses
   - **Metadata**: Tags for categorization, difficulty levels, scenario types

3. **Progressive Dataset Expansion**
   - Add production traces that revealed issues
   - Incorporate user feedback signals
   - Use synthetic data generation to supplement gaps
   - Regularly prune outdated or redundant examples

**Dataset Categories to Include:**

For the canonical category list — Happy Path, Edge Cases, Adversarial, Ambiguous, Multi-Tool, Error Recovery, Multi-Turn, and Out-of-Scope, with the "When to Include" column and NeMo Agent toolkit-specific dataset config — see [`methodology.md § Step 2`](methodology.md#step-2-build-the-evaluation-golden-dataset).

### 3.3 Step 3: Design Your Evaluator Suite

Design evaluators that match your quality dimensions. A comprehensive evaluator suite combines multiple approaches for thorough coverage.

**Evaluator Architecture:**

**LLM-as-Judge Configuration:**

When using LLM-as-Judge evaluators, configure these components:

1. **Prompt Setup**: Define clear assessment criteria and instructions
2. **Variable Mapping**: Connect agent inputs/outputs to evaluator prompts
3. **Feedback Type**: Choose scoring format:
   - Boolean (true/false)
   - Categorical (predefined options)
   - Continuous (numerical ranges)
4. **Few-Shot Examples**: Include human-corrected examples to improve alignment

### 3.4 Step 4: Build & Configure Evaluators

Agents require evaluation approaches beyond simple input-output testing. Modern evaluation platforms support three complementary strategies:

**A. Final Response Evaluation:**

Treats the agent as a black box, assessing only the end result.

| Aspect | Details |
| :---- | :---- |
| **Inputs** | User query, available tool list |
| **Outputs** | Agent's final response |
| **Evaluators** | LLM-as-judge for quality, helpfulness |
| **Limitation** | No visibility into internal failures |
| **Best For** | High-level quality assurance |

**B. Single Step Evaluation:**

Tests individual agent decisions in isolation.

| Aspect | Details |
| :---- | :---- |
| **Inputs** | Single step context (with or without prior steps) |
| **Outputs** | Tool selection and arguments |
| **Evaluators** | Binary scoring on correct tool choice |
| **Advantage** | Fast execution (single LLM call per test) |
| **Best For** | Tool selection accuracy, parameter extraction |

**C. Trajectory Evaluation:**

Assesses the complete sequence of actions taken by the agent.

| Aspect | Details |
| :---- | :---- |
| **Inputs** | User query, tool list |
| **Outputs** | Full sequence of tool calls |
| **Evaluators** | Exact match, edit distance, LLM-as-judge |
| **Complexity** | Most challenging to create reference trajectories |
| **Best For** | Efficiency analysis, path optimality |

**Evaluation Strategy Matrix:**

| What You Want to Measure | Evaluation Approach |
| :---- | :---- |
| Does the agent solve the task? | Final Response |
| Does the agent pick the right tools? | Single Step |
| Does the agent take an efficient path? | Trajectory |
| Does the agent reason correctly? | Trajectory \+ Single Step |
| Does the agent recover from errors? | Trajectory |

### 3.5 Step 5: Run Experiments and Capture Results

Experiments systematically test your agent against datasets while capturing comprehensive results.

**Experiment Workflow:**

**Experiment Execution Checklist:**

1. **Version Control**: Tag the agent version being tested
2. **Environment Capture**: Record model versions, configurations, prompts
3. **Full Trace Logging**: Capture intermediate steps, not just final outputs
4. **Metadata Tagging**: Add experiment context (date, trigger, hypothesis)
5. **Baseline Comparison**: Always compare against a known baseline

**What Experiments Capture:**

- Agent outputs for each dataset example
- Evaluator scores across all configured metrics
- Execution traces showing internal agent behavior
- Timing and resource consumption data
- Error logs and failure modes

### 3.6 Step 6: Compare and Iterate

Use experiment results to drive systematic improvement of your agent.

**Comparison Dimensions:**

| Dimension | What to Compare |
| :---- | :---- |
| **Version vs. Version** | How did changes affect performance? |
| **Time-Based** | Is performance stable or degrading? |
| **Segment-Based** | Which query types perform best/worst? |
| **Evaluator-Based** | Which quality dimensions need work? |

**Iteration Process:**

1. **Identify Failures**: Filter experiments to find failing examples
2. **Categorize Issues**: Group failures by root cause
3. **Prioritize Fixes**: Focus on high-impact, frequent failure modes
4. **Implement Changes**: Modify prompts, tools, or agent logic
5. **Re-Evaluate**: Run experiments to verify improvements
6. **Update Dataset**: Add failure cases to prevent regression

**Regression Prevention:**

- Add every production failure to your test dataset
- Create "golden" examples that must always pass
- Set up automated experiment runs on code changes
- Configure alerts for score degradation

### 3.7 Step 7: Establish Offline and Online Evaluation Cycles

Production-ready agents require both pre-deployment testing (offline) and live monitoring (online).

**Offline Evaluation (Pre-Deployment):**

| Aspect | Details |
| :---- | :---- |
| **Purpose** | Validate changes before release |
| **Data Source** | Curated datasets with reference outputs |
| **Evaluators** | Full suite including reference-based metrics |
| **Frequency** | On pre-release |
| **Decision** | Gate release on quality thresholds |

**Online Evaluation (Production):**

| Aspect | Details |
| :---- | :---- |
| **Purpose** | Monitor live agent behavior |
| **Data Source** | Real production traffic (runs and threads) |
| **Evaluators** | Reference-free metrics only |
| **Frequency** | Continuous or sampled |
| **Decision** | Alert on anomalies, trigger investigations |

### 3.8 Step 8: Scale Your Evaluation Practice

As your agent matures, evolve your evaluation practice to match.

**Maturity Stages:**

| Stage | Dataset Size\* (subjective) | Evaluator Coverage | Automation Level |
| :---- | :---- | :---- | :---- |
| **Prototype** | 10-20 examples | Basic correctness | Manual runs |
| **Alpha** | 50-100 examples | Multi-dimensional | PR-triggered |
| **Beta** | 200-500 examples | Comprehensive | CI/CD integrated |
| **Production** | 500+ examples | Full pyramid | Continuous \+ alerts |

**Scaling Checklist:**

- [ ] Integrate experiments into CI/CD pipeline
- [ ] Set quality thresholds for deployment gates
- [ ] Configure automated alerts for score degradation
- [ ] Establish human review queues for ambiguous cases
- [ ] Create dashboards for evaluation metrics over time
- [ ] Document evaluation criteria and scoring rubrics
- [ ] Train team members on evaluation interpretation

---

## 4\. Practical Implementation Guide

### 4.1 Getting Started Checklist

- [ ] Define 3-5 critical quality dimensions for your agent
- [ ] Curate initial dataset of 10-20 representative examples
- [ ] Implement at least one evaluator per quality dimension
- [ ] Run baseline experiment and record scores
- [ ] Set up automated experiment execution on code changes

### 4.2 Common Pitfalls to Avoid

| Pitfall | Why It's Problematic | Better Approach |
| :---- | :---- | :---- |
| Testing only happy paths | Misses critical failure modes | Include adversarial and edge cases |
| Over-relying on LLM-as-Judge | Expensive and potentially biased | Layer with static and code evaluators |
| Infrequent evaluation | Issues accumulate undetected | Continuous offline \+ online evaluation |
| Ignoring trajectories | Hides inefficiency and reasoning failures | Evaluate paths, not just outputs |
| Static datasets | Become stale and miss new patterns | Continuously add production failures |

---

## 5\. Conclusion and Future Directions

### 5.1 Key Takeaways

1. **Embrace non-determinism**: Accept that agents produce variable outputs and design evaluations around semantic equivalence rather than exact matches.

2. **Build the pyramid from bottom up**: Establish solid foundations of cheap, fast static tests before investing in expensive E2E evaluations.

3. **Evaluate trajectories, not just outputs**: The path an agent takes matters as much as the final answer for quality assessment.

4. **Match evaluator to need**: Use static assertions for structure, custom code for metrics, and LLM-as-Judge only for semantic quality.

5. **Continuous evaluation is essential**: Production monitoring catches issues that offline tests miss; build feedback loops.

6. **Human judgment remains crucial**: LLM-as-Judge augments but doesn't replace human evaluation for nuanced quality assessment.

*This document provides a framework for evaluating AI agents and multi-agentic workflows. Implementations should be adapted to specific use cases, agent architectures, and organizational requirements.*
