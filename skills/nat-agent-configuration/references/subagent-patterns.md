# Sub-Agent Composition Patterns

These patterns show how to compose agents hierarchically in NeMo Agent toolkit.

---

## Pattern: Router with Specialized Sub-Agents

A top-level router agent routes incoming requests to specialized sub-agents, each with their own tools:

```yaml
functions:
  # Sub-agent 1: handles data analysis tasks
  data_analyst:
    _type: react_agent
    tool_names: [query_database, plot_chart]
    llm_name: nim_llm
    description: "Handles data analysis, SQL queries, and chart generation"
    verbose: true

  # Sub-agent 2: handles research tasks
  researcher:
    _type: react_agent
    tool_names: [wikipedia_search, web_search]
    llm_name: nim_llm
    description: "Handles research questions using web and encyclopedia sources"
    verbose: true

  # Sub-agent 3: handles code tasks
  coder:
    _type: tool_calling_agent
    tool_names: [code_execution]
    llm_name: nim_llm
    description: "Writes and executes Python code"
    verbose: true

  # Individual tools used by the sub-agents
  query_database:
    _type: query_database
  plot_chart:
    _type: plot_chart
  wikipedia_search:
    _type: wikipedia_search
  web_search:
    _type: web_search
  code_execution:
    _type: code_execution

llms:
  nim_llm:
    _type: openai
    model_name: aws/anthropic/claude-haiku-4-5-v1
    base_url: https://inference-api.nvidia.com/v1
    temperature: 0.0
    api_key: $NVIDIA_INFERENCE_API_KEY

workflow:
  _type: router_agent
  branches: [data_analyst, researcher, coder]
  llm_name: nim_llm
  description: "Routes incoming requests to the appropriate specialist agent"
```

## Pattern: Reasoning Agent Wrapping a Tool-Rich Agent

A reasoning agent provides high-level planning while an inner agent handles execution:

```yaml
functions:
  # The execution agent has access to all tools
  executor:
    _type: react_agent
    tool_names: [current_datetime, calculator, web_search]
    llm_name: fast_llm
    verbose: true

llms:
  thinking_llm:
    _type: openai
    model_name: deepseek/deepseek-r1
    base_url: https://inference-api.nvidia.com/v1
    api_key: $NVIDIA_INFERENCE_API_KEY
  fast_llm:
    _type: openai
    model_name: aws/anthropic/claude-haiku-4-5-v1
    base_url: https://inference-api.nvidia.com/v1
    api_key: $NVIDIA_INFERENCE_API_KEY

workflow:
  _type: reasoning_agent
  llm_name: thinking_llm
  augmented_fn: executor
  verbose: true
```

## Pattern: Parallel Sub-Agent Fan-Out

Run multiple sub-agents concurrently and combine their results:

```yaml
functions:
  sentiment_agent:
    _type: react_agent
    tool_names: [current_datetime]
    llm_name: nim_llm
    description: "Analyzes the sentiment of the input text"

  summary_agent:
    _type: react_agent
    tool_names: [current_datetime]
    llm_name: nim_llm
    description: "Summarizes the input text"

  keyword_agent:
    _type: react_agent
    tool_names: [current_datetime]
    llm_name: nim_llm
    description: "Extracts keywords from the input text"

workflow:
  _type: parallel_executor
  tool_list: [sentiment_agent, summary_agent, keyword_agent]
  description: "Runs sentiment, summary, and keyword extraction in parallel"
  detailed_logs: true
```

## Pattern: Sequential Pipeline of Agents

Chain agents where each one's output feeds the next:

```yaml
functions:
  research_agent:
    _type: react_agent
    tool_names: [web_search, wikipedia_search]
    llm_name: nim_llm
    description: "Gathers research on the input topic"

  analysis_agent:
    _type: react_agent
    tool_names: [calculator]
    llm_name: nim_llm
    description: "Analyzes the research findings"

  report_agent:
    _type: react_agent
    tool_names: [current_datetime]
    llm_name: nim_llm
    description: "Generates a final report from the analysis"

workflow:
  _type: sequential_executor
  tool_list: [research_agent, analysis_agent, report_agent]
```
