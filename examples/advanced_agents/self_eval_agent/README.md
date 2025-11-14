# Self-Evaluation Agent

An intelligent agent that aggregates contributions from JIRA, GitLab, and Confluence to generate comprehensive self-evaluation reports and top-contribution lists.

## Overview

The Self-Evaluation Agent helps you analyze your professional contributions across multiple platforms by:
- **Gathering data** from JIRA (issues, tickets), GitLab (commits, MRs), and Confluence (pages, documentation)
- **Analyzing contributions** by impact, complexity, and collaboration
- **Generating reports** suitable for quarterly reviews, performance evaluations, or quick summaries

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Examples](#basic-examples)
- [Architecture](#architecture)
- [Phase 1 Implementation](#phase-1-implementation)
- [Future Enhancements](#future-enhancements)

## Key Features

### Multi-Platform Integration
- **JIRA**: Issues created, updated, resolved, and comments
- **GitLab**: Commits, merge requests, and code reviews
- **Confluence**: Pages created, updated, and comments

### Intelligent Gatherers
- High-level tools that understand contribution patterns
- Generate platform-specific queries (JQL, CQL, API parameters)
- Work with MCP servers for secure, authenticated access

### Flexible Output
- Top-5 contributions list for quick updates
- Comprehensive self-evaluation reports for reviews
- Customizable time periods (30 days default)

### Extensible Design
- Can be exposed as an MCP server for other workflows
- Modular architecture for easy enhancements
- Built on NeMo Agent Toolkit patterns

## Installation

### Prerequisites

1. **NeMo Agent Toolkit**: Install from source following the [Install Guide](../../../docs/source/quick-start/installing.md)
2. **API Keys**: Obtain an NVIDIA API key for NIM access
3. **MCP Servers** (for full functionality): Access to JIRA, GitLab, and Confluence MCP servers

### Install This Workflow

From the NeMo Agent Toolkit root directory:

```bash
uv pip install -e examples/advanced_agents/self_eval_agent
```

### Set Up Environment Variables

```bash
# Required for all modes
export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>

# Required for full MCP integration
export CORPORATE_MCP_JIRA_URL="https://your-jira-mcp-server.com/mcp"
export CORPORATE_MCP_GITLAB_URL="https://your-gitlab-mcp-server.com/mcp"
export CORPORATE_MCP_CONFLUENCE_URL="https://your-confluence-mcp-server.com/mcp"
export NAT_USER_ID="your.email@company.com"

# Optional: For token storage
export MINIO_ENDPOINT_URL="http://localhost:9000"
export MINIO_ACCESS_KEY="your_minio_access_key"
export MINIO_SECRET_KEY="your_minio_secret_key"
export MINIO_BUCKET_NAME="nat-mcp-bucket"
```

## Configuration

### Main Configuration (`config.yml`)

The main configuration connects to JIRA, GitLab, and Confluence MCP servers and provides full contribution gathering capabilities:

- **MCP Clients**: Direct access to platform APIs
- **Gatherer Tools**: High-level contribution collection
- **OAuth2 Authentication**: Secure token management
- **ReAct Agent**: Intelligent orchestration

### Raw Configuration (`config_raw.yml`)

Direct MCP client access without the self-eval agent wrapper (for comparison).

## Usage

### Basic Examples

#### Generate Top-5 Contributions

```bash
nat run --config_file examples/advanced_agents/self_eval_agent/configs/config.yml \
  --input "Generate my top 5 contributions from the last 30 days"
```

#### Comprehensive Self-Evaluation

```bash
nat run --config_file examples/advanced_agents/self_eval_agent/configs/config.yml \
  --input "Create a self-evaluation report for my quarterly review covering the last 90 days for user john.doe@company.com"
```

#### Analyze Specific Time Period

```bash
nat run --config_file examples/advanced_agents/self_eval_agent/configs/config.yml \
  --input "What were my major contributions in the last 2 weeks?"
```

## Architecture

### Component Overview

```
Self-Evaluation Agent
├── ReAct Agent (Orchestration)
├── MCP Clients (JIRA, GitLab, Confluence)
├── Gatherer Tools
│   ├── jira_gatherer - Generates JIRA queries
│   ├── gitlab_gatherer - Generates GitLab queries
│   └── confluence_gatherer - Generates Confluence queries
└── Data Models
    ├── ContributionItem
    ├── JiraContribution
    ├── GitLabContribution
    └── ConfluenceContribution
```

### How It Works

1. **User Request**: "Generate my top 5 contributions"
2. **Agent Analysis**: Determines which platforms to query
3. **Gatherer Tools**: Generate platform-specific queries
4. **MCP Clients**: Execute queries against MCP servers
5. **Data Aggregation**: Combine results from all platforms
6. **Report Generation**: Format output based on request

### Gatherer Tools (Phase 1)

The gatherer tools provide a **high-level abstraction** over platform-specific queries:

- **Input**: User ID, time period
- **Output**: JSON with query information, parameters, and instructions
- **Purpose**: Guides the ReAct agent on what data to collect

Example output from `jira_gatherer`:

```json
{
  "platform": "JIRA",
  "user_id": "john.doe@company.com",
  "time_period_days": 30,
  "queries": {
    "issues_created": {
      "jql": "creator = \"john.doe@company.com\" AND created >= -30d",
      "description": "Issues created by the user"
    },
    ...
  },
  "instructions": "Use the JIRA MCP tools (jira.search_issues) with the provided JQL queries..."
}
```

## Phase 1 Implementation

### What's Complete ✅

- **Core Infrastructure**
  - Project structure and configuration
  - Data models for contributions
  - System prompts and agent framework

- **Gatherer Tools**
  - JIRA gatherer with JQL query generation
  - GitLab gatherer with API parameters
  - Confluence gatherer with CQL queries

- **ReAct Integration**
  - Agent orchestration using NAT ReAct agent
  - Tool registration and discovery
  - Configuration management

- **Documentation**
  - Complete README with examples
  - Configuration templates
  - Architecture overview

### Current Capabilities

The agent can:
- Generate appropriate queries for each platform
- Provide structured guidance for data collection
- Work with or without MCP server connections
- Be extended with additional analysis tools

### Limitations (Phase 1)

- Gatherers return query information rather than executing queries directly
- No automated ranking or scoring of contributions
- Report formatting is handled by the LLM (no custom templates yet)
- Single-user focused (no team aggregation)

## Future Enhancements

### Phase 2: Analysis & Ranking
- Direct MCP client integration in gatherers
- Contribution scoring algorithm (impact, complexity, collaboration)
- Automated deduplication across platforms
- Link correlation (JIRA tickets ↔ GitLab MRs)

### Phase 3: Report Generation
- Custom report templates (Markdown, HTML, PDF)
- Multiple report formats (top-N, full eval, timeline)
- Visualization support (charts, graphs)
- Metrics dashboard

### Phase 4: MCP Server Mode
- Expose self-eval agent as an MCP server
- Allow other workflows to consume it
- Team-level aggregation
- Historical trend analysis

## Troubleshooting

### Common Issues

**Issue**: "No module named 'nat_self_eval_agent'"
```bash
# Solution: Reinstall the package
uv pip install -e examples/advanced_agents/self_eval_agent
```

**Issue**: "MCP server connection failed"
```bash
# Solution: Check environment variables
echo $CORPORATE_MCP_JIRA_URL
# Verify MCP server is running and accessible
```

**Issue**: "OAuth authentication required"
```bash
# Solution: Ensure NAT_USER_ID is set and follow the OAuth flow
# The agent will open a browser for authentication
```

### Getting Help

- Check the [NeMo Agent Toolkit Documentation](../../../docs/source/index.md)
- Review [MCP Client Guide](../../../docs/source/workflows/mcp/mcp-client.md)
- See [MCP Authentication Guide](../../../docs/source/workflows/mcp/mcp-auth.md)

## Contributing

This is an advanced agent example. Contributions are welcome:

1. Follow the [NeMo Agent Toolkit contributing guidelines](../../../CONTRIBUTING.md)
2. Ensure all tests pass
3. Add documentation for new features
4. Follow the existing code structure

## License

Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0. See [LICENSE](../../../LICENSE) for details.
