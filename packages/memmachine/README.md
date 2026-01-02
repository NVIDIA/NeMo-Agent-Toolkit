# NVIDIA NeMo Agent Toolkit - MemMachine Integration

This package provides integration with MemMachine for memory management in NeMo Agent Toolkit.

## Overview

MemMachine is a unified memory management system that supports both episodic and semantic memory through a single interface. This integration allows you to use MemMachine as a memory backend for your NeMo Agent Toolkit workflows.

## Prerequisites

- Python 3.11+
- MemMachine server (install via `pip install memmachine-server`)

## Installation

Install the package:

```bash
pip install nvidia-nat-memmachine
```

Or for development:

```bash
uv pip install -e packages/memmachine
```

## MemMachine Server Setup

### Step 1: Install MemMachine Server

```bash
pip install memmachine-server
```

### Step 2: Run the Configuration Wizard

Before starting the server, you need to configure MemMachine using the interactive configuration wizard:

```bash
memmachine-configure
```

The wizard will guide you through setting up:

- **Neo4j Database**: Option to install Neo4j automatically or provide connection details for an existing instance
- **Large Language Model (LLM) Provider**: Choose from supported providers like OpenAI, AWS Bedrock, or Ollama
- **Model Selection**: Select specific LLM and embedding models
- **API Keys and Credentials**: Input necessary API keys for your selected LLM provider
- **Server Settings**: Configure server host and port

**Note**: 
- The wizard installs Neo4j and Java automatically when you choose to install Neo4j (platform-specific: Windows uses ZIP, macOS uses brew, Linux uses tar.gz)
- The wizard uses Neo4j as the vector database and SQLite as the relational database by default
- The configuration file will be generated at `<HOME>/.config/memmachine/cfg.yml`

### Step 3: Start the MemMachine Server

After completing the configuration wizard, start the server:

```bash
memmachine-server
```

The server will start on `http://localhost:8080` by default (or the port you configured).

For more details, see the [MemMachine Configuration Wizard Documentation](https://docs.memmachine.ai/open_source/configuration-wizard).

## Usage in NeMo Agent Toolkit

Add MemMachine memory to your workflow configuration:

```yaml
memory:
  memmachine_memory:
    base_url: "http://localhost:8080"  # MemMachine server URL
    org_id: "my_org"  # Optional: default organization ID
    project_id: "my_project"  # Optional: default project ID
```

## Additional Resources

- [MemMachine Documentation](https://docs.memmachine.ai/)
- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)

