# NVIDIA RAG Library Integration

This package provides seamless integration between the NVIDIA RAG Library and the NeMo Agent Toolkit, allowing developers to easily incorporate retrieval-augmented generation capabilities into their agent workflows.

## Features

- Direct access to NVIDIA RAG Library functionality
- Configurable environment setup and prerequisites verification
- Retry logic for robust operation
- Support for various deployment modes (on-premises, hosted, mixed)
- Integration with NAT's component reference system

## Installation

Install as part of the NeMo Agent Toolkit:

```bash
pip install "nvidia-nat[rag-lib]"
```

## Usage

Configure the RAG library in your workflow YAML file and use it as a function in your agent setup.
