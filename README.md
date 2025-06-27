# Simple Setup Guide for NeMo Agent Toolkit

This guide shows how to quickly set up and run the NeMo Agent Toolkit servers and UI based on a real installation.

## Prerequisites

- Python 3.12+
- Node.js v18+
- Git with Git LFS
- UV package manager

## Installation Steps

### 1. Setup Git LFS and Fetch Data

```bash
git lfs install
git lfs fetch
git lfs pull
```

### 2. Create Virtual Environment

```bash
uv venv --seed .venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Install All Dependencies

```bash
uv sync --all-groups --all-extras
```

This installs the complete NeMo Agent Toolkit with all frameworks and examples.

### 5. Set Environment Variables

```bash
# Required API Keys
$env:NVIDIA_API_KEY="your_nvidia_api_key_here"

# Optional API Keys (add as needed)
$env:GITLAB_API_TOKEN="your_gitlab_token"
$env:PAGERDUTY_API_TOKEN="your_pagerduty_token"
$env:PAGERDUTY_API_URL="https://api.pagerduty.com"
$env:CONFLUENCE_API_TOKEN="your_confluence_token"
$env:CONFLUENCE_BASE_URL="https://confluence.nvidia.com/"
$env:LOKI_API_TOKEN="your_loki_token"
```

### 6. Install Specific Examples (Optional)

```bash
# Install PDF RAG examples
uv pip install -e examples/pdf_rag_chatbot
uv pip install -e examples/pdf_rag_ingest

# Install Friday assistant
uv pip install -e examples/friday
```

## Starting the Servers

### Option 1: Using Batch Files (Windows)

**Start Backend Servers:**
```bash
./start_servers.bat
```

This starts:
- ON_CALL backend: http://localhost:8000
- FRIDAY backend: http://localhost:8001  
- SLACK backend: http://localhost:8002

**Start UI:**
```bash
./start_ui.bat
```

### Option 2: Manual Startup

**Start a Single Server:**
```bash
aiq serve --config_file=examples/simple_calculator/configs/config.yml --port 8000
```

**Start UI:**
```bash
cd external/aiqtoolkit-opensource-ui
npm install
npm run dev
```

## Access Points

- **Frontend UI**: http://localhost:3000
- **ON_CALL Backend**: http://localhost:8000
- **FRIDAY Backend**: http://localhost:8001
- **SLACK Backend**: http://localhost:8002

## Verification

Check if AIQ Toolkit is installed correctly:
```bash
aiq --version
```

## Notes

- The UI includes a mode switcher to toggle between different backends (ON_CALL, FRIDAY, SLACK)
- All servers run simultaneously and can be accessed independently
- Make sure all environment variables are set before starting servers
- The installation includes all major frameworks: LangChain, LlamaIndex, CrewAI, Semantic Kernel, etc.

## Quick Test

Test a simple calculation:
```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "What is 2 + 2?",
    "use_knowledge_base": true
  }'
```

## Troubleshooting

If you encounter issues:
1. Ensure all environment variables are set
2. Check that ports 8000, 8001, 8002, and 3000 are available
3. Verify Git LFS files are properly downloaded
4. Make sure Node.js is installed for the UI component 