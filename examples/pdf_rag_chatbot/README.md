# PDF RAG Chatbot

This package provides an intelligent RAG chatbot for PDF documents using AIQ Toolkit with conversation memory and ordered retrieval capabilities.

## Features

- **🔄 Maximum AIQ Toolkit Usage**: Leverages built-in components (95% existing code)
- **📊 Ordered Retrieval**: Returns steps/procedures in correct sequence using chunk_index
- **🧠 Conversation Memory**: Milvus-based session memory (replaces mem0 due to bugs)
- **🔍 Dual Search Modes**: 
  - Ordered search for step-by-step instructions
  - General semantic search for concepts/definitions
- **📁 Document Management**: List and track available PDF documents
- **🎯 Smart Tool Selection**: Automatically chooses best retrieval method based on query type

## Architecture

**Built-in AIQ Toolkit Components (No Custom Code):**
- `milvus_retriever` - Vector database integration
- `react_agent` - Orchestration framework
- `nim` LLM & embeddings - NVIDIA models
- `aiq_retriever` - General RAG queries
- `current_datetime` - Timestamp utility

**Minimal Custom Tools (3 functions only):**
- `ordered_pdf_search` - Ordered retrieval for step-by-step queries
- `document_manager` - Document listing and management
- `milvus_conversation_memory` - Session-based conversation memory using Milvus

## Installation

1. **Install the package** (from the pdf_rag_chatbot directory):
```bash
pip install -e .
```

2. **Set environment variables**:
```bash
export NVIDIA_API_KEY="your_nvidia_api_key"
```

3. **Install in AIQ Toolkit** (from the main aiqtoolkit directory):
```bash
# Install the package as a plugin
uv pip install -e ../pdf_rag_chatbot
```

## Usage

### Validate Configuration
```bash
# From the aiqtoolkit directory
python -m aiq.cli.main validate --config_file ../pdf_rag_chatbot/configs/rag_chatbot_config.yml
```

### Run the RAG Chatbot

```bash
# From the pdf_rag_chatbot directory
aiq run --config_file configs/rag_chatbot_config.yml --input "What documents do you have?"
```

### Interactive Mode (Recommended)
```bash
# Start interactive chat session
aiq run --config_file configs/rag_chatbot_config.yml

# Then you can chat naturally:
```

## Example Conversations

### Document Discovery
```
User: What documents do you have?
Agent: 📚 Available Documents (3 total):

📄 installation_guide.pdf
   └─ Pages: 25
   └─ Chunks: 67
   └─ Ingested: 2025-01-15 14:30

📄 user_manual.pdf
   └─ Pages: 45
   └─ Chunks: 123
   └─ Ingested: 2025-01-15 14:32

📄 troubleshooting_guide.pdf
   └─ Pages: 18
   └─ Chunks: 45
   └─ Ingested: 2025-01-15 14:35
```

### Ordered Step-by-Step Queries
```
User: What are the installation steps?
Agent: [Uses ordered_search - returns chunks in sequence]

**Result 1** [📄 installation_guide - Page 2 - Chunk 3]
Step 1: Download the software package from the official website...

**Result 2** [📄 installation_guide - Page 2 - Chunk 4]  
Step 2: Extract the downloaded archive to your desired location...

**Result 3** [📄 installation_guide - Page 3 - Chunk 5]
Step 3: Run the installation script with administrator privileges...
```

### General Semantic Queries
```
User: What is CUDA?
Agent: [Uses general_search - returns best semantic matches]

Based on the documentation, CUDA (Compute Unified Device Architecture) is...
**Sources:** user_manual (page 5), installation_guide (page 1)
```

### Document-Specific Queries
```
User: Show me troubleshooting steps for GPU issues
Agent: [Uses ordered_search with filename filtering]

**Result 1** [📄 troubleshooting_guide - Page 8 - Chunk 23]
GPU Troubleshooting Step 1: Check device manager for driver issues...

**Result 2** [📄 troubleshooting_guide - Page 8 - Chunk 24]
GPU Troubleshooting Step 2: Verify power supply specifications...
```

### Conversation Memory
```
User: What are the system requirements?
Agent: [Provides requirements from documents]

User: Do I meet those requirements if I have 16GB RAM?
Agent: Based on the requirements I just mentioned, yes, 16GB RAM exceeds the minimum requirement of 8GB that we discussed...
```

## Query Types & Tool Selection

The agent automatically chooses the best tool based on your query:

| Query Type | Tool Used | Purpose |
|------------|-----------|---------|
| "steps to", "how to", "procedure" | `ordered_search` | Sequential chunks in order |
| "what is", "explain", "define" | `general_search` | Best semantic matches |
| "what documents", "list files" | `doc_manager` | Document inventory |
| Follow-up questions | All tools + memory | Context-aware responses |

## Configuration

Edit `configs/rag_chatbot_config.yml` to customize:

- **Memory settings**: Conversation retention and context
- **Retrieval parameters**: Number of results, search behavior
- **LLM settings**: Temperature, max tokens, model selection
- **Milvus connection**: Database credentials and collection

## Built vs Custom Components

**✅ Built-in AIQ Toolkit (95% of functionality):**
- Vector database operations
- Conversation memory management
- LLM orchestration and reasoning
- Embedding generation
- Agent framework and tools
- Configuration management

**🔧 Custom Components (5% of functionality):**
- Ordered retrieval logic (preserves chunk_index)
- Document metadata extraction and formatting

## Next Steps

After testing the RAG chatbot:

1. **Customize the system prompt** for your specific domain
2. **Adjust retrieval parameters** based on your document types
3. **Add more custom tools** if needed for specialized operations
4. **Deploy with FastAPI frontend** using aiqtoolkit's built-in UI
5. **Scale with multiple collections** for different document types

## Troubleshooting

**Configuration validation fails:**
```bash
# Ensure package is installed
uv pip install -e ../pdf_rag_chatbot
python -c "from pdf_rag_chatbot.register import ordered_pdf_search; print('✅ Functions registered')"
```

**Memory errors:**
```bash
# Check Milvus connectivity for conversation memory
# Test connection to your Milvus instance
```

**No search results:**
```bash
# Verify your documents are ingested with metadata
aiq run --config_file configs/rag_chatbot_config.yml --input "What documents do you have?"
```

This RAG chatbot maximizes AIQ Toolkit's built-in capabilities while providing the ordered retrieval and conversation memory you need! 