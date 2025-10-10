"""Constants for Vanna text-to-SQL functionality."""

# Maximum limit size for vector search
MAX_LIMIT_SIZE = 1000

# Reasoning models that require special handling (think tags removal, JSON extraction)
REASONING_MODEL_VALUES = {
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3_3-nemotron-super-49b-v1_5",
    "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-r1",
}

# Chat models that use standard response handling
CHAT_MODEL_VALUES = {
    "meta/llama-3.1-70b-instruct",
}
