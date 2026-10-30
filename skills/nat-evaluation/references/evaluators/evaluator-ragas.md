# Ragas Integration Removed

The Ragas package and extra have been removed.
Do not install `nvidia-nat[ragas]` or generate `_type: ragas` configurations.

Use [the tunable RAG evaluator](evaluator-tunable-rag.md) with explicit `judge_llm_prompt` rubrics.
Set `include_context: true` for groundedness and context relevance, and allow enough output tokens
for JSON scores and explanations. `enable_atif_evaluator: true` supports ATIF samples.
These prompts are not equivalent to Ragas metrics; rebaseline quality and optimizer thresholds.
