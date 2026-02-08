# Dynamo LLM nvext.annotations Support

**Date:** 2026-02-08
**Status:** Approved

## Overview

Extend `dynamo_llm.py` to support sending prefix routing hints via `nvext.annotations` in the request body, in addition to the existing HTTP header mechanism. This provides compatibility with both generalized and optimized Dynamo Thompson Sampling setups.

## Background

The current implementation only sends prefix hints via HTTP headers (`x-prefix-*`). A colleague implemented nvext.annotations support in a stale branch using a custom transport approach. We need to integrate this mechanism while preserving all existing functionality (prediction trie support, depth-aware prefix context).

## Transport Mechanisms

The implementation will support two transport mechanisms simultaneously:

1. **HTTP Headers** (`x-prefix-*`): For generalized Thompson Sampling setup with custom frontend.py
2. **nvext.annotations** (request body): For optimized Thompson Sampling setup with default frontend + custom processor.py (preferred mechanism)

## Architecture

### Custom Transport Layer

Replace the current event hooks approach with a custom `_DynamoTransport` class that wraps the base httpx transport. This provides:

- More reliable request body modification (before httpx processes the request)
- Unified handling of both headers and annotations
- Single point of control for static values and prediction overrides

### Integration with Existing Features

**Depth-aware Prefix Context:**
- `DynamoPrefixContext.get()` continues to provide prefix IDs
- Supports both automatic depth-based IDs and manual overrides
- No changes needed to the context management

**Prediction Trie Support:**
- When `prediction_trie_path` is configured, predictions override BOTH headers AND annotations
- Keeps both transport mechanisms in sync
- Falls back to static config values when no prediction found

## Implementation Details

### _DynamoTransport Class

```python
class _DynamoTransport:
    def __init__(self, transport, total_requests, osl, iat, prediction_lookup):
        self._transport = transport
        self._total_requests = total_requests  # Static values from config
        self._osl = osl
        self._iat = iat
        self._prediction_lookup = prediction_lookup  # Optional

    async def handle_async_request(self, request):
        # 1. Get prefix_id from DynamoPrefixContext (depth-aware)
        # 2. Check prediction_lookup for dynamic override
        # 3. Inject HTTP headers
        # 4. Inject nvext.annotations in JSON body
        # 5. Forward to base transport
```

### Request Processing Flow

1. **Prefix ID:** Retrieved from `DynamoPrefixContext.get()`
2. **Prediction override (if configured):**
   - Look up prediction using function path + call index from context
   - Convert numeric predictions to categories (LOW/MEDIUM/HIGH)
   - Override both header and annotation values
3. **Header injection:** Set `x-prefix-id`, `x-prefix-total-requests`, `x-prefix-osl`, `x-prefix-iat`
4. **Body modification:**
   - Parse JSON body (if POST request)
   - Create/merge `nvext.annotations` array
   - Format: `["prefix_id:value", "total_requests:value", "osl:value", "iat:value"]`
   - Preserve existing annotations that don't conflict

### nvext.annotations Format

```json
{
  "model": "...",
  "messages": [...],
  "nvext": {
    "annotations": [
      "prefix_id:workflow-abc-d0",
      "total_requests:10",
      "osl:MEDIUM",
      "iat:LOW"
    ]
  }
}
```

## Changes to Existing Code

### Functions to Remove

- `_create_dynamo_request_hook()` - replaced by transport
- `_create_prediction_request_hook()` - replaced by transport
- `_create_dynamic_prediction_hook()` - replaced by transport
- `create_httpx_client_with_prediction_headers()` - obsolete, unified into main function

### Functions to Modify

- `create_httpx_client_with_dynamo_hooks()`: Use custom transport instead of event hooks

### Functions to Keep

- `_output_tokens_to_osl()` - category conversion helper
- `_interarrival_ms_to_iat()` - category conversion helper
- `DynamoPrefixContext` - no changes
- `DynamoModelConfig` - no changes

## Error Handling

**JSON parsing failures:**
- Log debug message, skip annotations injection
- Still inject HTTP headers (fallback)
- Don't fail the request

**Prediction lookup failures:**
- Log warning with error details
- Fall back to static config values
- Continue with request

**Context lookup failures:**
- Use default depth=0 if Context unavailable
- Generate fallback UUID-based prefix_id
- Same as current behavior

## Logging Strategy

**DEBUG level:**
- Prefix ID generation/retrieval
- Injected header values
- Injected annotation values with body size
- Prediction lookup results

**WARNING level:**
- JSON body modification failures
- Prediction lookup exceptions
- Missing workflow_run_id in context

## Testing

Unit tests in `packages/nvidia_nat_core/tests/nat/llm/`:

- Test `_DynamoTransport` with mock requests
- Verify both headers and annotations are injected
- Test prediction overrides affect both mechanisms
- Test graceful degradation when body modification fails
- Test depth-aware prefix IDs work correctly
- Test annotation merging with existing nvext.annotations

## Compatibility

**Backwards compatible:** Yes
- Existing code continues to work
- HTTP headers still sent (no breaking changes)
- New annotations mechanism added transparently

**Configuration changes:** None required
- Works with existing `DynamoModelConfig`
- Prediction trie support unchanged
