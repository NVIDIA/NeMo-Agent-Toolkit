<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prompt Optimizer Object Store Integration Design

**Date:** 2026-02-04
**Status:** Approved
**Author:** Design discussion with user

## Overview

This design introduces object store integration for the prompt optimizer, enabling developers to persist optimization checkpoints and final prompts to any storage backend (local files, S3, custom stores) while maintaining backward compatibility with the current file-based approach.

## Motivation

Currently, the prompt optimizer writes checkpoint and final prompt files directly to the filesystem using `output_path`. This design extends the system to:

1. Support pluggable storage backends through the existing object store abstraction
2. Enable remote storage for distributed optimization scenarios
3. Maintain backward compatibility for existing workflows
4. Set foundation for future resume/restart functionality

## Scope

**In Scope:**
- Prompt checkpoints (`optimized_prompts_gen{N}.json`)
- Final optimized prompts (`optimized_prompts.json`)
- New LocalFileObjectStore implementation
- PromptStorage abstraction layer

**Out of Scope (for now):**
- CSV history files (remain file-based using `output_path`)
- Resume/restart functionality (interface designed to support future implementation)
- Reading checkpoints during optimization

## Architecture

### Component Overview

#### 1. PromptStorage Protocol

Location: `src/nat/profiler/parameter_optimization/prompt_storage.py`

Abstract interface defining storage operations for prompt optimizer:

```python
class PromptStorage(Protocol):
    async def save_checkpoint(
        self,
        generation: int,
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """Save generation checkpoint.

        Args:
            generation: Generation number (1-indexed)
            prompts: Map of param_name -> (prompt_text, purpose)
        """

    async def save_final(
        self,
        prompts: dict[str, tuple[str, str]]
    ) -> None:
        """Save final optimized prompts."""

    async def load_checkpoint(
        self,
        generation: int
    ) -> dict[str, tuple[str, str]]:
        """Load specific checkpoint. Raises KeyError if not found."""

    async def load_latest_checkpoint(
        self
    ) -> tuple[int, dict[str, tuple[str, str]]]:
        """Load most recent checkpoint. Returns (generation, prompts).

        For future resume support.
        """
```

The protocol uses semantic methods that understand the prompt optimizer's domain rather than generic key-value operations.

#### 2. LocalFilePromptStorage

Location: `src/nat/profiler/parameter_optimization/prompt_storage.py`

Implements PromptStorage using direct filesystem writes (current behavior).

**Behavior:**
- Stores prompts as JSON files: `{base_path}/{prefix}/optimized_prompts_gen{N}.json`
- If `key_prefix` is None, writes directly to `base_path` for backward compatibility
- If `key_prefix` is provided, creates subdirectory
- Does not use ObjectStore interface (direct file I/O)

**Constructor:**
```python
LocalFilePromptStorage(
    base_path: Path,
    key_prefix: str | None = None
)
```

#### 3. ObjectStorePromptStorage

Location: `src/nat/profiler/parameter_optimization/prompt_storage.py`

Implements PromptStorage using ObjectStore interface.

**Behavior:**
- Constructs keys: `{prefix}/optimized_prompts_gen{N}.json`, `{prefix}/optimized_prompts.json`
- If `key_prefix` is None, generates timestamp-based prefix: `prompt_opt_{YYYYMMDD_HHMMSS}`
- Serializes prompts dict to JSON bytes
- Wraps in ObjectStoreItem with `content_type="application/json"`
- Delegates storage to object store client

**Constructor:**
```python
ObjectStorePromptStorage(
    object_store: ObjectStore,
    key_prefix: str | None = None
)
```

#### 4. LocalFileObjectStore

Location: `src/nat/object_store/local_file.py`

New ObjectStore implementation for local filesystem.

**Config:**
```python
class LocalFileObjectStoreConfig(ObjectStoreBaseConfig, name="local_file"):
    base_path: str  # Base directory for all storage
```

**Storage Format:**
- Data: `{base_path}/{key}`
- Metadata: `{base_path}/{key}.meta` (JSON sidecar)

Metadata file contains:
```json
{
  "content_type": "application/json",
  "metadata": {"key": "value"}
}
```

**Key Behaviors:**
- `put_object()`: Write data + metadata files, raise `KeyAlreadyExistsError` if exists
- `upsert_object()`: Write/overwrite both files
- `get_object()`: Read both files, construct ObjectStoreItem, raise `NoSuchKeyError` if missing
- `delete_object()`: Remove both files, raise `NoSuchKeyError` if not found
- Create parent directories as needed
- Handle missing `.meta` file gracefully (return None for content_type/metadata)

**Registration:**
```python
@register_object_store(config_type=LocalFileObjectStoreConfig)
async def local_file_object_store(
    config: LocalFileObjectStoreConfig,
    _builder: Builder
):
    from .local_file import LocalFileObjectStore
    yield LocalFileObjectStore(base_path=Path(config.base_path))
```

#### 5. Configuration Changes

**OptimizerConfig additions:**
```python
class ObjectStoreSettings(BaseModel):
    name: str  # Reference to object store in config
    key_prefix: str | None = None  # Optional prefix, auto-generated if None

class OptimizerConfig(BaseModel):
    # ... existing fields ...
    object_store: ObjectStoreSettings | None = None
```

### Data Flow

#### Storage Selection (Startup)

```python
async def optimize_prompts(...):
    # Determine storage backend
    if optimizer_config.object_store:
        # Use object store
        store = await builder.get_object_store_client(
            optimizer_config.object_store.name
        )
        storage = ObjectStorePromptStorage(
            object_store=store,
            key_prefix=optimizer_config.object_store.key_prefix
        )
    else:
        # Fallback to file-based (backward compatible)
        storage = LocalFilePromptStorage(
            base_path=optimizer_config.output_path,
            key_prefix=None
        )
```

#### During GA Loop

Replace current checkpoint writes:

```python
# Before (current):
checkpoint_path = out_dir / f"optimized_prompts_gen{gen}.json"
with checkpoint_path.open("w") as fh:
    json.dump(checkpoint, fh, indent=2)

# After (with storage):
await storage.save_checkpoint(generation=gen, prompts=checkpoint)
```

CSV history continues using direct file writes to `output_path`.

#### After Optimization

Replace final prompt writes:

```python
# Before (current):
final_prompts_path = out_dir / "optimized_prompts.json"
with final_prompts_path.open("w") as fh:
    json.dump(best_prompts, fh, indent=2)

# After (with storage):
await storage.save_final(prompts=best_prompts)
```

### Error Handling

**Storage Operation Failures:**
- Wrap storage calls in try/except
- Log warnings for checkpoint save failures but continue optimization
- Log errors prominently for final save failures but still mark optimization as completed
- Don't crash the GA loop due to storage issues

**Object Store Connection Errors:**
- Fail early during storage construction (before GA loop starts)
- Provide clear error message indicating which object store failed

**Backward Compatibility:**
- When `object_store` is not configured, behavior is identical to current implementation
- File paths and formats unchanged
- CSV history location unchanged

## Configuration Examples

### Example 1: Using LocalFileObjectStore

```yaml
object_stores:
  local_store:
    _type: local_file
    base_path: /tmp/my_prompts

optimizer:
  output_path: ./optimizer_outputs  # Used for CSV history
  object_store:
    name: local_store
    key_prefix: experiment_001  # Optional, creates subdir
  eval_metrics:
    accuracy:
      evaluator_name: exact_match
      direction: maximize
      weight: 1.0
  prompt:
    ga_population_size: 10
    ga_generations: 5
    # ... other GA params ...
```

**Result:**
- Checkpoints: `/tmp/my_prompts/experiment_001/optimized_prompts_gen1.json`, etc.
- Final: `/tmp/my_prompts/experiment_001/optimized_prompts.json`
- CSV: `./optimizer_outputs/ga_history_prompts.csv`

### Example 2: Backward Compatible (No Object Store)

```yaml
optimizer:
  output_path: ./optimizer_outputs
  # object_store not specified
  eval_metrics: {...}
  prompt: {...}
```

**Result:**
- Checkpoints: `./optimizer_outputs/optimized_prompts_gen1.json`, etc.
- Final: `./optimizer_outputs/optimized_prompts.json`
- CSV: `./optimizer_outputs/ga_history_prompts.csv`

### Example 3: Using S3 (Future Use Case)

```yaml
object_stores:
  s3_store:
    _type: s3
    bucket_name: my-prompts
    endpoint_url: https://s3.amazonaws.com
    access_key: ${AWS_ACCESS_KEY}
    secret_key: ${AWS_SECRET_KEY}

optimizer:
  output_path: ./optimizer_outputs  # Still used for CSV
  object_store:
    name: s3_store
    key_prefix: null  # Auto-generate timestamp prefix
  eval_metrics: {...}
  prompt: {...}
```

**Result:**
- Checkpoints: `s3://my-prompts/prompt_opt_20260204_123456/optimized_prompts_gen1.json`
- Final: `s3://my-prompts/prompt_opt_20260204_123456/optimized_prompts.json`
- CSV: `./optimizer_outputs/ga_history_prompts.csv`

## Testing Strategy

### Unit Tests

**LocalFileObjectStore** (`tests/nat/object_store/test_local_file.py`):
- Test all operations: put, upsert, get, delete
- Test error conditions: KeyAlreadyExistsError, NoSuchKeyError
- Test metadata handling (with and without metadata)
- Test nested key paths (`foo/bar/baz.json`)
- Test missing `.meta` file handling

**PromptStorage Implementations** (`tests/nat/profiler/parameter_optimization/test_prompt_storage.py`):
- Test LocalFilePromptStorage:
  - Checkpoint and final save/load
  - Key prefix behavior (None vs provided)
  - JSON serialization correctness
- Test ObjectStorePromptStorage:
  - Mock object store backend
  - Verify key construction
  - Verify ObjectStoreItem wrapping
  - Test auto-generated prefix format

### Integration Tests

Extend existing prompt optimizer tests:
- Run small GA optimization (2 generations, 4 individuals) with LocalFileObjectStore
- Verify checkpoint files created with correct content
- Verify final prompts file created
- Verify CSV history still written to output_dir
- Test with and without key_prefix

## Implementation Plan

1. **Create LocalFileObjectStore**
   - Implement `src/nat/object_store/local_file.py`
   - Add config class and registration
   - Write unit tests

2. **Create PromptStorage abstraction**
   - Define protocol in `src/nat/profiler/parameter_optimization/prompt_storage.py`
   - Implement LocalFilePromptStorage
   - Implement ObjectStorePromptStorage
   - Write unit tests

3. **Update OptimizerConfig**
   - Add ObjectStoreSettings model
   - Add object_store field to OptimizerConfig
   - Update config validation

4. **Integrate into prompt_optimizer.py**
   - Add storage selection logic at startup
   - Replace checkpoint writes with `storage.save_checkpoint()`
   - Replace final writes with `storage.save_final()`
   - Add error handling around storage operations
   - Preserve CSV history writes

5. **Add integration tests**
   - Create test config with LocalFileObjectStore
   - Run end-to-end optimization test
   - Verify outputs

6. **Update documentation**
   - Add object store section to optimizer docs
   - Add LocalFileObjectStore to object store provider docs
   - Update examples with object store configs

## Future Extensions

**Resume/Restart Support:**
The PromptStorage interface already includes `load_checkpoint()` and `load_latest_checkpoint()` methods. Future work:
- Add `resume_from_generation` config option
- Load checkpoint at startup if configured
- Initialize population from checkpoint
- Skip already-completed generations

**Additional Storage Backends:**
The abstraction supports any backend that implements ObjectStore:
- Redis (key-value store)
- PostgreSQL (with JSONB columns)
- Cloud storage (Azure Blob, GCS)

**Metadata Enhancements:**
Store additional metadata with checkpoints:
- Fitness scores
- Generation timestamp
- Configuration hash
- Random seed

## Questions and Decisions

**Q: Should CSV history also use object store?**
A: No, keep it file-based for now. CSV is useful for local analysis and doesn't benefit from object store as much as prompts.

**Q: How to handle backward compatibility?**
A: Make object_store config optional. When not provided, use current file-based behavior.

**Q: Where to generate timestamp prefix?**
A: In ObjectStorePromptStorage constructor when key_prefix is None.

**Q: Should LocalFileObjectStore be the automatic default?**
A: No, maintain current direct file write behavior as default. LocalFileObjectStore is opt-in for testing/consistency.

**Q: How to store ObjectStoreItem metadata?**
A: Use sidecar files (`.meta`) to keep data files human-readable while maintaining full ObjectStore interface fidelity.
