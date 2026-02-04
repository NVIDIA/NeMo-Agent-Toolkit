# Prompt Optimizer Object Store: Implementer's Guide

**Audience**: Engineers implementing new object store backends that integrate with the prompt optimizer
**Last Updated**: 2026-02-04

## Overview

This guide explains how the prompt optimizer uses object stores and what your implementation needs to support. If you're building a new object store backend (e.g., Azure Blob, GCS, PostgreSQL), this document will help you understand the integration requirements.

## Architecture

```
┌─────────────────────────┐
│  Prompt Optimizer       │
│  (prompt_optimizer.py)  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  PromptStorage Protocol │  ← Domain-specific abstraction
└──────────┬──────────────┘
           │
           ├──────────────────────────────────┐
           ▼                                  ▼
┌──────────────────────┐      ┌───────────────────────────┐
│ LocalFilePrompt      │      │ ObjectStorePrompt         │
│ Storage              │      │ Storage                   │
└──────────────────────┘      └──────────┬────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────────┐
                              │  ObjectStore Interface  │ ← Your implementation goes here
                              └──────────┬──────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
           ┌────────────────┐  ┌──────────────┐   ┌──────────────────┐
           │ LocalFileOS    │  │  S3Store     │   │  YourNewStore    │
           └────────────────┘  └──────────────┘   └──────────────────┘
```

**Key Insight**: You don't interact directly with the prompt optimizer. You implement the `ObjectStore` interface, and `ObjectStorePromptStorage` handles the prompt-specific logic.

## ObjectStore Interface Contract

Your object store implementation must provide these async methods:

```python
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem

class YourObjectStore(ObjectStore):
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Store an object. Fail if key already exists.

        Args:
            key: Unique identifier (e.g., "my_experiment/optimized_prompts_gen1.json")
            item: ObjectStoreItem containing data, content_type, and metadata
        """
        ...

    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        """
        Store or update an object. Overwrite if key exists.

        Args:
            key: Unique identifier
            item: ObjectStoreItem containing data, content_type, and metadata
        """
        ...

    async def get_object(self, key: str) -> ObjectStoreItem:
        """
        Retrieve an object by key.

        Returns:
            ObjectStoreItem with data, content_type, and metadata

        Raises:
            KeyError: If key doesn't exist
        """
        ...

    async def delete_object(self, key: str) -> None:
        """
        Delete an object by key.

        Raises:
            KeyError: If key doesn't exist
        """
        ...
```

## Data Model: ObjectStoreItem

Every read/write uses this Pydantic model:

```python
from pydantic import BaseModel

class ObjectStoreItem(BaseModel):
    data: bytes                      # The actual content (JSON-encoded prompts)
    content_type: str                # Always "application/json" for prompts
    metadata: dict[str, str] | None  # String-keyed, string-valued metadata
```

### Metadata Field Requirements

**CRITICAL**: All metadata values must be strings. The prompt optimizer stores:

```python
metadata = {
    "generation": "1",                              # Generation number (string)
    "fitness_score": "0.8542",                      # Overall fitness (string float)
    "evaluator_scores": '{"accuracy": 0.85, ...}'   # JSON-encoded dict of scores
}
```

If your backend (like S3 or HTTP headers) only supports string metadata, this is already handled. If your backend supports rich types (like PostgreSQL JSON columns), you'll need to serialize to strings on write and deserialize on read.

## What the Prompt Optimizer Writes

### Write Pattern

During optimization, `ObjectStorePromptStorage` will call:

```python
# Generation checkpoints (called once per generation)
await your_object_store.upsert_object(
    key="my_experiment/optimized_prompts_gen1.json",
    item=ObjectStoreItem(
        data=b'{"functions.my_tool.prompt": ["optimized prompt text", "purpose"]}',
        content_type="application/json",
        metadata={
            "generation": "1",
            "fitness_score": "0.8542",
            "evaluator_scores": '{"accuracy": 0.85, "latency": 3.2}'
        }
    )
)

# Final prompts (called once at end)
await your_object_store.upsert_object(
    key="my_experiment/optimized_prompts.json",
    item=ObjectStoreItem(
        data=b'{"functions.my_tool.prompt": ["final prompt", "purpose"]}',
        content_type="application/json",
        metadata={
            "type": "final"
        }
    )
)
```

### Key Structure

Keys are constructed as: `{key_prefix}/{filename}`

- `key_prefix`: User-configured experiment identifier (e.g., `experiments/chatbot_v2/baseline`)
- `filename`: Fixed patterns:
  - `optimized_prompts_gen{N}.json` for generation N checkpoint
  - `optimized_prompts.json` for final result

**Example keys**:
- `experiments/chatbot_v2/baseline/optimized_prompts_gen1.json`
- `experiments/chatbot_v2/baseline/optimized_prompts_gen2.json`
- `experiments/chatbot_v2/baseline/optimized_prompts.json`

### Data Format

The `data` field is UTF-8 encoded JSON:

```json
{
  "functions.email_analyzer.prompt": [
    "Analyze the email content for phishing indicators...",
    "Detect phishing attempts in emails"
  ],
  "functions.email_analyzer.system_prompt": [
    "You are a security expert...",
    "System-level instructions"
  ]
}
```

Each entry is: `parameter_name -> [prompt_text, purpose]`

## Implementation Patterns

### Pattern 1: Native Metadata Support (S3, Azure, GCS)

If your backend supports object metadata natively:

```python
class S3ObjectStore(ObjectStore):
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        await self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=item.data,
            ContentType=item.content_type,
            Metadata=item.metadata or {}  # S3 metadata must be dict[str, str]
        )

    async def get_object(self, key: str) -> ObjectStoreItem:
        response = await self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return ObjectStoreItem(
            data=await response['Body'].read(),
            content_type=response['ContentType'],
            metadata=response.get('Metadata')  # S3 returns dict[str, str]
        )
```

### Pattern 2: Sidecar Metadata Files (LocalFileObjectStore)

If your backend doesn't support metadata, store it separately:

```python
class LocalFileObjectStore(ObjectStore):
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        # Write data file
        data_path = self.base_path / key
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(item.data)

        # Write metadata sidecar as {key}.meta
        if item.metadata:
            meta_path = self.base_path / f"{key}.meta"
            meta_data = {
                "content_type": item.content_type,
                "metadata": item.metadata
            }
            meta_path.write_text(json.dumps(meta_data, indent=2))

    async def get_object(self, key: str) -> ObjectStoreItem:
        # Read data file
        data_path = self.base_path / key
        if not data_path.exists():
            raise KeyError(f"Object not found: {key}")
        data = data_path.read_bytes()

        # Read metadata sidecar
        meta_path = self.base_path / f"{key}.meta"
        if meta_path.exists():
            meta_data = json.loads(meta_path.read_text())
            content_type = meta_data.get("content_type", "application/octet-stream")
            metadata = meta_data.get("metadata")
        else:
            content_type = "application/octet-stream"
            metadata = None

        return ObjectStoreItem(
            data=data,
            content_type=content_type,
            metadata=metadata
        )
```

### Pattern 3: Database Storage (Redis, MySQL, PostgreSQL)

If using a database, store as columns/fields:

```python
class PostgreSQLObjectStore(ObjectStore):
    # Table schema:
    # CREATE TABLE objects (
    #     key TEXT PRIMARY KEY,
    #     data BYTEA NOT NULL,
    #     content_type TEXT NOT NULL,
    #     metadata JSONB
    # );

    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        # Convert metadata dict to JSONB (PostgreSQL handles this)
        await self.pool.execute(
            """
            INSERT INTO objects (key, data, content_type, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (key) DO UPDATE SET
                data = EXCLUDED.data,
                content_type = EXCLUDED.content_type,
                metadata = EXCLUDED.metadata
            """,
            key,
            item.data,
            item.content_type,
            json.dumps(item.metadata) if item.metadata else None
        )

    async def get_object(self, key: str) -> ObjectStoreItem:
        row = await self.pool.fetchrow(
            "SELECT data, content_type, metadata FROM objects WHERE key = $1",
            key
        )
        if not row:
            raise KeyError(f"Object not found: {key}")

        return ObjectStoreItem(
            data=bytes(row['data']),
            content_type=row['content_type'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )
```

## Metadata Handling Guidelines

### String Conversion

The prompt optimizer stores numeric metadata as strings. Your implementation should:

1. **Accept strings as-is**: Don't parse or validate metadata values
2. **Return strings as-is**: Don't convert to numbers on read
3. **Preserve all fields**: Don't drop unknown metadata keys

**Correct**:
```python
# Store: {"fitness_score": "0.8542"}
# Return: {"fitness_score": "0.8542"}
```

**Incorrect**:
```python
# Store: {"fitness_score": "0.8542"}
# Return: {"fitness_score": 0.8542}  # ❌ Changed type!
```

### JSON-in-String Handling

The `evaluator_scores` metadata value is a JSON string:

```python
metadata = {
    "evaluator_scores": '{"accuracy": 0.85, "latency": 3.2}'  # Already a string!
}
```

**Don't parse it**. Treat it as an opaque string. The application layer will parse it.

### Optional Metadata

Not all objects have metadata. Your implementation must handle:

```python
# With metadata
item = ObjectStoreItem(data=..., content_type="...", metadata={"key": "value"})

# Without metadata
item = ObjectStoreItem(data=..., content_type="...", metadata=None)
```

## Testing Your Implementation

### Required Test Cases

1. **Write and Read Cycle**:
   ```python
   item = ObjectStoreItem(
       data=b'{"test": "data"}',
       content_type="application/json",
       metadata={"generation": "1", "fitness_score": "0.85"}
   )
   await store.upsert_object("test/key.json", item)
   retrieved = await store.get_object("test/key.json")
   assert retrieved.data == item.data
   assert retrieved.metadata == item.metadata
   ```

2. **Metadata String Preservation**:
   ```python
   # Ensure numeric strings aren't converted to numbers
   item = ObjectStoreItem(
       data=b'{}',
       content_type="application/json",
       metadata={"fitness_score": "0.123456789"}
   )
   await store.upsert_object("test/key.json", item)
   retrieved = await store.get_object("test/key.json")
   assert retrieved.metadata["fitness_score"] == "0.123456789"
   assert isinstance(retrieved.metadata["fitness_score"], str)
   ```

3. **JSON-in-String Preservation**:
   ```python
   # Ensure JSON strings aren't parsed
   scores_json = '{"accuracy": 0.85, "latency": 3.2}'
   item = ObjectStoreItem(
       data=b'{}',
       content_type="application/json",
       metadata={"evaluator_scores": scores_json}
   )
   await store.upsert_object("test/key.json", item)
   retrieved = await store.get_object("test/key.json")
   assert retrieved.metadata["evaluator_scores"] == scores_json
   assert isinstance(retrieved.metadata["evaluator_scores"], str)
   ```

4. **Upsert Overwrite**:
   ```python
   # First write
   item1 = ObjectStoreItem(data=b'{"v": 1}', content_type="...", metadata={"gen": "1"})
   await store.upsert_object("test/key.json", item1)

   # Overwrite
   item2 = ObjectStoreItem(data=b'{"v": 2}', content_type="...", metadata={"gen": "2"})
   await store.upsert_object("test/key.json", item2)

   # Should return item2
   retrieved = await store.get_object("test/key.json")
   assert retrieved.data == b'{"v": 2}'
   assert retrieved.metadata["gen"] == "2"
   ```

5. **Key Not Found**:
   ```python
   with pytest.raises(KeyError):
       await store.get_object("nonexistent/key.json")
   ```

### Integration Test with Prompt Optimizer

The NAT repository includes an integration test at:
`examples/evaluation_and_profiling/email_phishing_analyzer/configs/config_optimizer_test.yml`

To test your object store with the prompt optimizer:

1. Register your object store backend
2. Configure it in the test config:
   ```yaml
   object_stores:
     test_store:
       _type: your_backend_type
       # ... your backend config

   optimizer:
     object_store:
       name: test_store
       key_prefix: integration_test
   ```
3. Run: `nat optimize --config_file config_optimizer_test.yml`
4. Verify checkpoint files are created with metadata

## Common Pitfalls

### ❌ Converting Metadata Types

**Wrong**:
```python
# Don't do this!
if key == "fitness_score":
    metadata[key] = float(value)  # ❌ Changes type
```

**Right**:
```python
# Store as-is
metadata[key] = value  # ✓ Keep as string
```

### ❌ Parsing JSON Metadata Values

**Wrong**:
```python
# Don't do this!
if key == "evaluator_scores":
    metadata[key] = json.loads(value)  # ❌ Parses JSON
```

**Right**:
```python
# Treat as opaque string
metadata[key] = value  # ✓ Keep as JSON string
```

### ❌ Dropping Metadata on Update

**Wrong**:
```python
# Don't do this!
async def upsert_object(self, key: str, item: ObjectStoreItem):
    # Only updates data, loses metadata
    self.storage[key] = item.data  # ❌ Lost metadata!
```

**Right**:
```python
# Store everything
async def upsert_object(self, key: str, item: ObjectStoreItem):
    self.storage[key] = item  # ✓ Full object
```

### ❌ Not Handling Missing Keys

**Wrong**:
```python
async def get_object(self, key: str) -> ObjectStoreItem:
    return self.storage[key]  # ❌ Wrong exception type
```

**Right**:
```python
async def get_object(self, key: str) -> ObjectStoreItem:
    try:
        return self.storage[key]
    except KeyError:
        raise KeyError(f"Object not found: {key}")  # ✓ Correct exception
```

## Registration and Configuration

Once implemented, register your object store:

```python
from nat.object_store.interfaces import ObjectStoreBaseConfig, register_object_store

class YourObjectStoreConfig(ObjectStoreBaseConfig, name="your_backend"):
    # Your backend-specific config fields
    connection_string: str
    database: str

@register_object_store(config_type=YourObjectStoreConfig)
async def create_your_object_store(config: YourObjectStoreConfig, builder):
    yield YourObjectStore(
        connection_string=config.connection_string,
        database=config.database
    )
```

Users can then configure it:

```yaml
object_stores:
  my_store:
    _type: your_backend
    connection_string: "..."
    database: "..."

optimizer:
  object_store:
    name: my_store
    key_prefix: my_experiment
```

## Summary Checklist

- [ ] Implement `put_object`, `upsert_object`, `get_object`, `delete_object`
- [ ] Use `ObjectStoreItem` model with `data`, `content_type`, `metadata` fields
- [ ] Preserve metadata as `dict[str, str]` (string keys and values)
- [ ] Never convert metadata values to other types
- [ ] Handle missing keys with `KeyError`
- [ ] Support optional metadata (can be `None`)
- [ ] Test with numeric strings, JSON strings, and special characters
- [ ] Register with `@register_object_store` decorator
- [ ] Run integration test with prompt optimizer

## Questions?

If you have questions about integration requirements or encounter edge cases not covered here, please:

1. Check the reference implementations:
   - `packages/nvidia_nat_core/src/nat/object_store/local_file.py` (sidecar pattern)
   - `nat.plugins.s3` (native metadata pattern)
2. Review the tests:
   - `packages/nvidia_nat_core/tests/nat/profiler/parameter_optimization/test_prompt_storage.py`
3. Contact the NAT team

## Appendix: Real-World Example

Here's what a complete optimization run looks like from your object store's perspective:

```python
# Generation 1
await store.upsert_object(
    key="exp_v1/optimized_prompts_gen1.json",
    item=ObjectStoreItem(
        data=b'{"functions.analyzer.prompt": ["Check for phishing...", "Security"]}',
        content_type="application/json",
        metadata={
            "generation": "1",
            "fitness_score": "0.6234",
            "evaluator_scores": '{"accuracy": 0.65, "token_efficiency": 492.4, "llm_latency": 3.58}'
        }
    )
)

# Generation 2
await store.upsert_object(
    key="exp_v1/optimized_prompts_gen2.json",
    item=ObjectStoreItem(
        data=b'{"functions.analyzer.prompt": ["Analyze email for threats...", "Security"]}',
        content_type="application/json",
        metadata={
            "generation": "2",
            "fitness_score": "0.8124",
            "evaluator_scores": '{"accuracy": 0.82, "token_efficiency": 475.2, "llm_latency": 2.91}'
        }
    )
)

# ... more generations ...

# Final result
await store.upsert_object(
    key="exp_v1/optimized_prompts.json",
    item=ObjectStoreItem(
        data=b'{"functions.analyzer.prompt": ["Examine the email content thoroughly...", "Security"]}',
        content_type="application/json",
        metadata={"type": "final"}
    )
)
```

Your implementation should handle this sequence correctly, preserving all data and metadata across writes.
