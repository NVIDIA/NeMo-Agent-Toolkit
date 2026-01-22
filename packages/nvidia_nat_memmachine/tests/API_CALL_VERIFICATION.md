# API Call Verification Tests

## Overview

The tests in `test_memmachine_api_calls.py` verify that the MemMachine integration makes **correct API calls** to the MemMachine SDK. These tests use spies to capture and validate:

1. **Exact SDK methods called** - Verifies the right methods are invoked
2. **Parameter correctness** - Verifies parameters match SDK expectations
3. **Data transformations** - Verifies NAT format → MemMachine format conversion
4. **Memory type handling** - Verifies all memories are added to both episodic and semantic types

## What Gets Tested

### 1. Add Operations (`add_items`)

#### Conversation Handling
- ✅ Each message in conversation calls `memory.add()` separately
- ✅ Messages preserve their roles (user/assistant/system)
- ✅ Messages are added in the correct order
- ✅ All memories are added to both episodic and semantic memory types

#### Direct Memory (No Conversation)
- ✅ When `conversation=None`, uses `memory` field as content
- ✅ All memories are added to both episodic and semantic memory types
- ✅ Default `role` is "user"

#### Metadata Handling
- ✅ Tags are included in metadata dict
- ✅ Custom metadata fields are preserved
- ✅ Special fields (session_id, agent_id, project_id, org_id) are extracted and NOT passed to `add()`
- ✅ Empty metadata becomes `None`, not empty dict

#### Project/Org Handling
- ✅ Custom `project_id`/`org_id` in metadata triggers `create_project()` call
- ✅ `project.memory()` is called with correct user_id, session_id, agent_id, group_id

### 2. Search Operations (`search`)

#### Parameter Mapping
- ✅ NAT's `top_k` parameter is converted to SDK's `limit` parameter
- ✅ `query` parameter is passed correctly
- ✅ `project.memory()` is called with user_id, session_id, agent_id, group_id

#### Custom Project/Org
- ✅ Custom `project_id`/`org_id` triggers `create_project()` call

### 3. Remove Operations (`remove_items`)

#### Episodic Deletion
- ✅ Calls `memory.delete_episodic(episodic_id=...)` with correct ID

#### Semantic Deletion
- ✅ Calls `memory.delete_semantic(semantic_id=...)` with correct ID

### 4. API Call Format

#### Keyword Arguments
- ✅ All SDK methods are called with keyword arguments, not positional
- ✅ Parameter names match SDK exactly (`limit` not `top_k`, `episodic_id` not `memory_id`)

## Test Structure

Each test class focuses on a specific aspect:

- **`TestAddItemsAPICalls`** - Verifies `add_items()` makes correct `memory.add()` calls
- **`TestSearchAPICalls`** - Verifies `search()` makes correct `memory.search()` calls  
- **`TestRemoveItemsAPICalls`** - Verifies `remove_items()` makes correct delete calls
- **`TestAPICallParameterValidation`** - Verifies parameter names and formats
- **`TestDataTransformation`** - Verifies data transformations are correct

## Running the Tests

```bash
# Run all API call verification tests
pytest tests/test_memmachine_api_calls.py -v

# Run a specific test class
pytest tests/test_memmachine_api_calls.py::TestAddItemsAPICalls -v

# Run a specific test
pytest tests/test_memmachine_api_calls.py::TestAddItemsAPICalls::test_add_conversation_calls_add_with_correct_parameters -v
```

## Key Differences from Unit Tests

| Aspect | Unit Tests (`test_memmachine_editor.py`) | API Call Tests (`test_memmachine_api_calls.py`) |
|--------|------------------------------------------|--------------------------------------------------|
| **Focus** | Integration logic, error handling | Exact API calls and parameters |
| **Verification** | Mocks return values | Spies capture actual calls |
| **What's Tested** | Code flow, edge cases | Parameter correctness, data transformation |
| **SDK Methods** | Mocked, not verified | Spied, parameters validated |

## Example: What Gets Verified

### Adding a Conversation

**Input (NAT format):**
```python
MemoryItem(
    conversation=[
        {"role": "user", "content": "I like pizza"},
        {"role": "assistant", "content": "Great!"}
    ],
    user_id="user123",
    metadata={"session_id": "s1", "agent_id": "a1"},
    tags=["food"]
)
```

**Verified API Calls:**
1. ✅ `create_project(org_id="default-org", project_id="default-project", ...)`
2. ✅ `project.memory(user_id="user123", session_id="s1", agent_id="a1", group_id="default")`
3. ✅ `memory.add(content="I like pizza", role="user", memory_types=[Episodic, Semantic], metadata={"tags": ["food"]})`
4. ✅ `memory.add(content="Great!", role="assistant", memory_types=[Episodic, Semantic], metadata={"tags": ["food"]})`

**What's Verified:**
- ✅ Two `add()` calls (one per message)
- ✅ Correct roles preserved
- ✅ All memories added to both episodic and semantic types
- ✅ Tags in metadata
- ✅ session_id/agent_id NOT in metadata (used for memory instance instead)
- ✅ All parameters are keyword arguments

## Integration with Real Server

For tests that verify actual API calls work with a real MemMachine server, see:
- `test_memmachine_integration.py` - Tests with real server
- `TESTING.md` - Guide for integration testing

