# Per-User Workflow Example

This example demonstrates the **per-user workflow pattern** in NeMo Agent Toolkit (NAT). With this pattern, each user gets their own isolated workflow and function instances with separate state.

## Overview

The per-user workflow pattern is useful when you need:
- **User-isolated state**: Each user's data is completely separate from other users
- **Stateful functions**: Functions that maintain state across requests for the same user
- **Session-based personalization**: User preferences, history, or context that persists within a session

## Components

### Per-User Functions

1. **`per_user_notepad`**: A simple notepad that stores notes per user
   - Each user has their own list of notes
   - Notes added by one user are not visible to other users

2. **`per_user_preferences`**: A preferences store per user
   - Each user has their own preference settings
   - Changes by one user don't affect other users

### Per-User Workflow

**`per_user_assistant`**: A workflow that combines the notepad and preferences functions
- Tracks session statistics per user
- Provides a unified command interface

## Usage

### 1. Install the Example

First, install the example package:

```bash
cd examples/per_user_workflow
pip install -e .
```

### 2. Start the Server

```bash
nat serve --config_file=examples/per_user_workflow/configs/config.yml
```

### 2. Test with Different Users

Each user is identified by the `nat-session` cookie. Different session IDs represent different users.

#### User 1 Operations

```bash
# Add a note as User 1
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "note", "action": "add", "param1": "Alices first note"}'

# List notes as User 1
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "note", "action": "list"}'
```

#### User 2 Operations

```bash
# List notes as User 2 (should be empty - isolated from User 1)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d '{"command": "note", "action": "list"}'

# Add a note as User 2
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d '{"command": "note", "action": "add", "param1": "Bobs note"}'
```

#### Preferences

```bash
# Set a preference as User 1
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "pref", "action": "set", "param1": "theme", "param2": "light"}'

# Check User 2's theme (should still be "dark" from defaults)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d '{"command": "pref", "action": "get", "param1": "theme"}'
```

#### Help and Stats

```bash
# Get help
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "help"}'

# Get session stats (tracks commands per user)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "stats"}'
```

## Available Commands

| Command | Action | Parameters | Description |
|---------|--------|------------|-------------|
| `note` | `add` | `param1`: content | Add a note |
| `note` | `list` | - | List all notes |
| `note` | `clear` | - | Clear all notes |
| `note` | `count` | - | Count notes |
| `pref` | `set` | `param1`: key, `param2`: value | Set a preference |
| `pref` | `get` | `param1`: key | Get a preference |
| `pref` | `list` | - | List all preferences |
| `help` | - | - | Show help message |
| `stats` | - | - | Show session statistics |

## Configuration

The `config.yml` file configures:

- **`per_user_workflow_timeout`**: How long inactive user sessions are kept (default: 30 minutes)
- **`per_user_workflow_cleanup_interval`**: How often to check for inactive sessions (default: 5 minutes)
- **`max_notes`**: Maximum notes per user (default: 50)
- **`default_preferences`**: Default preferences for new users

## How It Works

1. **User Identification**: Users are identified by the `nat-session` cookie
2. **On-Demand Creation**: Per-user workflow builders are created when a user first makes a request
3. **State Isolation**: Each user's functions maintain separate state
4. **Automatic Cleanup**: Inactive user sessions are automatically cleaned up based on the configured timeout

## Key Files

- `src/nat_per_user_workflow/per_user_functions.py`: Per-user function definitions
- `src/nat_per_user_workflow/per_user_workflow.py`: Per-user workflow definition
- `src/nat_per_user_workflow/register.py`: Plugin registration
- `configs/config.yml`: Configuration file
- `pyproject.toml`: Package configuration with entry points

## Using `@register_per_user_function`

The `@register_per_user_function` decorator marks a function or workflow as per-user:

```python
@register_per_user_function(
    config_type=MyConfig,
    input_schema=MyInput,
    single_output_schema=MyOutput
)
async def my_per_user_function(config: MyConfig, builder: Builder):
    # This state is unique per user
    user_state = {"counter": 0}

    async def _impl(inp: MyInput) -> MyOutput:
        user_state["counter"] += 1
        return MyOutput(count=user_state["counter"])

    yield FunctionInfo.from_fn(_impl)
```

## Constraints

- **Shared functions cannot depend on per-user functions**: A shared function cannot call `builder.get_function()` on a per-user function
- **Per-user functions can depend on shared functions**: A per-user function can access shared functions via the builder
- **Per-user functions can depend on other per-user functions**: The dependency will be resolved within the same user's builder
