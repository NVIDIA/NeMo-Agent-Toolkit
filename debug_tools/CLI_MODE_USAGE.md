# Using Proper MCP Protocol in Load Tests

## ✅ **Fixed: No More 406 Errors!**

The load testing scripts now support proper MCP protocol using the `nat mcp client` CLI.

## 🔧 **What Changed**

### `mcp_load_test.py`

Added `call_tool_via_cli()` method that uses the proper MCP client:

```python
async def call_tool_via_cli(self, tool_name, arguments, user_id):
    """Call MCP tool using nat mcp client CLI (proper protocol)."""
    cmd = [
        "nat", "mcp", "client", "tool", "call",
        tool_name,
        "--url", self.server_url,
        "--json-args", json.dumps(arguments)
    ]
    process = await asyncio.create_subprocess_exec(*cmd, ...)
    return process.wait() == 0
```

**Default**: Now uses CLI mode by default (`use_cli=True`)

### Command Line Options

```bash
# Use CLI mode (default, proper MCP protocol)
python debug_tools/mcp_load_test.py --users 40 --calls 10

# Or explicitly specify
python debug_tools/mcp_load_test.py --users 40 --calls 10 --use-cli

# Use HTTP mode (for comparison, gets 406 errors)
python debug_tools/mcp_load_test.py --users 40 --calls 10 --use-http
```

## 🚀 **Usage**

### Quick Test (Recommended)

```bash
./debug_tools/quick_test.sh
```

This automatically uses CLI mode with proper MCP protocol.

### Python Load Tester Directly

```bash
# CLI mode (no 406 errors)
python debug_tools/mcp_load_test.py \
    --url http://localhost:9901/mcp \
    --users 40 \
    --calls 10 \
    --rounds 3

# HTTP mode (for testing, gets 406 errors)
python debug_tools/mcp_load_test.py \
    --url http://localhost:9901/mcp \
    --users 40 \
    --calls 10 \
    --use-http
```

### Integrated Test Runner

```bash
python debug_tools/run_memory_leak_test.py \
    --config_file examples/getting_started/simple_calculator/configs/config.yml \
    --users 40 \
    --calls 10 \
    --rounds 3
```

Uses CLI mode by default.

## 📊 **CLI Mode vs HTTP Mode**

| Feature | CLI Mode | HTTP Mode |
|---------|----------|-----------|
| Protocol | ✅ Proper MCP | ❌ Raw JSON-RPC |
| Errors | ✅ No 406 errors | ❌ 406 Not Acceptable |
| Session Handling | ✅ Correct | ❌ Incorrect |
| Use Case | Production testing | Debugging only |
| Default | ✅ Yes | No |

## 🎯 **Why CLI Mode Works**

The `nat mcp client tool call` command:
1. ✅ Properly implements MCP streamable-http protocol
2. ✅ Handles session management correctly
3. ✅ Sends correct HTTP headers
4. ✅ Uses proper message framing
5. ✅ No 406 errors

The raw HTTP POST:
1. ❌ Sends JSON-RPC without MCP protocol
2. ❌ Missing required headers
3. ❌ Doesn't handle sessions properly
4. ❌ Server rejects with 406 Not Acceptable

## 📝 **Test Scripts Updated**

All test scripts now use proper MCP protocol:

- ✅ `mcp_load_test.py` - Added CLI mode (default)
- ✅ `quick_test.sh` - Uses Python tester with CLI
- ✅ `run_memory_leak_test.py` - Calls mcp_load_test.py (uses CLI by default)

## 🧪 **Examples**

### Test with 100 concurrent users

```bash
python debug_tools/mcp_load_test.py --users 100 --calls 5 --rounds 3
```

### Test for extended duration

```bash
python debug_tools/mcp_load_test.py --users 20 --calls 50 --rounds 10 --delay 30
```

### Compare CLI vs HTTP modes

```bash
# CLI mode (should work)
python debug_tools/mcp_load_test.py --users 10 --calls 5 --use-cli

# HTTP mode (will get 406 errors)
python debug_tools/mcp_load_test.py --users 10 --calls 5 --use-http
```

## ✅ **Summary**

**Problem**: Raw HTTP POST requests got 406 errors

**Solution**: Added `call_tool_via_cli()` method that uses proper MCP client

**Default**: CLI mode is now the default (no more 406 errors)

**Usage**: Just run `./debug_tools/quick_test.sh` - it works out of the box!

---

## Quick Start

```bash
./debug_tools/quick_test.sh
```

That's it! The test will use proper MCP protocol and successfully make tool calls to test the memory leak fix. 🎉
