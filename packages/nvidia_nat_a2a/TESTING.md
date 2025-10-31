# A2A Client Testing Guide

This guide explains how to test the NAT A2A client implementation using the official A2A samples.

## Setup: External HelloWorld A2A Agent

We use the official [HelloWorld example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/helloworld) from the A2A samples repository as a test server.

### 1. Clone A2A Samples Repository

```bash
# Clone in a separate directory (outside NAT repo)
cd /tmp  # or any location outside NAT
git clone https://github.com/a2aproject/a2a-samples.git
cd a2a-samples/samples/python/agents/helloworld
```

### 2. Start HelloWorld Agent

```bash
# Start the agent server (runs on default port, typically 8000)
uv run .
```

**Expected output**:
```
Starting HelloWorld A2A Agent...
Server running on http://localhost:8000
Agent Card available at http://localhost:8000/.well-known/agent-card
```

### 3. Verify Agent is Running

In another terminal:

```bash
# Check agent card (discovery endpoint)
curl http://localhost:8000/.well-known/agent-card | jq

# Expected: JSON with agent metadata, skills, capabilities
```

---

## Testing NAT A2A Client

Once the HelloWorld agent is running, you can test NAT's A2A client implementation.

### Test Discovery

```bash
# From NAT repo root
nat a2a client discover --url http://localhost:8000
```

**Expected output**:
```
Agent: HelloWorld Agent
Description: A simple greeting agent
URL: http://localhost:8000

Skills:
  - greet: Generate a greeting message
  - ...
```

### Test Task Submission

```bash
# Submit a task to the agent
nat a2a client task submit greet \
  --url http://localhost:8000 \
  --json-args '{"name": "Alice"}'
```

**Expected output** (without --wait):
```
Task submitted successfully
Task ID: task_abc123
Status: queued

Check status with:
  nat a2a client task status task_abc123 --url http://localhost:8000
```

**With --wait flag**:
```bash
nat a2a client task submit greet \
  --url http://localhost:8000 \
  --json-args '{"name": "Alice"}' \
  --wait
```

**Expected output**:
```
Task submitted: task_abc123
Status: working... (2s elapsed)
Status: completed (5s elapsed)

Result:
{
  "greeting": "Hello, Alice!"
}
```

### Test Task Status

```bash
# Check status of submitted task
nat a2a client task status task_abc123 --url http://localhost:8000
```

### Test Health Check

```bash
nat a2a client ping --url http://localhost:8000
```

**Expected output**:
```
Agent at http://localhost:8000 is healthy (response time: 15ms)
```

---

## Using in NAT Workflow

Once the client is working via CLI, you can use it in a NAT workflow:

```yaml
# config.yml
function_groups:
  helloworld_agent:
    _type: a2a_client
    agent:
      url: http://localhost:8000
      task_timeout: 60

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names: [helloworld_agent.greet]
```

```bash
nat run --config_file config.yml --input "Greet Bob"
```

---

## Integration Testing

For automated tests during development:

```python
# tests/integration/test_a2a_client.py
import pytest
from nat.plugins.a2a.client_base import A2ABaseClient

@pytest.mark.integration
@pytest.mark.asyncio
async def test_discover_helloworld_agent():
    """Test discovery against external HelloWorld agent.

    Requires HelloWorld agent running on http://localhost:8000
    """
    client = A2ABaseClient("http://localhost:8000")

    # Discover agent
    card = await client.discover()

    assert card.name == "HelloWorld Agent"
    assert len(card.skills) > 0
    assert "greet" in [skill.name for skill in card.skills]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_submit_task_to_helloworld():
    """Test task submission to HelloWorld agent."""
    client = A2ABaseClient("http://localhost:8000")

    # Submit task
    task_id = await client.submit_task("greet", {"name": "TestUser"})
    assert task_id.startswith("task_")

    # Wait for completion
    result = await client.wait_for_task(task_id, timeout=30)
    assert "greeting" in result
    assert "TestUser" in result["greeting"]
```

**Run integration tests**:
```bash
# Requires HelloWorld agent running
pytest -m integration
```

---

## Troubleshooting

### HelloWorld Agent Not Starting

**Issue**: Port already in use
```bash
# Check what's using port 8000
lsof -i :8000

# Start on different port
uv run . --port 8001
```

Then update NAT client URL: `--url http://localhost:8001`

### Connection Refused

**Issue**: Agent not running or wrong URL
```bash
# Verify agent is accessible
curl http://localhost:8000/.well-known/agent-card

# If fails, check:
# 1. Agent server is running
# 2. Port is correct
# 3. No firewall blocking
```

### Task Timeout

**Issue**: Task takes too long
```bash
# Increase timeout
nat a2a client task submit greet \
  --url http://localhost:8000 \
  --json-args '{"name": "Alice"}' \
  --wait \
  --timeout 300  # 5 minutes
```

---

## Next Steps

After NAT A2A server support is implemented, you can test NAT-to-NAT A2A communication:

1. **NAT as A2A Server**: Expose NAT workflow as A2A agent
2. **NAT as A2A Client**: Connect to NAT A2A server
3. **Multi-agent workflows**: Multiple NAT agents coordinating via A2A

---

## References

- [A2A HelloWorld Example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/helloworld)
- [A2A Samples Repository](https://github.com/a2aproject/a2a-samples)
- [A2A Protocol Documentation](https://a2a-protocol.org/latest/)
- [A2A Python SDK](https://github.com/a2aproject/a2a-python)
