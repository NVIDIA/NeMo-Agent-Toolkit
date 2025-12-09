# Simple Protected A2A Server - Testing Authentication

This example demonstrates a minimal protected A2A server for testing NeMo Agent toolkit CLI authentication. It uses simple Bearer token validation to test all three NAT CLI authentication methods.

## Overview

This server provides:
- ✅ Simple Bearer token authentication
- ✅ Proper A2A `securitySchemes` configuration
- ✅ Public agent card discovery
- ✅ Protected agent operations
- ✅ Clear test commands printed on startup

## Installation

Install the example:

```bash
uv pip install -e examples/A2A/simple_auth_a2a
```

## Quick Start

### 1. Start the Protected Server

```bash
export SIMPLE_PROTECTED_SERVER_HOST=localhost
export SIMPLE_PROTECTED_SERVER_PORT=8000
export TEST_BEARER_TOKEN=test-token-12345

python examples/A2A/simple_auth_a2a/src/nat_simple_auth_a2a/scripts/simple_protected_server.py
```

The server will start on `http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT}` and print test commands.

### 2. Test NAT CLI Authentication

#### Test 1: No Authentication (Should FAIL with 401)

```bash
nat a2a client call --url http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT} --message "Hello"
```

**Expected:** `401 Unauthorized: Missing Authorization header`

#### Test 2: Invalid Token (Should FAIL with 403)

```bash
nat a2a client call --url http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT} \
  --message "Hello" \
  --bearer-token "wrong-token"
```

**Expected:** `403 Forbidden: Invalid Bearer token`

#### Test 3: Valid Bearer Token (Should SUCCEED)

```bash
nat a2a client call --url http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT} \
  --message "Hello" \
  --bearer-token "test-token-12345"
```

**Expected:** `✅ Authentication successful! You said: Hello`

#### Test 4: Token from Environment Variable

```bash
export MY_TOKEN="test-token-12345"
nat a2a client call --url http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT} \
  --message "Hello" \
  --bearer-token-env MY_TOKEN
```

#### Test 5: Config-Based Authentication

```bash
nat a2a client call --url http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT} \
  --message "Hello" \
  --auth-config examples/A2A/simple_auth_a2a/src/nat_simple_auth_a2a/configs/test_auth.yml \
  --auth-provider test_bearer
```

#### Test 6: Inline JSON Authentication

```bash
nat a2a client call --url http://${SIMPLE_PROTECTED_SERVER_HOST}:${SIMPLE_PROTECTED_SERVER_PORT} \
  --message "Hello" \
  --auth-json '{"_type": "api_key", "raw_key": "test-token-12345", "auth_scheme": "BEARER"}'
```


## Architecture

### Agent Card Security Configuration

The server uses the modern `securitySchemes` format:

```python
securitySchemes={
    "bearer_auth": SecurityScheme(
        root=HTTPAuthSecurityScheme(
            type="http",
            scheme="bearer",
            description="Bearer token authentication"
        )
    )
},
security=[{"bearer_auth": []}]  # Required for all operations
```

### Authentication Flow

```mermaid
sequenceDiagram
    participant CLI as NAT CLI
    participant Agent as Protected Agent

    CLI->>Agent: GET /.well-known/agent.json (public)
    Agent-->>CLI: AgentCard with securitySchemes

    CLI->>CLI: Obtain Bearer token (bearer/config/JSON)

    CLI->>Agent: POST /message (Authorization: Bearer token)
    Agent->>Agent: Validate token

    alt Valid Token
        Agent-->>CLI: 200 OK with response
    else Invalid/Missing Token
        Agent-->>CLI: 401/403 Error
    end
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_BEARER_TOKEN` | `test-token-12345` | Valid bearer token for testing |
| `SIMPLE_PROTECTED_SERVER_HOST` | `localhost` | Server host |
| `SIMPLE_PROTECTED_SERVER_PORT` | `8000` | Server port |


## References

- [A2A Protocol Specification](https://github.com/NVIDIA/a2a)
- [NAT Authentication Documentation](../../../docs/source/reference/api-authentication.md)
