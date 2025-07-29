<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Authenticated Requests Example

This example demonstrates the AuthProviders extensibility in the NeMo Agent Toolkit, showing how to make authenticated HTTP
requests.

## Available AuthProviders Request Methods

The AuthProviderMixin provides the following HTTP methods that can be called:

### **Unified Request Method**
```python
# The main unified method that supports all HTTP methods
response = await auth_provider.request(
    method="GET",
    url="https://api.atlassian.com/oauth/token/accessible-resources",
    user_id="Test",
    apply_auth=True,  # Default
    headers={"Custom-Header": "value"},
    timeout=30
)
```

### **GET Requests**
```python
# Authenticated GET request (default behavior)
response = await auth_provider.get(
    url="https://api.atlassian.com/oauth/token/accessible-resources",
    user_id="Test"
)

# Public GET request (unauthenticated)
response = await auth_provider.get(
    url="https://api.github.com/repos/nvidia/agent-iq-toolkit",
    apply_auth=False
)
```

### **POST Requests**
```python
# POST with JSON data (authenticated)
response = await auth_provider.post(
    url="https://api.atlassian.com/rest/api/2/project",
    user_id="admin",
    json={"name": "New Project", "key": "NP", "projectTypeKey": "software"}
)

# POST with form data (authenticated)
response = await auth_provider.post(
    url="https://api.example.com/form",
    user_id="user123",
    data={"field1": "value1", "field2": "value2"}
)

# Public POST request (unauthenticated)
response = await auth_provider.post(
    url="https://httpbin.org/post",
    apply_auth=False,
    json={"test": "data"}
)
```

### **PUT Requests**
```python
# PUT with JSON data to update a resource (authenticated)
response = await auth_provider.put(
    url="https://api.atlassian.com/rest/api/2/issue/ABC-123",
    user_id="editor",
    json={"fields": {"summary": "Updated Issue Title"}}
)

# PUT with string data (authenticated)
response = await auth_provider.put(
    url="https://api.example.com/documents/doc123",
    user_id="writer",
    data="Updated document content"
)
```

### **DELETE Requests**
```python
# Delete a resource (authenticated)
response = await auth_provider.delete(
    url="https://api.atlassian.com/rest/api/2/project/PROJECT-123",
    user_id="admin"
)

# Delete with query parameters (authenticated)
response = await auth_provider.delete(
    url="https://api.example.com/items/item456",
    user_id="moderator",
    params={"force": "true", "reason": "cleanup"}
)
```

### **PATCH Requests**
```python
# PATCH to partially update a resource (authenticated)
response = await auth_provider.patch(
    url="https://api.atlassian.com/rest/api/2/issue/ABC-123",
    user_id="editor",
    json={"fields": {"priority": {"name": "High"}}}
)

# PATCH with form data (authenticated)
response = await auth_provider.patch(
    url="https://api.example.com/profile/user123",
    user_id="user123",
    data={"email": "newemail@example.com"}
)
```

### **HEAD Requests**
```python
# HEAD request to check resource existence (authenticated)
response = await auth_provider.head(
    url="https://api.atlassian.com/rest/api/2/issue/ABC-123",
    user_id="viewer"
)

# Public HEAD request to check headers
response = await auth_provider.head(
    url="https://httpbin.org/status/200",
    apply_auth=False
)
```

### **OPTIONS Requests**
```python
# OPTIONS request to check allowed methods (authenticated)
response = await auth_provider.options(
    url="https://api.atlassian.com/rest/api/2/project",
    user_id="developer"
)

# Public OPTIONS request
response = await auth_provider.options(
    url="https://httpbin.org/",
    apply_auth=False
)
```

## HTTPResponse Object

All request methods return an `HTTPResponse` object that contains the response data and metadata:

### **HTTPResponse Properties**

- **`status_code`** (int): HTTP status code returned by the server (200, 404, 500, etc.)
- **`headers`** (dict[str, str]): HTTP response headers as key-value pairs
- **`body`** (dict | list | str | bytes | None): Response body content - automatically parsed JSON (dict/list), plain text (str), or raw bytes
- **`cookies`** (dict[str, str] | None): Cookies returned by the server, if any
- **`content_type`** (str | None): Content-Type header value (e.g., "application/json", "text/html")
- **`url`** (str | None): Final URL after any redirects
- **`elapsed`** (float | None): Request duration in seconds
- **`auth_result`** (AuthResult | None): Authentication result used for this request, contains credentials and token info

## OAuth 2.0 Setup for Jira

This example uses **OAuth 2.0 Authorization Code Flow** to authenticate with Atlassian's Jira API.

### Environment Variables Required

```bash
export AIQ_OAUTH_CLIENT_ID=your_jira_client_id
export AIQ_OAUTH_CLIENT_SECRET=your_jira_client_secret
```

### Jira OAuth App Setup

To use this example, you need to configure the following parameters:

**Required Environment Variables:**
- `AIQ_OAUTH_CLIENT_ID` - Your Jira OAuth app client ID
- `AIQ_OAUTH_CLIENT_SECRET` - Your Jira OAuth app client secret

**Pre-configured Settings in config.yml:**
- `redirect_uri`: `http://localhost:8000/auth/redirect`
- `authorization_url`: `https://auth.atlassian.com/authorize`
- `token_url`: `https://auth.atlassian.com/oauth/token`
- Required Jira API scopes are already included

## Running the Example

### 1. Install Dependencies
```bash
cd examples/front_ends/authenticated_requests
pip install -e .
```

### 2. Set Environment Variables
```bash
export AIQ_OAUTH_CLIENT_ID=your_client_id_here
export AIQ_OAUTH_CLIENT_SECRET=your_client_secret_here
```

### 3. Start the Agent
```bash
aiq serve --config_file=examples/front_ends/authenticated_requests/src/authenticated_requests/configs/config.yml
```

### 4. Deploy the UI
Follow the instructions at [NeMo Agent Toolkit UI](../../../external/aiqtoolkit-opensource-ui/) to deploy the web interface.

### 5. Test the Functions

Once connected to the UI, you can test the authenticated requests:
