<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# FastAPI Frontend

Serving a NeMo Agent toolkit workflow over HTTP via `nat serve` / `nat start fastapi`.

Deploy a NeMo Agent toolkit workflow behind a FastAPI server by setting `front_end._type: fastapi`. Define custom endpoints that map to specific functions:

```yaml
general:
  front_end:
    _type: fastapi
    enable_streaming: true
    endpoints:
      - path: /api/generate
        method: POST
        description: "Run the main generation workflow"
        function_name: my_workflow
      - path: /api/health
        method: GET
        description: "Health check"
        function_name: health_check
```

For production deployments with CORS and WebSocket support:

```yaml
general:
  front_end:
    _type: fastapi
    host: "0.0.0.0"
    port: ${PORT:-8080}
    workflow:
      method: POST
      path: /generate
      websocket_path: /websocket
    cors:
      allow_origins: ["https://my-app.nvidia.com", "http://localhost:3000"]
      allow_methods: ["GET", "POST", "OPTIONS"]
      allow_headers: ["*"]
      allow_credentials: true
```
