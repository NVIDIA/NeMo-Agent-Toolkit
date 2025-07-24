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

# Redis Examples

These examples use the redis memory backend.

## Start redis with docker compose

Run redis on `localhost:6379` and Redis Insight on `localhost:5540` with:

```bash
docker compose -f examples/deploy/docker-compose.redis.yml up
```

## Start phoenix with docker compose

The examples are configured to use the Phoenix observability tool. Start phoenix on `localhost:6006` with:

```bash
docker compose -f examples/deploy/docker-compose.phoenix.yml up
```

## Simple chat with the ability to create and recall memories

This examples shows how to have a simple chat that uses a redis memory backend for creating and retrieving memories.

An embeddings model is used to create embeddings for queries and for stored memories. Uses HNSW and L2 distance metric.

Try the chat by running:

```
aiq run --config_file=examples/memory/redis/configs/memory-example.yml --input "my favorite flavor is strawberry"

--------------------------------------------------
Workflow Result:
["The user's favorite flavor has been stored as strawberry."]
--------------------------------------------------
```

```
aiq run --config_file=examples/memory/redis/configs/memory-example.yml --input "what flavor of ice-cream should I get?"

--------------------------------------------------
Workflow Result:
['You should get strawberry ice cream, as it is your favorite flavor.']
--------------------------------------------------
```

## Test

```bash
pytest packages/aiqtoolkit_redis/tests/test_redis_editor.py -v
```
