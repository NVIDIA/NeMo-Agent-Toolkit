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

# Object Store for NVIDIA NeMo Agent Toolkit

The NeMo Agent toolkit Object Store subsystem provides a standardized interface for storing and retrieving binary data with associated metadata. This is particularly useful for building applications that need to manage files, documents, images, or any other binary content within these workflows.

The object store module is extensible, which allows developers to create custom object store backends. The providers in NeMo Agent toolkit terminology supports different storage systems.

## Features
- **Standard Interface**: Object stores implement a standard key-value interface, allowing for compatibility across different storage implementations.
- **Metadata Support**: Objects can be stored with content type and custom metadata for better management and organization.
- **Extensible Via Plugins**: Additional object stores can be added as plugins by developers to support more storage systems.
- **File Server Integration**: Object stores can be integrated with the NeMo Agent file server for direct HTTP access to stored objects.

## Core Components

### ObjectStoreItem
The `ObjectStoreItem` model represents an object in the store.
```python
class ObjectStoreItem:
    data: bytes  # The binary data to store
    content_type: str | None  # The MIME type of the data (optional)
    metadata: dict[str, str] | None  # Custom key-value metadata (optional)
```

### ObjectStoreListItem
The `ObjectStoreListItem` model represents only an object's metadata.
```python
class ObjectStoreListItem:
    key: str                           # The object's unique key
    size: int                          # Size in bytes
    content_type: str | None           # MIME type (e.g., "video/mp4")
    metadata: dict[str, str] | None    # Custom metadata
    last_modified: datetime | None     # Last modification timestamp
```

Note: Unlike `ObjectStoreItem`, this model does not include the `data` field, making listings fast and memory-efficient.

### ObjectStore Interface
The `ObjectStore` abstract interface defines the five standard operations:

- **put_object(key, item)**: Store a new object with a unique key. Raises if the key already exists.
- **upsert_object(key, item)**: Update (or inserts) an object with the given key.
- **get_object(key)**: Retrieve an object by its key. Raises if the key doesn't exist.
- **delete_object(key)**: Remove an object from the store. Raises if the key doesn't exist.
- **list_objects(prefix)**: List objects in the store, optionally filtered by key prefix.

```python
class ObjectStore(ABC):
    @abstractmethod
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        ...

    @abstractmethod
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        ...

    @abstractmethod
    async def get_object(self, key: str) -> ObjectStoreItem:
        ...

    @abstractmethod
    async def delete_object(self, key: str) -> None:
        ...

    @abstractmethod
    async def list_objects(self, prefix: str | None = None) -> list["ObjectStoreListItem"]:
        ...
```

## Included Object Stores
The NeMo Agent toolkit includes several object store providers:

- **In-Memory Object Store**: In-memory storage for development and testing. See `src/nat/object_store/in_memory_object_store.py`
- **S3 Object Store**: Amazon S3 and S3-compatible storage (like MinIO). See `packages/nvidia_nat_s3/src/nat/plugins/s3/s3_object_store.py`
- **MySQL Object Store**: MySQL database-backed storage. See `packages/nvidia_nat_mysql/src/nat/plugins/mysql/mysql_object_store.py`
- **Redis Object Store**: Redis key-value store. See `packages/nvidia_nat_redis/src/nat/plugins/redis/redis_object_store.py`

## Usage

### Configuration
Object stores are configured similarly to other NeMo Agent toolkit components. Each object store provider has a Pydantic config object that defines its configurable parameters. These parameters can then be configured in the config file under the `object_stores` section.

Example configuration for the in-memory object store:
```yaml
object_stores:
  my_object_store:
    _type: in_memory
    bucket_name: my-bucket
```

Example configuration for S3-compatible storage (like MinIO):
```yaml
object_stores:
  my_object_store:
    _type: s3
    endpoint_url: http://localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket_name: my-bucket
```

Example configuration for MySQL storage:
```yaml
object_stores:
  my_object_store:
    _type: mysql
    host: localhost
    port: 3306
    username: root
    password: my_password
    bucket_name: my-bucket
```

Example configuration for Redis storage:
```yaml
object_stores:
  my_object_store:
    _type: redis
    host: localhost
    port: 6379
    db: 0
    bucket_name: my-bucket
```

### Using Object Stores in Functions
Object stores can be used as components in custom functions. You can instantiate an object store client using the builder:

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):
    # Get an object store client
    object_store = await builder.get_object_store_client(object_store_name=config.object_store)

    # Store an object
    item = ObjectStoreItem(
        data=b"Hello, World!",
        content_type="text/plain",
        metadata={"author": "user123"}
    )
    await object_store.put_object("greeting.txt", item)

    # Retrieve an object
    retrieved_item = await object_store.get_object("greeting.txt")
    print(retrieved_item.data.decode("utf-8"))

    # Update (or insert) an object
    await object_store.upsert_object("greeting.txt", ObjectStoreItem(
        data=b"Goodbye, World!",
        content_type="text/plain",
        metadata={"author": "user123"}
    ))

    # Retrieve an object
    retrieved_item = await object_store.get_object("greeting.txt")
    print(retrieved_item.data.decode("utf-8"))

    # List objects with optional prefix filtering
    all_objects = await object_store.list_objects()
    for obj in all_objects:
        print(f"{obj.key}: {obj.size} bytes, {obj.content_type}")
    
    # List only objects with specific prefix
    greetings = await object_store.list_objects(prefix="greeting")
    
    # Delete an object
    await object_store.delete_object("greeting.txt")
```

The `list_objects()` method returns metadata for stored objects without downloading their content. This is efficient for building file browsers, galleries, or managing large files. The optional `prefix` parameter filters objects by key prefix, similar to listing files in a directory.

### File Server Integration
By adding the `object_store` field in the `general.front_end` block of the configuration, clients can directly download and upload files to the connected object store:

```yaml
general:
  front_end:
    object_store: my_object_store
    _type: fastapi
    cors:
      allow_origins: ['*']

object_stores:
  my_object_store:
    _type: s3
    endpoint_url: http://localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket_name: my-bucket
```

This enables HTTP endpoints for object store operations:
- **PUT** `/static/{file_path}` - Create or replace an object at the given path (upsert)
  ```console
  $ curl -X PUT --data-binary @data.txt http://localhost:9000/static/folder/data.txt
  ```
- **GET** `/static/{file_path}` - Download an object
  ```console
  $ curl -X GET http://localhost:9000/static/folder/data.txt
  ```
- **POST** `/static/{file_path}` - Upload a new object
  ```console
  $ curl -X POST --data-binary @data_new.txt http://localhost:9000/static/folder/data.txt
  ```
- **DELETE** `/static/{file_path}` - Delete an object
  ```console
  $ curl -X DELETE http://localhost:9000/static/folder/data.txt
  ```

### Video Upload Integration

When `object_store` is configured in the FastAPI front end, the UI also exposes video upload endpoints. Uploaded videos are stored with the `videos/` prefix and can be accessed from your workflow functions using the same ObjectStore instance.

```yaml
general:
  front_end:
    object_store: my_video_store  # Enables video routes
    _type: fastapi

object_stores:
  my_video_store:
    _type: s3
    endpoint_url: http://localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket_name: my-video-bucket

functions:
  my_video_function:
    _type: my_video_processor
    object_store: my_video_store  # Same store as frontend
```

This enables HTTP endpoints for object store operations:

- **GET** `/videos` - List uploaded videos
  ```console
  $ curl -X GET http://localhost:8000/videos
  ```
- **POST** `/videos` - Upload a new video
  ```console
  $ curl -X POST -F "file=@my_video.mp4;type=video/mp4" http://localhost:8000/videos
  ```
- **DELETE** `/videos/{video_key}` - Delete a video
  ```console
  $ curl -X DELETE http://localhost:8000/videos/videos_12345.mp4
  ```

Your workflow functions can access uploaded videos using `list_objects(prefix="videos/")` and `get_object(video_key)` as shown in the usage examples above.

## Examples
The following examples demonstrate how to use the object store module in the NeMo Agent toolkit:
* `examples/object_store/user_report` - A complete workflow that stores and retrieves user diagnostic reports using different object store backends

## Running Tests

Run these from the repository root after installing dependencies:

```bash
# In-memory object store unit tests
pytest tests/nat/object_store/test_in_memory_object_store.py -v

# FastAPI video upload routes
pytest tests/nat/front_ends/fastapi/test_video_upload_routes.py -v

# S3 provider integration tests (requires MinIO or S3 running and S3 plugin installed)
# Install S3 plugin first: uv pip install -e packages/nvidia_nat_s3
pytest packages/nvidia_nat_s3/tests/test_s3_object_store.py --run_integration -v
```

## Error Handling
Object stores may raise specific exceptions:
- **KeyAlreadyExistsError**: When trying to store an object with a key that already exists (for `put_object`)
- **NoSuchKeyError**: When trying to retrieve or delete an object with a non-existent key

## Additional Resources
For information on how to write a new object store provider, see the [Adding an Object Store Provider](../extend/object-store.md) document.
