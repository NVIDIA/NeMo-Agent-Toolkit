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
The `ObjectStoreItem` model represents an object in the store with the following attributes:

- **`data`**: The binary data to store
- **`content_type`**: The MIME type of the data (optional)
- **`metadata`**: Custom key-value metadata for the object (optional)

### ObjectStore Interface
The `ObjectStore` abstract interface defines the standard operations:

- **put_object(key, item)**: Store a new object with a unique key
- **upsert_object(key, item)**: Store or update an object with the given key
- **get_object(key)**: Retrieve an object by its key
- **delete_object(key)**: Remove an object from the store

## Included Object Stores
The AIQ toolkit includes several object store providers:

- **In-Memory Object Store**: In-memory storage for development and testing
- **S3 Object Store**: Amazon S3 and S3-compatible storage (like MinIO)
- **MySQL Object Store**: MySQL database-backed storage

## Usage

### Configuration
Object stores are configured similarly to other NeMo Agent toolkit components. Each object store provider has a Pydantic config object that defines its configurable parameters. These parameters can then be configured in the config file under the `object_stores` section.

Below is an example configuration for the in-memory object store:
```yaml
object_stores:
  my_object_store:
    _type: in_memory
    bucket_name: my-bucket
```

For S3-compatible storage (like MinIO):
```yaml
object_stores:
  my_object_store:
    _type: s3
    endpoint_url: http://localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
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
```

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
- **GET** `/static/{file_path}` - Download an object
- **POST** `/static/{file_path}` - Upload a new object
- **PUT** `/static/{file_path}` - Update an existing object
- **DELETE** `/static/{file_path}` - Delete an object

## Examples
The following examples demonstrate how to use the object store module in the AIQ toolkit:
* `examples/object_store/user_report` - A complete workflow that stores and retrieves user diagnostic reports using different object store backends

## Error Handling
Object stores may raise specific exceptions:
- **KeyAlreadyExistsError**: When trying to store an object with a key that already exists (for `put_object`)
- **NoSuchKeyError**: When trying to retrieve or delete an object with a non-existent key

## Additional Resources
For information on how to write a new object store provider, see the [Adding an Object Store Provider](../extend/object-store.md) document.
