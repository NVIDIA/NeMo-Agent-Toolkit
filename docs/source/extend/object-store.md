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

# Adding an Object Store Provider With NVIDIA NeMo Agent Toolkit

This documentation presumes familiarity with the NeMo Agent toolkit plugin architecture, the concept of "function registration" using `@register_function`, and how we define tool/workflow configurations in the NeMo Agent toolkit config described in the [Creating a New Tool and Workflow](../tutorials/create-a-new-workflow.md) tutorial.

## Key Object Store Module Components

* **Object Store Data Models**
   - **{py:class}`~nat.data_models.object_store.ObjectStoreBaseConfig`**: A Pydantic base class that all object store config classes must extend. This is used for specifying object store registration in the NeMo Agent toolkit config file.
   - **{py:class}`~nat.data_models.object_store.ObjectStoreBaseConfigT`**: A generic type alias for object store config classes.

* **Object Store Interfaces**
   - **{py:class}`~nat.object_store.interfaces.ObjectStore`** (abstract interface): The core interface for object store operations, including put, upsert, get, and delete operations.
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
     ```

* **Object Store Models**
   - **{py:class}`~nat.object_store.models.ObjectStoreItem`**: The main object representing an item in the object store.
     ```python
     class ObjectStoreItem:
        data: bytes  # The binary data to store
        content_type: str | None  # The MIME type of the data (optional)
        metadata: dict[str, str] | None  # Custom key-value metadata (optional)
     ```

* **Object Store Exceptions**
   - **{py:class}`~nat.data_models.object_store.KeyAlreadyExistsError`**: Raised when trying to store an object with a key that already exists (for `put_object`)
   - **{py:class}`~nat.data_models.object_store.NoSuchKeyError`**: Raised when trying to retrieve or delete an object with a non-existent key

## Adding an Object Store Provider

In the NeMo Agent toolkit system, anything that extends {py:class}`~nat.data_models.object_store.ObjectStoreBaseConfig` and is declared with a `name="some_object_store"` can be discovered as an *Object Store type* by the NeMo Agent toolkit global type registry. This allows you to define a custom object store class to handle your own backends (for example, Redis, custom database, or cloud storage). Then your object store class can be selected in the NeMo Agent toolkit config YAML using `_type: <your object store type>`.

### Basic Steps

1. **Create a config Class** that extends {py:class}`~nat.data_models.object_store.ObjectStoreBaseConfig`:
   ```python
   from nat.data_models.object_store import ObjectStoreBaseConfig

   class MyCustomObjectStoreConfig(ObjectStoreBaseConfig, name="my_custom_object_store"):
       # You can define any fields you want. For example:
       connection_url: str
       api_key: str
       bucket_name: str
   ```
   > **Note**: The `name="my_custom_object_store"` ensures that NeMo Agent toolkit can recognize it when the user places `_type: my_custom_object_store` in the object store config.

2. **Implement an {py:class}`~nat.object_store.interfaces.ObjectStore`** that uses your backend:
   ```python
   from nat.object_store.interfaces import ObjectStore
   from nat.object_store.models import ObjectStoreItem
   from nat.data_models.object_store import KeyAlreadyExistsError, NoSuchKeyError
   from nat.utils.type_utils import override

   class MyCustomObjectStore(ObjectStore):
       def __init__(self, config: MyCustomObjectStoreConfig):
           self._api_key = config.api_key
           self._conn_url = config.connection_url
           self._bucket_name = config.bucket_name
           # Set up connections to your backend here

       @override
       async def put_object(self, key: str, item: ObjectStoreItem) -> None:
           # Check if key already exists
           if await self._key_exists(key):
               raise KeyAlreadyExistsError(key)

           # Store the object in your backend
           await self._store_object(key, item)

       @override
       async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
           # Store or update the object in your backend
           await self._store_object(key, item)

       @override
       async def get_object(self, key: str) -> ObjectStoreItem:
           # Retrieve the object from your backend
           item = await self._retrieve_object(key)
           if item is None:
               raise NoSuchKeyError(key)
           return item

       @override
       async def delete_object(self, key: str) -> None:
           # Delete the object from your backend
           if not await self._delete_object(key):
               raise NoSuchKeyError(key)

       # Helper methods for your specific backend
       async def _key_exists(self, key: str) -> bool:
           # Implementation specific to your backend
           pass

       async def _store_object(self, key: str, item: ObjectStoreItem) -> None:
           # Implementation specific to your backend
           pass

       async def _retrieve_object(self, key: str) -> ObjectStoreItem | None:
           # Implementation specific to your backend
           pass

       async def _delete_object(self, key: str) -> bool:
           # Implementation specific to your backend
           pass
   ```

3. **Register your object store with NeMo Agent toolkit** using the `@register_object_store` decorator:
   ```python
   from nat.builder.builder import Builder
   from nat.cli.register_workflow import register_object_store

   @register_object_store(config_type=MyCustomObjectStoreConfig)
   async def my_custom_object_store(config: MyCustomObjectStoreConfig, builder: Builder):
       yield MyCustomObjectStore(config)
   ```

4. **Use in config**: In your NeMo Agent toolkit config, you can do something like:
   ```yaml
   object_stores:
     my_store:
       _type: my_custom_object_store
       connection_url: "http://localhost:1234"
       api_key: "some-secret"
       bucket_name: "my-bucket"
   ```

> The user can then reference `my_store` in their function or workflow config (for example, in a function that uses an object store).

---

## Bringing Your Own Object Store Implementation

A typical pattern is:
- You define a *config class* that extends {py:class}`~nat.data_models.object_store.ObjectStoreBaseConfig` (giving it a unique `_type` / name).
- You define the actual *runtime logic* in an "Object Store" class that implements {py:class}`~nat.object_store.interfaces.ObjectStore`.
- You connect them together using the `@register_object_store` decorator.

### Example: Minimal Skeleton

File Structure:
```
my_custom_object_store
├── my_custom_object_store.py
├── object_store.py
└── register.py
```

`my_custom_object_store.py` contents:
```python
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override

class MyCustomObjectStore(ObjectStore):
    def __init__(self, cfg: MyCustomObjectStoreConfig):
        self._url = cfg.url
        self._token = cfg.token
        self._bucket_name = cfg.bucket_name

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        # Check if key exists and raise KeyAlreadyExistsError if it does
        # Store the object
        pass

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        # Store or update the object
        pass

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        # Retrieve the object, raise NoSuchKeyError if not found
        pass

    @override
    async def delete_object(self, key: str) -> None:
        # Delete the object, raise NoSuchKeyError if not found
        pass
```

`object_store.py` contents:
```python
from nat.data_models.object_store import ObjectStoreBaseConfig

class MyCustomObjectStoreConfig(ObjectStoreBaseConfig, name="my_custom_object_store"):
    url: str
    token: str
    bucket_name: str


@register_object_store(config_type=MyCustomObjectStoreConfig)
async def my_custom_object_store(config: MyCustomObjectStoreConfig, builder: Builder):

    from .my_custom_object_store import MyCustomObjectStore
    yield MyCustomObjectStore(config)
```


`register.py` contents:
```python
from . import object_store
```

---

## Using Object Stores in a Workflow

**At runtime**, you typically see code like:

```python
object_store_client = await builder.get_object_store_client(<object_store_config_name>)
await object_store_client.put_object("my-key", ObjectStoreItem(data=b"Hello, World!"))
```

or

```python
item = await object_store_client.get_object("my-key")
print(item.data.decode("utf-8"))
```

**Inside Functions**: Functions that read or write to object stores simply call the object store client. For example:

```python
from nat.object_store.models import ObjectStoreItem
from langchain_core.tools import ToolException

async def store_file_tool_action(file_data: bytes, key: str, object_store_name: str):
    object_store_client = await builder.get_object_store_client(object_store_name)
    try:
        item = ObjectStoreItem(
            data=file_data,
            content_type="application/octet-stream",
            metadata={"uploaded_by": "user123"}
        )
        await object_store_client.put_object(key, item)
        return "File stored successfully"
    except KeyAlreadyExistsError as e:
        raise ToolException(f"File already exists: {e}")
    except Exception as e:
        raise ToolException(f"Error storing file: {e}")
```

### Example Configuration

Here are the relevant sections from the `examples/object_store/user_report/configs/config_s3.yml` in the source code repository:

```yaml
object_stores:
  report_object_store:
    _type: s3
    endpoint_url: http://localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket_name: my-bucket
```

```yaml
functions:
  get_user_report:
    _type: get_user_report
    object_store: report_object_store
    description: >
      Fetches user diagnostic report from object store given a user ID and date.
      Args:
        user_id: str: The user ID to fetch the report for.
        date: str | null: The date to fetch the report for. Format: YYYY-MM-DD. If not provided, the latest report will be fetched.

  put_user_report:
    _type: put_user_report
    object_store: report_object_store
    description: >
      Puts user diagnostic report into object store given a user ID and date.
      Args:
        report: str: The report to put into the object store.
        user_id: str: The user ID to put the report for.
        date: str | null: The date to put the report for. Format: YYYY-MM-DD. If not provided, the report will be named "latest".
```

## Error Handling Best Practices

When implementing your object store provider, follow these error handling guidelines:

- **Use the provided exceptions**: Always use `KeyAlreadyExistsError` and `NoSuchKeyError` for the appropriate scenarios.

- **Handle backend-specific errors**: Wrap backend-specific exceptions and convert them to the appropriate NeMo Agent toolkit exceptions.

- **Provide meaningful error messages**: Include context in your error messages to help with debugging.

- **Implement idempotent operations**: Ensure that `upsert_object` can be called multiple times with the same key without causing issues.

## Testing Your Object Store Provider

When developing your object store provider, consider testing:

- **Basic operations**: Test all four main operations (put, upsert, get, delete)
- **Error conditions**: Test with non-existent keys, duplicate keys, and invalid data
- **Concurrent access**: Test with multiple concurrent operations
- **Large objects**: Test with objects of various sizes
- **Metadata handling**: Test with and without metadata and content types

## Plugin Integration

To integrate your object store provider as a plugin, follow the standard NeMo Agent toolkit plugin structure:

1. Create a plugin package with the appropriate structure
2. Include your config, implementation, and registration code
3. Add the necessary dependencies to your plugin's `pyproject.toml`
4. Ensure your plugin is discoverable by NeMo Agent toolkit

For more information on creating plugins, see the [Plugins](../extend/plugins.md) documentation.
