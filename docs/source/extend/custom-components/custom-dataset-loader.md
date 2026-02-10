<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

<!-- path-check-skip-begin -->

# Custom Data Sources for Evaluation

:::{note}
We recommend reading the [Evaluating NeMo Agent Toolkit Workflows](../../improve-workflows/evaluate.md) guide before proceeding with this detailed documentation.
:::

NeMo Agent Toolkit loads evaluation datasets through the [ObjectStore](../../build-workflows/object-store.md) subsystem. Built-in support covers common file formats (CSV, JSON, JSONL, Parquet, and Excel) and storage backends (local files, S3, Redis, MySQL). This guide shows how to configure dataset loading and how to create a custom ObjectStore implementation for novel data sources.

## Loading from a Local File

The simplest way to specify an evaluation dataset is the `file_path` shorthand. NeMo Agent Toolkit creates a transient `FileObjectStore` behind the scenes and infers the data format from the file extension.

```yaml
eval:
  general:
    dataset:
      file_path: /data/eval.csv
      structure:
        question_key: input
        answer_key: expected_output
```

## Loading from a Remote ObjectStore

To load data from a configured ObjectStore (for example, S3), reference the store by name and provide the object key:

```yaml
object_stores:
  s3_data:
    _type: s3
    bucket: my-eval-datasets

eval:
  general:
    dataset:
      object_store: s3_data
      key: v2/eval.parquet
```

The named ObjectStore must be declared in the top-level `object_stores` section of your configuration file. Any ObjectStore backend that NeMo Agent Toolkit supports (S3, Redis, MySQL, or a custom implementation) can be used here.

## Format Inference and Explicit Override

The data format is inferred from the file extension of the `file_path` or `key`:

| Extension | Format |
|-----------|--------|
| `.csv` | csv |
| `.json` | json |
| `.jsonl` | jsonl |
| `.parquet` | parquet |
| `.xls`, `.xlsx` | xls |

If the file extension does not match the actual format, or if there is no extension, you can specify the format explicitly:

```yaml
eval:
  general:
    dataset:
      file_path: /data/eval_data
      format: csv
```

## Creating a Custom ObjectStore for Novel Data Sources

If your evaluation data lives in a source not covered by the built-in ObjectStore providers (for example, a REST API, a database query, or a proprietary storage system), you can create a custom ObjectStore implementation. This replaces the former DatasetLoader plugin approach.

### Example: API-backed ObjectStore

The following example shows how to create an ObjectStore that fetches data from a custom REST API.

<!-- path-check-skip-begin -->
```python
# my_plugin/api_object_store.py
import httpx
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override


class ApiObjectStore(ObjectStore):
    """ObjectStore that reads data from a REST API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url
        self._api_key = api_key

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._base_url}/datasets/{key}",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if response.status_code == 404:
                raise NoSuchKeyError(key)
            response.raise_for_status()
            return ObjectStoreItem(
                data=response.content,
                content_type=response.headers.get("content-type"),
            )

    @override
    async def put_object(self, key: str, item: ObjectStoreItem) -> None:
        raise NotImplementedError("Read-only store")

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:
        raise NotImplementedError("Read-only store")

    @override
    async def delete_object(self, key: str) -> None:
        raise NotImplementedError("Read-only store")
```
<!-- path-check-skip-end -->

For dataset loading, only `get_object` needs a real implementation. The base `ObjectStore.read_dataframe()` method will call `get_object` to fetch the raw bytes and then parse them into a pandas DataFrame using the inferred (or explicit) format.

### Overriding `read_dataframe()` for Efficient Native Reads

If your backend can produce a DataFrame directly (for example, via a SQL query or a native API), you can override `read_dataframe()` to skip the bytes-to-DataFrame parsing:

<!-- path-check-skip-begin -->
```python
class ApiObjectStore(ObjectStore):
    # ... (same as above)

    @override
    async def read_dataframe(self, key: str, format: str | None = None, **kwargs):
        """Fetch data directly as a DataFrame from the API."""
        import pandas as pd

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self._base_url}/datasets/{key}/records",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            return pd.DataFrame(response.json())
```
<!-- path-check-skip-end -->

### Registering the Custom ObjectStore

Create a config class and registration function following the standard ObjectStore plugin pattern:

<!-- path-check-skip-begin -->
```python
# my_plugin/register.py
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.object_store import ObjectStoreBaseConfig


class ApiObjectStoreConfig(ObjectStoreBaseConfig, name="api_store"):
    base_url: str
    api_key: str


@register_object_store(config_type=ApiObjectStoreConfig)
async def api_object_store(config: ApiObjectStoreConfig, _builder: Builder):
    from .api_object_store import ApiObjectStore
    yield ApiObjectStore(base_url=config.base_url, api_key=config.api_key)
```
<!-- path-check-skip-end -->

Add an entry point in your `pyproject.toml` so that NeMo Agent Toolkit discovers the plugin automatically:

```toml
[project.entry-points.'nat.plugins']
my_plugin = "my_plugin.register"
```

### Using the Custom ObjectStore for Evaluation

Once registered, reference the custom ObjectStore in your evaluation configuration:

```yaml
object_stores:
  my_api:
    _type: api_store
    base_url: https://data.example.com
    api_key: ${API_KEY}

eval:
  general:
    dataset:
      object_store: my_api
      key: eval-set-v3
      format: json
```

## Built-in Format Support

The following formats are supported for parsing evaluation datasets:

| Format | Reader | Notes |
|--------|--------|-------|
| `csv` | `pandas.read_csv` | Default for `.csv` files |
| `json` | `pandas.read_json` | Expects a JSON array of records |
| `jsonl` | Custom JSONL reader | One JSON object per line |
| `parquet` | `pandas.read_parquet` | Binary columnar format |
| `xls` | `pandas.read_excel` | Requires `openpyxl`; covers `.xls` and `.xlsx` |

For more details on ObjectStore configuration and the available built-in providers, see the [Object Stores](../../build-workflows/object-store.md) documentation.

For details on how to create a custom ObjectStore provider, see [Adding an Object Store Provider](object-store.md).

<!-- path-check-skip-end -->
