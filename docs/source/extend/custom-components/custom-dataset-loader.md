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

# Adding a Custom Dataset Loader

:::{note}
We recommend reading the [Evaluating NeMo Agent Toolkit Workflows](../../improve-workflows/evaluate.md) guide before proceeding with this detailed documentation.
:::

NeMo Agent Toolkit provides built-in dataset loaders for common file formats (`json`, `jsonl`, `csv`, `xls`, `parquet`, and `custom`). In addition, the toolkit provides a plugin system to add custom dataset loaders for new file formats or data sources.

## Summary
This guide provides a step-by-step process to create and register a custom dataset loader with NeMo Agent Toolkit. A TSV (tab-separated values) dataset loader is used as an example to demonstrate the process.

## Existing Dataset Loaders
You can view the list of existing dataset loaders by running the following command:
```bash
nat info components -t dataset_loader
```

## Extending NeMo Agent Toolkit with Custom Dataset Loaders
To extend NeMo Agent Toolkit with custom dataset loaders, you need to create a dataset loader configuration class and a registration function, then register it with NeMo Agent Toolkit using the `register_dataset_loader` decorator.

### Dataset Loader Configuration
The dataset loader configuration defines the dataset type name and any format-specific parameters. This configuration is paired with a registration function that yields a `DatasetLoaderInfo` object containing the load function.

The following example shows how to define and register a custom dataset loader for TSV files:

<!-- path-check-skip-begin -->
```python
# my_plugin/dataset_loader_register.py
import pandas as pd
from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader
from nat.data_models.dataset_handler import EvalDatasetBaseConfig


class EvalDatasetTsvConfig(EvalDatasetBaseConfig, name="tsv"):
    """Configuration for TSV dataset loader."""
    separator: str = Field(default="\t", description="Column separator character.")


@register_dataset_loader(config_type=EvalDatasetTsvConfig)
async def register_tsv_dataset_loader(config: EvalDatasetTsvConfig, builder: EvalBuilder):
    """Register TSV dataset loader."""

    def load_tsv(file_path, **kwargs):
        return pd.read_csv(file_path, sep=config.separator, **kwargs)

    yield DatasetLoaderInfo(config=config, load_fn=load_tsv, description="TSV file dataset loader")
```
<!-- path-check-skip-end -->

- The `EvalDatasetTsvConfig` class extends `EvalDatasetBaseConfig` with the `name="tsv"` parameter, which sets the `_type` value used in YAML configuration files.
- The `register_tsv_dataset_loader` function uses the `@register_dataset_loader` decorator to register the dataset loader with NeMo Agent Toolkit.
- The function yields a `DatasetLoaderInfo` object, which binds the config, load function, and a human-readable description.

### Understanding `DatasetLoaderInfo`

The `DatasetLoaderInfo` class contains the following fields:
- `config`: The dataset loader configuration object (an instance of `EvalDatasetBaseConfig` or a subclass).
- `load_fn`: A callable that takes a file path and optional keyword arguments and returns a `pandas.DataFrame`. This function is used by the evaluation framework to load the dataset.
- `description`: A human-readable description of the dataset loader.

### Importing for Registration
To ensure the dataset loader is registered at runtime, import the registration function in your project's `register.py` file -- even if the function is not called directly.

<!-- path-check-skip-begin -->
```python
# my_plugin/register.py
from .dataset_loader_register import register_tsv_dataset_loader
```
<!-- path-check-skip-end -->

### Entry Point
Add an entry point in your `pyproject.toml` so that NeMo Agent Toolkit discovers the plugin automatically:

```toml
[project.entry-points.'nat.plugins']
my_plugin = "my_plugin.register"
```

### Display All Dataset Loaders
To display all registered dataset loaders, run the following command:
```bash
nat info components -t dataset_loader
```
This will now display the custom dataset loader `tsv` in the list of dataset loaders.

### Using the Custom Dataset Loader
Once registered, you can use the custom dataset loader in your evaluation configuration:

```yaml
eval:
  general:
    dataset:
      _type: tsv
      file_path: <path to your file>
      separator: "\t"
```

The `_type` field specifies the dataset loader name. All fields defined in the configuration class are available as YAML keys.

### Running the Evaluation
Run the evaluation using the standard command:
```bash
nat eval --config_file <path to file>
```

## Built-in Dataset Loaders

The following dataset loaders are included with NeMo Agent Toolkit:

| Type | Description | Load Function |
|------|-------------|---------------|
| `json` | JSON file dataset | `pandas.read_json` |
| `jsonl` | JSON Lines file dataset | Custom JSONL reader |
| `csv` | CSV file dataset | `pandas.read_csv` |
| `parquet` | Parquet file dataset | `pandas.read_parquet` |
| `xls` | Excel file dataset | `pandas.read_excel`  |
| `custom` | Custom parser function | User-provided function via `function` config key |

<!-- path-check-skip-end -->

For more details on the built-in dataset formats and their configuration options, see the [Using Datasets](../../improve-workflows/evaluate.md#using-datasets) section in the evaluation guide.
