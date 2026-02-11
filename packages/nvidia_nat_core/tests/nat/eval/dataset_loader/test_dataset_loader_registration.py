# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pytest

from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import RegisteredDatasetLoaderInfo
from nat.data_models.dataset_handler import EvalDatasetBaseConfig
from nat.data_models.dataset_handler import EvalDatasetCsvConfig
from nat.data_models.dataset_handler import EvalDatasetCustomConfig
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.dataset_handler import EvalDatasetJsonlConfig
from nat.data_models.dataset_handler import EvalDatasetParquetConfig
from nat.data_models.dataset_handler import EvalDatasetXlsConfig
from nat.data_models.discovery_metadata import DiscoveryMetadata


def test_builtin_dataset_loaders_registered():
    """Verify all 6 built-in dataset types are registered in the TypeRegistry."""
    import nat.eval.dataset_loader.register  # noqa: F401

    registry = GlobalTypeRegistry.get()

    for config_type in [
            EvalDatasetJsonConfig,
            EvalDatasetJsonlConfig,
            EvalDatasetCsvConfig,
            EvalDatasetParquetConfig,
            EvalDatasetXlsConfig,
            EvalDatasetCustomConfig,
    ]:
        info = registry.get_dataset_loader(config_type)
        assert info is not None
        assert info.build_fn is not None


def test_compute_annotation_for_dataset_base():
    """Verify compute_annotation returns a valid union type for datasets."""
    import nat.eval.dataset_loader.register  # noqa: F401

    registry = GlobalTypeRegistry.get()
    annotation = registry.compute_annotation(EvalDatasetBaseConfig)
    assert annotation is not None


def test_yaml_backward_compat_csv():
    """Verify _type: csv in YAML still parses to EvalDatasetCsvConfig."""
    import nat.eval.dataset_loader.register  # noqa: F401
    from nat.data_models.evaluate import EvalConfig
    from nat.data_models.evaluate import EvalGeneralConfig

    EvalConfig.rebuild_annotations()

    config = EvalGeneralConfig.model_validate({"dataset": {"_type": "csv", "file_path": "/tmp/test.csv"}})
    assert isinstance(config.dataset, EvalDatasetCsvConfig)


def test_yaml_backward_compat_json():
    """Verify _type: json in YAML still parses to EvalDatasetJsonConfig."""
    import nat.eval.dataset_loader.register  # noqa: F401
    from nat.data_models.evaluate import EvalConfig
    from nat.data_models.evaluate import EvalGeneralConfig

    EvalConfig.rebuild_annotations()

    config = EvalGeneralConfig.model_validate({"dataset": {"_type": "json", "file_path": "/tmp/test.json"}})
    assert isinstance(config.dataset, EvalDatasetJsonConfig)


def test_registered_dataset_loader_info_fields():
    """Verify RegisteredDatasetLoaderInfo has the correct structure."""

    def mock_fn(config, builder):
        pass

    info = RegisteredDatasetLoaderInfo(
        full_type="nat_core/csv",
        config_type=EvalDatasetCsvConfig,
        build_fn=mock_fn,
    )
    assert info.full_type == "nat_core/csv"
    assert info.config_type is EvalDatasetCsvConfig
    assert info.module_name == "nat_core"
    assert info.local_name == "csv"


def test_duplicate_registration_raises():
    """Verify that registering the same config type twice raises ValueError."""

    with GlobalTypeRegistry.push() as registry:

        class TestDuplicateConfig(EvalDatasetBaseConfig, name="test_dup_ds"):
            pass

        def mock_fn(config, builder):
            pass

        info = RegisteredDatasetLoaderInfo(
            full_type="test/test_dup_ds",
            config_type=TestDuplicateConfig,
            build_fn=mock_fn,
            discovery_metadata=DiscoveryMetadata(),
        )
        registry.register_dataset_loader(info)

        with pytest.raises(ValueError, match="already been registered"):
            registry.register_dataset_loader(info)


def test_dataset_loader_info_creation():
    """Verify DatasetLoaderInfo dataclass works correctly."""
    from nat.builder.dataset_loader import DatasetLoaderInfo

    config = EvalDatasetCsvConfig(file_path="/tmp/test.csv")
    info = DatasetLoaderInfo(config=config, load_fn=pd.read_csv, description="Test CSV loader")
    assert info.config is config
    assert info.load_fn is pd.read_csv
    assert info.description == "Test CSV loader"
