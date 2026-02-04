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

from nat.data_models.optimizer import ObjectStoreSettings
from nat.data_models.optimizer import OptimizerConfig


class TestObjectStoreSettings:

    def test_create_with_name(self):
        """Test ObjectStoreSettings creation with name only."""
        settings = ObjectStoreSettings(name="local_store")

        assert settings.name == "local_store"
        assert settings.key_prefix is None

    def test_create_with_prefix(self):
        """Test ObjectStoreSettings creation with name and prefix."""
        settings = ObjectStoreSettings(name="s3_store", key_prefix="experiment_001")

        assert settings.name == "s3_store"
        assert settings.key_prefix == "experiment_001"


class TestOptimizerConfig:

    def test_create_without_object_store(self):
        """Test OptimizerConfig can be created without object_store."""
        config = OptimizerConfig(
            eval_metrics={"accuracy": {
                "evaluator_name": "exact_match", "direction": "maximize", "weight": 1.0
            }})

        assert config.object_store is None

    def test_create_with_object_store(self):
        """Test OptimizerConfig accepts object_store settings."""
        config = OptimizerConfig(
            eval_metrics={"accuracy": {
                "evaluator_name": "exact_match", "direction": "maximize", "weight": 1.0
            }},
            object_store={
                "name": "local_store", "key_prefix": "experiment_001"
            })

        assert config.object_store is not None
        assert config.object_store.name == "local_store"
        assert config.object_store.key_prefix == "experiment_001"

    def test_create_with_object_store_no_prefix(self):
        """Test OptimizerConfig with object_store but no key_prefix."""
        config = OptimizerConfig(
            eval_metrics={"accuracy": {
                "evaluator_name": "exact_match", "direction": "maximize", "weight": 1.0
            }},
            object_store={"name": "s3_store"})

        assert config.object_store is not None
        assert config.object_store.name == "s3_store"
        assert config.object_store.key_prefix is None
