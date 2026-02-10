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

import pytest
from pydantic import ValidationError


class TestEvalDatasetConfig:

    def test_file_path_shorthand(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        config = EvalDatasetConfig(file_path="/data/eval.csv")
        assert str(config.file_path) == "/data/eval.csv"
        assert config.object_store is None

    def test_object_store_reference(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        config = EvalDatasetConfig(object_store="my_s3", key="datasets/eval.csv")
        assert config.object_store == "my_s3"
        assert config.key == "datasets/eval.csv"
        assert config.file_path is None

    def test_explicit_format(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        config = EvalDatasetConfig(file_path="data.csv", format="csv")
        assert config.format == "csv"

    def test_format_defaults_to_none(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        config = EvalDatasetConfig(file_path="data.csv")
        assert config.format is None  # will be inferred at load time

    def test_structure_defaults(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        config = EvalDatasetConfig(file_path="data.csv")
        assert config.structure.question_key == "question"
        assert config.structure.answer_key == "answer"
        assert config.id_key == "id"

    def test_filter_defaults(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        config = EvalDatasetConfig(file_path="data.csv")
        assert config.filter is not None

    def test_must_specify_source(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        with pytest.raises(ValidationError):
            EvalDatasetConfig()

    def test_object_store_requires_key(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        with pytest.raises(ValidationError):
            EvalDatasetConfig(object_store="my_s3")

    def test_extra_fields_forbidden(self):
        from nat.data_models.dataset_handler import EvalDatasetConfig
        with pytest.raises(ValidationError):
            EvalDatasetConfig(file_path="data.csv", unknown_field="value")
