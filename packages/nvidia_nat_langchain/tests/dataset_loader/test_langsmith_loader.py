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

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from nat.plugins.langchain.dataset_loader.langsmith import load_langsmith_dataset
from nat.plugins.langchain.dataset_loader.register import EvalDatasetLangSmithConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(inputs: dict, outputs: dict | None = None, example_id: str | None = None):
    """Create a mock LangSmith Example object."""
    return SimpleNamespace(
        id=uuid.UUID(example_id) if example_id else uuid.uuid4(),
        inputs=inputs,
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestEvalDatasetLangSmithConfig:

    def test_config_with_dataset_id(self):
        config = EvalDatasetLangSmithConfig(dataset_id="abc-123")
        assert config.dataset_id == "abc-123"
        assert config.dataset_name is None

    def test_config_with_dataset_name(self):
        config = EvalDatasetLangSmithConfig(dataset_name="my-dataset")
        assert config.dataset_name == "my-dataset"
        assert config.dataset_id is None

    def test_config_with_both(self):
        config = EvalDatasetLangSmithConfig(dataset_id="abc-123", dataset_name="my-dataset")
        assert config.dataset_id == "abc-123"
        assert config.dataset_name == "my-dataset"

    def test_config_requires_id_or_name(self):
        with pytest.raises(ValueError, match="At least one of"):
            EvalDatasetLangSmithConfig()

    def test_parser_returns_callable(self):
        config = EvalDatasetLangSmithConfig(dataset_id="abc-123")
        load_fn, kwargs = config.parser()
        assert callable(load_fn)
        assert isinstance(kwargs, dict)
        assert kwargs["dataset_id"] == "abc-123"

    def test_parser_includes_structure_keys(self):
        config = EvalDatasetLangSmithConfig(
            dataset_id="abc-123",
            structure={
                "question_key": "q", "answer_key": "a"
            },
        )
        _, kwargs = config.parser()
        assert kwargs["question_col"] == "q"
        assert kwargs["answer_col"] == "a"

    def test_parser_includes_optional_fields(self):
        config = EvalDatasetLangSmithConfig(
            dataset_id="abc-123",
            split="test",
            as_of="v2",
            limit=50,
        )
        _, kwargs = config.parser()
        assert kwargs["split"] == "test"
        assert kwargs["as_of"] == "v2"
        assert kwargs["limit"] == 50


# ---------------------------------------------------------------------------
# Loader function tests
# ---------------------------------------------------------------------------


class TestLoadLangSmithDataset:

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_prefers_id_over_name(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_client_cls.return_value = mock_client

        load_langsmith_dataset(
            None,
            dataset_id="id-123",
            dataset_name="name-456",
        )

        mock_client.list_examples.assert_called_once_with(dataset_id="id-123")

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_basic(self, mock_client_cls):
        examples = [
            _make_example({"input": f"q{i}"}, {"output": f"a{i}"}, f"00000000-0000-0000-0000-00000000000{i}")
            for i in range(1, 4)
        ]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_client_cls.return_value = mock_client

        df = load_langsmith_dataset(None, dataset_id="test-id")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns[:3]) == ["id", "question", "answer"]
        assert df["question"].tolist() == ["q1", "q2", "q3"]
        assert df["answer"].tolist() == ["a1", "a2", "a3"]

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_custom_keys(self, mock_client_cls):
        examples = [
            _make_example({"prompt": "hello"}, {"response": "world"}),
        ]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_client_cls.return_value = mock_client

        df = load_langsmith_dataset(
            None,
            dataset_id="test-id",
            input_key="prompt",
            output_key="response",
            question_col="q",
            answer_col="a",
        )

        assert df["q"].tolist() == ["hello"]
        assert df["a"].tolist() == ["world"]

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_limit(self, mock_client_cls):
        examples = [_make_example({"input": f"q{i}"}, {"output": f"a{i}"}) for i in range(10)]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_client_cls.return_value = mock_client

        df = load_langsmith_dataset(None, dataset_id="test-id", limit=2)

        assert len(df) == 2

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_split(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_client_cls.return_value = mock_client

        load_langsmith_dataset(None, dataset_id="test-id", split="test")

        call_kwargs = mock_client.list_examples.call_args[1]
        assert call_kwargs["splits"] == ["test"]

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_as_of(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_client_cls.return_value = mock_client

        load_langsmith_dataset(None, dataset_id="test-id", as_of="v2")

        call_kwargs = mock_client.list_examples.call_args[1]
        assert call_kwargs["as_of"] == "v2"

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter([])
        mock_client_cls.return_value = mock_client

        df = load_langsmith_dataset(None, dataset_id="test-id")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["id", "question", "answer"]

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_null_outputs(self, mock_client_cls):
        examples = [
            _make_example({"input": "q1"}, None),
        ]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_client_cls.return_value = mock_client

        df = load_langsmith_dataset(None, dataset_id="test-id")

        assert len(df) == 1
        assert df["question"].tolist() == ["q1"]
        assert df["answer"].tolist() == [""]

    def test_load_raises_without_id_or_name(self):
        with pytest.raises(ValueError, match="At least one of"):
            load_langsmith_dataset(None)

    @patch("nat.plugins.langchain.dataset_loader.langsmith.Client")
    def test_load_extra_fields_preserved(self, mock_client_cls):
        examples = [
            _make_example(
                {
                    "input": "q1", "context": "ctx1"
                },
                {
                    "output": "a1", "score": 0.9
                },
            ),
        ]
        mock_client = MagicMock()
        mock_client.list_examples.return_value = iter(examples)
        mock_client_cls.return_value = mock_client

        df = load_langsmith_dataset(None, dataset_id="test-id")

        assert "context" in df.columns
        assert "score" in df.columns
        assert df["context"].tolist() == ["ctx1"]
        assert df["score"].tolist() == [0.9]


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:

    def test_registration(self):
        import nat.plugins.langchain.dataset_loader.register  # noqa: F401
        from nat.cli.type_registry import GlobalTypeRegistry

        registry = GlobalTypeRegistry.get()
        info = registry.get_dataset_loader(EvalDatasetLangSmithConfig)
        assert info is not None
        assert info.build_fn is not None

    def test_yaml_backward_compat(self):
        import nat.plugins.langchain.dataset_loader.register  # noqa: F401
        from nat.data_models.evaluate import EvalConfig
        from nat.data_models.evaluate import EvalGeneralConfig

        EvalConfig.rebuild_annotations()

        config = EvalGeneralConfig.model_validate({
            "dataset": {
                "_type": "langsmith",
                "dataset_name": "my-test-dataset",
            },
        })
        assert isinstance(config.dataset, EvalDatasetLangSmithConfig)
        assert config.dataset.dataset_name == "my-test-dataset"
