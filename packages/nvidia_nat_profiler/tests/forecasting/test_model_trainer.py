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

from nat.data_models.atif import Agent
from nat.data_models.atif import Metrics
from nat.data_models.atif import Step
from nat.data_models.atif import Trajectory
from nat.plugins.profiler.atif_dataframe import create_dataframe_from_atif
from nat.plugins.profiler.forecasting.model_trainer import ModelTrainer
from nat.plugins.profiler.forecasting.model_trainer import create_model
from nat.plugins.profiler.forecasting.models import ForecastingBaseModel
from nat.plugins.profiler.forecasting.models import LinearModel
from nat.plugins.profiler.forecasting.models import RandomForestModel
from nat.plugins.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor


@pytest.mark.parametrize("model_type, expected_model_class", [
    ("linear", LinearModel),
    ("randomforest", RandomForestModel),
],
                         ids=["linear", "randomforest"])
def test_create_model(model_type: str, expected_model_class: type[ForecastingBaseModel]):
    assert isinstance(create_model(model_type), expected_model_class)


def test_create_model_invalid_type():
    with pytest.raises(ValueError, match="Unsupported model_type: unsupported_model"):
        create_model("unsupported_model")


@pytest.mark.parametrize("model_trainer_kwargs", [
    {},
    {
        "model_type": "linear"
    },
    {
        "model_type": "randomforest"
    },
])
def test_model_trainer_initialization(model_trainer_kwargs: dict):
    mt = ModelTrainer(**model_trainer_kwargs)
    if "model_type" in model_trainer_kwargs:
        assert mt.model_type == model_trainer_kwargs["model_type"]


@pytest.mark.parametrize("model_type, expected_model_class", [("linear", LinearModel),
                                                              ("randomforest", RandomForestModel)],
                         ids=["linear", "randomforest"])
def test_model_trainer_train(model_type: str,
                             expected_model_class: type[ForecastingBaseModel],
                             rag_intermediate_property_adaptor: list[list[IntermediatePropertyAdaptor]]):
    mt = ModelTrainer(model_type=model_type)
    model = mt.train(rag_intermediate_property_adaptor)
    assert isinstance(model, expected_model_class)


def test_model_trainer_train_with_dataframe():
    """ModelTrainer.train accepts DataFrame from create_dataframe_from_atif (RandomForestModel supports it)."""
    traj = Trajectory(
        agent=Agent(name="test", version="0.0.0"),
        steps=[
            Step(
                step_id=1,
                source="user",
                message="Hi",
                timestamp="2024-01-01T12:00:00+00:00",
            ),
            Step(
                step_id=2,
                source="agent",
                message="Hello!",
                timestamp="2024-01-01T12:00:01+00:00",
                model_name="gpt-4",
                metrics=Metrics(prompt_tokens=100, completion_tokens=20),
            ),
            Step(
                step_id=3,
                source="user",
                message="Again",
                timestamp="2024-01-01T12:00:02+00:00",
            ),
            Step(
                step_id=4,
                source="agent",
                message="Hi again",
                timestamp="2024-01-01T12:00:03+00:00",
                model_name="gpt-4",
                metrics=Metrics(prompt_tokens=80, completion_tokens=15),
            ),
        ],
    )
    df = create_dataframe_from_atif([traj])
    mt = ModelTrainer(model_type="randomforest")
    model = mt.train(df)
    assert isinstance(model, RandomForestModel)
