# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import BaseModel

from aiq.data_models.step_adaptor import StepAdaptorConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

ENDPOINT_BASE_ALL_VALUES = {
    "method": "GET",
    "description": "all values provided",
    "path": "/test",
    "websocket_path": "/ws",
    "openai_api_path": "/openai"
}

ENDPOINT_BASE_REQUIRED_VALUES = {"method": "POST", "description": "only required values"}

ENDPOINT_ALL_VALUES = ENDPOINT_BASE_ALL_VALUES | {'function_name': 'apples'}

ENDPOINT_REQUIRED_VALUES = ENDPOINT_BASE_REQUIRED_VALUES | {'function_name': 'oranges'}

CORS_ALL_VALUES = {
    "allow_origins": ["http://example.com", "https://example.com"],
    "allow_origin_regex": r"^https?://.*\.example\.com$",
    "allow_methods": ["GET", "POST"],
    "allow_headers": ["Content-Type"],
    "allow_credentials": True,
    "expose_headers": ["X-Custom-Header"],
    "max_age": 3600
}

CORS_REQUIRED_VALUES = {}

FAST_API_FRONT_END_CONFIG_ALL_VALUES = {
    "root_path": "/endpoint",
    "host": "testhost",
    "port": 8080,
    "reload": True,
    "workers": 4,
    "step_adaptor": {
        "mode": "custom", "custom_event_types": ["CUSTOM_START", "CUSTOM_END"]
    },
    "workflow": ENDPOINT_BASE_ALL_VALUES.copy(),
    "endpoints": [ENDPOINT_ALL_VALUES.copy()],
    "cors": CORS_ALL_VALUES.copy(),
    "use_gunicorn": True,
    "runner_class": "test_runner_class",
    "object_store": "test_object_store",
}

FAST_API_FRONT_END_CONFIG_REQUIRES_VALUES = {}


def _test_model_instantiation(model_class, model_kwargs):
    """
    Helper function to test instantiation of a Pydantic model.
    """
    model = model_class(**model_kwargs)
    assert model.model_fields_set == model_kwargs.keys()
    for (key, expected_value) in model_kwargs.items():
        actual_value = getattr(model, key)
        if isinstance(actual_value, BaseModel) and isinstance(expected_value, dict):
            _test_model_instantiation(actual_value.__class__, expected_value)
        elif isinstance(actual_value, list) and isinstance(expected_value, list):
            for (i, v) in enumerate(actual_value):
                if isinstance(v, BaseModel) and isinstance(expected_value[i], dict):
                    _test_model_instantiation(v.__class__, expected_value[i])
                else:
                    assert v == expected_value[i]
        else:
            assert actual_value == expected_value

    return model


@pytest.mark.parametrize("endpoint_kwargs", [ENDPOINT_BASE_ALL_VALUES.copy(), ENDPOINT_BASE_REQUIRED_VALUES.copy()],
                         ids=["all-values", "required-values"])
def test_endpoint_base(endpoint_kwargs: dict):
    _test_model_instantiation(FastApiFrontEndConfig.EndpointBase, endpoint_kwargs)


def test_endpoint_base_invalid_method():
    with pytest.raises(ValueError, match=r"validation error for EndpointBase\s+method"):
        FastApiFrontEndConfig.EndpointBase(method="INVALID", description="test")


@pytest.mark.parametrize("endpoint_kwargs", [ENDPOINT_ALL_VALUES.copy(), ENDPOINT_REQUIRED_VALUES.copy()],
                         ids=["all-values", "required-values"])
def test_endpoint(endpoint_kwargs: dict):
    _test_model_instantiation(FastApiFrontEndConfig.Endpoint, endpoint_kwargs)


@pytest.mark.parametrize("cors_kwargs", [CORS_ALL_VALUES.copy(), CORS_REQUIRED_VALUES.copy()],
                         ids=["all-values", "required-values"])
def test_cross_origin_resource_sharing(cors_kwargs: dict):
    model = _test_model_instantiation(FastApiFrontEndConfig.CrossOriginResourceSharing, cors_kwargs)

    if len(model.model_fields_set) == 0:
        # Make sure that the defaults appear reasonable
        assert model.allow_methods == ["GET"]
        assert isinstance(model.allow_headers, list)
        assert isinstance(model.allow_credentials, bool)
        assert isinstance(model.expose_headers, list)
        assert isinstance(model.max_age, int)


@pytest.mark.parametrize(
    "config_kwargs", [FAST_API_FRONT_END_CONFIG_ALL_VALUES.copy(), FAST_API_FRONT_END_CONFIG_REQUIRES_VALUES.copy()],
    ids=["all-values", "required-values"])
def test_fast_api_front_end_config(config_kwargs: dict):
    model = _test_model_instantiation(FastApiFrontEndConfig, config_kwargs)

    if len(model.model_fields_set) == 0:
        # Make sure that the defaults appear reasonable
        assert isinstance(model.root_path, str)
        assert isinstance(model.host, str)
        assert isinstance(model.port, int)
        assert model.port >= 0
        assert model.port <= 65535
        assert isinstance(model.reload, bool)
        assert isinstance(model.workers, int)
        assert model.workers >= 1
        assert isinstance(model.step_adaptor, StepAdaptorConfig)
        assert isinstance(model.workflow, FastApiFrontEndConfig.EndpointBase)
        assert isinstance(model.endpoints, list)
        assert isinstance(model.cors, FastApiFrontEndConfig.CrossOriginResourceSharing)
        assert isinstance(model.use_gunicorn, bool)
        assert (isinstance(model.runner_class, str) or model.runner_class is None)
        assert (isinstance(model.object_store, str) or model.object_store is None)
