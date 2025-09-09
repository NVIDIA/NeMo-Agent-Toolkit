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


class PredictorRegistry:
    """A registry for storing and retrieving predictor classes."""
    _registry = {}

    @classmethod
    def register(cls, predictor_name: str, predictor_cls):
        """
        Register a predictor class with a unique name.
        Args:
            predictor_name (str): The name of the predictor type.
            predictor_cls (type): The predictor class to register.
        """
        if predictor_name in cls._registry:
            raise ValueError(f"predictor '{predictor_name}' is already registered.")
        cls._registry[predictor_name] = predictor_cls

    @classmethod
    def get(cls, predictor_name: str):
        """
        Retrieve a predictor class by name.
        Args:
            predictor_name (str): The name of the predictor type.
        Returns:
            The predictor class associated with the predictor_name.
        """
        if predictor_name not in cls._registry:
            raise KeyError(f"predictor '{predictor_name}' is not registered.")
        return cls._registry[predictor_name]

    @classmethod
    def all(cls):
        """
        Retrieve all registered workflows.
        Returns:
            dict: All registered workflows.
        """
        return cls._registry


def register_predictor(predictor_name: str):
    """Decorator to register a predictor class."""

    def decorator(cls):
        PredictorRegistry.register(predictor_name, cls)
        return cls

    return decorator
