# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

import yaml
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def schema_exception_handler(func, **kwargs):
    """
    A decorator that handles `ValidationError` exceptions for schema validation functions.

    This decorator wraps a function that performs schema validation using Pydantic.
    If a `ValidationError` is raised, it logs detailed error messages and raises a `ValueError` with the combined error
    messages.

    Parameters
    ----------
    func : callable
        The function to be decorated. This function is expected to perform schema validation.

    kwargs : dict
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    callable
        The wrapped function that executes `func` with exception handling.

    Raises
    ------
    ValueError
        If a `ValidationError` is caught, this decorator logs the error details and raises a `ValueError` with the
        combined error messages.

    Notes
    -----
    This decorator is particularly useful for functions that validate configurations or data models,
    ensuring that any validation errors are logged and communicated clearly.

    Examples
    --------
    >>> @schema_exception_handler
    ... def validate_config(config_data):
    ...     schema = MySchema(**config_data)
    ...     return schema
    ...
    >>> try:
    ...     validate_config(invalid_config)
    ... except ValueError as e:
    ...     logger.error("Caught error: %s", e)
    Caught error: Invalid configuration: field1: value is not a valid integer; field2: field required
    """

    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            error_messages = "; ".join([f"{error['loc'][0]}: {error['msg']}" for error in e.errors()])
            log_error_message = f"Invalid configuration: {error_messages}"
            logger.error(log_error_message)
            raise ValueError(log_error_message) from e

    return inner_function


def yaml_exception_handler(func):
    """
    A decorator that handles YAML parsing exceptions.

    This decorator wraps a function that performs YAML file operations.
    If a YAML-related error occurs, it logs the error and raises a ValueError
    with a clear error message.

    Returns
    -------
    callable
        The wrapped function that executes `func` with YAML exception handling.

    Raises
    ------
    ValueError
        If a YAML error is caught, with details about the parsing failure.
    """

    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except yaml.YAMLError as e:
            log_error_message = f"Invalid YAML configuration: {str(e)}"
            logger.error(log_error_message)
            raise ValueError(log_error_message) from e

        except Exception as e:
            log_error_message = f"Error reading YAML file: {str(e)}"
            logger.error(log_error_message)
            raise ValueError(log_error_message) from e

    return inner_function
