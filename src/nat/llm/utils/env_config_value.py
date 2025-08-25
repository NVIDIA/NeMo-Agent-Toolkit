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

import os
from abc import ABC
from enum import Enum


class EnvConfigValueSource(Enum):
    ENV_DEFAULT = 1
    CONSTRUCTOR = 2
    ENV_OVERRIDE = 3


class EnvConfigValue(ABC):
    """
    A wrapper for a string used as a configuration value which can be loaded from the system environment or injected via
    the constructor. This class should be subclassed and the class fields `_ENV_KEY` and `_ENV_KEY_OVERRIDE` can be set
    to enable environment-loading functionality. Convienience properties are available to check from where the value was
    loaded.
    """

    _ENV_KEY: str | None = None
    _ENV_KEY_OVERRIDE: str | None = None
    _ALLOW_NONE: bool = False

    def __init__(self, value: str | None = None, use_env: bool = True):
        """
        Parameters
        ----------
        value : str, optional
            The value to be contained in the EnvConfigValue. If the value is `None`, an attempt will be made to load it
            from the environment using `_ENV_KEY`. if the `_ENV_KEY_OVERRIDE` field is not `None`, an attempt will be
            made to load that environment variable in place of the passed-in value.
        use_env : bool
            If False, all environment-loading logic will be bypassed and the passed-in value will be used as-is.
            defaults to True.
        """

        self._source = EnvConfigValueSource.CONSTRUCTOR

        if use_env:
            if value is None and self.__class__._ENV_KEY is not None:
                value = os.environ.get(self.__class__._ENV_KEY, None)
                self._source = EnvConfigValueSource.ENV_DEFAULT

            if self.__class__._ENV_KEY_OVERRIDE is not None and self.__class__._ENV_KEY_OVERRIDE in os.environ:
                value = os.environ[self.__class__._ENV_KEY_OVERRIDE]
                self._source = EnvConfigValueSource.ENV_OVERRIDE

            if not self.__class__._ALLOW_NONE and value is None:

                message = ("value must not be None, but provided value was None and no environment-based default or "
                           "override was found.")

                if self.__class__._ENV_KEY is None:
                    raise ValueError(message)

                raise ValueError(
                    f"{message} Try passing a value to the constructor, or setting the `{self.__class__._ENV_KEY}` "
                    "environment variable.")

        elif not self.__class__._ALLOW_NONE and value is None:
            raise ValueError("value must not be none")

        assert isinstance(value, str) or value is None

        self._value = value
        self._use_env = use_env

    @property
    def source(self) -> EnvConfigValueSource:
        return self._source

    @property
    def use_env(self) -> bool:
        return self._use_env

    @property
    def value(self) -> str | None:
        return self._value
