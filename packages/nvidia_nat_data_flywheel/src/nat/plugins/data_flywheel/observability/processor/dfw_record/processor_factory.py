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

from typing import Any

from nat.observability.processor.processor import Processor


def processor_factory(processor_class: type, from_type: type[Any], to_type: type[Any]) -> type[Processor]:

    class ConcreteProcessor(processor_class[from_type, to_type]):  # type: ignore
        pass

    return ConcreteProcessor


def processor_factory_from_type(processor_class: type, from_type: type[Any]) -> type[Processor]:

    class ConcreteProcessor(processor_class[from_type]):  # type: ignore
        pass

    return ConcreteProcessor


def processor_factory_to_type(processor_class: type, to_type: type[Any]) -> type[Processor]:

    class ConcreteProcessor(processor_class[to_type | None]):  # type: ignore
        pass

    return ConcreteProcessor
