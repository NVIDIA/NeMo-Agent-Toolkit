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
"""Re-exported from :mod:`nat.builder.framework_instrumentation`, where the framework hook now lives."""

from nat.builder.framework_instrumentation import _library_instrumented  # noqa: F401
from nat.builder.framework_instrumentation import callback_handler_var  # noqa: F401
from nat.builder.framework_instrumentation import chain_wrapped_build_fn  # noqa: F401
from nat.builder.framework_instrumentation import set_framework_profiler_handler  # noqa: F401
