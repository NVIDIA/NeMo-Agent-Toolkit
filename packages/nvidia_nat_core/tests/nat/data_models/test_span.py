# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.data_models.span import Span
from nat.data_models.span import SpanContext


def test_span_without_context_gets_a_default_context():
    """A span created without a context gets a default SpanContext."""
    # The before-validator that fills in a SpanContext does not run on field
    # defaults, so omitting the argument used to leave the context as None.
    span = Span(name="x")

    assert isinstance(span.context, SpanContext)


def test_span_default_contexts_are_distinct():
    """Each span gets its own default context rather than a shared one."""
    assert Span(name="a").context.span_id != Span(name="b").context.span_id


def test_span_explicit_none_context_still_gets_a_default_context():
    """Passing context=None explicitly also gets a default SpanContext."""
    assert isinstance(Span(name="x", context=None).context, SpanContext)
