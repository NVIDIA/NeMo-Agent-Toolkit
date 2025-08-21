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

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for evaluation call objects that have an optional 'id' attribute
EvalCallType = Any  # Could be Weave Call object or other tracing framework objects


class EvalTraceContext:
    """
    Evaluation trace context manager for coordinating traces.

    This class provides a framework-agnostic way to:
    1. Track evaluation calls/contexts
    2. Ensure proper parent-child relationships in traces
    """

    def __init__(self):
        self.eval_call: EvalCallType | None = None  # Store the evaluation call/context for propagation

    def set_eval_call(self, eval_call: EvalCallType | None) -> None:
        """Set the evaluation call/context for propagation to traces."""
        self.eval_call = eval_call
        if eval_call:
            logger.debug("Set evaluation call context: %s", getattr(eval_call, 'id', str(eval_call)))

    def get_eval_call(self) -> EvalCallType | None:
        """Get the current evaluation call/context."""
        return self.eval_call

    @contextmanager
    def evaluation_context(self):
        """
        Context manager that can be overridden by framework-specific implementations.
        Default implementation is a no-op.
        """
        yield


class WeaveEvalTraceContext(EvalTraceContext):
    """
    Weave-specific implementation of evaluation trace context.
    """

    def __init__(self):
        super().__init__()
        self.available = False
        self.set_call_stack: Callable[[list[EvalCallType]], Any] | None = None

        try:
            from weave.trace.context.call_context import set_call_stack
            self.set_call_stack = set_call_stack
            self.available = True
        except ImportError:
            self.available = False
            logger.debug("Weave not available for trace context")

    @contextmanager
    def evaluation_context(self):
        """Set the evaluation call as active context for Weave traces."""
        if self.available and self.eval_call and self.set_call_stack:
            try:
                with self.set_call_stack([self.eval_call]):
                    logger.debug("Set Weave evaluation call context: %s",
                                 getattr(self.eval_call, 'id', str(self.eval_call)))
                    yield
            except Exception as e:
                logger.warning("Failed to set Weave evaluation call context: %s", e)
                yield
        else:
            yield
