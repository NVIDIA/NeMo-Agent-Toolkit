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
"""
General-purpose compilation framework for agent optimization.

The package provides a layered architecture:

- ``AbstractCompiler`` -- framework-agnostic base (any source -> any artifact)
- ``AbstractPipelinedCompiler`` -- ordered stage pipeline over a ``CompilationContext``
- ``CompilationStage`` -- protocol for individual pipeline stages
- ``CompilationContext`` -- shared mutable state between stages

Graph-specific optimization is provided as a concrete specialization:

- ``DefaultGraphCompiler`` -- the standard 6-stage graph pipeline
- ``context_to_result`` -- convert a graph compilation context to a
  ``TransformationResult``

For simple graph use cases, ``GraphOptimizer`` wraps
``DefaultGraphCompiler`` with a one-call API.

For custom pipelines (graph or otherwise), subclass
``AbstractPipelinedCompiler`` with your own stages:

    from nat_app.compiler import AbstractPipelinedCompiler, CompilationStage

    class MyCompiler(AbstractPipelinedCompiler[MySource, MyArtifact]):
        def default_stages(self): ...
        def prepare(self, source, **kw): ...
"""

from nat_app.compiler.compilation_context import CompilationContext
from nat_app.compiler.compilation_stage import CompilationStage
from nat_app.compiler.compiler import AbstractCompiler
from nat_app.compiler.compiler import UnsupportedSourceError
from nat_app.compiler.compiler import compile_with
from nat_app.compiler.default_graph_compiler import DefaultGraphCompiler
from nat_app.compiler.default_graph_compiler import context_to_result
from nat_app.compiler.errors import GraphValidationError
from nat_app.compiler.optimizer import GraphOptimizer
from nat_app.compiler.pipelined_compiler import AbstractPipelinedCompiler

__all__ = [
    "AbstractCompiler",
    "AbstractPipelinedCompiler",
    "CompilationContext",
    "CompilationStage",
    "compile_with",
    "context_to_result",
    "DefaultGraphCompiler",
    "GraphOptimizer",
    "GraphValidationError",
    "UnsupportedSourceError",
]
