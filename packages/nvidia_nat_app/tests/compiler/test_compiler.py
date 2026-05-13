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
"""Tests for AbstractCompiler, compile_with, and UnsupportedSourceError."""

import pytest

from nat_app.compiler.compiler import AbstractCompiler
from nat_app.compiler.compiler import UnsupportedSourceError
from nat_app.compiler.compiler import compile_with


class _ConcreteCompiler(AbstractCompiler):

    def compile(self, source, **kwargs):
        return f"compiled:{source}"


class _RejectingCompiler(_ConcreteCompiler):

    def validate(self, source):
        return False


class TestUnsupportedSourceError:

    def test_is_value_error(self):
        assert issubclass(UnsupportedSourceError, ValueError)

    def test_message(self):
        err = UnsupportedSourceError("bad format")
        assert "bad format" in str(err)


class TestAbstractCompiler:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AbstractCompiler()

    def test_validate_default_true(self):
        c = _ConcreteCompiler()
        assert c.validate("anything") is True

    def test_export_default_not_implemented(self):
        c = _ConcreteCompiler()
        with pytest.raises(NotImplementedError):
            c.export("compiled", "/tmp/out")

    def test_compile(self):
        c = _ConcreteCompiler()
        assert c.compile("test") == "compiled:test"


class TestCompileWith:

    def test_success(self):
        c = _ConcreteCompiler()
        result = compile_with("input", c)
        assert result == "compiled:input"

    def test_validation_fails(self):
        c = _RejectingCompiler()
        with pytest.raises(UnsupportedSourceError):
            compile_with("input", c)

    def test_kwargs_forwarded(self):

        class _KwargsCompiler(AbstractCompiler):

            def compile(self, source, **kwargs):
                return kwargs.get("extra")

        result = compile_with("x", _KwargsCompiler(), extra="val")
        assert result == "val"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
