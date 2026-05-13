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
"""Tests for constraint data models: NodeConstraints, ResolvedConstraints, OptimizationConfig."""

import pytest

from nat_app.constraints.models import NodeConstraints
from nat_app.constraints.models import OptimizationConfig
from nat_app.constraints.models import ResolvedConstraints


class TestNodeConstraints:

    def test_defaults(self):
        nc = NodeConstraints(name="my_node")
        assert nc.name == "my_node"
        assert nc.force_sequential is False
        assert nc.depends_on == set()
        assert nc.reason is None
        assert nc.has_side_effects is False

    def test_mutable_depends_on(self):
        nc = NodeConstraints(name="a")
        nc.depends_on.add("b")
        assert "b" in nc.depends_on

    def test_instances_do_not_share_depends_on(self):
        nc1 = NodeConstraints(name="a")
        nc2 = NodeConstraints(name="b")
        nc1.depends_on.add("x")
        assert "x" not in nc2.depends_on


class TestResolvedConstraints:

    def test_defaults(self):
        rc = ResolvedConstraints(name="my_node")
        assert rc.name == "my_node"
        assert rc.force_sequential is False
        assert rc.explicit_dependencies == set()
        assert rc.has_side_effects is False
        assert rc.reasons == []
        assert rc.source == "analysis"

    def test_instances_do_not_share_reasons(self):
        rc1 = ResolvedConstraints(name="a")
        rc2 = ResolvedConstraints(name="b")
        rc1.reasons.append("test reason")
        assert rc2.reasons == []


class TestOptimizationConfig:

    def test_defaults(self):
        config = OptimizationConfig()
        assert config.force_sequential == set()
        assert config.explicit_dependencies == {}
        assert config.side_effect_nodes == set()
        assert config.disable_parallelization is False
        assert config.trust_analysis is False
        assert len(config.side_effect_keywords) > 0

    def test_default_keywords_contain_common_terms(self):
        config = OptimizationConfig()
        assert "write" in config.side_effect_keywords
        assert "delete" in config.side_effect_keywords
        assert "send" in config.side_effect_keywords

    def test_conservative_factory(self):
        config = OptimizationConfig.conservative()
        assert config.disable_parallelization is True
        assert config.trust_analysis is False

    def test_aggressive_factory(self):
        config = OptimizationConfig.aggressive()
        assert config.trust_analysis is True
        assert config.side_effect_keywords == set()

    def test_instances_do_not_share_sets(self):
        c1 = OptimizationConfig()
        c2 = OptimizationConfig()
        c1.force_sequential.add("node_x")
        assert "node_x" not in c2.force_sequential


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
