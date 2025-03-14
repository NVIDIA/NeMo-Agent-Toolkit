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

import pandas as pd
import pytest

from aiq.data_models.dataset_handler import EvalFilterConfig
from aiq.data_models.dataset_handler import EvalFilterEntryConfig
from aiq.eval.dataset_handler.dataset_filter import DatasetFilter

# pylint: disable=redefined-outer-name


@pytest.fixture
def sample_df():
    """
    Fixture for a sample DataFrame.
    repos: iproute2, frr, vxlan
    """
    return pd.DataFrame([
        {
            "instance_id": "iproute2_101", "repo": "iproute2", "version": 1, "problem:": "incorrect output"
        },
        {
            "instance_id": "frr_101", "repo": "frr", "version": 2, "problem:": "NULL ptr access"
        },
        {
            "instance_id": "vxlan_101", "repo": "vxlan", "version": 1, "problem:": "bridge driver inaccessibility"
        },
        {
            "instance_id": "iproute2_99", "repo": "iproute2", "version": 4, "problem:": "memory leak"
        },
        {
            "instance_id": "vxlan_102", "repo": "vxlan", "version": 2, "problem": "kernel panic"
        },
    ])


@pytest.fixture
def allowlist_filter():
    """Fixture for repo-based allowlist filter config."""
    return EvalFilterConfig(
        allowlist=EvalFilterEntryConfig(field={"repo": ["iproute2"]}),  # Keep only repo "iproute2"
        denylist=None)


@pytest.fixture
def denylist_filter():
    """Fixture for a repo-based denylist filter."""
    return EvalFilterConfig(
        allowlist=None,
        denylist=EvalFilterEntryConfig(field={"repo": ["vxlan"]})  # Remove rows where repo is "vxlan"
    )


@pytest.fixture
def combined_filter():
    """
    Fixture for a combined allowlist & denylist filter config.
    This filters on the repo and instance_id columns.
    """
    return EvalFilterConfig(
        allowlist=EvalFilterEntryConfig(field={"repo": ["iproute2", "vxlan"]}),  # Keep repos "iproute2" and "vxlan"
        denylist=EvalFilterEntryConfig(field={"instance_id": ["iproute2_99"]})  # Remove one specific instance
    )


def test_apply_filters_allowlist(sample_df, allowlist_filter):
    """Test that the allowlist filter correctly keeps only the specified repo."""
    dataset_filter = DatasetFilter(allowlist_filter)
    filtered_df = dataset_filter.apply_filters(sample_df)

    # Check that only the "iproute2" rows remain
    assert len(filtered_df) == 2, "Only two rows should remain"
    assert set(filtered_df["repo"]) == {"iproute2"}, "Only repo iproute2 should be present"


def test_apply_filters_denylist(sample_df, denylist_filter):
    """Test that the denylist filter removes the specified repo."""
    dataset_filter = DatasetFilter(denylist_filter)
    filtered_df = dataset_filter.apply_filters(sample_df)

    assert len(filtered_df) == 3, "Three rows should remain after removing repo 'vxlan'"
    assert "vxlan" not in filtered_df["repo"].values, "Repo 'vxlan' should be removed"


def test_apply_filters_combined(sample_df, combined_filter):
    """Test that the combined allowlist & denylist filter correctly applies both."""
    dataset_filter = DatasetFilter(combined_filter)
    filtered_df = dataset_filter.apply_filters(sample_df)

    assert len(filtered_df) == 3, "Only three rows should remain"
    assert "iproute2_99" not in filtered_df["instance_id"].values, "Instance 'iproute2_99' should be removed"
    assert set(filtered_df["repo"]) == {"iproute2", "vxlan"}, "Only repo 'iproute2' and 'vxlan' should remain"
