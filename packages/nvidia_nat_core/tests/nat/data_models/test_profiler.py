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

from nat.data_models.profiler import PredictionTrieConfig


def test_prediction_trie_config_defaults():
    config = PredictionTrieConfig()
    assert config.enable is False
    assert config.output_filename == "prediction_trie.json"
    assert config.auto_sensitivity is True
    assert config.sensitivity_scale == 5
    assert config.w_critical == 0.5
    assert config.w_fanout == 0.3
    assert config.w_position == 0.2


def test_prediction_trie_config_custom_weights():
    config = PredictionTrieConfig(
        auto_sensitivity=True,
        sensitivity_scale=10,
        w_critical=0.6,
        w_fanout=0.2,
        w_position=0.2,
    )
    assert config.sensitivity_scale == 10
    assert config.w_critical == 0.6


def test_prediction_trie_config_auto_sensitivity_disabled():
    config = PredictionTrieConfig(auto_sensitivity=False)
    assert config.auto_sensitivity is False
