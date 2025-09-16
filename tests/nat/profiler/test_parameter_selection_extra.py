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

import optuna
from optuna.study import StudyDirection

from nat.profiler.parameter_optimization.parameter_selection import pick_trial


def _make_study_with_trials(values_list):  # noqa: ANN001
    study = optuna.create_study(directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE])
    for vals in values_list:
        t = optuna.trial.create_trial(values=list(vals), params={}, distributions={})
        study.add_trial(t)
    return study


def test_pick_trial_sum_and_chebyshev_selects_center_point():
    # three Pareto-optimal points: none dominates the others
    vals = [(0.1, 0.9), (0.2, 0.2), (0.9, 0.1)]
    study = _make_study_with_trials(vals)

    # sum should favor the balanced point (0.2, 0.2)
    trial_sum = pick_trial(study, mode="sum")
    assert tuple(trial_sum.values) == (0.2, 0.2)

    # chebyshev should also favor the balanced point
    trial_cheb = pick_trial(study, mode="chebyshev")
    assert tuple(trial_cheb.values) == (0.2, 0.2)


def test_pick_trial_weights_mismatch_raises():
    vals = [(0.1, 0.9), (0.2, 0.2), (0.9, 0.1)]
    study = _make_study_with_trials(vals)

    try:
        pick_trial(study, mode="sum", weights=[1.0])
        assert False, "Expected ValueError for weights length"
    except ValueError:
        pass


def test_pick_trial_unknown_mode_raises():
    vals = [(0.1, 0.9), (0.2, 0.2)]
    study = _make_study_with_trials(vals)

    try:
        pick_trial(study, mode="unknown_mode")
        assert False, "Expected ValueError for unknown mode"
    except ValueError:
        pass


def test_pick_trial_empty_front_raises():
    study = optuna.create_study(directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE])
    try:
        pick_trial(study, mode="sum")
        assert False, "Expected ValueError for empty Pareto front"
    except ValueError:
        pass
