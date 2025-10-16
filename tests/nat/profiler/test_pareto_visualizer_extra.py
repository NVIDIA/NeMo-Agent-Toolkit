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

from pathlib import Path

import optuna
import pandas as pd

from nat.profiler.parameter_optimization.pareto_visualizer import create_pareto_visualization


def _make_two_obj_study():
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.add_trial(optuna.trial.create_trial(values=[0.1, 0.9], params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(values=[0.2, 0.2], params={}, distributions={}))
    study.add_trial(optuna.trial.create_trial(values=[0.9, 0.1], params={}, distributions={}))
    return study


def test_create_pareto_visualization_from_study(tmp_path: Path):
    study = _make_two_obj_study()
    figs = create_pareto_visualization(
        data_source=study,
        metric_names=["m1", "m2"],
        directions=["minimize", "minimize"],
        output_dir=tmp_path,
        title_prefix="T",
        show_plots=False,
    )
    # Should include 2D scatter and other plots when 2 metrics
    assert "2d_scatter" in figs
    assert (tmp_path / "pareto_front_2d.png").exists()


def test_create_pareto_visualization_from_csv(tmp_path: Path):
    # build a small dataframe matching expected 'values_' columns
    df = pd.DataFrame({"values_0": [1.0, 0.5], "values_1": [0.5, 1.0]})
    csv = tmp_path / "trials.csv"
    df.to_csv(csv, index=False)

    figs = create_pareto_visualization(
        data_source=csv,
        metric_names=["a", "b"],
        directions=["minimize", "minimize"],
        output_dir=None,
        show_plots=False,
    )
    assert isinstance(figs, dict)
