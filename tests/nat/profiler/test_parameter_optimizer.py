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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nat.data_models.config import Config
from nat.data_models.optimizable import SearchSpace
from nat.data_models.optimizer import OptimizerConfig
from nat.data_models.optimizer import OptimizerMetric
from nat.data_models.optimizer import OptimizerRunConfig
from nat.profiler.parameter_optimization.parameter_optimizer import optimize_parameters


class _FakeTrial:

    def __init__(self, trial_id: int):
        self._trial_id = trial_id
        self.user_attrs: dict[str, object] = {}

    # Optuna Trial API subset used by SearchSpace.suggest()
    def suggest_categorical(self, _name: str, choices):  # noqa: ANN001
        return choices[0]

    def suggest_int(
            self,
            name: str,  # noqa: ANN001
            low: int,
            high: int,  # noqa: ANN001
            log: bool = False,  # noqa: FBT001, ANN001
            step: float | None = None):  # noqa: ANN001
        _ = (name, high, log, step)
        return low

    def suggest_float(
            self,
            name: str,  # noqa: ANN001
            low: float,
            high: float,  # noqa: ANN001
            log: bool = False,  # noqa: FBT001, ANN001
            step: float | None = None):  # noqa: ANN001
        _ = (name, log, step)
        return (low + high) / 2.0

    def set_user_attr(self, key: str, value):  # noqa: ANN001
        self.user_attrs[key] = value


class _FakeDF:

    def __init__(self):
        # include rep_scores so the optimizer's flattening branch is skipped
        self.columns = ["rep_scores"]

    def __getitem__(self, key):  # noqa: ANN001
        raise KeyError(key)

    def __setitem__(self, key, value):  # noqa: ANN001
        # no-op for tests
        return None

    def drop(self, columns=None):  # noqa: ANN001, D401
        return self

    def to_csv(self, fh, index: bool = False):  # noqa: ANN001, FBT001
        fh.write("trial_id,params\n0,{}\n")


class _FakeStudy:

    def __init__(self, directions: list[str]):
        self.directions = directions
        self.trials: list[_FakeTrial] = []
        self.optimize_calls = 0

    def optimize(self, objective, n_trials: int):  # noqa: ANN001, D401
        for i in range(n_trials):
            trial = _FakeTrial(i)
            objective(trial)
            self.trials.append(trial)
            self.optimize_calls += 1

    def trials_dataframe(self, *args, **kwargs):  # noqa: ANN001, D401
        return _FakeDF()


def _make_optimizer_config(tmp_path: Path) -> OptimizerConfig:
    return OptimizerConfig(
        output_path=tmp_path,
        eval_metrics={
            "acc": OptimizerMetric(evaluator_name="Accuracy", direction="maximize", weight=1.0),
            "lat": OptimizerMetric(evaluator_name="Latency", direction="minimize", weight=0.5),
        },
        reps_per_param_set=2,
    )


def _make_run_config(_cfg: Config) -> OptimizerRunConfig:
    return OptimizerRunConfig(
        config_file=_cfg,  # pass instantiated model (allowed by type)
        dataset=None,
        result_json_path="$",
        endpoint=None,
        endpoint_timeout=5,
    )


def test_optimize_parameters_happy_path(tmp_path: Path):
    base_cfg = Config()
    out_dir = tmp_path / "opt"

    optimizer_config = _make_optimizer_config(out_dir)
    optimizer_config.numeric.n_trials = 2

    best_params = {"lr": 0.02, "arch": "A"}

    # Define full search space including a prompt param which must be filtered out
    full_space = {
        "lr": SearchSpace(low=0.001, high=0.1, log=False, step=None),
        "arch": SearchSpace(values=["A", "B"], high=None),
        "prompt_text": SearchSpace(is_prompt=True),
    }

    run_cfg = _make_run_config(base_cfg)

    # Prepare stubs/spies
    apply_calls: list[dict[str, object]] = []
    intermediate_cfg = Config()
    final_cfg = Config()

    def fake_apply_suggestions(_cfg: Config, suggestions: dict[str, object]) -> Config:  # noqa: ANN001
        apply_calls.append(suggestions)
        # Return distinct objects to ensure the function uses the return values
        return final_cfg if suggestions == best_params else intermediate_cfg

    def fake_create_study(directions: list[str]):  # noqa: ANN001
        # Validate directions are forwarded correctly from metrics
        assert directions == ["maximize", "minimize"]
        return _FakeStudy(directions)

    class _DummyEvalRun:

        def __init__(self, config):  # noqa: ANN001
            self.config = config

        async def run_and_evaluate(self):
            # Provide metrics by evaluator_name
            return SimpleNamespace(evaluation_results=[
                ("Accuracy", SimpleNamespace(average_score=0.8)),
                ("Latency", SimpleNamespace(average_score=0.5)),
            ])

    with patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
               side_effect=fake_apply_suggestions) as apply_mock, \
         patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
               return_value=SimpleNamespace(params=best_params)) as pick_mock, \
         patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization") as viz_mock, \
         patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
               side_effect=fake_create_study) as study_mock, \
         patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
               _DummyEvalRun) as eval_run_mock:

        tuned = optimize_parameters(base_cfg=base_cfg,
                                    full_space=full_space,
                                    optimizer_config=optimizer_config,
                                    opt_run_config=run_cfg)

        # Returned config should be what apply_suggestions returned for best_params
        assert tuned is final_cfg

        # Study created with correct directions
        study_mock.assert_called_once()

        # pick_trial used to choose final params
        pick_mock.assert_called_once()
        assert pick_mock.call_args.kwargs["mode"] == optimizer_config.multi_objective_combination_mode

        # apply_suggestions called at least once during trials and once for final params
        assert any("lr" in c and "arch" in c and "prompt_text" not in c for c in apply_calls)
        assert any(c == best_params for c in apply_calls)

        # Files should be written
        assert (out_dir / "optimized_config.yml").exists()
        assert (out_dir / "trials_dataframe_params.csv").exists()
        # Trial artifacts for each trial
        for i in range(optimizer_config.numeric.n_trials):
            assert (out_dir / f"config_numeric_trial_{i}.yml").exists()

        # Pareto visualization called with expected signature
        viz_mock.assert_called_once()
        viz_kwargs = viz_mock.call_args.kwargs
        assert viz_kwargs["data_source"].directions == ["maximize", "minimize"]
        assert viz_kwargs["metric_names"] == ["Accuracy", "Latency"]
        assert viz_kwargs["directions"] == ["maximize", "minimize"]
        assert viz_kwargs["output_dir"] == out_dir / "plots"
        assert viz_kwargs["show_plots"] is False

        # Trials should have rep_scores recorded
        study = viz_kwargs["data_source"]
        assert all("rep_scores" in t.user_attrs for t in study.trials)

    # Silence unused warnings
    assert apply_mock and pick_mock and viz_mock and eval_run_mock


def test_optimize_parameters_requires_output_path(tmp_path: Path):
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)
    optimizer_config.output_path = None
    run_cfg = _make_run_config(base_cfg)

    with pytest.raises(ValueError):
        optimize_parameters(base_cfg=base_cfg, full_space={}, optimizer_config=optimizer_config, opt_run_config=run_cfg)


def test_optimize_parameters_requires_eval_metrics(tmp_path: Path):
    base_cfg = Config()
    optimizer_config = _make_optimizer_config(tmp_path)
    optimizer_config.eval_metrics = None
    run_cfg = _make_run_config(base_cfg)

    with pytest.raises(ValueError):
        optimize_parameters(base_cfg=base_cfg, full_space={}, optimizer_config=optimizer_config, opt_run_config=run_cfg)
