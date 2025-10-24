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
        self.number = trial_id  # Add number attribute for Pareto optimal identification
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
        self.columns = ["rep_scores", "number"]
        self._data = {}

    def __getitem__(self, key):  # noqa: ANN001
        if key == "number":
            # Return a fake series-like object that supports .isin()
            return _FakeSeries([0, 1])
        if key in self._data:
            return self._data[key]
        raise KeyError(key)

    def __setitem__(self, key, value):  # noqa: ANN001
        # Store values so they can be used later
        self._data[key] = value

    def drop(self, columns=None):  # noqa: ANN001, D401
        return self

    def to_csv(self, fh, index: bool = False):  # noqa: ANN001, FBT001
        fh.write("trial_id,params\n0,{}\n")


class _FakeSeries:

    def __init__(self, values):  # noqa: ANN001
        self.values = values

    def isin(self, other):  # noqa: ANN001
        # Return a fake boolean array
        return [v in other for v in self.values]


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

    @property
    def best_trials(self):  # noqa: D401
        """Return Pareto optimal trials (for multi-objective optimization)."""
        # For testing purposes, consider all trials as Pareto optimal
        return self.trials


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

    def fake_create_study(directions: list[str], sampler=None, **kwargs):  # noqa: ANN001
        # Validate directions are forwarded correctly from metrics
        assert directions == ["maximize", "minimize"]
        # This test uses default sampler (None)
        assert sampler is None
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


# Integration tests for sampler selection and grid search
class TestSamplerSelection:
    """Test sampler selection logic based on optimizer config."""

    def test_default_sampler_is_none(self, tmp_path: Path):
        """Test that default sampler (None) is passed to Optuna."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.n_trials = 1
        # Default should be None
        assert optimizer_config.numeric.sampler is None

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg
            sampler_arg = kwargs.get("sampler")
            return _FakeStudy(kwargs.get("directions", []))

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Should pass None to let Optuna choose
        assert sampler_arg is None

    def test_none_sampler_single_objective_uses_tpe(self, tmp_path: Path):
        """Test that None sampler with single objective allows Optuna to use TPE."""
        base_cfg = Config()
        optimizer_config = OptimizerConfig(
            output_path=tmp_path / "opt",
            eval_metrics={
                "acc": OptimizerMetric(evaluator_name="Accuracy", direction="maximize", weight=1.0),
            },
            reps_per_param_set=1,
        )
        optimizer_config.numeric.sampler = None
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None
        directions_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg, directions_arg
            sampler_arg = kwargs.get("sampler")
            directions_arg = kwargs.get("directions", [])
            return _FakeStudy(directions_arg)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Single objective: Optuna will use TPE with sampler=None
        assert sampler_arg is None
        assert directions_arg is not None
        assert len(directions_arg) == 1
        assert directions_arg == ["maximize"]

    def test_none_sampler_multi_objective_uses_nsga2(self, tmp_path: Path):
        """Test that None sampler with multi-objective allows Optuna to use NSGA-II."""
        base_cfg = Config()
        optimizer_config = OptimizerConfig(
            output_path=tmp_path / "opt",
            eval_metrics={
                "acc": OptimizerMetric(evaluator_name="Accuracy", direction="maximize", weight=1.0),
                "lat": OptimizerMetric(evaluator_name="Latency", direction="minimize", weight=0.5),
            },
            reps_per_param_set=1,
        )
        optimizer_config.numeric.sampler = None
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None
        directions_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg, directions_arg
            sampler_arg = kwargs.get("sampler")
            directions_arg = kwargs.get("directions", [])
            return _FakeStudy(directions_arg)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Multi-objective: Optuna will use NSGA-II with sampler=None
        assert sampler_arg is None
        assert directions_arg is not None
        assert len(directions_arg) == 2
        assert directions_arg == ["maximize", "minimize"]

    def test_grid_sampler_selected(self, tmp_path: Path):
        """Test that GridSampler is created when sampler='grid'."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "temp": SearchSpace(values=[0.1, 0.5, 0.9]),
            "top_p": SearchSpace(values=[0.8, 1.0]),
        }
        run_cfg = _make_run_config(base_cfg)

        sampler_instance = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal sampler_instance
            sampler_instance = original_grid_sampler(search_space)
            return sampler_instance

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize", "minimize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Should create a GridSampler instance
        assert isinstance(sampler_instance, optuna.samplers.GridSampler)

    def test_enum_sampler_type_with_correct_value(self, tmp_path: Path):
        """Test that sampler enum works with correct enum value."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        # Use the actual enum value
        from nat.data_models.optimizer import SamplerType
        optimizer_config.numeric.sampler = SamplerType.GRID
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_instance = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal sampler_instance
            sampler_instance = original_grid_sampler(search_space)
            return sampler_instance

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize", "minimize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        assert isinstance(sampler_instance, optuna.samplers.GridSampler)

    def test_bayesian_sampler_passes_none_to_optuna(self, tmp_path: Path):
        """Test that 'bayesian' sampler explicitly passes None to Optuna."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "bayesian"
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg
            sampler_arg = kwargs.get("sampler")
            return _FakeStudy(kwargs.get("directions", []))

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # "bayesian" should pass None to let Optuna choose (TPE or NSGA-II)
        assert sampler_arg is None

    def test_bayesian_sampler_single_objective(self, tmp_path: Path):
        """Test that 'bayesian' sampler with single objective lets Optuna use TPE."""
        base_cfg = Config()
        optimizer_config = OptimizerConfig(
            output_path=tmp_path / "opt",
            eval_metrics={
                "acc": OptimizerMetric(evaluator_name="Accuracy", direction="maximize", weight=1.0),
            },
            reps_per_param_set=1,
        )
        optimizer_config.numeric.sampler = "bayesian"
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None
        directions_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg, directions_arg
            sampler_arg = kwargs.get("sampler")
            directions_arg = kwargs.get("directions", [])
            return _FakeStudy(directions_arg)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Single objective with "bayesian": should pass None for TPE
        assert sampler_arg is None
        assert directions_arg == ["maximize"]

    def test_bayesian_sampler_multi_objective(self, tmp_path: Path):
        """Test that 'bayesian' sampler with multi-objective lets Optuna use NSGA-II."""
        base_cfg = Config()
        optimizer_config = OptimizerConfig(
            output_path=tmp_path / "opt",
            eval_metrics={
                "acc": OptimizerMetric(evaluator_name="Accuracy", direction="maximize", weight=1.0),
                "lat": OptimizerMetric(evaluator_name="Latency", direction="minimize", weight=0.5),
            },
            reps_per_param_set=1,
        )
        optimizer_config.numeric.sampler = "bayesian"
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None
        directions_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg, directions_arg
            sampler_arg = kwargs.get("sampler")
            directions_arg = kwargs.get("directions", [])
            return _FakeStudy(directions_arg)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Multi-objective with "bayesian": should pass None for NSGA-II
        assert sampler_arg is None
        assert directions_arg == ["maximize", "minimize"]

    def test_bayesian_sampler_with_enum_value(self, tmp_path: Path):
        """Test that 'bayesian' sampler works with enum value."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        # Use the actual enum value
        from nat.data_models.optimizer import SamplerType
        optimizer_config.numeric.sampler = SamplerType.BAYESIAN
        optimizer_config.numeric.n_trials = 1

        full_space = {"param": SearchSpace(values=[1, 2])}
        run_cfg = _make_run_config(base_cfg)

        sampler_arg = None

        def capture_sampler(**kwargs):
            nonlocal sampler_arg
            sampler_arg = kwargs.get("sampler")
            return _FakeStudy(kwargs.get("directions", []))

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   side_effect=capture_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # BAYESIAN enum value should pass None to Optuna
        assert sampler_arg is None


class TestGridSearchIntegration:
    """Integration tests for grid search with various parameter configurations."""

    def test_grid_search_with_multiple_categorical_params_static_pass(self, tmp_path: Path):
        """Test grid search with multiple categorical parameters."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        from nat.data_models.optimizer import SamplerType
        optimizer_config.numeric.sampler = SamplerType.GRID
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "model": SearchSpace(values=["gpt-3.5", "gpt-4"]),
            "temperature": SearchSpace(values=[0.0, 0.5, 1.0]),
        }
        run_cfg = _make_run_config(base_cfg)

        grid_space = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal grid_space
            grid_space = search_space
            return original_grid_sampler(search_space)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize", "minimize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Verify grid space was created with correct values
        assert grid_space is not None
        assert "model" in grid_space
        assert "temperature" in grid_space
        assert grid_space["model"] == ["gpt-3.5", "gpt-4"]
        assert grid_space["temperature"] == [0.0, 0.5, 1.0]

    def test_grid_search_with_multiple_categorical_params_runtime_pass(self, tmp_path: Path):
        """Test grid search with multiple categorical parameters."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "model": SearchSpace(values=["gpt-3.5", "gpt-4"]),
            "temperature": SearchSpace(values=[0.0, 0.5, 1.0]),
        }
        run_cfg = _make_run_config(base_cfg)

        grid_space = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal grid_space
            grid_space = search_space
            return original_grid_sampler(search_space)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize", "minimize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Verify grid space was created with correct values
        assert grid_space is not None
        assert "model" in grid_space
        assert "temperature" in grid_space
        assert grid_space["model"] == ["gpt-3.5", "gpt-4"]
        assert grid_space["temperature"] == [0.0, 0.5, 1.0]

    def test_grid_search_with_integer_range(self, tmp_path: Path):
        """Test grid search with integer range and step."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "batch_size": SearchSpace(low=8, high=32, step=8),
        }
        run_cfg = _make_run_config(base_cfg)

        grid_space = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal grid_space
            grid_space = search_space
            return original_grid_sampler(search_space)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        assert grid_space is not None
        assert grid_space["batch_size"] == [8, 16, 24, 32]

    def test_grid_search_with_float_range(self, tmp_path: Path):
        """Test grid search with float range and step."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "learning_rate": SearchSpace(low=0.001, high=0.01, step=0.003),
        }
        run_cfg = _make_run_config(base_cfg)

        grid_space = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal grid_space
            grid_space = search_space
            return original_grid_sampler(search_space)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["minimize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        assert grid_space is not None
        assert len(grid_space["learning_rate"]) == 4  # 0.001, 0.004, 0.007, 0.01
        assert grid_space["learning_rate"][0] == pytest.approx(0.001)
        assert grid_space["learning_rate"][-1] == pytest.approx(0.01)

    def test_grid_search_mixed_categorical_and_ranges(self, tmp_path: Path):
        """Test grid search with mix of categorical values and numeric ranges."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "temperature": SearchSpace(values=[0.0, 0.5, 1.0]),  # Explicit values
            "max_tokens": SearchSpace(low=100, high=500, step=200),  # Integer range
            "model": SearchSpace(values=["fast", "accurate"]),  # Categorical
        }
        run_cfg = _make_run_config(base_cfg)

        grid_space = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal grid_space
            grid_space = search_space
            return original_grid_sampler(search_space)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize", "minimize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Verify all parameter types are handled correctly
        assert grid_space is not None
        assert grid_space["temperature"] == [0.0, 0.5, 1.0]
        assert grid_space["max_tokens"] == [100, 300, 500]
        assert grid_space["model"] == ["fast", "accurate"]

    def test_grid_search_filters_prompt_parameters(self, tmp_path: Path):
        """Test that prompt parameters are filtered out for grid search."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        full_space = {
            "temperature": SearchSpace(values=[0.5, 1.0]),
            "system_prompt": SearchSpace(is_prompt=True, prompt="You are helpful"),
            "max_tokens": SearchSpace(values=[100, 200]),
        }
        run_cfg = _make_run_config(base_cfg)

        grid_space = None

        # Save original GridSampler before patching
        import optuna
        original_grid_sampler = optuna.samplers.GridSampler

        def capture_grid_sampler(search_space):
            nonlocal grid_space
            grid_space = search_space
            return original_grid_sampler(search_space)

        class _DummyEvalRun:

            def __init__(self, config):  # noqa: ANN001
                self.config = config

            async def run_and_evaluate(self):
                return SimpleNamespace(evaluation_results=[
                    ("Accuracy", SimpleNamespace(average_score=0.8)),
                    ("Latency", SimpleNamespace(average_score=0.5)),
                ])

        with patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.samplers.GridSampler",
                   side_effect=capture_grid_sampler), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.optuna.create_study",
                   return_value=_FakeStudy(["maximize"])), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun",
                   _DummyEvalRun), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                   return_value=base_cfg), \
             patch("nat.profiler.parameter_optimization.parameter_optimizer.pick_trial",
                   return_value=SimpleNamespace(params={})), \
             patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

            optimize_parameters(base_cfg=base_cfg,
                                full_space=full_space,
                                optimizer_config=optimizer_config,
                                opt_run_config=run_cfg)

        # Prompt parameters should be filtered out
        assert grid_space is not None
        assert "temperature" in grid_space
        assert "max_tokens" in grid_space
        assert "system_prompt" not in grid_space

    def test_grid_search_range_without_step_raises_error(self, tmp_path: Path):
        """Test that ranges without step raise clear error for grid search."""
        base_cfg = Config()
        optimizer_config = _make_optimizer_config(tmp_path / "opt")
        optimizer_config.numeric.sampler = "grid"
        optimizer_config.numeric.n_trials = 1

        # Missing step for range
        full_space = {
            "temperature": SearchSpace(low=0.0, high=1.0),  # No step!
        }
        run_cfg = _make_run_config(base_cfg)

        with pytest.raises(ValueError, match="requires 'step' to be specified"):
            with patch("nat.profiler.parameter_optimization.parameter_optimizer.EvaluationRun"), \
                 patch("nat.profiler.parameter_optimization.parameter_optimizer.apply_suggestions",
                       return_value=base_cfg), \
                 patch("nat.profiler.parameter_optimization.pareto_visualizer.create_pareto_visualization"):

                optimize_parameters(base_cfg=base_cfg,
                                    full_space=full_space,
                                    optimizer_config=optimizer_config,
                                    opt_run_config=run_cfg)
