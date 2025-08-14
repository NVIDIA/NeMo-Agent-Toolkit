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

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class OptimizerMetric(BaseModel):
    """
    Parameters used by the aiq workflow optimizer.
    """
    evaluator_name: str = Field(description="Name of the metric to optimize.")
    direction: str = Field(description="Direction of the optimization. Can be 'maximize' or 'minimize'.")
    weight: float = Field(description="Weight of the metric in the optimization process.", default=1.0)


class OptimizerConfig(BaseModel):
    """
    Parameters used by the aiq workflow optimizer.
    """
    output_path: Path | None = Field(
        default=None,
        description="Path to the output directory where the results will be saved.",
    )

    eval_metrics: dict[str, OptimizerMetric] | None = Field(description="List of evaluation metrics to optimize.",
                                                            default=None)

    n_trials_numeric: int = Field(description="Number of trials for the optimization.", default=20)

    n_trials_prompt: int = Field(description="Number of trials for the prompt optimization.", default=20)

    reps_per_param_set: int = Field(default=3,
                                    description="Number of repetitions per parameter set for the optimization.")

    target: float | None = Field(description="Target value for the optimization. "
                                 "If set, the optimization will stop when this value is reached.",
                                 default=None)

    do_prompt_optimization: bool = Field(
        description="Flag to indicate if prompt optimization should be performed.",
        default=False,
    )

    do_numeric_optimization: bool = Field(
        description="Flag to indicate if numeric optimization should be performed.",
        default=True,
    )

    prompt_optimization_function: str | None = Field(
        default=None,
        description="Name of the function to use for prompt evaluation.",
    )

    trajectory_eval_metric_name: str | None = Field(default=None,
                                                    description="Name of the trajectory evaluation metric to use.")

    num_feedback: int = Field(default=3, description="Number of feedbacks to use for the optimization.")

    multi_objective_combination_mode: str = Field(
        description="Method to combine multiple objectives into a single score.",
        default="harmonic",
    )


class OptimizerRunConfig(BaseModel):
    """
    Parameters used for an Optimizer R=run
    """
    # Eval parameters

    config_file: Path | BaseModel  # allow for instantiated configs to be passed in
    dataset: str | Path | None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    endpoint: str | None = None  # only used when running the workflow remotely
    endpoint_timeout: int = 300
    override: tuple[tuple[str, str], ...] = ()
