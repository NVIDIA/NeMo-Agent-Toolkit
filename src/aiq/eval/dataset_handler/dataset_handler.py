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

import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from aiq.data_models.dataset_handler import EvalDatasetConfig
from aiq.data_models.dataset_handler import EvalDatasetCsvConfig
from aiq.data_models.dataset_handler import EvalDatasetJsonConfig
from aiq.data_models.dataset_handler import EvalDatasetJsonlConfig
from aiq.data_models.dataset_handler import EvalDatasetParquetConfig
from aiq.data_models.dataset_handler import EvalDatasetXlsConfig
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepType
from aiq.eval.dataset_handler.dataset_downloader import DatasetDownloader
from aiq.eval.dataset_handler.dataset_filter import DatasetFilter
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


class DatasetHandler:
    """
    Read the datasets and pre-process (apply filters, deduplicate etc.) before turning them into EvalInput objects.
    One DatasetHandler object is needed for each dataset to be evaluated.
    """

    def __init__(self, dataset_config: EvalDatasetConfig, reps: int):
        from aiq.eval.intermediate_step_adapter import IntermediateStepAdapter

        self.dataset_config = dataset_config
        self.dataset_filter = DatasetFilter(dataset_config.filter)
        self.reps = reps
        # Helpers
        self.intermediate_step_adapter = IntermediateStepAdapter()

    def is_structured_input(self) -> bool:
        '''Check if the input is structured or unstructured'''
        return not self.dataset_config.structure.disable

    @property
    def id_key(self) -> str:
        return self.dataset_config.id_key

    @property
    def question_key(self) -> str:
        return self.dataset_config.structure.question_key

    @property
    def answer_key(self) -> str:
        return self.dataset_config.structure.answer_key

    @property
    def generated_answer_key(self) -> str:
        return self.dataset_config.structure.generated_answer_key

    @property
    def trajectory_key(self) -> str:
        return self.dataset_config.structure.trajectory_key

    @property
    def expected_trajectory_key(self) -> str:
        return self.dataset_config.structure.expected_trajectory_key

    def get_eval_input_from_df(self, input_df: pd.DataFrame) -> EvalInput:

        def create_eval_item(row: pd.Series, structured: bool) -> EvalInputItem:
            """Helper function to create EvalInputItem."""
            return EvalInputItem(
                id=row.get(self.id_key, ""),
                input_obj=row.to_json() if not structured else row.get(self.question_key, ""),
                expected_output_obj=row.get(self.answer_key, "") if structured else "",
                output_obj=row.get(self.generated_answer_key, "") if structured else "",
                trajectory=row.get(self.trajectory_key, []) if structured else [],
                expected_trajectory=row.get(self.expected_trajectory_key, []) if structured else [],
            )

        # if input dataframe is empty return an empty list
        if input_df.empty:
            return EvalInput(eval_input_items=[])

        structured = self.is_structured_input()
        if structured:
            # For structured input, question is mandatory. Ignore rows with missing or empty questions
            input_df = input_df[input_df[self.question_key].notnull() & input_df[self.question_key].str.strip().ne("")]
        eval_input_items = [create_eval_item(row, structured) for _, row in input_df.iterrows()]

        return EvalInput(eval_input_items=eval_input_items)

    def setup_reps(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """replicate the rows and update the id to id_key + "_rep" + rep_number"""
        # Replicate the rows
        input_df = pd.concat([input_df] * self.reps, ignore_index=True)
        # Compute repetition index
        rep_index = input_df.groupby(self.dataset_config.id_key).cumcount().astype(str)
        # Convert id_key to string (id can be integer) if needed and update IDs
        input_df[self.dataset_config.id_key] = input_df[self.dataset_config.id_key].astype(str) + "_rep" + rep_index
        # Ensure unique ID values after modification
        input_df.drop_duplicates(subset=[self.dataset_config.id_key], inplace=True)

        return input_df

    @staticmethod
    def run_custom_script(dataset_config: EvalDatasetConfig) -> Path | None:
        """
        Run a custom script to transform the dataset.
        Passes the original dataset (--input_path) and (--output_path/--output_format) as
        arguments along with the kwargs provided in the dataset config. The custom script is
        expected to write the new dataset to the output_path with the output_format.

        If the custom script fails an exception is raised
        """
        if not dataset_config.custom_script:
            return None

        script_config = dataset_config.custom_script
        script_path = script_config.script

        if not script_path.exists():
            raise FileNotFoundError(f"Custom script {script_path} does not exist.")

        output_path = script_config.output_path
        # make the output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        args = [
            sys.executable,
            str(script_path),
            "--input_path",
            str(dataset_config.file_path),
            "--input_format",
            str(dataset_config.type),
            "--output_path",
            str(output_path),
            "--output_format",
            str(script_config.output_format),
        ]

        if script_config.kwargs:
            for key, value in script_config.kwargs.items():
                args.extend([f"--{key}", str(value)])

        display_args = " ".join(f'"{arg}"' if " " in arg else arg for arg in args[1:])
        logger.info("Running custom script: %s %s", script_path, display_args)

        try:
            output = subprocess.run(args, check=True, text=True, capture_output=True)
            logger.info("Custom script output: %s", output.stdout)
            if not output_path.exists():
                logger.warning("Script completed but did not write to expected output path: %s", output_path)
                return None
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error("Custom script failed: %s", e)
            raise

    @staticmethod
    def _get_dataset_config_class_for_format(output_format: str):
        """Get the appropriate dataset config class based on the output format."""
        format_to_config_class = {
            "json": EvalDatasetJsonConfig,
            "jsonl": EvalDatasetJsonlConfig,
            "csv": EvalDatasetCsvConfig,
            "parquet": EvalDatasetParquetConfig,
            "xls": EvalDatasetXlsConfig,
        }
        if output_format not in format_to_config_class:
            raise ValueError(
                f"Unsupported output format: {output_format}. Supported formats: {list(format_to_config_class.keys())}")
        return format_to_config_class[output_format]

    def get_eval_input_from_dataset(self, dataset: str) -> EvalInput:
        # read the dataset and convert it to EvalInput

        # if a dataset file has been provided in the command line, use that
        dataset_config = EvalDatasetJsonConfig(file_path=dataset) if dataset else self.dataset_config

        # Download the dataset if it is remote
        downloader = DatasetDownloader(dataset_config=dataset_config)
        downloader.download_dataset()

        # Run a custom script to transform the dataset
        new_file_path = self.run_custom_script(dataset_config)

        # Parse the dataset into a DataFrame
        if new_file_path:
            # parser for the new file is dependent on the output format
            # create a new dataset config with the new file path and format
            output_format = dataset_config.custom_script.output_format
            if not output_format:
                raise ValueError("Output format is not provided in the custom script")
            dataset_config_class = self._get_dataset_config_class_for_format(output_format)
            dataset_config = dataset_config_class(file_path=new_file_path)

        parser, kwargs = dataset_config.parser()
        input_df = parser(dataset_config.file_path, **kwargs)

        # Apply filters and deduplicate
        input_df = self.dataset_filter.apply_filters(input_df)
        input_df.drop_duplicates(subset=[self.dataset_config.id_key], inplace=True)

        # If more than one repetition is needed, replicate the rows
        if self.reps > 1:
            input_df = self.setup_reps(input_df)

        # Convert the DataFrame to a list of EvalInput objects
        return self.get_eval_input_from_df(input_df)

    def filter_intermediate_steps(self,
                                  intermediate_steps: list[IntermediateStep],
                                  event_filter: list[IntermediateStepType] = None) -> list[dict]:
        """
        Filter out the intermediate steps that are not relevant for evaluation.
        The output is written with with the intention of re-running the evaluation using the original config file.
        """
        if event_filter is None:
            event_filter = self.intermediate_step_adapter.DEFAULT_EVENT_FILTER
        filtered_steps = self.intermediate_step_adapter.filter_intermediate_steps(intermediate_steps, event_filter)
        return self.intermediate_step_adapter.serialize_intermediate_steps(filtered_steps)

    def publish_eval_input(self, eval_input, workflow_output_step_filter: list[IntermediateStepType] = None) -> str:
        """
        Convert the EvalInput object to a JSON output for storing in a file. Use the orginal keys to
        allow re-running evaluation using the orignal config file and '--skip_workflow' option.
        """

        indent = 2
        if self.is_structured_input():
            # Extract structured data from EvalInputItems
            data = [{
                self.id_key: item.id,
                self.question_key: item.input_obj,
                self.answer_key: item.expected_output_obj,
                self.generated_answer_key: item.output_obj,
                self.trajectory_key: self.filter_intermediate_steps(item.trajectory, workflow_output_step_filter),
                self.expected_trajectory_key: self.filter_intermediate_steps(item.expected_trajectory),
            } for item in eval_input.eval_input_items]
        else:
            # Unstructured case: return only raw output objects as a JSON array
            data = [json.loads(item.output_obj) for item in eval_input.eval_input_items]

        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
