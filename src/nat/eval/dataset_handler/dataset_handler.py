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
import math

import pandas as pd

from nat.data_models.dataset_handler import EvalDatasetConfig
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepType
from nat.eval.dataset_handler.dataset_downloader import DatasetDownloader
from nat.eval.dataset_handler.dataset_filter import DatasetFilter
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem


class DatasetHandler:
    """
    Read the datasets and pre-process (apply filters, deduplicate etc.) before turning them into EvalInput objects.
    One DatasetHandler object is needed for each dataset to be evaluated.
    """

    def __init__(self,
                 dataset_config: EvalDatasetConfig,
                 reps: int,
                 concurrency: int,
                 num_passes: int | None = None,
                 adjust_dataset_size: bool = False):
        from nat.eval.intermediate_step_adapter import IntermediateStepAdapter

        self.dataset_config = dataset_config
        self.dataset_filter = DatasetFilter(dataset_config.filter)
        self.reps = reps

        # number of passes at specific concurrency
        self.concurrency = concurrency
        self.num_passes = num_passes
        self.adjust_dataset_size = adjust_dataset_size

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
                full_dataset_entry=row.to_dict(),
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

    def adjust_dataset(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust the dataset so its length is a multiple of concurrency.

        If num_passes > 0:
            dataset size is adjusted to concurrency * num_passes
        else:
            dataset size is adjusted to the largest multiple of concurrency
            that is less than or equal to the current dataset size
        """
        if self.concurrency <= 0:
            raise ValueError("Concurrency must be > 0")

        if self.num_passes < 0:
            raise ValueError("num_passes must be >= 0")

        original_size = input_df.shape[0]

        # Calculate target size
        if self.num_passes > 0:
            # When num_passes is specified, always use concurrency * num_passes
            # This respects the user's intent for exact number of passes
            target_size = self.concurrency * self.num_passes
        else:
            # When num_passes = 0, use the largest multiple of concurrency <= original_size
            # If original_size < concurrency, we need at least concurrency rows
            if original_size >= self.concurrency:
                target_size = (original_size // self.concurrency) * self.concurrency
            else:
                target_size = self.concurrency

        if target_size == 0:
            raise ValueError("Input dataset too small for even one batch at given concurrency.")

        id_col = self.dataset_config.id_key

        # If we need more rows than we have, replicate the dataset
        if original_size < target_size:
            # Clean existing _rep suffix if present
            input_df[id_col] = input_df[id_col].astype(str).str.replace(r"_rep\d+$", "", regex=True)

            # Calculate how many complete copies we need
            copies_needed = math.ceil(target_size / original_size)

            # Create the replicated dataframe
            replicated_dfs = []
            for i in range(copies_needed):
                df_copy = input_df.copy()
                if i > 0:  # Add suffix to all but the first copy
                    df_copy[id_col] = df_copy[id_col].astype(str) + f"_rep{i}"
                replicated_dfs.append(df_copy)

            input_df = pd.concat(replicated_dfs, ignore_index=True)

        # Return exactly the target size
        return input_df.head(target_size)

    def get_eval_input_from_dataset(self, dataset: str) -> EvalInput:
        # read the dataset and convert it to EvalInput

        # if a dataset file has been provided in the command line, use that
        dataset_config = EvalDatasetJsonConfig(file_path=dataset) if dataset else self.dataset_config

        # Download the dataset if it is remote
        downloader = DatasetDownloader(dataset_config=dataset_config)
        downloader.download_dataset()

        parser, kwargs = dataset_config.parser()
        # Parse the dataset into a DataFrame
        input_df = parser(dataset_config.file_path, **kwargs)

        # Apply filters and deduplicate
        input_df = self.dataset_filter.apply_filters(input_df)
        input_df.drop_duplicates(subset=[self.dataset_config.id_key], inplace=True)

        if self.reps > 1 and self.adjust_dataset_size:
            raise ValueError("reps and adjust_dataset_size are mutually exclusive")

        # If more than one repetition is needed, replicate the rows
        if self.reps > 1:
            input_df = self.setup_reps(input_df)
        elif self.adjust_dataset_size:
            input_df = self.adjust_dataset(input_df)

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

        def parse_if_json_string(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            if hasattr(value, "model_dump"):
                return value.model_dump()
            return value

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
            data = [parse_if_json_string(item.output_obj) for item in eval_input.eval_input_items]

        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
