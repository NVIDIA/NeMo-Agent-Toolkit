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
import os
import shutil
from pathlib import Path

from nat.data_models.swe_bench_model import SWEBenchInput
from nat.data_models.swe_bench_model import SWEBenchOutput
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutput

try:
    import swebench.harness.run_evaluation as swebench_eval
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
except ImportError as exc:
    raise ImportError("Please install swebench to use this evaluator") from exc

logger = logging.getLogger(__name__)


class SweBenchEvaluator:

    def __init__(self, run_id: str, max_workers: int, output_dir: Path):

        self.run_id = run_id
        self.max_workers = max_workers
        self.output_dir = output_dir

        # metadata
        self._unsupported_repos = []
        self._swe_bench_inputs = []
        self._swe_bench_outputs = []
        self._model_name_or_path = "no_llm"

    def get_model_name_from_output(self, workflow_output: list[dict]) -> str | None:
        """Fetch the `model_name_or_path` from the first entry in the list."""
        return workflow_output[0].get("model_name_or_path") if workflow_output else None

    @staticmethod
    def empty_report_dir(report_dir: Path):
        """Remove the current contents of the report directory."""
        os.makedirs(report_dir, exist_ok=True)

        # Iterate through all files in the directory and remove them
        for item in report_dir.iterdir():
            if item.is_file():  # Remove files only
                item.unlink()
            elif item.is_dir():  # Remove subdirectories and their contents
                shutil.rmtree(item)

    @staticmethod
    def move_report_and_logs(swe_bench_report_file: str, logs_dir: str, report_dir: Path):
        """ Temorary function to move the report and logs to the output directory"""
        try:
            shutil.move(swe_bench_report_file, report_dir)
        except Exception as e:
            logger.exception("Error moving report file: %s", e)

        try:
            dest_logs_dir = os.path.join(report_dir, 'logs')
            shutil.move(logs_dir, dest_logs_dir)
        except Exception as e:
            logger.exception("Error moving logs directory: %s", e)

    def is_repo_supported(self, repo: str, version: str) -> bool:
        """Check if the repo is supported by swebench"""

        try:
            _ = MAP_REPO_VERSION_TO_SPECS[repo][str(version)]
        except KeyError:
            self._unsupported_repos.append({repo, version})
            return False
        return True

    def process_eval_input(self, eval_input: EvalInput) -> tuple[Path, Path]:
        """Converts EvalInput into lists of SWEBenchInput and SWEBenchOutput models and applies filtering."""
        # Convert input_obj and output_obj JSON strings to SWEBenchInput and SWEBenchOutput models
        swebench_inputs = []
        swebench_outputs = []

        for item in eval_input.eval_input_items:
            try:
                swebench_input = SWEBenchInput.model_validate_json(item.input_obj)  # Convert input JSON to model
                swebench_input.version = str(swebench_input.version)  # Convert version to string
                swebench_inputs.append(swebench_input)

                if item.output_obj:  # Convert output JSON to model if available
                    swebench_output = SWEBenchOutput.model_validate_json(item.output_obj)
                    swebench_outputs.append(swebench_output)
                    # this is bit of a hack to match the swe_bench harness
                    self._model_name_or_path = swebench_output.model_name_or_path

            except Exception as e:
                logger.exception("Failed to parse EvalInputItem %s: %s", item.id, e)

        # Filter out repos/version not supported by SWEBench
        supported_inputs = [
            swebench for swebench in swebench_inputs if self.is_repo_supported(swebench.repo, swebench.version)
        ]

        if not supported_inputs:
            logger.exception("No supported instances; nothing to evaluate")
            return None, None

        if len(supported_inputs) < len(swebench_inputs):
            logger.warning("The following repos are not supported by SWEBench and were skipped:\n %s",
                           {s.repo
                            for s in swebench_inputs if s not in supported_inputs})

        # Write SWEBenchInput to file
        workflow_input_file = self.output_dir / "nat_workflow_input.json"
        workflow_input_file.parent.mkdir(parents=True, exist_ok=True)
        Path(workflow_input_file).write_text(json.dumps([swebench.model_dump() for swebench in supported_inputs],
                                                        indent=2),
                                             encoding="utf-8")
        logger.info("Workflow input written to %s", workflow_input_file)

        # Filter SWEBenchOutput to include only instance_ids present in SWEBenchInput
        valid_instance_ids = {swebench.instance_id for swebench in supported_inputs}
        filtered_outputs = [output for output in swebench_outputs if output.instance_id in valid_instance_ids]

        if not filtered_outputs:
            logger.error("No supported outputs; nothing to evaluate", exc_info=True)
            return None, None

        # Write SWEBenchOutput to file
        workflow_output_file = self.output_dir / "nat_workflow_output.json"
        Path(workflow_output_file).write_text(json.dumps([output.model_dump() for output in filtered_outputs],
                                                         indent=2),
                                              encoding="utf-8")
        logger.info("Workflow output written to %s", workflow_output_file)

        self._swe_bench_inputs = supported_inputs
        self._swe_bench_outputs = filtered_outputs
        return workflow_input_file, workflow_output_file

    def build_eval_output(self):
        """Builds the EvalOutput object from the SWEBenchOutput models and the average score."""
        # WIP: Build a score based on eval run logs
        for swebench_output in self._swe_bench_outputs:
            yield {"id": swebench_output.instance_id, "score": "-", "reasoning": "-"}

    @staticmethod
    def compute_score(success_cnt: int, total_cnt: int) -> float:
        if total_cnt == 0:
            return 0.0
        score = success_cnt / total_cnt
        return min(max(score, 0.0), 1.0)

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        '''Run the swebench evaluation and store the report in the output directory'''

        # Process the EvalInput
        workflow_input_file, workflow_output_file = self.process_eval_input(eval_input)
        if not workflow_input_file or not workflow_output_file:
            # nothing to evaluate
            return EvalOutput(average_score=0.0, eval_output_items=[])

        report_dir = self.output_dir / "swe_bench_reports"
        self.empty_report_dir(report_dir)

        logger.info("Starting swe_bench run %s", self.run_id)
        swebench_eval.main(dataset_name=str(workflow_input_file),
                           split="dev",
                           instance_ids=[],
                           predictions_path=str(workflow_output_file),
                           max_workers=self.max_workers,
                           force_rebuild=False,
                           cache_level="env",
                           clean=False,
                           open_file_limit=4096,
                           run_id=self.run_id,
                           timeout=1800,
                           namespace=None,
                           rewrite_reports=False,
                           modal=False,
                           instance_image_tag='latest',
                           report_dir=str(report_dir))
        logger.info("Completed swe_bench run %s", self.run_id)

        swe_bench_report_file = f"{self._model_name_or_path}.{self.run_id}.json"

        # There is a bug in swebench because of which report_dir is being ignored. Copy the report to the output dir
        self.move_report_and_logs(swe_bench_report_file=swe_bench_report_file, logs_dir="logs", report_dir=report_dir)
        logger.info("SWE_bench report and logs written to %s directory", report_dir)

        # read the swe_bench report file
        report_file = report_dir / swe_bench_report_file
        # if report file is not present, return empty EvalOutput
        avg_score = 0.0
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
                resolved_instances = report.get("resolved_instances", 0)
                total_instances = report.get("total_instances", 0)
                avg_score = self.compute_score(resolved_instances, total_instances)

        # Build the EvalOutput from self._swe_bench_outputs and avg_score
        eval_output_items = list(self.build_eval_output())
        return EvalOutput(average_score=avg_score, eval_output_items=eval_output_items)
