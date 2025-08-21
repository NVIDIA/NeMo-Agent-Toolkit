# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from tqdm import tqdm

from nat.data_models.evaluate import EvalConfig
from nat.data_models.evaluate import JobEvictionPolicy
from nat.eval.config import EvaluationRunConfig
from nat.eval.config import EvaluationRunOutput
from nat.eval.dataset_handler.dataset_handler import DatasetHandler
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.usage_stats import UsageStats
from nat.eval.usage_stats import UsageStatsItem
from nat.eval.usage_stats import UsageStatsLLM
from nat.eval.utils.output_uploader import OutputUploader
from nat.eval.utils.weave_eval import WeaveEvaluationIntegration
from nat.profiler.data_models import ProfilerResults
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class EvaluationRun:  # pylint: disable=too-many-public-methods
    """
    Instantiated for each evaluation run and used to store data for that single run.

    .. warning::
        **Experimental Feature**: The Evaluation API is experimental and may change in future releases.
        Future versions may introduce breaking changes without notice.
    """

    def __init__(self, config: EvaluationRunConfig):
        """
        Initialize an EvaluationRun with configuration.
        """
        from nat.eval.intermediate_step_adapter import IntermediateStepAdapter

        # Run-specific configuration
        self.config: EvaluationRunConfig = config
        self.eval_config: EvalConfig | None = None

        # Helpers
        self.intermediate_step_adapter: IntermediateStepAdapter = IntermediateStepAdapter()

        # Create evaluation trace context
        try:
            from nat.eval.utils.eval_trace_ctx import WeaveEvalTraceContext
            self.eval_trace_context = WeaveEvalTraceContext()
        except Exception:
            from nat.eval.utils.eval_trace_ctx import EvalTraceContext
            self.eval_trace_context = EvalTraceContext()

        self.weave_eval: WeaveEvaluationIntegration = WeaveEvaluationIntegration(self.eval_trace_context)
        # Metadata
        self.eval_input: EvalInput | None = None
        self.workflow_interrupted: bool = False

        # evaluation_results is list of tuples (evaluator_name, EvalOutput)
        self.evaluation_results: list[tuple[str, EvalOutput]] = []

        # usage stats
        self.usage_stats: UsageStats = UsageStats()

        # workflow output file
        self.workflow_output_file: Path | None = None

        # evaluation output files
        self.evaluator_output_files: list[Path] = []

    def _compute_usage_stats(self, item: EvalInputItem):
        """Compute usage stats for a single item using the intermediate steps"""
        # get the prompt and completion tokens from the intermediate steps
        from nat.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor
        steps = [IntermediatePropertyAdaptor.from_intermediate_step(step) for step in item.trajectory]
        usage_stats_per_llm = {}
        total_tokens = 0
        for step in steps:
            if step.event_type == "LLM_END":
                llm_name = step.llm_name
                if llm_name not in usage_stats_per_llm:
                    usage_stats_per_llm[llm_name] = UsageStatsLLM()
                usage_stats_per_llm[llm_name].prompt_tokens += step.token_usage.prompt_tokens
                usage_stats_per_llm[llm_name].completion_tokens += step.token_usage.completion_tokens
                usage_stats_per_llm[llm_name].total_tokens += step.token_usage.total_tokens
                total_tokens += step.token_usage.total_tokens

        # find min and max event timestamps
        if item.trajectory:
            min_timestamp = min(step.event_timestamp for step in item.trajectory)
            max_timestamp = max(step.event_timestamp for step in item.trajectory)
            runtime = max_timestamp - min_timestamp
        else:
            min_timestamp = 0.0
            max_timestamp = 0.0
            runtime = 0.0

        # find llm latency by calculating p95 of all llm calls
        llm_latencies = []
        previous_llm_start_time = None
        for step in steps:
            if step.event_type == "LLM_START":
                previous_llm_start_time = step.event_timestamp
            elif step.event_type == "LLM_END" and previous_llm_start_time is not None:
                llm_latencies.append(step.event_timestamp - previous_llm_start_time)
                previous_llm_start_time = None

        # Calculate p95 LLM latency (or 0 if no LLM calls)
        if llm_latencies:
            import numpy as np
            llm_latency = float(np.percentile(llm_latencies, 95))
        else:
            llm_latency = 0.0

        # add the usage stats to the usage stats dict
        self.usage_stats.usage_stats_items[item.id] = UsageStatsItem(usage_stats_per_llm=usage_stats_per_llm,
                                                                     runtime=runtime,
                                                                     total_tokens=total_tokens,
                                                                     min_timestamp=min_timestamp,
                                                                     max_timestamp=max_timestamp,
                                                                     llm_latency=llm_latency)
        return self.usage_stats.usage_stats_items[item.id]

    async def run_workflow_local(self, session_manager: SessionManager):
        '''
        Launch the workflow with the specified questions and extract the output using the jsonpath
        '''
        # import function level dependencies
        from jsonpath_ng import parse

        from nat.eval.runtime_event_subscriber import pull_intermediate

        # Run the workflow
        jsonpath_expr = parse(self.config.result_json_path)
        stop_event = asyncio.Event()

        async def run_one(item: EvalInputItem):
            if stop_event.is_set():
                return "", []

            async with session_manager.run(item.input_obj) as runner:
                if not session_manager.workflow.has_single_output:
                    # raise an error if the workflow has multiple outputs
                    raise NotImplementedError("Multiple outputs are not supported")

                runner_result = None
                intermediate_future = None

                try:

                    # Start usage stats and intermediate steps collection in parallel
                    intermediate_future = pull_intermediate()
                    runner_result = runner.result()
                    base_output = await runner_result
                    intermediate_steps = await intermediate_future
                except NotImplementedError as e:
                    # raise original error
                    raise e
                except Exception as e:
                    logger.exception("Failed to run the workflow: %s", e, exc_info=True)
                    # stop processing if a workflow error occurs
                    self.workflow_interrupted = True

                    # Cancel any coroutines that are still running, avoiding a warning about unawaited coroutines
                    # (typically one of these two is what raised the exception and the other is still running)
                    for coro in (runner_result, intermediate_future):
                        if coro is not None:
                            asyncio.ensure_future(coro).cancel()

                    stop_event.set()
                    return

                try:
                    base_output = runner.convert(base_output, to_type=str)
                except ValueError:
                    pass

                # if base_output is a pydantic model dump it to json
                if isinstance(base_output, BaseModel):
                    output = base_output.model_dump_json(indent=2)
                else:
                    m = jsonpath_expr.find(base_output)
                    if (not m):
                        raise RuntimeError(f"Failed to extract output using jsonpath: {self.config.result_json_path}")
                    if (len(m) > 1):
                        logger.warning("Multiple matches found for jsonpath at row '%s'. Matches: %s. Using the first",
                                       base_output,
                                       m)
                    output = m[0].value

                item.output_obj = output
                item.trajectory = self.intermediate_step_adapter.validate_intermediate_steps(intermediate_steps)
                usage_stats_item = self._compute_usage_stats(item)

                self.weave_eval.log_prediction(item, output)
                await self.weave_eval.log_usage_stats(item, usage_stats_item)

        async def wrapped_run(item: EvalInputItem) -> None:
            await run_one(item)
            pbar.update(1)

        # if self.config.skip_complete is set skip eval_input_items with a non-empty output_obj
        if self.config.skip_completed_entries:
            eval_input_items = [item for item in self.eval_input.eval_input_items if not item.output_obj]
            if not eval_input_items:
                logger.warning("All items have a non-empty output. Skipping workflow pass altogether.")
                return
        else:
            eval_input_items = self.eval_input.eval_input_items
        pbar = tqdm(total=len(eval_input_items), desc="Running workflow")
        await asyncio.gather(*[wrapped_run(item) for item in eval_input_items])
        pbar.close()

    async def run_workflow_remote(self):
        from nat.eval.remote_workflow import EvaluationRemoteWorkflowHandler
        handler = EvaluationRemoteWorkflowHandler(self.config, self.eval_config.general.max_concurrency)
        await handler.run_workflow_remote(self.eval_input)
        for item in self.eval_input.eval_input_items:
            usage_stats_item = self._compute_usage_stats(item)
            self.weave_eval.log_prediction(item, item.output_obj)
            await self.weave_eval.log_usage_stats(item, usage_stats_item)

    async def profile_workflow(self) -> ProfilerResults:
        """
        Profile a dataset
        """

        if not self.eval_config.general.profiler:
            logger.info("Profiler is not enabled. Skipping profiling.")
            return ProfilerResults()

        from nat.profiler.profile_runner import ProfilerRunner

        all_stats = []
        for input_item in self.eval_input.eval_input_items:
            all_stats.append(input_item.trajectory)

        profiler_runner = ProfilerRunner(self.eval_config.general.profiler,
                                         self.eval_config.general.output_dir,
                                         write_output=self.config.write_output)

        return await profiler_runner.run(all_stats)

    def cleanup_output_directory(self):
        '''Remove contents of the output directory if it exists'''
        output_config = self.eval_config.general.output
        output_dir = output_config.dir

        if not (output_config and output_dir.exists()):
            return

        # If cleanup is true, remove the entire directory and we are done
        if output_config.cleanup:
            logger.info("Cleaning up entire output directory: %s", output_config.dir)
            shutil.rmtree(output_config.dir)
            return

        if output_config.job_management.max_jobs == 0:
            # No eviction policy
            return

        base_dir = output_dir / "jobs"
        if not base_dir.exists():
            return

        # Get all subdirectories, which represent individual job runs
        job_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if len(job_dirs) <= output_config.job_management.max_jobs:
            return

        # Determine sort key based on eviction_policy, defaulting to creation time
        if output_config.job_management.eviction_policy == JobEvictionPolicy.TIME_MODIFIED:

            def sort_key(x):
                return x.stat().st_mtime

            logger.info("Using last modified time for job eviction policy.")
        else:

            def sort_key(x):
                return x.stat().st_ctime

            logger.info("Using creation time for job eviction policy.")

        # Sort directories (oldest first)
        job_dirs.sort(key=sort_key)
        num_to_delete = len(job_dirs) - output_config.job_management.max_jobs

        logger.info("Found %d jobs, exceeding limit of %d. Removing %d oldest jobs.",
                    len(job_dirs),
                    output_config.job_management.max_jobs,
                    num_to_delete)

        for dir_to_delete in job_dirs[:num_to_delete]:
            try:
                logger.info("Deleting old job directory: %s", dir_to_delete)
                shutil.rmtree(dir_to_delete)
            except Exception as e:
                logger.exception("Failed to delete old job directory: %s: %s", dir_to_delete, e, exc_info=True)

    def write_output(self, dataset_handler: DatasetHandler, profiler_results: ProfilerResults):  # pylint: disable=unused-argument  # noqa: E501
        workflow_output_file = self.eval_config.general.output_dir / "workflow_output.json"
        workflow_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the workflow output to a file (this can be used for re-running the evaluation)

        step_filter = self.eval_config.general.output.workflow_output_step_filter \
            if self.eval_config.general.output else None
        workflow_output = dataset_handler.publish_eval_input(self.eval_input, step_filter)
        with open(workflow_output_file, "w", encoding="utf-8") as f:
            # set indent to 2 for pretty printing
            f.write(workflow_output)
        self.workflow_output_file = workflow_output_file
        logger.info("Workflow output written to %s", workflow_output_file)

        # Write the output of each evaluator to a separate json file
        for evaluator_name, eval_output in self.evaluation_results:
            output_file = self.eval_config.general.output_dir / f"{evaluator_name}_output.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # create json content using the evaluation results
            output = eval_output.model_dump_json(indent=2)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            self.evaluator_output_files.append(output_file)
            logger.info("Evaluation results written to %s", output_file)

    def publish_output(self, dataset_handler: DatasetHandler, profiler_results: ProfilerResults):
        """Publish the output"""
        if self.config.write_output:
            self.write_output(dataset_handler, profiler_results)

        if self.workflow_interrupted:
            # Issue a warning if the workflow was not completed on all datasets
            msg = ("Workflow execution was interrupted due to an error. The results may be incomplete. "
                   "You can re-execute evaluation for incomplete results by running "
                   "`eval` with the --skip_completed_entries flag.")
            logger.warning(msg)

        self.weave_eval.log_summary(self.usage_stats, self.evaluation_results, profiler_results)

    async def run_single_evaluator(self, evaluator_name: str, evaluator: Any):
        """Run a single evaluator and store its results."""
        try:
            eval_output = await evaluator.evaluate_fn(self.eval_input)
            self.evaluation_results.append((evaluator_name, eval_output))

            await self.weave_eval.alog_score(eval_output, evaluator_name)
        except Exception as e:
            logger.exception("An error occurred while running evaluator %s: %s", evaluator_name, e, exc_info=True)

    async def run_evaluators(self, evaluators: dict[str, Any]):
        """Run all configured evaluators asynchronously."""
        tasks = [self.run_single_evaluator(name, evaluator) for name, evaluator in evaluators.items() if evaluator]

        if not tasks:
            logger.warning("All evaluators were empty or invalid.")
            return

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.exception("An error occurred while running evaluators: %s", e, exc_info=True)
            raise
        finally:
            # Finish prediction loggers in Weave
            await self.weave_eval.afinish_loggers()

    def apply_overrides(self):
        from nat.cli.cli_utils.config_override import load_and_override_config
        from nat.data_models.config import Config
        from nat.runtime.loader import PluginTypes
        from nat.runtime.loader import discover_and_register_plugins
        from nat.utils.data_models.schema_validator import validate_schema

        # Register plugins before validation
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

        config_dict = load_and_override_config(self.config.config_file, self.config.override)
        config = validate_schema(config_dict, Config)
        return config

    def _get_workflow_alias(self, workflow_type: str | None = None):
        """Get the workflow alias for displaying in evaluation UI."""
        if self.eval_config.general.workflow_alias:
            return self.eval_config.general.workflow_alias

        if not workflow_type or workflow_type == "EmptyFunctionConfig":
            return "nat-eval"

        return workflow_type

    async def wait_for_all_export_tasks_local(self, session_manager: SessionManager, timeout: float) -> None:
        """Wait for all trace export tasks to complete for local workflows.

        This only works for local workflows where we have direct access to the
        SessionManager and its underlying workflow with exporter manager.
        """
        try:
            workflow = session_manager.workflow
            all_exporters = await workflow.get_all_exporters()
            if not all_exporters:
                logger.debug("No exporters to wait for")
                return

            logger.info("Waiting for export tasks from %d local exporters (timeout: %ds)", len(all_exporters), timeout)

            for name, exporter in all_exporters.items():
                try:
                    await exporter.wait_for_tasks(timeout=timeout)
                    logger.info("Export tasks completed for exporter: %s", name)
                except Exception as e:
                    logger.warning("Error waiting for export tasks from %s: %s", name, e)

            logger.info("All local export task waiting completed")

        except Exception as e:
            logger.warning("Failed to wait for local export tasks: %s", e)

    async def run_and_evaluate(self,
                               session_manager: SessionManager | None = None,
                               job_id: str | None = None) -> EvaluationRunOutput:
        """
        Run the workflow with the specified config file and evaluate the dataset
        """
        logger.info("Starting evaluation run with config file: %s", self.config.config_file)

        from nat.builder.eval_builder import WorkflowEvalBuilder
        from nat.runtime.loader import load_config

        # Load and override the config
        if self.config.override:
            config = self.apply_overrides()
        else:
            config = load_config(self.config.config_file)
        self.eval_config = config.eval
        workflow_alias = self._get_workflow_alias(config.workflow.type)
        logger.debug("Loaded %s evaluation configuration: %s", workflow_alias, self.eval_config)

        # Cleanup the output directory
        if self.eval_config.general.output:
            self.cleanup_output_directory()

        # Generate a job_id if append_job_id_to_output_dir is enabled and no job_id provided
        if (self.eval_config.general.output
                and self.eval_config.general.output.job_management.append_job_id_to_output_dir and not job_id):
            job_id = "job_" + str(uuid4())
            logger.info("Generated job ID for output directory: %s", job_id)

        # If a job id is provided keep the data per-job
        if job_id:
            self.eval_config.general.output_dir = self.eval_config.general.output_dir / f"jobs/{job_id}"
            if self.eval_config.general.output:
                self.eval_config.general.output.dir = self.eval_config.general.output_dir

        # Load the input dataset
        # For multiple datasets, one handler per dataset can be created
        dataset_config = self.eval_config.general.dataset  # Currently only one dataset is supported
        if not dataset_config:
            logger.info("No dataset found, nothing to evaluate")
            return EvaluationRunOutput(workflow_output_file=self.workflow_output_file,
                                       evaluator_output_files=self.evaluator_output_files,
                                       workflow_interrupted=self.workflow_interrupted,
                                       eval_input=EvalInput(eval_input_items=[]),
                                       evaluation_results=[],
                                       usage_stats=UsageStats(),
                                       profiler_results=ProfilerResults())

        dataset_handler = DatasetHandler(dataset_config=dataset_config,
                                         reps=self.config.reps,
                                         concurrency=self.eval_config.general.max_concurrency,
                                         num_passes=self.config.num_passes,
                                         adjust_dataset_size=self.config.adjust_dataset_size)
        self.eval_input = dataset_handler.get_eval_input_from_dataset(self.config.dataset)
        if not self.eval_input.eval_input_items:
            logger.info("Dataset is empty. Nothing to evaluate.")
            return EvaluationRunOutput(workflow_output_file=self.workflow_output_file,
                                       evaluator_output_files=self.evaluator_output_files,
                                       workflow_interrupted=self.workflow_interrupted,
                                       eval_input=self.eval_input,
                                       evaluation_results=self.evaluation_results,
                                       usage_stats=self.usage_stats,
                                       profiler_results=ProfilerResults())

        # Run workflow and evaluate
        async with WorkflowEvalBuilder.from_config(config=config) as eval_workflow:
            # Initialize Weave integration
            self.weave_eval.initialize_logger(workflow_alias, self.eval_input, config)

            # Run workflow
            with self.eval_trace_context.evaluation_context():
                if self.config.endpoint:
                    await self.run_workflow_remote()
                else:
                    if not self.config.skip_workflow:
                        if session_manager is None:
                            session_manager = SessionManager(eval_workflow.build(),
                                                             max_concurrency=self.eval_config.general.max_concurrency)
                        await self.run_workflow_local(session_manager)

                # Evaluate
                evaluators = {name: eval_workflow.get_evaluator(name) for name in self.eval_config.evaluators}
                await self.run_evaluators(evaluators)

                # Wait for all trace export tasks to complete (local workflows only)
                if session_manager and not self.config.endpoint:
                    await self.wait_for_all_export_tasks_local(session_manager, timeout=self.config.export_timeout)

        # Profile the workflow
        profiler_results = await self.profile_workflow()

        # compute total runtime
        if self.usage_stats.usage_stats_items:
            self.usage_stats.total_runtime = max(self.usage_stats.usage_stats_items.values(),
                                                 key=lambda x: x.max_timestamp).max_timestamp - \
                min(self.usage_stats.usage_stats_items.values(), key=lambda x: x.min_timestamp).min_timestamp
        else:
            self.usage_stats.total_runtime = 0.0

        # Publish the results
        self.publish_output(dataset_handler, profiler_results)

        # Run custom scripts and upload evaluation outputs to S3
        if self.eval_config.general.output:
            output_uploader = OutputUploader(self.eval_config.general.output, job_id=job_id)
            output_uploader.run_custom_scripts()
            await output_uploader.upload_directory()

        return EvaluationRunOutput(workflow_output_file=self.workflow_output_file,
                                   evaluator_output_files=self.evaluator_output_files,
                                   workflow_interrupted=self.workflow_interrupted,
                                   eval_input=self.eval_input,
                                   evaluation_results=self.evaluation_results,
                                   usage_stats=self.usage_stats,
                                   profiler_results=profiler_results)
