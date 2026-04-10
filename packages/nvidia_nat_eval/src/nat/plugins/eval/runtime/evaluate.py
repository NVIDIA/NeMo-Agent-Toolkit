# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import inspect
import json
import logging
import shutil
from collections.abc import Awaitable
from contextlib import nullcontext
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel
from pydantic import SecretStr
from tqdm import tqdm

from nat.plugins.eval.dataset_handler.dataset_handler import DatasetHandler
from nat.plugins.eval.eval_callbacks import EvalCallbackManager
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import LegacyEvaluator
from nat.plugins.eval.runtime.eval_harness import EvaluationHarness
from nat.plugins.eval.runtime.llm_validator import validate_llm_endpoints
from nat.plugins.eval.utils.output_uploader import OutputUploader

FULL_EVAL_INSTALL_HINT = ("Full workflow evaluation requires optional dependencies that are not installed. "
                          "Install with: pip install \"nvidia-nat[eval]\" "
                          "(or pip install \"nvidia-nat-eval[full]\")")


def _raise_full_eval_dependency_error(error: Exception):
    raise ModuleNotFoundError(FULL_EVAL_INSTALL_HINT) from error


try:
    from nat.builder.context import ContextState
    from nat.data_models.config import Config
    from nat.data_models.evaluate_config import EvalConfig
    from nat.data_models.evaluate_config import JobEvictionPolicy
    from nat.data_models.evaluate_runtime import EvaluationRunConfig
    from nat.data_models.evaluate_runtime import EvaluationRunOutput
    from nat.data_models.evaluate_runtime import ProfilerResults
    from nat.data_models.evaluate_runtime import UsageStats
    from nat.data_models.evaluate_runtime import UsageStatsItem
    from nat.data_models.evaluate_runtime import UsageStatsLLM
    from nat.data_models.evaluator import EvalInput
    from nat.data_models.evaluator import EvalInputItem
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.data_models.user_info import BasicUserInfo
    from nat.data_models.user_info import UserInfo
    from nat.plugins.eval.data_models.evaluator_io import EvalOutput
    from nat.runtime.session import SessionManager
except ImportError as import_error:  # pragma: no cover - guarded runtime path
    _raise_full_eval_dependency_error(import_error)

if TYPE_CHECKING:
    from starlette.requests import HTTPConnection

    from nat.plugins.eval.eval_callbacks import EvalCallbackManager
    from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
    from nat.plugins.eval.exporters.file_eval_callback import FileEvalCallback

logger = logging.getLogger(__name__)


class EvaluationRun:
    """
    Instantiated for each evaluation run and used to store data for that single run.

    .. warning::
        **Experimental Feature**: The Evaluation API is experimental and may change in future releases.
        Future versions may introduce breaking changes without notice.
    """

    def __init__(self, config: EvaluationRunConfig, callback_manager: "EvalCallbackManager | None" = None):
        """
        Initialize an EvaluationRun with configuration.
        """
        from nat.plugins.eval.utils.intermediate_step_adapter import IntermediateStepAdapter

        # Run-specific configuration
        self.config: EvaluationRunConfig = config
        self.callback_manager: EvalCallbackManager = callback_manager or EvalCallbackManager()
        if self.config.write_output:
            from nat.plugins.eval.exporters.file_eval_callback import FileEvalCallback
            if not any(isinstance(cb, FileEvalCallback) for cb in self.callback_manager._callbacks):
                # Keep direct `EvaluationRun(...)` behavior consistent with CLI usage.
                self.callback_manager.register(FileEvalCallback())
        self.eval_config: EvalConfig | None = None
        self.effective_config: Config | None = None  # Stores the complete config after applying overrides

        # Helpers
        self.intermediate_step_adapter: IntermediateStepAdapter = IntermediateStepAdapter()
        from nat.plugins.eval.runtime.atif_adapter import EvalAtifAdapter
        self.atif_adapter = EvalAtifAdapter()
        self.evaluation_harness = EvaluationHarness()
        # Metadata
        self.eval_input: EvalInput | None = None
        self.atif_eval_samples: AtifEvalSampleList = []
        self.workflow_interrupted: bool = False

        # evaluation_results is list of tuples (evaluator_name, EvalOutput)
        self.evaluation_results: list[tuple[str, EvalOutput]] = []

        # usage stats
        self.usage_stats: UsageStats = UsageStats()

        # workflow output file
        self.workflow_output_file: Path | None = None

        # evaluation output files
        self.evaluator_output_files: list[Path] = []

        # configuration output files
        self.config_original_file: Path | None = None
        self.config_effective_file: Path | None = None
        self.config_metadata_file: Path | None = None

        # Pre-generated OTEL root span_ids for eager trace linking (item_id -> span_id)
        self._item_span_ids: dict[str, int] = {}

    def _compute_usage_stats(self, item: EvalInputItem):
        """Compute usage stats for a single item using the intermediate steps"""
        usage_stats_per_llm = {}
        total_tokens = 0
        for step in item.trajectory:
            if step.event_type == IntermediateStepType.LLM_END:
                llm_name = step.name or step.function_ancestry.function_name or "unknown"
                if llm_name not in usage_stats_per_llm:
                    usage_stats_per_llm[llm_name] = UsageStatsLLM()

                token_usage = step.usage_info.token_usage if step.usage_info else None
                if token_usage is not None:
                    usage_stats_per_llm[llm_name].prompt_tokens += token_usage.prompt_tokens
                    usage_stats_per_llm[llm_name].completion_tokens += token_usage.completion_tokens
                    usage_stats_per_llm[llm_name].total_tokens += token_usage.total_tokens
                    usage_stats_per_llm[llm_name].reasoning_tokens += token_usage.reasoning_tokens
                    usage_stats_per_llm[llm_name].cached_tokens += token_usage.cached_tokens
                    total_tokens += token_usage.total_tokens

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
        for step in item.trajectory:
            if step.event_type == IntermediateStepType.LLM_START:
                previous_llm_start_time = step.event_timestamp
            elif step.event_type == IntermediateStepType.LLM_END and previous_llm_start_time is not None:
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

    async def run_workflow_local(self,
                                 session_manager: SessionManager,
                                 http_connection: "HTTPConnection | None" = None):
        '''
        Launch the workflow with the specified questions and extract the output using the jsonpath
        '''
        # import function level dependencies
        from jsonpath_ng import parse

        from nat.builder.runtime_event_subscriber import pull_intermediate

        # Run the workflow
        jsonpath_expr = parse(self.config.result_json_path)
        stop_event = asyncio.Event()

        async def run_one(item: EvalInputItem):
            if stop_event.is_set():
                return "", []

            # Only pre-generate root span_ids when callbacks need them
            # (e.g. LangSmith eager linking). This avoids touching core
            # observability code paths for non-LangSmith eval runs.
            pre_span_id = None
            if self.callback_manager and self.callback_manager.needs_root_span_ids:
                from nat.data_models.span import _generate_nonzero_span_id
                pre_span_id = _generate_nonzero_span_id()
                self._item_span_ids[str(item.id)] = pre_span_id

            eval_username: str = "nat_eval_user"
            if self.eval_config.general.per_input_user_id:
                eval_username += f"-{uuid4()}"
            eval_user_id: str = UserInfo(
                basic_user=BasicUserInfo(username=eval_username, password=SecretStr("nat_eval_user"))).get_user_id()

            # Set the pre-generated span_id in the ContextVar BEFORE entering
            # the session/runner context. asyncio.create_task() copies ContextVars,
            # so the Runner's task will inherit this value.
            ctx_state = ContextState.get()
            root_span_token = ctx_state._root_span_id.set(pre_span_id) if pre_span_id is not None else None
            try:
                async with session_manager.session(user_id=eval_user_id, http_connection=http_connection) as session:
                    async with session.run(item.input_obj) as runner:
                        if not session.workflow.has_single_output:
                            # raise an error if the workflow has multiple outputs
                            raise NotImplementedError("Multiple outputs are not supported")

                        runner_task = None
                        intermediate_task = None

                        async def cancel_pending_tasks():
                            pending = []
                            for awaitable in (runner_task, intermediate_task):
                                if awaitable is not None:
                                    if not awaitable.done():
                                        awaitable.cancel()
                                    pending.append(awaitable)
                            if pending:
                                await asyncio.gather(*pending, return_exceptions=True)

                        try:
                            # Start usage stats and intermediate steps collection in parallel
                            intermediate_task = asyncio.ensure_future(pull_intermediate())
                            runner_task = asyncio.create_task(runner.result())
                            base_output = await runner_task
                            intermediate_steps = await intermediate_task
                        except NotImplementedError as e:
                            logger.error("Failed to run the workflow: %s", e)
                            await cancel_pending_tasks()
                            # raise original error
                            raise
                        except Exception as e:
                            logger.exception("Failed to run the workflow: %s", e)
                            # stop processing if a workflow error occurs
                            self.workflow_interrupted = True
                            await cancel_pending_tasks()
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
                                raise RuntimeError(
                                    f"Failed to extract output using jsonpath: {self.config.result_json_path}")
                            if (len(m) > 1):
                                logger.warning(
                                    "Multiple matches found for jsonpath at row '%s'. Matches: %s. Using the first",
                                    base_output,
                                    m)
                            output = m[0].value

                        item.output_obj = output
                        item.trajectory = self.intermediate_step_adapter.validate_intermediate_steps(intermediate_steps)
                        usage_stats_item = self._compute_usage_stats(item)
                        if self.callback_manager:
                            self.callback_manager.on_prediction(item=item, output=output)
                            await self.callback_manager.a_on_usage_stats(item=item, usage_stats_item=usage_stats_item)
            finally:
                if root_span_token is not None:
                    ctx_state._root_span_id.reset(root_span_token)

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
        from nat.plugins.eval.runtime.remote_workflow import EvaluationRemoteWorkflowHandler
        handler = EvaluationRemoteWorkflowHandler(self.config, self.eval_config.general.max_concurrency)
        await handler.run_workflow_remote(self.eval_input)
        for item in self.eval_input.eval_input_items:
            usage_stats_item = self._compute_usage_stats(item)
            if self.callback_manager:
                self.callback_manager.on_prediction(item=item, output=item.output_obj)
                await self.callback_manager.a_on_usage_stats(item=item, usage_stats_item=usage_stats_item)

    async def profile_workflow(self) -> ProfilerResults:
        """
        Profile a dataset
        """

        if not self.eval_config.general.profiler:
            logger.info("Profiler is not enabled. Skipping profiling.")
            return ProfilerResults()

        from nat.plugins.profiler.profile_runner import ProfilerRunner

        all_stats = [item.trajectory for item in self.eval_input.eval_input_items]

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
                logger.exception("Failed to delete old job directory: %s: %s", dir_to_delete, e)

    def get_file_exporter(self) -> "FileEvalCallback | None":
        """Return the registered ``FileEvalCallback``, if any."""
        from nat.plugins.eval.exporters.file_eval_callback import FileEvalCallback
        for cb in self.callback_manager._callbacks:
            if isinstance(cb, FileEvalCallback):
                return cb
        return None

    def write_configuration(self) -> None:
        """Save the configuration used for this evaluation run to the output directory.

        This saves three files:
        1. config_original.yml - The original configuration file
        2. config_effective.yml - The configuration with all overrides applied
        3. config_metadata.json - Metadata about the evaluation run and overrides
        """
        output_dir = self.eval_config.general.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save original configuration
            config_original_file = output_dir / "config_original.yml"
            if isinstance(self.config.config_file, Path):
                # Copy original file if it exists
                if self.config.config_file.exists():
                    shutil.copy2(self.config.config_file, config_original_file)
                    self.config_original_file = config_original_file
                    logger.info("Original config file copied to %s", config_original_file)
                else:
                    logger.warning("Original config file not found at %s", self.config.config_file)
            elif isinstance(self.config.config_file, BaseModel):
                # Serialize programmatic config, using mode='json' to handle special types like timedelta
                config_dict = self.config.config_file.model_dump(mode='json')
                with open(config_original_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
                self.config_original_file = config_original_file
                logger.info("Programmatic config saved to %s", config_original_file)

            # 2. Save effective configuration (with overrides applied)
            config_effective_file = output_dir / "config_effective.yml"
            if self.effective_config is not None:
                effective_config_dict = self.effective_config.model_dump(mode='json') if self.effective_config else {}
                with open(config_effective_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(effective_config_dict, f, default_flow_style=False, sort_keys=False)
                self.config_effective_file = config_effective_file
                logger.info("Effective config (with overrides) saved to %s", config_effective_file)
            else:
                logger.warning("Effective config not available, skipping config_effective.yml")

            # 3. Save metadata about the run
            config_metadata_file = output_dir / "config_metadata.json"
            metadata = {
                "config_file":
                    str(self.config.config_file),
                "config_file_type":
                    "Path" if isinstance(self.config.config_file, Path) else "BaseModel",
                "overrides": [{
                    "path": path, "value": value
                } for path, value in self.config.override] if self.config.override else [],
                "dataset":
                    self.config.dataset,
                "result_json_path":
                    self.config.result_json_path,
                "skip_workflow":
                    self.config.skip_workflow,
                "skip_completed_entries":
                    self.config.skip_completed_entries,
                "reps":
                    self.config.reps,
                "endpoint":
                    self.config.endpoint,
                "endpoint_timeout":
                    self.config.endpoint_timeout,
                "adjust_dataset_size":
                    self.config.adjust_dataset_size,
                "num_passes":
                    self.config.num_passes,
                "export_timeout":
                    self.config.export_timeout,
                "user_id":
                    self.config.user_id,
                "timestamp":
                    datetime.now(tz=UTC).isoformat(),
            }

            with open(config_metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            self.config_metadata_file = config_metadata_file
            logger.info("Configuration metadata saved to %s", config_metadata_file)

        except Exception:
            logger.exception("Failed to write configuration files")
            # Don't raise - this is not critical enough to fail the entire evaluation

    def write_output(self, dataset_handler: DatasetHandler, profiler_results: ProfilerResults):
        workflow_output_file = self.eval_config.general.output_dir / "workflow_output.json"
        workflow_output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the configuration files (original, effective, and metadata)
        self.write_configuration()

        # Write the workflow output to a file (this can be used for re-running the evaluation)

        step_filter = self.eval_config.general.output.workflow_output_step_filter \
            if self.eval_config.general.output else None
        workflow_output = dataset_handler.publish_eval_input(self.eval_input, step_filter)
        with open(workflow_output_file, "w", encoding="utf-8") as f:
            # set indent to 2 for pretty printing
            f.write(workflow_output)
        self.workflow_output_file = workflow_output_file
        logger.info("Workflow output written to %s", workflow_output_file)

        output_config = self.eval_config.general.output
        if output_config and output_config.write_atif_workflow_output:
            atif_workflow_output_file = self.eval_config.general.output_dir / "workflow_output_atif.json"
            atif_workflow_output = json.dumps([sample.model_dump(mode="json") for sample in self.atif_eval_samples],
                                              indent=2)
            with open(atif_workflow_output_file, "w", encoding="utf-8") as f:
                f.write(atif_workflow_output)
            logger.info("ATIF workflow output written to %s", atif_workflow_output_file)

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
        if self.callback_manager:
            self.callback_manager.on_eval_summary(usage_stats=self.usage_stats,
                                                  evaluation_results=self.evaluation_results,
                                                  profiler_results=profiler_results)

    async def run_single_evaluator(self, evaluator_name: str, evaluator: Any):
        """Run a single evaluator and store its results."""
        if isinstance(evaluator, AtifEvaluator):
            harness_results = await self.evaluation_harness.evaluate({evaluator_name: evaluator},
                                                                     self.atif_eval_samples)
            eval_output = harness_results.get(evaluator_name)
            if eval_output is None:
                return
            self.evaluation_results.append((evaluator_name, eval_output))
            if self.callback_manager:
                await self.callback_manager.a_on_evaluator_score(eval_output=eval_output, evaluator_name=evaluator_name)
            return
        await self._run_single_legacy_evaluator(evaluator_name, evaluator)

    async def _run_single_legacy_evaluator(self, evaluator_name: str, evaluator: Any):
        """Run one evaluator through the legacy `evaluate_fn` lane."""
        try:
            evaluate_fn = getattr(evaluator, "evaluate_fn", None)
            if not isinstance(evaluator, LegacyEvaluator):
                raise TypeError(f"Evaluator '{evaluator_name}' is missing callable evaluate_fn and evaluate_atif_fn")
            eval_result = evaluate_fn(self.eval_input)
            if not inspect.isawaitable(eval_result):
                raise TypeError(f"Evaluator '{evaluator_name}' evaluate_fn must return an awaitable")
            eval_output = await eval_result
            self.evaluation_results.append((evaluator_name, eval_output))
            if self.callback_manager:
                await self.callback_manager.a_on_evaluator_score(eval_output=eval_output, evaluator_name=evaluator_name)
        except Exception as e:
            logger.exception("An error occurred while running evaluator %s: %s", evaluator_name, e)

    async def run_evaluators(self, evaluators: dict[str, Any]):
        """Run all configured evaluators asynchronously."""
        atif_evaluators: dict[str, AtifEvaluator] = {}
        legacy_evaluators: dict[str, LegacyEvaluator] = {}
        for name, evaluator in evaluators.items():
            if not evaluator:
                continue
            if isinstance(evaluator, AtifEvaluator):
                atif_evaluators[name] = evaluator
            elif isinstance(evaluator, LegacyEvaluator):
                legacy_evaluators[name] = evaluator
            else:
                logger.warning("Skipping evaluator %s: missing ATIF and legacy evaluator interfaces", name)

        if not atif_evaluators and not legacy_evaluators:
            logger.warning("All evaluators were empty or invalid.")
            return

        try:
            if atif_evaluators:
                harness_results = await self.evaluation_harness.evaluate(atif_evaluators, self.atif_eval_samples)
                for evaluator_name, eval_output in harness_results.items():
                    self.evaluation_results.append((evaluator_name, eval_output))
                    if self.callback_manager:
                        await self.callback_manager.a_on_evaluator_score(eval_output=eval_output,
                                                                         evaluator_name=evaluator_name)

            if legacy_evaluators:
                tasks: list[Awaitable[None]] = [
                    self._run_single_legacy_evaluator(evaluator_name=name, evaluator=evaluator)
                    for name, evaluator in legacy_evaluators.items()
                ]
                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error("An error occurred while running evaluators: %s", e)
            raise
        finally:
            if self.callback_manager:
                await self.callback_manager.a_on_export_flush()

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

    def _on_eval_complete(self, dataset_handler: DatasetHandler | None = None) -> None:
        """Build an EvalResult from collected data and fire the on_eval_complete callback."""
        if not self.evaluation_results:
            return
        try:
            from nat.plugins.eval.eval_callbacks import build_eval_result

            workflow_output_json: str | None = None
            atif_workflow_output_json: str | None = None
            if dataset_handler is not None and self.eval_input is not None:
                step_filter = (self.eval_config.general.output.workflow_output_step_filter
                               if self.eval_config and self.eval_config.general.output else None)
                workflow_output_json = dataset_handler.publish_eval_input(self.eval_input, step_filter)
                if self.eval_config.general.output and self.eval_config.general.output.write_atif_workflow_output:
                    atif_workflow_output_json = json.dumps(
                        [sample.model_dump(mode="json") for sample in self.atif_eval_samples], indent=2)

            scores = {name: output.average_score for name, output in self.evaluation_results}
            result = build_eval_result(
                eval_input_items=self.eval_input.eval_input_items,
                evaluation_results=self.evaluation_results,
                metric_scores=scores,
                usage_stats=self.usage_stats,
                item_span_ids=self._item_span_ids,
                workflow_output_json=workflow_output_json,
                atif_workflow_output_json=atif_workflow_output_json,
                run_config=self.config,
                effective_config=self.effective_config,
                output_dir=(self.eval_config.general.output_dir if self.eval_config else None),
            )
            self.callback_manager.on_eval_complete(result)
        except Exception:
            logger.warning("Failed to fire on_eval_complete callback", exc_info=True)

    async def run_and_evaluate(self,
                               session_manager: SessionManager | None = None,
                               job_id: str | None = None,
                               http_connection: "HTTPConnection | None" = None) -> EvaluationRunOutput:
        """
        Run the workflow with the specified config file and evaluate the dataset
        """
        logger.info("Starting evaluation run with config file: %s", self.config.config_file)

        from nat.plugins.eval.runtime.builder import WorkflowEvalBuilder
        from nat.runtime.loader import load_config

        # Load and override the config
        config: Config | None = None
        if isinstance(self.config.config_file, BaseModel):
            config = self.config.config_file
        elif self.config.override:
            config = self.apply_overrides()
        else:
            config = load_config(self.config.config_file)

        # Store the effective configuration for later saving to output directory
        self.effective_config = config
        self.eval_config = config.eval
        workflow_alias = self._get_workflow_alias(config.workflow.type)
        logger.debug("Loaded %s evaluation configuration: %s", workflow_alias, self.eval_config)

        # Cleanup the output directory (skip when reusing existing workflow output)
        if self.eval_config.general.output:
            if self.config.skip_workflow:
                logger.info("Skipping output directory cleanup because --skip_workflow is set")
            else:
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
                                       profiler_results=ProfilerResults(),
                                       config_original_file=self.config_original_file,
                                       config_effective_file=self.config_effective_file,
                                       config_metadata_file=self.config_metadata_file)

        custom_pre_eval_process_function = self.eval_config.general.output.custom_pre_eval_process_function \
            if self.eval_config.general.output else None
        dataset_handler = DatasetHandler(dataset_config=dataset_config,
                                         reps=self.config.reps,
                                         concurrency=self.eval_config.general.max_concurrency,
                                         num_passes=self.config.num_passes,
                                         adjust_dataset_size=self.config.adjust_dataset_size,
                                         custom_pre_eval_process_function=custom_pre_eval_process_function)
        self.eval_input = dataset_handler.get_eval_input_from_dataset(self.config.dataset)
        if self.eval_input.eval_input_items:
            try:
                file_path = getattr(dataset_config, 'file_path', 'nat-eval-dataset')
                dataset_name = Path(file_path).stem if file_path else 'nat-eval-dataset'
                self.callback_manager.on_dataset_loaded(dataset_name=dataset_name,
                                                        items=self.eval_input.eval_input_items)
            except Exception:
                logger.warning("Failed to fire on_dataset_loaded callback", exc_info=True)

        if self.callback_manager:
            try:
                self.callback_manager.on_eval_started(workflow_alias=workflow_alias,
                                                      eval_input=self.eval_input,
                                                      config=config,
                                                      job_id=job_id)
            except Exception:
                logger.warning("Failed to initialize eval export callbacks", exc_info=True)
        if not self.eval_input.eval_input_items:
            logger.info("Dataset is empty. Nothing to evaluate.")
            return EvaluationRunOutput(workflow_output_file=self.workflow_output_file,
                                       evaluator_output_files=self.evaluator_output_files,
                                       workflow_interrupted=self.workflow_interrupted,
                                       eval_input=self.eval_input,
                                       evaluation_results=self.evaluation_results,
                                       usage_stats=self.usage_stats,
                                       profiler_results=ProfilerResults(),
                                       config_original_file=self.config_original_file,
                                       config_effective_file=self.config_effective_file,
                                       config_metadata_file=self.config_metadata_file)

        # Validate LLM endpoints before running evaluation (opt-in via config)
        if (not self.config.skip_workflow and not self.config.endpoint and config.eval.general.validate_llm_endpoints):
            try:
                logger.info("Validating LLM endpoints before evaluation (enabled via config)...")
                await validate_llm_endpoints(config)
            except RuntimeError as e:
                # Critical validation errors (404, connection failures) - fail fast
                logger.error("LLM endpoint validation failed: %s", e)
                raise
            except Exception as e:
                # Non-critical errors (missing packages, config issues) - warn but continue
                logger.warning("LLM endpoint validation incomplete: %s. Continuing with evaluation...",
                               e,
                               exc_info=True)

        # Run workflow and evaluate
        async with WorkflowEvalBuilder.from_config(config=config) as eval_workflow:
            eval_context = self.callback_manager.evaluation_context() if self.callback_manager else nullcontext()
            with eval_context:
                # Run workflow
                local_session_manager: SessionManager | None = None
                try:
                    if self.config.endpoint:
                        await self.run_workflow_remote()
                    elif not self.config.skip_workflow:
                        if session_manager is None:
                            session_manager = await SessionManager.create(
                                config=config,
                                shared_builder=eval_workflow,
                                max_concurrency=self.eval_config.general.max_concurrency)
                            local_session_manager = session_manager
                        await self.run_workflow_local(session_manager, http_connection=http_connection)

                    # Pre-evaluation process the workflow output
                    self.eval_input = dataset_handler.pre_eval_process_eval_input(self.eval_input)
                    evaluators = {name: eval_workflow.get_evaluator(name) for name in self.eval_config.evaluators}
                    needs_atif = (any(isinstance(ev, AtifEvaluator) for ev in evaluators.values())
                                  or (self.eval_config.general.output
                                      and self.eval_config.general.output.write_atif_workflow_output))
                    if needs_atif:
                        self.atif_eval_samples = self.atif_adapter.build_samples(self.eval_input)
                    else:
                        self.atif_eval_samples = []

                    # Evaluate
                    await self.run_evaluators(evaluators)

                    # Wait for all trace export tasks to complete (local workflows only)
                    if session_manager and not self.config.endpoint:
                        await self.wait_for_all_export_tasks_local(session_manager, timeout=self.config.export_timeout)
                finally:
                    if local_session_manager is not None:
                        await local_session_manager.shutdown()

        # Profile the workflow
        profiler_results = await self.profile_workflow()

        # compute total runtime
        if self.usage_stats.usage_stats_items:
            self.usage_stats.total_runtime = max(self.usage_stats.usage_stats_items.values(),
                                                 key=lambda x: x.max_timestamp).max_timestamp - \
                min(self.usage_stats.usage_stats_items.values(), key=lambda x: x.min_timestamp).min_timestamp
        else:
            self.usage_stats.total_runtime = 0.0

        # Fire eval-complete callbacks (including FileEvalCallback for file export)
        self._on_eval_complete(dataset_handler)

        if self.workflow_interrupted:
            msg = ("Workflow execution was interrupted due to an error. The results may be incomplete. "
                   "You can re-execute evaluation for incomplete results by running "
                   "`eval` with the --skip_completed_entries flag.")
            logger.warning(msg)

        # Retrieve file paths written by FileEvalCallback (if registered)
        file_exporter = self.get_file_exporter()
        if file_exporter is not None:
            self.workflow_output_file = file_exporter.workflow_output_file
            self.evaluator_output_files = file_exporter.evaluator_output_files
            self.config_original_file = file_exporter.config_original_file
            self.config_effective_file = file_exporter.config_effective_file
            self.config_metadata_file = file_exporter.config_metadata_file

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
                                   profiler_results=profiler_results,
                                   config_original_file=self.config_original_file,
                                   config_effective_file=self.config_effective_file,
                                   config_metadata_file=self.config_metadata_file)
