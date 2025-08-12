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
import math
import os
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nat.data_models.evaluate import ProfilerConfig
from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.data_models import ProfilerResults
from nat.profiler.forecasting.model_trainer import ModelTrainer
from nat.profiler.inference_metrics_model import InferenceMetricsModel
from nat.profiler.utils import create_standardized_dataframe
from nat.utils.type_converter import TypeConverter

logger = logging.getLogger(__name__)


class SimpleMetricsHolder(BaseModel):
    workflow_run_time_confidence_intervals: Any
    llm_latency_confidence_intervals: Any
    throughput_estimate_confidence_interval: Any


class InferenceOptimizationHolder(BaseModel):
    confidence_intervals: SimpleMetricsHolder
    common_prefixes: Any
    token_uniqueness: Any
    workflow_runtimes: Any


class ProfilerRunner:
    """
    A utility to run a series of prompts through an AIQ Toolkit workflow for profiling:

    - can load prompts from a file
    - or generate them via an LLM
    - collect usage stats for each run
    - store them in a configured directory

    Updated version with additional metrics:

    - For each request, we collect a list of UsageStatistic objects, store them individually,
      and also keep a final large JSON of all requests.
    - We then compute:
       1. 90, 95, 99% confidence intervals for the mean total workflow run time.
       2. 90, 95, 99% confidence intervals for the mean LLM latency.
       3. 90, 95, 99% estimates of throughput.

      All computed metrics are saved to a metrics JSON file at the end.
    """

    def __init__(self, profiler_config: ProfilerConfig, output_dir: Path, write_output: bool = True):
        self.profile_config = profiler_config
        self.output_dir = output_dir
        self.write_output = write_output
        self._converter = TypeConverter([])

        # Holds per-request data (prompt, output, usage_stats, etc.)
        # This will be saved at the end to a big JSON file
        self.all_requests_data: list[dict] = []
        self.all_steps = []

        # Ensure output directory
        os.makedirs(output_dir, exist_ok=True)

    async def run(self, all_steps: list[list[IntermediateStep]]) -> ProfilerResults:
        """
        Main entrypoint: Works on Input DataFrame generated from eval to fit forecasting model,
        writes out combined requests JSON, then computes and saves additional metrics,
        and optionally fits a forecasting model.
        """
        from nat.profiler.inference_optimization.bottleneck_analysis.nested_stack_analysis import \
            multi_example_call_profiling
        from nat.profiler.inference_optimization.bottleneck_analysis.simple_stack_analysis import \
            profile_workflow_bottlenecks
        from nat.profiler.inference_optimization.experimental.concurrency_spike_analysis import \
            concurrency_spike_analysis
        from nat.profiler.inference_optimization.experimental.prefix_span_analysis import \
            prefixspan_subworkflow_with_text
        from nat.profiler.inference_optimization.llm_metrics import LLMMetrics
        from nat.profiler.inference_optimization.prompt_caching import get_common_prefixes
        from nat.profiler.inference_optimization.token_uniqueness import compute_inter_query_token_uniqueness_by_llm
        from nat.profiler.inference_optimization.workflow_runtimes import compute_workflow_runtime_metrics
        from nat.profiler.intermediate_property_adapter import IntermediatePropertyAdaptor

        # Convert the incoming DataFrame to a list of dicts and store
        all_steps = [[IntermediatePropertyAdaptor.from_intermediate_step(step) for step in steps]
                     for steps in all_steps]  # Add adapter properties to each step

        self.all_steps = all_steps
        self.all_requests_data = []
        for i, steps in enumerate(all_steps):
            request_data = []
            for step in steps:
                request_data.append(step.model_dump())
            self.all_requests_data.append({"request_number": i, "intermediate_steps": request_data})

        # Write the final big JSON (all requests)
        if self.write_output:
            final_path = os.path.join(self.output_dir, "all_requests_profiler_traces.json")
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_requests_data, f, indent=2, default=str)
            logger.info("Wrote combined data to: %s", final_path)

        # ------------------------------------------------------------
        # Generate one standardized dataframe for all usage stats
        # ------------------------------------------------------------
        merged_df = create_standardized_dataframe(all_steps)

        if self.profile_config.compute_llm_metrics and not merged_df.empty:
            merged_df = LLMMetrics.compute_profiling_metrics(all_steps)

        output_df = merged_df.copy()

        if self.profile_config.csv_exclude_io_text and not output_df.empty:
            # Exclude text fields from CSV
            output_df = output_df.drop(columns=['llm_text_input', 'llm_text_output', 'llm_new_token'])

        # Write this single CSV
        csv_path = os.path.join(self.output_dir, "standardized_data_all.csv")
        output_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info("Wrote merged standardized DataFrame to %s", csv_path)

        # ------------------------------------------------------------
        # Compute and save additional performance metrics
        # ------------------------------------------------------------
        workflow_run_time_ci: InferenceMetricsModel = self._compute_workflow_run_time_confidence_intervals()

        # 2. 90, 95, 99% confidence intervals of mean LLM latency
        llm_latency_ci: InferenceMetricsModel = self._compute_llm_latency_confidence_intervals()

        # 3. 90, 95, 99% estimates of throughput
        throughput_ci: InferenceMetricsModel = self._compute_throughput_estimates()

        # Collect all computed metrics
        simple_metrics = SimpleMetricsHolder(workflow_run_time_confidence_intervals=workflow_run_time_ci.model_dump(),
                                             llm_latency_confidence_intervals=llm_latency_ci.model_dump(),
                                             throughput_estimate_confidence_interval=throughput_ci.model_dump())

        common_prefix_results = token_uniqueness_results = workflow_runtimes_results = None

        if self.profile_config.prompt_caching_prefixes.enable:
            # ------------------------------------------------------------
            # Compute and save common prefixes
            # ------------------------------------------------------------

            prefixes = get_common_prefixes(all_steps, self.profile_config.prompt_caching_prefixes.min_frequency)
            common_prefix_results = prefixes

        if self.profile_config.token_uniqueness_forecast:
            # ------------------------------------------------------------
            # Compute and save inter-query token uniqueness
            # ------------------------------------------------------------

            uniqueness = compute_inter_query_token_uniqueness_by_llm(all_steps)
            token_uniqueness_results = uniqueness

        if self.profile_config.workflow_runtime_forecast or self.profile_config.base_metrics:
            # ------------------------------------------------------------
            # Compute and save workflow runtime metrics
            # ------------------------------------------------------------

            workflow_runtimes = compute_workflow_runtime_metrics(all_steps)
            workflow_runtimes_results = workflow_runtimes

        inference_optimization_results = InferenceOptimizationHolder(confidence_intervals=simple_metrics,
                                                                     common_prefixes=common_prefix_results,
                                                                     token_uniqueness=token_uniqueness_results,
                                                                     workflow_runtimes=workflow_runtimes_results)

        if self.write_output and inference_optimization_results:
            # Save to JSON
            optimization_results_path = os.path.join(self.output_dir, "inference_optimization.json")
            with open(optimization_results_path, 'w', encoding='utf-8') as f:
                json.dump(inference_optimization_results.model_dump(), f, indent=2)
            logger.info("Wrote inference optimization results to: %s", optimization_results_path)

        workflow_profiling_reports = ""
        workflow_profiling_metrics = {}

        if self.profile_config.bottleneck_analysis.enable_simple_stack:
            # ------------------------------------------------------------
            # Profile workflow bottlenecks
            # ------------------------------------------------------------

            workflow_bottlenecks = profile_workflow_bottlenecks(all_steps)
            workflow_bottlenecks = workflow_bottlenecks.model_dump()
            workflow_profiling_reports += "\n\n\n" + workflow_bottlenecks["summary"]
            workflow_profiling_metrics["simple_stack_analysis"] = workflow_bottlenecks["stats"]
            logger.info("Simple stack analysis complete")

        if self.profile_config.bottleneck_analysis.enable_nested_stack:
            # ------------------------------------------------------------
            # Profile workflow bottlenecks with nested stack analysis
            # ------------------------------------------------------------
            nested_bottlenecks = multi_example_call_profiling(all_steps, output_dir=str(self.output_dir))
            workflow_profiling_reports += "\n\n\n" + nested_bottlenecks.textual_report
            workflow_profiling_metrics["nested_stack_analysis"] = nested_bottlenecks.model_dump(
                exclude=["textual_report"])
            logger.info("Nested stack analysis complete")

        if self.profile_config.concurrency_spike_analysis.enable:
            # ------------------------------------------------------------
            # Profile concurrency spikes
            # ------------------------------------------------------------
            concurrency_metrics = concurrency_spike_analysis(
                all_steps, self.profile_config.concurrency_spike_analysis.spike_threshold)
            workflow_profiling_reports += "\n\n\n" + concurrency_metrics.textual_report
            workflow_profiling_metrics["concurrency_spike_analysis"] = concurrency_metrics.model_dump(
                exclude=["textual_report"])
            logger.info("Concurrency spike analysis complete")

        if self.profile_config.prefix_span_analysis.enable:
            # ------------------------------------------------------------
            # Profile prefix span analysis
            # ------------------------------------------------------------
            prefix_list = []
            if (self.profile_config.prefix_span_analysis.chain_with_common_prefixes
                    and "common_prefixes" in inference_optimization_results):
                logger.info("Using common prefixes for prefix span analysis")
                for _, llm_data in inference_optimization_results["common_prefixes"].items():
                    for prefix_data in llm_data["prefix_info"]:
                        prefix_list.append(prefix_data["prefix"])

            prefix_span_analysis = prefixspan_subworkflow_with_text(
                all_steps,
                **self.profile_config.prefix_span_analysis.model_dump(exclude=["enable", "chain_with_common_prefixes"]),
                prefix_list=prefix_list)

            workflow_profiling_reports += "\n\n\n" + prefix_span_analysis.textual_report
            workflow_profiling_metrics["prefix_span_analysis"] = prefix_span_analysis.model_dump(
                exclude=["textual_report"])
            logger.info("Prefix span analysis complete")

        if self.write_output and workflow_profiling_reports:
            # Save to text file
            profiling_report_path = os.path.join(self.output_dir, "workflow_profiling_report.txt")
            with open(profiling_report_path, 'w', encoding='utf-8') as f:
                f.write(workflow_profiling_reports)
            logger.info("Wrote workflow profiling report to: %s", profiling_report_path)

        if self.write_output and workflow_profiling_metrics:
            # Save to JSON
            profiling_metrics_path = os.path.join(self.output_dir, "workflow_profiling_metrics.json")
            with open(profiling_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(workflow_profiling_metrics, f, indent=2)
            logger.info("Wrote workflow profiling metrics to: %s", profiling_metrics_path)

        if self.profile_config.token_usage_forecast:
            # ------------------------------------------------------------
            # Fit forecasting model and save
            # ------------------------------------------------------------

            logger.info("Fitting model for forecasting.")
            model_trainer = ModelTrainer()

            try:
                fitted_model = model_trainer.train(all_steps)
                logger.info("Fitted model for forecasting.")
            except Exception as e:
                logger.exception("Fitting model failed. %s", e, exc_info=True)
                return ProfilerResults()

            if self.write_output:
                os.makedirs(self.output_dir, exist_ok=True)

                import pickle
                with open(os.path.join(self.output_dir, "fitted_model.pkl"), 'wb') as f:
                    pickle.dump(fitted_model, f)

            logger.info("Saved fitted model to disk.")

        return ProfilerResults(workflow_runtime_metrics=workflow_runtimes_results, llm_latency_ci=llm_latency_ci)

    # -------------------------------------------------------------------
    # Confidence Intervals / Metrics
    # -------------------------------------------------------------------
    def _compute_workflow_run_time_confidence_intervals(self) -> InferenceMetricsModel:
        """
        Computes 90, 95, 99% confidence intervals for the mean total workflow run time (in seconds).
        The total workflow run time for each request is the difference between the last and first
        event timestamps in usage_stats.
        """
        run_times = []
        for req_data in self.all_steps:
            # Find the min and max event_timestamp
            timestamps = [u.event_timestamp for u in req_data]
            if not timestamps:
                continue

            start_time = min(timestamps)
            end_time = max(timestamps)
            run_times.append(end_time - start_time)

        return self._compute_confidence_intervals(run_times, "Workflow Run Time")

    def _compute_llm_latency_confidence_intervals(self) -> InferenceMetricsModel:
        """
        Computes 90, 95, 99% confidence intervals for the mean LLM latency.
        LLM latency is defined as the difference between an LLM_END event_timestamp and
        the immediately preceding LLM_START event_timestamp, across all usage_stats.
        """
        latencies = []
        for req_data in self.all_steps:

            usage_stats_sorted = sorted(req_data, key=lambda x: x.event_timestamp)

            previous_llm_start_time = None
            for u in usage_stats_sorted:
                event_type = u.event_type.value
                ts = u.event_timestamp
                if event_type == "LLM_START":
                    previous_llm_start_time = ts
                elif event_type == "LLM_END" and previous_llm_start_time is not None:
                    latencies.append(ts - previous_llm_start_time)
                    previous_llm_start_time = None

        return self._compute_confidence_intervals(latencies, "LLM Latency")

    def _compute_throughput_estimates(self) -> InferenceMetricsModel:
        """
        Computes 90, 95, 99% confidence intervals for throughput, defined as:

        | throughput = (total number of requests) / (total time window),

        where total time window is from the earliest usage_stats event across all requests
        to the latest usage_stats event.
        Note: This is a simple approximate measure of overall throughput for the entire run.
        """
        # Gather min timestamp and max timestamp across ALL requests
        all_timestamps = []
        for req_data in self.all_steps:
            for u in req_data:
                all_timestamps.append(u.event_timestamp)

        if not all_timestamps:
            return InferenceMetricsModel()

        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)
        total_time = max_ts - min_ts
        if total_time <= 0:
            # Can't compute a meaningful throughput if time <= 0
            return InferenceMetricsModel()

        total_requests = len(self.all_requests_data)
        # Single estimate of throughput
        throughput_value = total_requests / total_time

        # For confidence intervals of throughput, we do a simplistic assumption:
        # We treat each request's contribution as 1 occurrence, and approximate
        # the distribution as if these arrivals were uniform. This is quite simplified.
        # We can compute a standard error:  SE = sqrt(throughput_value / total_time)
        # However, a more accurate approach might require a different method (e.g., Poisson).
        # We'll do a naive normal approximation here.

        # We'll guess that the standard deviation of #requests is sqrt(N), so stdev_n ~ sqrt(N).
        # stdev_time is quite small though. We'll do a naive approach:
        # We'll treat the throughput as a sample mean with n=total_requests.
        # Then standard error is (throughput_value / sqrt(n)).
        # This is purely heuristic.
        n = total_requests
        if n <= 1:
            return InferenceMetricsModel()

        # A rough standard error for throughput:
        standard_error = throughput_value / math.sqrt(n)

        # Build confidence intervals using z-scores for 90%, 95%, 99%
        intervals = {'n': total_requests, 'mean': throughput_value}
        for confidence, zvalue in \
                [("ninetieth_interval", 1.645), ("ninety_fifth_interval", 1.96), ("ninety_ninth_interval", 2.576)]:
            ci_lower = throughput_value - zvalue * standard_error
            ci_upper = throughput_value + zvalue * standard_error
            intervals[confidence] = (max(ci_lower, 0.0), ci_upper)

        return InferenceMetricsModel(**intervals)

    def _compute_confidence_intervals(self, data: list[float], metric_name: str) -> InferenceMetricsModel:
        """
        Helper to compute 90, 95, 99 % confidence intervals **and** the empirical
        90th/95th/99th percentiles (p90/p95/p99) for the mean of a dataset.
        Uses a z-score from the normal approximation for large samples.

        Returns a dict like::

            {
              'ninetieth_interval': (lower, upper),
              'ninety_fifth_interval': (lower, upper),
              'ninety_ninth_interval': (lower, upper),
            }
        """
        if not data:
            logger.warning("No data points for %s, cannot compute intervals.", metric_name)
            return InferenceMetricsModel()

        n = len(data)
        mean_val = statistics.mean(data)
        if n <= 1:
            return InferenceMetricsModel(
                n=n,
                mean=mean_val,
                ninetieth_interval=(mean_val, mean_val),
                ninety_fifth_interval=(mean_val, mean_val),
                ninety_ninth_interval=(mean_val, mean_val),
                p90=mean_val,
                p95=mean_val,
                p99=mean_val,
            )

        stdev_val = statistics.pstdev(data)  # population stdev or use stdev for sample
        # standard error
        se = stdev_val / math.sqrt(n)

        intervals = {}
        for confidence, zvalue in \
                [("ninetieth_interval", 1.645), ("ninety_fifth_interval", 1.96), ("ninety_ninth_interval", 2.576)]:
            margin = zvalue * se
            lower = mean_val - margin
            upper = mean_val + margin
            intervals[confidence] = (lower, upper)

        # Optionally, store more info
        intervals["n"] = n
        intervals["mean"] = mean_val

        # ------------------------------------------------------------------
        # Percentiles
        # ------------------------------------------------------------------
        sorted_data = sorted(data)

        def _percentile(arr: list[float], pct: float) -> float:
            """
            Linear interpolation between closest ranks.
            pct is given from 0‑100 (e.g. 90 for p90).
            """
            if not arr:
                return 0.0
            k = (len(arr) - 1) * (pct / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return arr[int(k)]
            return arr[f] + (arr[c] - arr[f]) * (k - f)

        p90_val = _percentile(sorted_data, 90)
        p95_val = _percentile(sorted_data, 95)
        p99_val = _percentile(sorted_data, 99)

        intervals["p90"] = p90_val
        intervals["p95"] = p95_val
        intervals["p99"] = p99_val

        return InferenceMetricsModel(**intervals)
