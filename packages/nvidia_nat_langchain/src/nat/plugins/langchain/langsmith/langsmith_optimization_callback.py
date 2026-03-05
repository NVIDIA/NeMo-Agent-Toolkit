# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import re
from typing import Any

import langsmith

from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult

from .langsmith_evaluation_callback import _LS_RETRY_DELAY_S
from .langsmith_evaluation_callback import _backfill_feedback_for_unlinked_items
from .langsmith_evaluation_callback import _eager_link_run_to_item
from .langsmith_evaluation_callback import _estimate_indexing_time
from .langsmith_evaluation_callback import _humanize_dataset_name
from .langsmith_evaluation_callback import _match_and_link_otel_runs
from .langsmith_evaluation_callback import _retry_unlinked_references
from .langsmith_evaluation_callback import _span_id_to_langsmith_run_id

logger = logging.getLogger(__name__)


class LangSmithOptimizationCallback:
    """Per-trial experiment projects with OTEL trace linking and prompt management.

    Each optimizer trial gets its own experiment project linked to a shared dataset.
    OTEL traces are routed to per-trial projects via get_trial_project_name(), which
    also pre-creates the project with reference_dataset_id. After eval, OTEL runs are
    retroactively linked to dataset examples with feedback and parameter metadata.
    """

    needs_root_span_ids = True

    def __init__(self, *, project: str, experiment_prefix: str = "NAT", dataset_name: str | None = None) -> None:
        self._client = langsmith.Client()
        self._project = project
        self._experiment_prefix = experiment_prefix
        self._dataset_name_hint = dataset_name
        self._dataset_id: str | None = None
        self._dataset_name: str | None = None
        self._run_number: int | None = None
        self._example_ids: dict[Any, str] = {}
        self._prompt_commit_urls: dict[tuple[str, int], str] = {}
        self._prompt_repo_names: dict[str, str] = {}
        self._prompt_trial_counter: int = 0
        self._prompt_param_names: list[str] = []

    def set_prompt_param_names(self, names: list[str]) -> None:
        self._prompt_param_names = list(names)

    # ------------------------------------------------------------------ #
    # Run numbering
    # ------------------------------------------------------------------ #

    def _build_base_name(self) -> str:
        """Build the base name used for datasets and run numbering.

        Format: ``Optimization Benchmark (<dataset>) (<project>)``
        """
        ds_label = self._dataset_name_hint or "eval"
        pretty_ds = _humanize_dataset_name(ds_label)
        return f"Optimization Benchmark ({pretty_ds}) ({self._project})"

    def _get_run_number(self) -> int:
        """Get the run number for this optimization execution (cached)."""
        if self._run_number is not None:
            return self._run_number
        base_name = self._build_base_name()
        run_pattern = re.compile(re.escape(base_name) + r" \(Run #(\d+)\)")
        max_run = 0
        for ds in self._client.list_datasets(dataset_name_contains=base_name, ):
            match = run_pattern.match(ds.name)
            if match:
                max_run = max(max_run, int(match.group(1)))
        self._run_number = max_run + 1
        return self._run_number

    # ------------------------------------------------------------------ #
    # Per-trial project management
    # ------------------------------------------------------------------ #

    def get_trial_project_name(self, trial_number: int) -> str:
        """Return the per-trial OTEL project name and pre-create it as an experiment.

        Called by the parameter/prompt optimizer BEFORE the eval run starts.
        Pre-creates the project with reference_dataset_id so OTEL traces land
        in an experiment project (visible in Datasets & Experiments UI).
        """
        run_num = self._get_run_number()
        base_name = self._build_base_name()
        trial_project = f"{base_name} (Run #{run_num}, Trial {trial_number + 1})"

        # Pre-create as experiment if dataset exists
        if self._dataset_id:
            try:
                self._client.create_project(
                    trial_project,
                    reference_dataset_id=self._dataset_id,
                    description=f"Trial {trial_number + 1}",
                )
            except langsmith.utils.LangSmithConflictError:
                pass  # Already exists from a previous attempt

        return trial_project

    # ------------------------------------------------------------------ #
    # Dataset management
    # ------------------------------------------------------------------ #

    def _create_dataset_with_examples(
        self,
        items: list[tuple[str, str, str]],
    ) -> None:
        """Create the LangSmith dataset and populate it with examples.

        Args:
            items: List of ``(item_id, question, expected)`` tuples.
        """
        if self._dataset_id is not None:
            return
        run_num = self._get_run_number()
        base_name = self._build_base_name()
        dataset_name = f"{base_name} (Run #{run_num})"
        ds = self._client.create_dataset(dataset_name=dataset_name, description="NAT optimizer eval dataset")
        self._dataset_id = str(ds.id)
        self._dataset_name = dataset_name
        for item_id, question, expected in items:
            example = self._client.create_example(
                inputs={
                    "nat_item_id": item_id, "question": question
                },
                outputs={"expected": expected},
                dataset_id=self._dataset_id,
            )
            self._example_ids[item_id] = str(example.id)
        logger.info("Created LangSmith dataset '%s' with %d examples", dataset_name, len(items))

    def _ensure_dataset(self, eval_result: Any) -> None:
        """Create the dataset for this optimization run (once)."""
        self._create_dataset_with_examples([(str(item.item_id), str(item.input_obj), str(item.expected_output))
                                            for item in eval_result.items])

    def pre_create_experiment(self, dataset_items: list) -> None:
        """Create the dataset upfront (before any trials run).

        Must be called BEFORE get_trial_project_name() so the dataset exists
        when per-trial projects are pre-created with reference_dataset_id.
        Accepts list[EvalInputItem] from the eval framework.
        """
        self._create_dataset_with_examples([(
            str(item.id),
            str(item.input_obj) if item.input_obj else "",
            str(item.expected_output_obj) if item.expected_output_obj else "",
        ) for item in dataset_items])

    # ------------------------------------------------------------------ #
    # OTEL run linking (per-trial project)
    # ------------------------------------------------------------------ #

    # Retry budget scaling for substring matching (Phase 2).
    _LS_SAFETY_MULTIPLIER: float = 3.0
    _LS_MIN_RETRIES: int = 10
    _LS_MAX_RETRIES: int = 60
    _LS_WARN_ITEM_THRESHOLD: int = 5000

    @classmethod
    def _estimate_retry_budget(cls, expected_count: int) -> tuple[int, float]:
        """Estimate the retry budget for OTEL run linking based on dataset size.

        Uses the shared indexing constants from ``langsmith_evaluation_callback``
        (pipeline latency, throughput, retry delay) with a safety multiplier
        to scale the retry window proportionally.

        Formula::

            indexing_time = pipeline_latency + (expected_count / throughput)
            total_budget  = indexing_time × safety_multiplier
            max_retries   = clamp(total_budget / retry_delay, min=10, max=60)

        ========== ============= ============= ============ =============
        Items      Indexing Est.  ×3 Safety     Max Retries  Total Budget
        ========== ============= ============= ============ =============
        5          10.5 s        31.5 s        10 (floor)   100 s
        150        25.0 s        75.0 s        10 (floor)   100 s
        600        70.0 s        210.0 s       21           210 s
        5 000      510.0 s       1 530.0 s     60 (cap)     600 s
        ========== ============= ============= ============ =============

        .. warning::
            Datasets above 5 000 items per trial may exceed the maximum
            retry window (600 s). Some runs may not be linked in the
            LangSmith UI, although all traces will have been delivered.

        Returns:
            (max_retries, retry_delay) tuple for ``_match_and_link_otel_runs``.
        """
        if expected_count > cls._LS_WARN_ITEM_THRESHOLD:
            logger.warning(
                "Dataset has %d items (> %d). LangSmith may not index all "
                "runs within the retry window — some experiments may appear "
                "incomplete in the UI despite all traces being delivered.",
                expected_count,
                cls._LS_WARN_ITEM_THRESHOLD,
            )

        indexing_time = _estimate_indexing_time(expected_count)
        total_budget = indexing_time * cls._LS_SAFETY_MULTIPLIER
        retries = int(total_budget / _LS_RETRY_DELAY_S)
        retries = max(cls._LS_MIN_RETRIES, min(cls._LS_MAX_RETRIES, retries))

        return retries, _LS_RETRY_DELAY_S

    def _link_otel_runs(
        self,
        trial_number: int,
        eval_result: Any,
        parameters: dict[str, Any] | None = None,
        prompt_commit_tags: dict[str, str] | None = None,
    ) -> None:
        """Link OTEL runs in the trial's project to dataset examples and attach feedback."""
        trial_project = self.get_trial_project_name(trial_number)
        formatted_params = self._format_params(parameters or {})

        # Include prompt commit tags in experiment metadata
        if prompt_commit_tags:
            for param_name, tag in prompt_commit_tags.items():
                key = f"prompt_tag_{param_name.replace('.', '_')}"
                formatted_params[key] = tag
        else:
            for param_name in self._prompt_param_names:
                key = f"prompt_tag_{param_name.replace('.', '_')}"
                formatted_params[key] = "original"

        # Update experiment metadata with parameters
        if formatted_params:
            try:
                self._client.update_project(
                    self._client.read_project(project_name=trial_project).id,
                    metadata=formatted_params,
                )
            except Exception:
                logger.debug("Could not update project metadata for '%s'", trial_project, exc_info=True)

        expected_count = len(eval_result.items)

        # Phase 1: Eager linking for items with pre-generated span_ids.
        eagerly_linked = 0
        fallback_items = []
        for item in eval_result.items:
            root_span_id = getattr(item, 'root_span_id', None)
            if isinstance(root_span_id, int):
                run_id = _span_id_to_langsmith_run_id(root_span_id)
                if _eager_link_run_to_item(self._client, run_id, item, self._example_ids):
                    eagerly_linked += 1
                else:
                    fallback_items.append(item)
            else:
                fallback_items.append(item)

        # Phase 2: Fallback to substring matching for remaining items.
        if fallback_items:
            from nat.plugins.eval.eval_callbacks import EvalResult
            max_retries, retry_delay = self._estimate_retry_budget(len(fallback_items))
            matched = _match_and_link_otel_runs(
                client=self._client,
                project_name=trial_project,
                eval_result=EvalResult(metric_scores=eval_result.metric_scores, items=fallback_items),
                example_ids=self._example_ids,
                expected_count=len(fallback_items),
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            eagerly_linked += matched

        # Phase 3a: Retry reference_example_id for ALL items.
        # update_run() can return 200 OK before the run is fully indexed,
        # silently dropping the reference_example_id.
        retried = _retry_unlinked_references(
            client=self._client,
            project_name=trial_project,
            items=eval_result.items,
            example_ids=self._example_ids,
        )

        # Phase 3b: Attach fallback feedback only for items that failed
        # eager linking AND substring matching (avoid duplicate feedback).
        fallback_item_count = _backfill_feedback_for_unlinked_items(
            client=self._client,
            project_name=trial_project,
            items=fallback_items,
            example_ids=self._example_ids,
        )

        logger.info("Linked %d/%d OTEL runs for trial %d in '%s'",
                    eagerly_linked,
                    expected_count,
                    trial_number + 1,
                    trial_project)
        if retried:
            logger.info("Retried reference linking for %d items in trial %d '%s'",
                        retried,
                        trial_number + 1,
                        trial_project)
        if fallback_item_count:
            logger.info("Recorded fallback feedback for %d unlinked items in '%s'", fallback_item_count, trial_project)

    # ------------------------------------------------------------------ #
    # Parameter formatting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_params(parameters: dict[str, Any]) -> dict[str, Any]:
        """Sanitize parameter names (dots->underscores) and round floats."""
        formatted = {}
        for k, v in parameters.items():
            key = k.replace(".", "_")
            if isinstance(v, float):
                v = round(v, 4)
            formatted[key] = v
        return formatted

    # ------------------------------------------------------------------ #
    # Prompt management
    # ------------------------------------------------------------------ #

    @staticmethod
    def _humanize_param_name(param_name: str) -> str:
        """Convert 'functions.email_phishing_analyzer.prompt' to 'Email Phishing Analyzer Prompt'."""
        name = param_name
        for prefix in ("functions.", "llms.", "workflow."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return _humanize_dataset_name(name)

    def _get_prompt_repo_name(self, param_name: str) -> str:
        """Get or create a unique prompt repo name for this optimization run.

        Format: ``<project>-<param>-run-<N>``
        e.g. ``aiq-shallow-researcher-full-optimization-system-prompt-run-1``
        """
        if param_name in self._prompt_repo_names:
            return self._prompt_repo_names[param_name]

        # Sanitize param name
        param_slug = param_name
        for prefix in ("functions.", "llms.", "workflow."):
            if param_slug.startswith(prefix):
                param_slug = param_slug[len(prefix):]
                break
        param_slug = param_slug.lower().replace(".", "-").replace("_", "-")

        # Prefix with project name
        project_slug = (self._project.lower().replace(" ", "-").replace("_", "-"))
        base = f"{project_slug}-{param_slug}"

        pattern = re.compile(re.escape(base) + r"-run-(\d+)$")
        max_run = 0
        try:
            for prompt in self._client.list_prompts(query=base).repos:
                match = pattern.match(prompt.repo_handle)
                if match:
                    max_run = max(max_run, int(match.group(1)))
        except Exception:
            logger.debug("Could not list existing prompts for '%s'", base, exc_info=True)
        repo_name = f"{base}-run-{max_run + 1}"
        self._prompt_repo_names[param_name] = repo_name
        return repo_name

    VALID_TEMPLATE_FORMATS = frozenset({"f-string", "jinja2", "mustache"})

    # Jinja2-only markers (never appear in mustache or f-string)
    _JINJA2_MARKERS = ("{%", "{#")
    # Jinja2 constructs inside {{ }} (e.g. {{ x if y }}, {{ x | filter }})
    _JINJA2_EXPR_KEYWORDS = ("| ", " if ", " else ", " for ")

    # Mustache-only markers: {{#section}}, {{/section}}, {{>partial}}, {{^inverted}}
    _MUSTACHE_MARKERS = ("{{#", "{{/", "{{>", "{{^")

    @classmethod
    def _detect_template_format(cls, text: str) -> str:
        """Auto-detect template format from prompt content.

        Detection priority (first match wins):
            1. Jinja2 block/comment tags (``{%``, ``{#``) → ``"jinja2"``
            2. Mustache section markers (``{{#``, ``{{/``, ``{{>``, ``{{^``) → ``"mustache"``
            3. Jinja2 expression keywords inside ``{{ }}``
               (pipes, conditionals, loops) → ``"jinja2"``
            4. Plain ``{{ }}`` without keywords → ``"jinja2"``
               (ambiguous with mustache, but Jinja2 is far more common
               in Python/LangChain prompts)
            5. No curly-brace templating detected → ``"f-string"``

        Used as a fallback when ``SearchSpace.prompt_format`` is not
        explicitly set.
        """
        # 1. Unambiguous Jinja2: block tags {% %}
        if "{%" in text:
            return "jinja2"

        # 2. Mustache section/partial markers: {{#, {{/, {{>, {{^
        #    Check BEFORE Jinja2 comments because {# is a substring of {{#
        if any(marker in text for marker in cls._MUSTACHE_MARKERS):
            return "mustache"

        # 3. Jinja2 comments {# #} (now safe — mustache already checked)
        if "{#" in text:
            return "jinja2"

        # 4. {{ }} present — disambiguate via expression keywords
        if "{{" in text:
            if any(kw in text for kw in cls._JINJA2_EXPR_KEYWORDS):
                return "jinja2"
            # Plain {{ }} — default to jinja2 (more common in Python)
            return "jinja2"

        # 5. No template markers found
        return "f-string"

    @classmethod
    def _validate_template_format(cls, fmt: str) -> str:
        """Validate that a template format string is supported.

        Raises ``ValueError`` with the list of valid options if not.
        """
        if fmt not in cls.VALID_TEMPLATE_FORMATS:
            raise ValueError(f"Invalid template_format '{fmt}'. "
                             f"Must be one of: {sorted(cls.VALID_TEMPLATE_FORMATS)}")
        return fmt

    def _resolve_template_format(
        self,
        param_name: str,
        prompt_text: str,
        result: Any,
    ) -> str:
        """Resolve the LangChain template_format for a prompt.

        Priority:
            1. Explicit ``prompt_formats`` from TrialResult
               (set via ``SearchSpace.prompt_format``)
            2. Auto-detection from prompt content

        Supported values: ``"f-string"``, ``"jinja2"``, ``"mustache"``.
        """
        # Check explicit format from SearchSpace → TrialResult
        if hasattr(result, "prompt_formats") and result.prompt_formats:
            fmt = result.prompt_formats.get(param_name)
            if fmt:
                return self._validate_template_format(fmt)
        # Fallback to auto-detection
        return self._detect_template_format(prompt_text)

    def _push_prompt(self, result: Any, commit_tags: list[str] | None = None) -> dict[str, str]:
        """Push a trial's prompts to LangSmith with full metadata."""
        from langchain_core.prompts import ChatPromptTemplate

        if not result.prompts:
            return {}

        repo_tags: list[str] = []
        if self._dataset_name:
            repo_tags.append(f"dataset:{self._dataset_name}")
        elif self._dataset_name_hint:
            repo_tags.append(f"dataset:{self._dataset_name_hint}")

        prompt_urls: dict[str, str] = {}
        for param_name, prompt_text in result.prompts.items():
            repo_name = self._get_prompt_repo_name(param_name)
            try:
                metadata: dict[str, Any] = {
                    "trial_number": result.trial_number + 1,
                    "param_name": param_name,
                }
                if self._dataset_name:
                    metadata["dataset"] = self._dataset_name
                elif self._dataset_name_hint:
                    metadata["dataset"] = self._dataset_name_hint
                if result.metric_scores:
                    metadata["metrics"] = {k: round(v, 4) for k, v in result.metric_scores.items()}
                if result.parameters:
                    metadata["parameters"] = self._format_params(result.parameters)
                if result.is_best:
                    metadata["is_best"] = True

                template_format = self._resolve_template_format(
                    param_name,
                    prompt_text,
                    result,
                )
                template = ChatPromptTemplate.from_template(
                    prompt_text,
                    template_format=template_format,
                    metadata=metadata,
                )
                pretty_name = self._humanize_param_name(param_name)
                url = self._client.push_prompt(repo_name,
                                               object=template,
                                               tags=repo_tags,
                                               commit_tags=commit_tags,
                                               description=f"Optimized prompt for {pretty_name}")
                prompt_urls[param_name] = url
                self._prompt_commit_urls[(param_name, result.trial_number)] = url
                logger.debug("Pushed prompt '%s' trial %d to %s", param_name, result.trial_number + 1, url)
            except langsmith.utils.LangSmithConflictError:
                prompt_urls[param_name] = repo_name
                logger.debug("Prompt '%s' unchanged for trial %d", param_name, result.trial_number + 1)
                # Retroactively tag the existing latest commit
                if commit_tags:
                    try:
                        response = self._client.request_with_retries("GET",
                                                                     f"/commits/-/{repo_name}/",
                                                                     params={
                                                                         "limit": 1, "offset": 0
                                                                     })
                        commits = response.json().get("commits", [])
                        if commits:
                            self._client._create_commit_tags(f"-/{repo_name}", commits[0]["id"], commit_tags)
                    except Exception:
                        logger.debug("Could not tag existing commit for '%s'", param_name, exc_info=True)
            except Exception:
                logger.warning("Failed to push prompt '%s' to LangSmith", param_name, exc_info=True)
                prompt_urls[param_name] = prompt_text
        return prompt_urls

    # ------------------------------------------------------------------ #
    # Callback interface
    # ------------------------------------------------------------------ #

    def on_trial_end(self, result: TrialResult) -> None:
        prompt_commit_tags: dict[str, str] = {}

        # Push prompts with commit tags (GA trials only — numeric trials don't have prompts)
        if result.prompts:
            self._prompt_trial_counter += 1
            commit_tag = f"trial-{self._prompt_trial_counter}"
            self._push_prompt(result, commit_tags=[commit_tag])
            for param_name in result.prompts:
                prompt_commit_tags[param_name] = commit_tag

        # Link OTEL runs in the per-trial project to dataset examples
        if result.eval_result and hasattr(result.eval_result, 'items') and result.eval_result.items:
            self._ensure_dataset(result.eval_result)
            self._link_otel_runs(result.trial_number,
                                 result.eval_result,
                                 result.parameters,
                                 prompt_commit_tags=prompt_commit_tags)

    def on_study_end(self, *, best_trial: TrialResult, total_trials: int) -> None:
        # Tag the best trial's prompt commit with "best" by re-pushing it.
        # Re-push ensures the correct commit is tagged even if it's not the
        # latest (e.g., best=trial 3 but last pushed=trial 9).
        if best_trial.prompts:
            self._push_prompt(best_trial, commit_tags=["best"])
        self._client.flush()
        logger.info("Optimization study complete (%d trials). Best: trial %d",
                    total_trials,
                    best_trial.trial_number + 1)
