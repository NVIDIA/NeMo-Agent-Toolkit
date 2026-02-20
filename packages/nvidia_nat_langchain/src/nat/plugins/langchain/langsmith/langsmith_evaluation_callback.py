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
import time
from typing import Any

import langsmith

from nat.eval.eval_callbacks import EvalResult

logger = logging.getLogger(__name__)


def _humanize_dataset_name(name: str) -> str:
    """Convert a raw dataset name to title case (underscores and hyphens become spaces)."""
    return name.replace("_", " ").replace("-", " ").title()


def _span_id_to_langsmith_run_id(span_id: int) -> str:
    """Derive LangSmith run_id from OTEL span_id.

    LangSmith deterministically maps OTEL span_ids to run UUIDs:
    the first 8 bytes are zeroed, the last 8 bytes are the span_id.
    """
    hex_str = format(span_id, "016x")
    return f"00000000-0000-0000-{hex_str[:4]}-{hex_str[4:]}"


def _eager_link_run_to_item(
    client: Any,
    run_id: str,
    item: Any,
    example_ids: dict[Any, str],
) -> bool:
    """Link a run to an eval item using a pre-computed run_id (no polling required).

    Uses the deterministic span_id-to-run_id mapping to call update_run()
    immediately via LangSmith's write path, bypassing the indexing delay.
    Returns True if the linking succeeded.
    """
    example_id = example_ids.get(str(item.item_id))
    if not example_id:
        return False
    try:
        client.update_run(run_id, reference_example_id=example_id)
    except Exception:
        logger.debug("Eager link failed for run %s (item %s) to example %s",
                     run_id,
                     item.item_id,
                     example_id,
                     exc_info=True)
        return False
    for metric_name, score in item.scores.items():
        try:
            client.create_feedback(
                run_id=run_id,
                key=metric_name,
                score=score,
                comment=str(item.reasoning.get(metric_name, "")),
            )
        except Exception:
            logger.debug("Could not attach feedback %s to run %s", metric_name, run_id)
    return True


def _get_run_input_str(run: Any) -> str:
    """Extract a comparable input string from an OTEL run.

    OTEL spans store inputs in various formats depending on the framework.
    This normalizes to a plain string for comparison.
    """
    if isinstance(run.inputs, dict):
        return str(run.inputs.get("input", ""))
    return str(run.inputs or "")


def _link_run_to_item(client: Any, run: Any, item: Any, example_ids: dict[Any, str]) -> bool:
    """Link a single OTEL run to an eval item in LangSmith.

    Sets reference_example_id on the run (links it to the dataset example)
    and attaches evaluator scores as feedback. Returns True if successful.
    """
    example_id = example_ids.get(str(item.item_id))
    if not example_id:
        return False
    try:
        client.update_run(run.id, reference_example_id=example_id)
    except Exception:
        logger.debug("Could not link run %s to example %s", run.id, example_id)
        return False
    for metric_name, score in item.scores.items():
        try:
            client.create_feedback(
                run_id=run.id,
                key=metric_name,
                score=score,
                comment=str(item.reasoning.get(metric_name, "")),
            )
        except Exception:
            logger.debug("Could not attach feedback %s to run %s", metric_name, run.id)
    return True


def _normalize_input(text: str) -> str:
    """Strip JSON quoting and whitespace for robust comparison."""
    text = text.strip()
    # Remove outer JSON string quotes (OTEL serializes plain strings as '"text"')
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            import json as _json
            text = _json.loads(text)
        except (ValueError, TypeError):
            pass
    return text.strip()


def _match_and_link_otel_runs(
    *,
    client: Any,
    project_name: str,
    eval_result: Any,
    example_ids: dict[Any, str],
    expected_count: int,
    max_retries: int = 10,
    retry_delay: float = 10.0,
    processed_run_ids: set[str] | None = None,
) -> int:
    """Match OTEL runs to eval items by substring and link them in LangSmith.

    OTEL traces are exported asynchronously in batches, so they may not all be
    available immediately. This function retries up to max_retries times, waiting
    retry_delay seconds between attempts.

    On each attempt, fetches all root runs in the project and matches them to
    eval items using substring comparison: if the eval item's input text appears
    anywhere in the OTEL run's input (or vice versa), they match. Matched runs
    get reference_example_id set and evaluator scores attached as feedback.

    Returns the number of successfully matched and linked runs.
    """
    if processed_run_ids is None:
        processed_run_ids = set()

    # Track which items have been successfully linked across ALL retry attempts.
    # This prevents an already-matched item from greedily stealing a run that
    # should go to a still-unmatched item on later retries.
    matched_item_ids: set[str] = set()

    total_matched = 0
    for attempt in range(1, max_retries + 1):
        # Wait for OTEL exporter to flush traces to LangSmith
        if attempt > 1:
            time.sleep(retry_delay)

        # Fetch root-level workflow runs in the project.
        # Filter to runs named '<workflow>' to exclude evaluator LLM calls
        # (e.g. judge model invocations from FreshQA) that land in the same
        # trial project as unparented root spans during GA optimization.
        try:
            all_root_runs = list(client.list_runs(project_name=project_name, is_root=True))
            otel_runs = [r for r in all_root_runs if getattr(r, "name", None) == "<workflow>"]
        except Exception:
            logger.warning("Could not query OTEL runs in '%s' (attempt %d/%d)",
                           project_name,
                           attempt,
                           max_retries,
                           exc_info=True)
            continue

        # Filter to runs we haven't already processed
        new_runs = [r for r in otel_runs if str(r.id) not in processed_run_ids]
        if not new_runs:
            if total_matched >= expected_count:
                break
            logger.debug("No new runs found (%d matched so far, need %d, attempt %d/%d)",
                         total_matched,
                         expected_count,
                         attempt,
                         max_retries)
            continue

        # Build mutable pools — only include items NOT yet matched
        unmatched_runs = {str(r.id): r for r in new_runs}
        unmatched_items = {
            str(item.item_id): item
            for item in eval_result.items if str(item.item_id) not in matched_item_ids
        }

        # Match by exact normalized comparison first (most reliable),
        # then fall back to substring for wrapped inputs.
        matches: list[tuple[str, str]] = []
        for run_id, run in list(unmatched_runs.items()):
            run_input_raw = _get_run_input_str(run)
            run_input_norm = _normalize_input(run_input_raw)

            best_match: str | None = None
            best_len = -1

            for item_id, item in list(unmatched_items.items()):
                item_input = str(item.input_obj).strip()

                # Exact match (after normalization)
                if item_input == run_input_norm:
                    best_match = item_id
                    break

                # Substring match — prefer longest match to avoid
                # short questions stealing runs from longer ones
                if item_input in run_input_norm or run_input_norm in item_input:
                    if len(item_input) > best_len:
                        best_len = len(item_input)
                        best_match = item_id

            if best_match is not None:
                matches.append((run_id, best_match))
                unmatched_items.pop(best_match, None)

        # Link matched pairs and attach feedback
        matched_this_round = 0
        for run_id, item_id in matches:
            run = unmatched_runs.pop(run_id, None)
            item = next((i for i in eval_result.items if str(i.item_id) == item_id), None)
            if run and item and _link_run_to_item(client, run, item, example_ids):
                matched_this_round += 1
                matched_item_ids.add(item_id)
            processed_run_ids.add(run_id)

        # Mark remaining unmatched runs as processed so we don't re-check them
        for run_id in unmatched_runs:
            processed_run_ids.add(run_id)

        total_matched += matched_this_round
        logger.debug("Attempt %d/%d: matched %d this round, %d/%d total",
                     attempt,
                     max_retries,
                     matched_this_round,
                     total_matched,
                     expected_count)

        if total_matched >= expected_count:
            break

    return total_matched


class LangSmithEvaluationCallback:
    """Links OTEL traces to LangSmith experiments for structured eval result viewing.

    Pre-creates the OTEL project as an experiment (with reference_dataset_id) so
    OTEL traces land in an experiment project. After eval completes, retroactively
    links OTEL runs to dataset examples and attaches evaluator feedback scores.
    """

    needs_root_span_ids = True

    def __init__(self, *, project: str, experiment_prefix: str = "NAT") -> None:
        self._client = langsmith.Client()
        self._project = project
        self._experiment_prefix = experiment_prefix
        self._dataset_id: str | None = None
        self._dataset_name: str | None = None
        self._example_ids: dict[Any, str] = {}  # item_id -> langsmith example UUID

    def get_eval_project_name(self) -> str:
        """Return a unique eval project name with auto-incrementing Run #.

        Called from evaluate.py BEFORE the OTEL exporter starts to set
        the project name on the config. Each eval run gets its own experiment.
        """
        import re
        base = self._project
        pattern = re.compile(re.escape(base) + r" \(Run #(\d+)\)")
        max_run = 0
        for proj in self._client.list_projects():
            match = pattern.match(proj.name)
            if match:
                max_run = max(max_run, int(match.group(1)))
        self._project = f"{base} (Run #{max_run + 1})"
        return self._project

    def on_dataset_loaded(self, *, dataset_name: str, items: list) -> None:
        self._dataset_name = dataset_name
        pretty_name = _humanize_dataset_name(dataset_name)
        ls_dataset_name = f"Benchmark Dataset ({pretty_name})"
        try:
            ds = self._client.create_dataset(dataset_name=ls_dataset_name, description="NAT eval dataset")
            self._dataset_id = str(ds.id)
        except langsmith.utils.LangSmithConflictError:
            existing = self._client.read_dataset(dataset_name=ls_dataset_name)
            self._dataset_id = str(existing.id)
            logger.info("Reusing existing LangSmith dataset: %s", ls_dataset_name)
            # Load existing example IDs so we can link runs to them
            for example in self._client.list_examples(dataset_id=self._dataset_id):
                inputs = example.inputs or {}
                item_id = inputs.get("nat_item_id", str(example.id))
                self._example_ids[str(item_id)] = str(example.id)
            # Still pre-create the OTEL project as experiment (may already exist)
            self._pre_create_experiment_project()
            return

        for item in items:
            item_id = str(item.id)
            question = str(item.input_obj) if item.input_obj else ""
            expected = str(item.expected_output_obj) if item.expected_output_obj else ""
            example = self._client.create_example(
                inputs={
                    "nat_item_id": item_id, "question": question
                },
                outputs={"expected": expected},
                dataset_id=self._dataset_id,
            )
            self._example_ids[item_id] = str(example.id)
        logger.info("Created LangSmith dataset '%s' with %d examples", ls_dataset_name, len(items))

        # Pre-create the OTEL project as an experiment BEFORE the OTEL exporter starts
        self._pre_create_experiment_project()

    def _pre_create_experiment_project(self) -> None:
        """Pre-create the OTEL project with reference_dataset_id so it's an experiment."""
        if not self._dataset_id:
            return
        try:
            self._client.create_project(
                self._project,
                reference_dataset_id=self._dataset_id,
                description=f"Evaluation using {self._experiment_prefix}",
            )
            logger.info("Pre-created experiment project '%s' linked to dataset", self._project)
        except langsmith.utils.LangSmithConflictError:
            logger.debug("Project '%s' already exists", self._project)

    def on_eval_complete(self, result: EvalResult) -> None:
        if not self._dataset_id:
            logger.warning("No dataset_id — skipping LangSmith experiment linking")
            return

        # Phase 1: Eager linking for items with pre-generated span_ids.
        # Derives the LangSmith run_id directly from the OTEL span_id and
        # calls update_run() immediately — no polling or indexing delay.
        eagerly_linked = 0
        fallback_items = []
        for item in result.items:
            if getattr(item, 'root_span_id', None) is not None:
                run_id = _span_id_to_langsmith_run_id(item.root_span_id)
                if _eager_link_run_to_item(self._client, run_id, item, self._example_ids):
                    eagerly_linked += 1
                else:
                    fallback_items.append(item)
            else:
                fallback_items.append(item)

        # Phase 2: Fallback to substring matching for remaining items
        # (e.g. remote workflows, or if eager linking failed).
        if fallback_items:
            logger.info("Falling back to substring matching for %d items", len(fallback_items))
            fallback_result = EvalResult(metric_scores=result.metric_scores, items=fallback_items)
            matched = _match_and_link_otel_runs(
                client=self._client,
                project_name=self._project,
                eval_result=fallback_result,
                example_ids=self._example_ids,
                expected_count=len(fallback_items),
            )
            eagerly_linked += matched

        self._client.flush()
        logger.info("Linked %d/%d OTEL runs to dataset examples in '%s'",
                    eagerly_linked,
                    len(result.items),
                    self._project)
