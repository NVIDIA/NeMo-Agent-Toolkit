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
"""File-based eval callback that writes evaluation output to local files."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from nat.eval.eval_callbacks import EvalResult
    from nat.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


class FileEvalCallback:
    """Eval callback that persists evaluation artifacts to the local filesystem.

    This replaces the direct file I/O previously embedded in ``EvaluationRun``,
    making file output opt-in and enabling eval as a clean Python API.
    """

    def __init__(self) -> None:
        self.workflow_output_file: Path | None = None
        self.atif_workflow_output_file: Path | None = None
        self.evaluator_output_files: list[Path] = []
        self.config_original_file: Path | None = None
        self.config_effective_file: Path | None = None
        self.config_metadata_file: Path | None = None

    def on_dataset_loaded(self, *, dataset_name: str, items: list[EvalInputItem]) -> None:
        pass

    def on_eval_complete(self, result: EvalResult) -> None:
        """Write evaluation artifacts to ``result.output_dir``."""
        output_dir = result.output_dir
        if output_dir is None:
            logger.debug("FileEvalCallback: no output_dir on EvalResult, skipping file export")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_configuration(result, output_dir)
        self._write_workflow_output(result, output_dir)
        self._write_evaluator_outputs(result, output_dir)

    def _write_configuration(self, result: EvalResult, output_dir: Path) -> None:
        """Save original config, effective config, and run metadata."""
        run_config = result.run_config
        effective_config = result.effective_config
        if run_config is None:
            return

        try:
            config_file = run_config.config_file

            config_original_file = output_dir / "config_original.yml"
            if isinstance(config_file, Path):
                if config_file.exists():
                    shutil.copy2(config_file, config_original_file)
                    self.config_original_file = config_original_file
                    logger.info("Original config file copied to %s", config_original_file)
                else:
                    logger.warning("Original config file not found at %s", config_file)
            elif isinstance(config_file, BaseModel):
                config_dict = config_file.model_dump(mode='json')
                with open(config_original_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
                self.config_original_file = config_original_file
                logger.info("Programmatic config saved to %s", config_original_file)

            config_effective_file = output_dir / "config_effective.yml"
            if effective_config is not None:
                effective_config_dict = effective_config.model_dump(mode='json') if effective_config else {}
                with open(config_effective_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(effective_config_dict, f, default_flow_style=False, sort_keys=False)
                self.config_effective_file = config_effective_file
                logger.info("Effective config (with overrides) saved to %s", config_effective_file)
            else:
                logger.warning("Effective config not available, skipping config_effective.yml")

            config_metadata_file = output_dir / "config_metadata.json"
            metadata = self._build_run_metadata(run_config)
            with open(config_metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            self.config_metadata_file = config_metadata_file
            logger.info("Configuration metadata saved to %s", config_metadata_file)

        except Exception:
            logger.exception("Failed to write configuration files")

    @staticmethod
    def _build_run_metadata(run_config: Any) -> dict[str, Any]:
        """Assemble the metadata dict from an ``EvaluationRunConfig``."""
        return {
            "config_file":
                str(run_config.config_file),
            "config_file_type":
                "Path" if isinstance(run_config.config_file, Path) else "BaseModel",
            "overrides": [{
                "path": path, "value": value
            } for path, value in run_config.override] if run_config.override else [],
            "dataset":
                run_config.dataset,
            "result_json_path":
                run_config.result_json_path,
            "skip_workflow":
                run_config.skip_workflow,
            "skip_completed_entries":
                run_config.skip_completed_entries,
            "reps":
                run_config.reps,
            "endpoint":
                run_config.endpoint,
            "endpoint_timeout":
                run_config.endpoint_timeout,
            "adjust_dataset_size":
                run_config.adjust_dataset_size,
            "num_passes":
                run_config.num_passes,
            "export_timeout":
                run_config.export_timeout,
            "user_id":
                run_config.user_id,
            "timestamp":
                datetime.now(tz=UTC).isoformat(),
        }

    def _write_workflow_output(self, result: EvalResult, output_dir: Path) -> None:
        """Write the serialized workflow output JSON."""
        if result.workflow_output_json is not None:
            workflow_output_file = output_dir / "workflow_output.json"
            with open(workflow_output_file, "w", encoding="utf-8") as f:
                f.write(result.workflow_output_json)
            self.workflow_output_file = workflow_output_file
            logger.info("Workflow output written to %s", workflow_output_file)

        if result.atif_workflow_output_json is None:
            return

        atif_workflow_output_file = output_dir / "workflow_output_atif.json"
        with open(atif_workflow_output_file, "w", encoding="utf-8") as f:
            f.write(result.atif_workflow_output_json)
        self.atif_workflow_output_file = atif_workflow_output_file
        logger.info("ATIF workflow output written to %s", atif_workflow_output_file)

    def _write_evaluator_outputs(self, result: EvalResult, output_dir: Path) -> None:
        """Write per-evaluator result files."""
        for evaluator_name, eval_output in result.evaluation_outputs:
            output_file = output_dir / f"{evaluator_name}_output.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output = eval_output.model_dump_json(indent=2)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output)
            self.evaluator_output_files.append(output_file)
            logger.info("Evaluation results written to %s", output_file)
