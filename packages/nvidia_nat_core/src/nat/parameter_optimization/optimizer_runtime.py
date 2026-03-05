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

from pydantic import BaseModel

from nat.data_models.optimizer import OptimizerRunConfig
from nat.experimental.decorators.experimental_warning_decorator import experimental
from nat.parameter_optimization.optimizable_utils import walk_optimizables
from nat.parameter_optimization.parameter_optimizer import optimize_parameters
from nat.parameter_optimization.prompt_optimizer import optimize_prompts
from nat.runtime.loader import load_config

logger = logging.getLogger(__name__)


def _build_optimizer_callback_manager(base_cfg):
    """Build optimizer callback manager from registered callbacks matching the tracing config."""
    try:
        from pathlib import Path

        from nat.cli.type_registry import GlobalTypeRegistry
        from nat.observability.utils.tracing_utils import get_tracing_configs
        from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallbackManager

        tracing = get_tracing_configs(base_cfg)
        if not tracing:
            return None

        # Extract dataset name from eval config (runtime concern, not plugin-specific)
        opt_dataset_name = None
        try:
            ds_cfg = base_cfg.eval.general.dataset
            file_path = getattr(ds_cfg, 'file_path', None)
            if file_path:
                opt_dataset_name = Path(file_path).stem
        except Exception:
            logger.debug("Could not extract dataset name from config", exc_info=True)

        manager = OptimizerCallbackManager()
        registry = GlobalTypeRegistry.get()

        for _name, exporter_config in tracing.items():
            try:
                registered = registry.get_optimizer_callback(type(exporter_config))
            except KeyError:
                continue
            cb = registered.factory_fn(exporter_config, dataset_name=opt_dataset_name)
            manager.register(cb)

        if not manager.has_callbacks:
            return None

        # Pre-create experiments for callbacks that support it (duck-typed).
        # Load raw dataset items from the eval dataset file as EvalInputItem objects.
        try:
            import csv
            import json

            from nat.plugins.eval.evaluator.evaluator_model import EvalInputItem

            ds_cfg = base_cfg.eval.general.dataset
            file_path = getattr(ds_cfg, 'file_path', None)
            if file_path:
                fp = Path(file_path)
                q_key = getattr(getattr(ds_cfg, 'structure', None), 'question_key', 'question')
                a_key = getattr(getattr(ds_cfg, 'structure', None), 'answer_key', 'expected_output')
                id_key = getattr(ds_cfg, 'id_key', None)
                dataset_items: list[EvalInputItem] = []
                if fp.suffix == '.csv':
                    with open(fp, encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    for row in rows:
                        item_id = row.get(id_key, row.get(q_key, "")) if id_key else row.get(q_key, "")
                        dataset_items.append(
                            EvalInputItem(
                                id=item_id,
                                input_obj=row.get(q_key, ""),
                                expected_output_obj=row.get(a_key, ""),
                                full_dataset_entry=row,
                            ))
                elif fp.suffix == '.json':
                    with open(fp, encoding="utf-8") as f:
                        raw_items = json.load(f)
                    for entry in raw_items:
                        if isinstance(entry, dict):
                            item_id = entry.get("id", entry.get(q_key, ""))
                            dataset_items.append(
                                EvalInputItem(
                                    id=item_id,
                                    input_obj=entry.get(q_key, ""),
                                    expected_output_obj=entry.get(a_key, ""),
                                    full_dataset_entry=entry,
                                ))
                if dataset_items:
                    manager.pre_create_experiment(dataset_items)
        except Exception:
            logger.debug("Could not pre-create experiment", exc_info=True)

        return manager
    except Exception:
        logger.debug("Optimizer callback not available", exc_info=True)
        return None


@experimental(feature_name="Optimizer")
async def optimize_config(opt_run_config: OptimizerRunConfig):
    """Entry-point called by the CLI or runtime."""
    # ---------------- 1. load / normalise ---------------- #
    if not isinstance(opt_run_config.config_file, BaseModel):
        from nat.data_models.config import Config  # guarded import
        base_cfg: Config = load_config(config_file=opt_run_config.config_file)
    else:
        base_cfg = opt_run_config.config_file  # already validated

    # Build optimizer callback manager from registered callbacks matching the tracing config
    callback_manager = _build_optimizer_callback_manager(base_cfg)

    # ---------------- 2. discover search space ----------- #
    full_space = walk_optimizables(base_cfg)
    if not full_space:
        logger.warning("No optimizable parameters found in the configuration. "
                       "Skipping optimization.")
        return base_cfg

    # Tell the callback manager which params are prompts (for tagging numeric trials as "original")
    if callback_manager:
        prompt_param_names = [k for k, v in full_space.items() if v.is_prompt]
        callback_manager.set_prompt_param_names(prompt_param_names)

    # ---------------- 3. numeric / enum tuning ----------- #
    tuned_cfg = base_cfg
    best_numeric_params: dict = {}
    _numeric_trial_count = 0
    if base_cfg.optimizer.numeric.enabled:
        tuned_cfg, best_numeric_params, _numeric_trial_count = optimize_parameters(
            base_cfg=base_cfg,
            full_space=full_space,
            optimizer_config=base_cfg.optimizer,
            opt_run_config=opt_run_config,
            callback_manager=callback_manager,
        )

    # ---------------- 4. prompt optimization ------------- #
    if base_cfg.optimizer.prompt.enabled:
        await optimize_prompts(
            base_cfg=tuned_cfg,
            full_space=full_space,
            optimizer_config=base_cfg.optimizer,
            opt_run_config=opt_run_config,
            callback_manager=callback_manager,
            trial_number_offset=_numeric_trial_count,
            frozen_params=best_numeric_params,
        )

    logger.info("All optimization phases complete.")
    return tuned_cfg
