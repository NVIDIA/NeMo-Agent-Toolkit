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

from collections.abc import Sequence

import numpy as np
import optuna
from optuna._hypervolume import compute_hypervolume
from optuna.study import Study
from optuna.study import StudyDirection


# ---------- helper ----------
def _to_minimisation_matrix(
    trials: Sequence[optuna.trial.FrozenTrial],
    directions: Sequence[StudyDirection],
) -> np.ndarray:
    """Return array (n_trials × n_objectives) where **all** objectives are ‘smaller-is-better’."""
    vals = np.asarray([t.values for t in trials], dtype=float)
    for j, d in enumerate(directions):
        if d == StudyDirection.MAXIMIZE:
            vals[:, j] *= -1.0  # flip sign
    return vals


# ---------- public API ----------
def pick_trial(
    study: Study,
    mode: str = "harmonic",
    *,
    weights: Sequence[float] | None = None,
    ref_point: Sequence[float] | None = None,
    eps: float = 1e-12,
) -> optuna.trial.FrozenTrial:
    """
    Collapse Optuna’s Pareto front (`study.best_trials`) to a single “best compromise”.

    Parameters
    ----------
    study      : completed **multi-objective** Optuna study
    mode       : {"harmonic", "sum", "chebyshev", "hypervolume"}
    weights    : per-objective weights (used only for "sum")
    ref_point  : reference point for hyper-volume (defaults to ones after normalisation)
    eps        : tiny value to avoid division by zero

    Returns
    -------
    optuna.trial.FrozenTrial
    """

    # ---- 1. Pareto front ----
    front = study.best_trials
    if not front:
        raise ValueError("`study.best_trials` is empty – no Pareto-optimal trials found.")

    # ---- 2. Convert & normalise objectives ----
    vals = _to_minimisation_matrix(front, study.directions)  # smaller is better
    span = np.ptp(vals, axis=0)
    norm = (vals - vals.min(axis=0)) / (span + eps)  # 0 = best, 1 = worst

    # ---- 3. Scalarise according to chosen mode ----
    mode = mode.lower()

    if mode == "harmonic":
        hmean = norm.shape[1] / (1.0 / (norm + eps)).sum(axis=1)
        best_idx = hmean.argmin()  # lower = better

    elif mode == "sum":
        w = np.ones(norm.shape[1]) if weights is None else np.asarray(weights, float)
        if w.size != norm.shape[1]:
            raise ValueError("`weights` length must equal number of objectives.")
        score = norm @ w
        best_idx = score.argmin()

    elif mode == "chebyshev":
        score = norm.max(axis=1)  # worst dimension
        best_idx = score.argmin()

    elif mode == "hypervolume":
        # Hyper-volume assumes points are *below* the reference point (minimisation space).
        if len(front) == 0:
            raise ValueError("Pareto front is empty - no trials to select from")
        elif len(front) == 1:
            best_idx = 0
        else:
            rp = np.ones(norm.shape[1]) if ref_point is None else np.asarray(ref_point, float)
            base_hv = compute_hypervolume(norm, rp)
            contrib = np.array([base_hv - compute_hypervolume(np.delete(norm, i, 0), rp) for i in range(len(front))])
            best_idx = contrib.argmax()  # bigger contribution wins

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from "
                         "'harmonic', 'sum', 'chebyshev', 'hypervolume'.")

    return front[best_idx]
