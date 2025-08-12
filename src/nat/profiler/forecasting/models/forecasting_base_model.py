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

# forecasting/models/base_model.py

from abc import ABC, abstractmethod
import numpy as np


class ForecastingBaseModel(ABC):
    """
    Abstract base class for all models in this package.
    """

    @abstractmethod
    def fit(self, raw_stats):
        """
        Train/fine-tune the model on the provided dataset.
        """
        pass

    @abstractmethod
    def predict(self, raw_stats) -> np.ndarray:
        """
        Predict using the trained model.
        Returns a np.ndarray, shape = (N, 4).
        """
        pass
