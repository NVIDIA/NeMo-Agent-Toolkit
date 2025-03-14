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

import importlib
import logging
import os

from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase
from aiq.runtime.loader import load_config

logger = logging.getLogger(__name__)


def get_app():

    config_file_path = os.getenv("AIQ_CONFIG_FILE")
    front_end_worker_full_name = os.getenv("AIQ_FRONT_END_WORKER")

    if (not config_file_path):
        raise ValueError("Config file not found in environment variable AIQ_CONFIG_FILE.")

    if (not front_end_worker_full_name):
        raise ValueError("Front end worker not found in environment variable AIQ_FRONT_END_WORKER.")

    # Try to import the front end worker class
    try:
        # Split the package from the class
        front_end_worker_parts = front_end_worker_full_name.split(".")

        front_end_worker_module_name = ".".join(front_end_worker_parts[:-1])
        front_end_worker_class_name = front_end_worker_parts[-1]

        front_end_worker_module = importlib.import_module(front_end_worker_module_name)

        if not hasattr(front_end_worker_module, front_end_worker_class_name):
            raise ValueError(f"Front end worker {front_end_worker_full_name} not found.")

        front_end_worker_class: type[FastApiFrontEndPluginWorkerBase] = getattr(front_end_worker_module,
                                                                                front_end_worker_class_name)

        if (not issubclass(front_end_worker_class, FastApiFrontEndPluginWorkerBase)):
            raise ValueError(
                f"Front end worker {front_end_worker_full_name} is not a subclass of FastApiFrontEndPluginWorker.")

        # Load the config
        abs_config_file_path = os.path.abspath(config_file_path)

        config = load_config(abs_config_file_path)

        # Create an instance of the front end worker class
        front_end_worker = front_end_worker_class(config)

        aiq_app = front_end_worker.build_app()

        return aiq_app

    except ImportError as e:
        raise ValueError(f"Front end worker {front_end_worker_full_name} not found.") from e
