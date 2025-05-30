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

import logging
import os
import tempfile
import time
import typing

from aiq.builder.front_end import FrontEndBase
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase
from aiq.front_ends.fastapi.main import get_app
from aiq.front_ends.fastapi.utils import get_class_name
from aiq.utils.io.yaml_tools import yaml_dump

if typing.TYPE_CHECKING:
    try:
        from dask.distributed import LocalCluster
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class FastApiFrontEndPlugin(FrontEndBase[FastApiFrontEndConfig]):

    # This attribute is set if dask is installed, and an external cluster is not used (scheduler_address is None)
    _cluster: "LocalCluster | None" = None

    def get_worker_class(self) -> type[FastApiFrontEndPluginWorkerBase]:
        from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

        return FastApiFrontEndPluginWorker

    @typing.final
    def get_worker_class_name(self) -> str:

        if (self.front_end_config.runner_class):
            return self.front_end_config.runner_class

        worker_class = self.get_worker_class()

        return get_class_name(worker_class)

    @staticmethod
    def _periodic_cleanup(scheduler_address: str, sleep_time_sec: int = 300):
        from aiq.front_ends.fastapi.job_store import JobStore
        job_store = JobStore(scheduler_address=scheduler_address)

        logger.info("Starting periodic cleanup of expired jobs every %d seconds", sleep_time_sec)
        while True:
            try:
                job_store.cleanup_expired_jobs()
                logger.debug("Expired jobs cleaned up")
            except Exception as e:
                logger.error("Error during job cleanup: %s", e)

            time.sleep(sleep_time_sec)

    def _submit_cleanup_task(self, scheduler_address: str):
        """Submit a cleanup task to the cluster to remove the job after expiry."""
        from dask.distributed import Client
        from dask.distributed import fire_and_forget

        client = Client(self._cluster)
        future = client.submit(self._periodic_cleanup, scheduler_address=scheduler_address)
        fire_and_forget(future)

    async def run(self):

        # Write the entire config to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", prefix="aiq_config", suffix=".yml", delete=True) as config_file:

            # Get as dict
            config_dict = self.full_config.model_dump(mode="json", by_alias=True, round_trip=True)

            using_dask = False
            if self.front_end_config.scheduler_address is None:
                try:
                    from dask.distributed import LocalCluster

                    self._cluster = LocalCluster(asynchronous=True)
                    if self._cluster.scheduler is not None:
                        config_dict["scheduler_address"] = self._cluster.scheduler.address
                        using_dask = True
                    else:
                        raise RuntimeError("Dask LocalCluster did not start correctly, no scheduler address available.")
                except ImportError:
                    logger.warning("Dask is not installed, async execution and evaluation will not be available.")
            else:
                using_dask = True

            if using_dask:
                from aiq.front_ends.fastapi.job_store import register_dask_serializers
                register_dask_serializers()
                self._submit_cleanup_task(config_dict["scheduler_address"])

            # Write to YAML file
            yaml_dump(config_dict, config_file)

            # Set the config file in the environment
            os.environ["AIQ_CONFIG_FILE"] = str(config_file.name)

            # Set the worker class in the environment
            os.environ["AIQ_FRONT_END_WORKER"] = self.get_worker_class_name()

            if not self.front_end_config.use_gunicorn:
                import uvicorn

                reload_excludes = ["./.*"]

                uvicorn.run("aiq.front_ends.fastapi.main:get_app",
                            host=self.front_end_config.host,
                            port=self.front_end_config.port,
                            workers=self.front_end_config.workers,
                            reload=self.front_end_config.reload,
                            factory=True,
                            reload_excludes=reload_excludes)

            else:
                app = get_app()

                from gunicorn.app.wsgiapp import WSGIApplication

                class StandaloneApplication(WSGIApplication):

                    def __init__(self, app, options=None):
                        self.options = options or {}
                        self.app = app
                        super().__init__()

                    def load_config(self):
                        config = {
                            key: value
                            for key, value in self.options.items() if key in self.cfg.settings and value is not None
                        }
                        for key, value in config.items():
                            self.cfg.set(key.lower(), value)

                    def load(self):
                        return self.app

                options = {
                    "bind": f"{self.front_end_config.host}:{self.front_end_config.port}",
                    "workers": self.front_end_config.workers,
                    "worker_class": "uvicorn.workers.UvicornWorker",
                }

                StandaloneApplication(app, options=options).run()
