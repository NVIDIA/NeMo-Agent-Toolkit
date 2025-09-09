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

import asyncio
import logging
import os
import tempfile
import traceback
import typing

from nat.builder.front_end import FrontEndBase
from nat.front_ends.fastapi.dask_client_mixin import DaskClientMangerMixin
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase
from nat.front_ends.fastapi.main import get_app
from nat.front_ends.fastapi.utils import get_class_name
from nat.utils.io.yaml_tools import yaml_dump

if (typing.TYPE_CHECKING):
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


class FastApiFrontEndPlugin(DaskClientMangerMixin, FrontEndBase[FastApiFrontEndConfig]):

    def __init__(self, full_config: "Config"):
        super().__init__(full_config)

        # This attribute is set if dask is installed, and an external cluster is not used (scheduler_address is None)
        self._cluster = None
        self._cleanup_future = None

    def get_worker_class(self) -> type[FastApiFrontEndPluginWorkerBase]:
        from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker

        return FastApiFrontEndPluginWorker

    @typing.final
    def get_worker_class_name(self) -> str:

        if (self.front_end_config.runner_class):
            return self.front_end_config.runner_class

        worker_class = self.get_worker_class()

        return get_class_name(worker_class)

    @staticmethod
    async def _periodic_cleanup(scheduler_address: str, db_url: str, sleep_time_sec: int = 300):
        from nat.front_ends.fastapi.job_store import JobStore

        job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)

        logger.info("Starting periodic cleanup of expired jobs every %d seconds", sleep_time_sec)
        await asyncio.sleep(sleep_time_sec)
        while True:
            try:
                await job_store.cleanup_expired_jobs()
                logger.debug("Expired jobs cleaned up")
            except Exception:
                logger.error("Error during job cleanup: %s", traceback.format_exc())

            await asyncio.sleep(sleep_time_sec)

    async def _submit_cleanup_task(self, scheduler_address: str, db_url: str):
        """Submit a cleanup task to the cluster to remove the job after expiry."""
        from dask.distributed import fire_and_forget

        async with self.client(self._cluster) as client:
            self._cleanup_future = client.submit(self._periodic_cleanup,
                                                 scheduler_address=scheduler_address,
                                                 db_url=db_url)
            fire_and_forget(self._cleanup_future)

    async def run(self):

        # Write the entire config to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", prefix="nat_config", suffix=".yml", delete=False) as config_file:

            # Get as dict
            config_dict = self.full_config.model_dump(mode="json", by_alias=True, round_trip=True)

            # Three possible cases:
            # 1. Dask is installed and scheduler_address is None, we create a LocalCluster
            # 2. Dask is installed and scheduler_address is set, we use the existing cluster
            # 3. Dask is not installed, we skip the cluster setup
            scheduler_address = self.front_end_config.scheduler_address
            if scheduler_address is None:
                try:
                    from dask.distributed import LocalCluster

                    self._cluster = await LocalCluster(asynchronous=True)

                    if self._cluster.scheduler is not None:
                        scheduler_address = self._cluster.scheduler.address
                    else:
                        raise RuntimeError("Dask LocalCluster did not start correctly, no scheduler address available.")
                except ImportError:
                    logger.warning("Dask is not installed, async execution and evaluation will not be available.")

            if scheduler_address is not None:

                from nat.front_ends.fastapi.job_store import Base
                from nat.front_ends.fastapi.job_store import get_db_engine

                db_engine = get_db_engine(self.front_end_config.db_url, use_async=True)
                async with db_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all, checkfirst=True)  # create tables if they do not exist

                # If self.front_end_config.db_url is None, then we need to get the actual url from the engine
                db_url = str(db_engine.url)
                await self._submit_cleanup_task(scheduler_address=scheduler_address, db_url=db_url)

                # Set environment variabls such that the worker subprocesses will know how to connect to dask and to
                # the database
                os.environ.update({"NAT_DASK_SCHEDULER_ADDRESS": scheduler_address, "NAT_JOB_STORE_DB_URL": db_url})

            # Write to YAML file
            yaml_dump(config_dict, config_file)

            # Save the config file path for cleanup (required on Windows due to delete=False workaround)
            config_file_name = config_file.name

            # Set the config file in the environment
            os.environ["NAT_CONFIG_FILE"] = str(config_file.name)

            # Set the worker class in the environment
            os.environ["NAT_FRONT_END_WORKER"] = self.get_worker_class_name()

        try:
            if not self.front_end_config.use_gunicorn:
                import uvicorn

                reload_excludes = ["./.*"]

                uvicorn.run("nat.front_ends.fastapi.main:get_app",
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

        finally:
            logger.debug("Shuting down")
            if self._cluster is not None:
                if self._cleanup_future is not None:

                    logger.info("Cancelling periodic cleanup task.")
                    async with self.client(self._cluster) as client:
                        await client.cancel([self._cleanup_future], force=True, reason="Shutting down")

                logger.info("Closing Dask cluster.")
                self._cluster.close()
            try:
                os.remove(config_file_name)
            except OSError as e:
                logger.exception(f"Warning: Failed to delete temp file {config_file_name}: {e}")
