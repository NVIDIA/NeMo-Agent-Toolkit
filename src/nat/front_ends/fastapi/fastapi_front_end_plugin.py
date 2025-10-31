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
import sys
import tempfile
import typing

from nat.builder.front_end import FrontEndBase
from nat.front_ends.fastapi.dask_client_mixin import DaskClientMixin
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorkerBase
from nat.front_ends.fastapi.main import get_app
from nat.front_ends.fastapi.utils import get_class_name
from nat.utils.io.yaml_tools import yaml_dump
from nat.utils.log_levels import LOG_LEVELS

if (typing.TYPE_CHECKING):
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


class FastApiFrontEndPlugin(DaskClientMixin, FrontEndBase[FastApiFrontEndConfig]):

    def __init__(self, full_config: "Config"):
        super().__init__(full_config)

        # This attribute is set if dask is installed, and an external cluster is not used (scheduler_address is None)
        self._cluster = None
        self._periodic_cleanup_future = None
        self._scheduler_address = None

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
    async def _periodic_cleanup(scheduler_address: str,
                                db_url: str,
                                sleep_time_sec: int = 300,
                                log_level: int = logging.INFO):
        from nat.front_ends.fastapi.job_store import JobStore

        job_store = JobStore(scheduler_address=scheduler_address, db_url=db_url)

        logging.basicConfig(level=log_level)
        logger.info("Starting periodic cleanup of expired jobs every %d seconds", sleep_time_sec)
        while True:
            await asyncio.sleep(sleep_time_sec)

            try:
                await job_store.cleanup_expired_jobs()
                logger.debug("Expired jobs cleaned up")
            except:  # noqa: E722
                logger.exception("Error during job cleanup")

    async def _submit_cleanup_task(self, scheduler_address: str, db_url: str, log_level: int = logging.INFO):
        """Submit a cleanup task to the cluster to remove the job after expiry."""
        logger.debug("Submitting periodic cleanup task to Dask cluster at %s", scheduler_address)
        async with self.client(self._scheduler_address) as client:
            self._periodic_cleanup_future = client.submit(self._periodic_cleanup,
                                                          scheduler_address=self._scheduler_address,
                                                          db_url=db_url,
                                                          log_level=log_level)

    @staticmethod
    def _setup_worker():
        """
        Setup function to be run in each worker process. This moves each worker into it's own process group.
        This fixes an issue where a Ctrl-C in the terminal sends a SIGINT to all workers, which then causes the
        workers to exit before the main process can shutdown the cluster gracefully.
        """
        os.setsid()

    async def run(self):

        # Write the entire config to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", prefix="nat_config", suffix=".yml", delete=False) as config_file:

            # Get as dict
            config_dict = self.full_config.model_dump(mode="json", by_alias=True, round_trip=True)

            # Three possible cases:
            # 1. Dask is installed and scheduler_address is None, we create a LocalCluster
            # 2. Dask is installed and scheduler_address is set, we use the existing cluster
            # 3. Dask is not installed, we skip the cluster setup
            dask_log_level = LOG_LEVELS.get(self.front_end_config.dask_log_level.upper(), logging.WARNING)
            dask_logger = logging.getLogger("distributed")
            dask_logger.setLevel(dask_log_level)

            self._scheduler_address = self.front_end_config.scheduler_address
            if self._scheduler_address is None:
                try:

                    from dask.distributed import LocalCluster

                    use_threads = self.front_end_config.dask_workers == 'threads'

                    # set n_workers to max_running_async_jobs + 1 to allow for one worker to handle the cleanup task
                    self._cluster = LocalCluster(processes=not use_threads,
                                                 silence_logs=dask_log_level,
                                                 protocol="tcp",
                                                 n_workers=self.front_end_config.max_running_async_jobs + 1)

                    self._scheduler_address = self._cluster.scheduler.address

                    if not use_threads and sys.platform != "win32":
                        with self.blocking_client(self._scheduler_address) as client:
                            # Client.run submits a function to be run on each worker
                            client.run(self._setup_worker)

                    logger.info("Created local Dask cluster with scheduler at %s using %s workers",
                                self._scheduler_address,
                                self.front_end_config.dask_workers)

                except ImportError:
                    logger.warning("Dask is not installed, async execution and evaluation will not be available.")

            if self._scheduler_address is not None:
                # If we are here then either the user provided a scheduler address, or we created a LocalCluster

                from nat.front_ends.fastapi.job_store import Base
                from nat.front_ends.fastapi.job_store import get_db_engine

                db_engine = get_db_engine(self.front_end_config.db_url, use_async=True)
                async with db_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all, checkfirst=True)  # create tables if they do not exist

                # If self.front_end_config.db_url is None, then we need to get the actual url from the engine
                db_url = str(db_engine.url)
                await self._submit_cleanup_task(scheduler_address=self._scheduler_address,
                                                db_url=db_url,
                                                log_level=dask_log_level)

                # Set environment variabls such that the worker subprocesses will know how to connect to dask and to
                # the database
                os.environ.update({
                    "NAT_DASK_SCHEDULER_ADDRESS": self._scheduler_address,
                    "NAT_JOB_STORE_DB_URL": db_url,
                })

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

                # By default, Uvicorn uses "auto" event loop policy, which prefers `uvloop` if installed. However,
                # uvloop’s event loop policy for macOS doesn’t provide a child watcher (which is needed for MCP server),
                # so setting loop="asyncio" forces Uvicorn to use the standard event loop, which includes child-watcher
                # support.
                if sys.platform == "darwin" or sys.platform.startswith("linux"):
                    # For macOS
                    event_loop_policy = "asyncio"
                else:
                    # For non-macOS platforms
                    event_loop_policy = "auto"

                uvicorn.run("nat.front_ends.fastapi.main:get_app",
                            host=self.front_end_config.host,
                            port=self.front_end_config.port,
                            workers=self.front_end_config.workers,
                            reload=self.front_end_config.reload,
                            factory=True,
                            reload_excludes=reload_excludes,
                            loop=event_loop_policy)

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
            logger.debug("Shutting down")
            if self._periodic_cleanup_future is not None:
                logger.info("Cancelling periodic cleanup task.")
                # Use the scheduler address, because self._cluster is None if an external cluster is used
                async with self.client(self._scheduler_address) as client:
                    await client.cancel([self._periodic_cleanup_future], asynchronous=True, force=True)

            if self._cluster is not None:
                # Only shut down the cluster if we created it
                logger.debug("Closing Local Dask cluster.")
                self._cluster.close()

            try:
                os.remove(config_file_name)
            except OSError as e:
                logger.exception(f"Warning: Failed to delete temp file {config_file_name}: {e}")
