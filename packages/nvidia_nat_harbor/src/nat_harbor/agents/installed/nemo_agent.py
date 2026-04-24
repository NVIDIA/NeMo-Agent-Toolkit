# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NAT bridge for Harbor's NemoAgent behavior gaps."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.base import CliFlag
from harbor.agents.installed.nemo_agent import NemoAgent as HarborNemoAgent
from harbor.environments.base import BaseEnvironment

from nat_harbor.agents.installed.local_install_policy import is_local_install_allowed
from nat_harbor.agents.installed.local_install_policy import resolve_local_install_policy


class NemoAgent(HarborNemoAgent):
    """Bridge agent that layers local-mode and setup parity over Harbor 0.5.0."""

    CLI_FLAGS = [
        *HarborNemoAgent.CLI_FLAGS,
        CliFlag(
            "workflow_packages",
            cli="--workflow-packages",
            type="str",
            env_fallback="NVIDIA_NAT_WORKFLOW_PACKAGES",
        ),
        CliFlag(
            "python_bin",
            cli="--python-bin",
            type="str",
            default="python3",
            env_fallback="NVIDIA_NAT_PYTHON_BIN",
        ),
        CliFlag(
            "allow_host_install",
            cli="--allow-host-install",
            type="bool",
            default=False,
            env_fallback="HARBOR_ALLOW_HOST_INSTALL",
        ),
        CliFlag(
            "local_install_policy",
            cli="--local-install-policy",
            type="enum",
            choices=["skip", "allow"],
            default="skip",
            env_fallback="HARBOR_LOCAL_INSTALL_POLICY",
        ),
    ]

    def _resolve_workflow_packages(self) -> list[str]:
        """Resolve workflow package install list from single + multi-package flags."""
        packages: list[str] = []

        single_package = self._resolved_flags.get("workflow_package")
        if single_package:
            packages.append(str(single_package).strip())

        multi_packages = self._resolved_flags.get("workflow_packages")
        if multi_packages:
            for package in str(multi_packages).split(","):
                candidate = package.strip()
                if candidate:
                    packages.append(candidate)

        deduped_packages: list[str] = []
        seen: set[str] = set()
        for package in packages:
            if package not in seen:
                seen.add(package)
                deduped_packages.append(package)

        return deduped_packages

    async def install(self, environment: BaseEnvironment) -> None:
        """Apply local install policy before delegating to Harbor install logic."""
        env_type_obj = environment.type()
        env_type = getattr(env_type_obj, "value", str(env_type_obj))
        local_install_policy = resolve_local_install_policy(
            self._resolved_flags.get("local_install_policy", "skip")
        )
        allow_host_install_raw = self._resolved_flags.get("allow_host_install")

        should_run_install = True
        if env_type == "local":
            should_run_install = is_local_install_allowed(
                local_install_policy,
                allow_host_install_raw,
            )

        install_metadata = {
            "environment_type": env_type,
            "local_install_policy": local_install_policy if env_type == "local" else None,
            "local_install_allowed": should_run_install if env_type == "local" else None,
            "install_executed": should_run_install,
        }
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "install-policy.json").write_text(
            json.dumps(install_metadata, indent=2),
            encoding="utf-8",
        )

        if not should_run_install:
            self.logger.warning(
                "Skipping agent install in local mode (safe default). "
                "Assuming dependencies are pre-provisioned."
            )
            return

        if env_type == "local":
            self.logger.warning(
                "Local mode host install is enabled for this run. "
                "Agent setup may mutate host packages/files."
            )
        await super().install(environment)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install base agent + wrapper and support multiple workflow packages."""
        await BaseInstalledAgent.setup(self, environment)

        wrapper_path = Path(__file__).parent / "nemo_agent_run_wrapper.py"
        await environment.upload_file(
            source_path=wrapper_path,
            target_path=self._CONTAINER_WRAPPER_PATH,
        )

        for index, workflow_package in enumerate(self._resolve_workflow_packages()):
            local_path = Path(workflow_package)
            if local_path.exists():
                container_pkg_dir = f"/installed-agent/workflow-package-{index}"
                await environment.upload_dir(
                    source_dir=local_path,
                    target_dir=container_pkg_dir,
                )
                result = await environment.exec(
                    command=(
                        'export PATH="/opt/nvidia-nat-venv/bin:$HOME/.local/bin:$PATH"; '
                        "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "
                        f"pip install --no-deps {shlex.quote(container_pkg_dir)}"
                    ),
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                )
            else:
                result = await environment.exec(
                    command=(
                        'export PATH="/opt/nvidia-nat-venv/bin:$HOME/.local/bin:$PATH"; '
                        "uv pip install --python /opt/nvidia-nat-venv/bin/python "
                        f"{shlex.quote(workflow_package)}"
                    ),
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to install workflow package '{workflow_package}': "
                    f"{result.stderr or result.stdout}"
                )

        config_file = self._resolved_flags.get("config_file")
        if config_file:
            host_config_path = Path(config_file)
            if not host_config_path.exists():
                raise FileNotFoundError(
                    f"NeMo Agent Toolkit config file not found: {config_file}"
                )
            await environment.upload_file(
                source_path=host_config_path,
                target_path=self._CONTAINER_CONFIG_PATH,
            )

    def _build_run_command(self, instruction: str) -> str:
        """Honor python_bin override while keeping Harbor command behavior."""
        run_command = super()._build_run_command(instruction)
        python_bin = shlex.quote(self._resolved_flags.get("python_bin", "python3"))
        return run_command.replace(
            f"python3 {self._CONTAINER_WRAPPER_PATH}",
            f"{python_bin} {self._CONTAINER_WRAPPER_PATH}",
            1,
        )

