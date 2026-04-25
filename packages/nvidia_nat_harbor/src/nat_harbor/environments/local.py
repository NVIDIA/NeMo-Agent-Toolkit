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

"""Host-local Harbor environment implementation for NAT workflows."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path, PurePosixPath

from harbor.environments.base import BaseEnvironment
from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class LocalEnvironment(BaseEnvironment):
    """Host-local environment for fast development/debug iteration.

    Path model:
      - Most container-like paths (``/app``, ``/workspace``, ``/tests``, etc.)
        are mapped under ``<trial_dir>/.local-env``.
      - Canonical Harbor log/artifact paths (``/logs/agent``, ``/logs/verifier``,
        ``/logs/artifacts``) are mapped to ``trial_paths`` directories.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )
        self._local_root = (self.trial_paths.trial_dir / ".local-env").resolve()
        self._logs_dir = self._local_root / "logs"
        self._tests_dir = self._local_root / "tests"
        self._solution_dir = self._local_root / "solution"
        self._workspace_dir = self._local_root / "workspace"
        self._app_dir = self._local_root / "app"
        self._opt_dir = self._local_root / "opt"
        self._installed_agent_dir = self._local_root / "installed-agent"
        self._agent_log_dir = self.trial_paths.agent_dir.resolve()
        self._verifier_log_dir = self.trial_paths.verifier_dir.resolve()
        self._artifacts_dir = self.trial_paths.artifacts_dir.resolve()
        self._allowed_write_roots: tuple[Path, ...] = (
            self.trial_paths.trial_dir.resolve(),
            self._local_root,
        )

        self._path_map: list[tuple[str, Path]] = [
            ("/logs/agent", self._agent_log_dir),
            ("/logs/verifier", self._verifier_log_dir),
            ("/logs/artifacts", self._artifacts_dir),
            ("/logs", self._logs_dir),
            ("/tests", self._tests_dir),
            ("/solution", self._solution_dir),
            ("/workspace", self._workspace_dir),
            ("/app", self._app_dir),
            ("/opt", self._opt_dir),
            ("/installed-agent", self._installed_agent_dir),
        ]

    @staticmethod
    def type() -> str:
        return "local"

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        # Local mode runs on the host and cannot reliably enforce egress blocking.
        return False

    def _validate_definition(self):
        return

    def _translate_path(self, raw_path: str) -> str:
        """Translate a container-style path to its host-mapped local-mode path."""
        raw = PurePosixPath(raw_path).as_posix()
        for src, dst in sorted(self._path_map, key=lambda item: len(item[0]), reverse=True):
            if raw == src:
                return str(dst)
            if raw.startswith(src + "/"):
                suffix = raw[len(src) + 1 :]
                return str(dst / suffix)
        return raw_path

    def _translate_command(self, command: str) -> str:
        """Translate mapped paths in a shell command and neutralize chown operations."""
        translated = command
        for src, dst in sorted(self._path_map, key=lambda item: len(item[0]), reverse=True):
            translated = translated.replace(src, str(dst))
        translated = re.sub(r"(^|&&)\s*chown\s+\S+\s+\S+", r"\1 true", translated)
        return translated

    @staticmethod
    def is_shell_profile_write(command: str) -> bool:
        profile_tokens = (".bashrc", ".zshrc", ".profile")
        write_tokens = (">", ">>", "tee", "sed -i")
        return any(token in command for token in profile_tokens) and any(
            token in command for token in write_tokens
        )

    def _rewrite_local_paths_in_file(self, file_path: Path) -> None:
        if not file_path.is_file():
            return
        if file_path.suffix.lower() not in {".sh", ".py", ".bash"}:
            return
        try:
            original = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return

        rewritten = self._translate_command(original)
        if rewritten != original:
            file_path.write_text(rewritten, encoding="utf-8")

    def _rewrite_local_paths_in_tree(self, root: Path) -> None:
        if not root.exists():
            return
        if root.is_file():
            self._rewrite_local_paths_in_file(root)
            return
        for file_path in root.rglob("*"):
            self._rewrite_local_paths_in_file(file_path)

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _assert_allowed_write_path(self, path: Path, operation: str) -> None:
        resolved = path.resolve()
        if any(self._is_within(resolved, root) for root in self._allowed_write_roots):
            return
        roots = ", ".join(str(root) for root in self._allowed_write_roots)
        raise PermissionError(
            f"Local mode policy violation during {operation}: write path '{resolved}' "
            f"is outside allowed roots [{roots}]"
        )

    async def start(self, force_build: bool) -> None:
        del force_build
        self._local_root.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._tests_dir.mkdir(parents=True, exist_ok=True)
        self._solution_dir.mkdir(parents=True, exist_ok=True)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        self._app_dir.mkdir(parents=True, exist_ok=True)
        self._opt_dir.mkdir(parents=True, exist_ok=True)
        self._installed_agent_dir.mkdir(parents=True, exist_ok=True)
        self.trial_paths.agent_dir.mkdir(parents=True, exist_ok=True)
        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.trial_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    async def stop(self, delete: bool):
        if delete and self._local_root.exists():
            shutil.rmtree(self._local_root, ignore_errors=True)

    async def upload_file(self, source_path: Path | str, target_path: str):
        source = Path(source_path)
        target = Path(self._translate_path(target_path))
        self._assert_allowed_write_path(target, "upload_file")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self._rewrite_local_paths_in_file(target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        source = Path(source_dir)
        target = Path(self._translate_path(target_dir))
        self._assert_allowed_write_path(target, "upload_dir")
        shutil.copytree(source, target, dirs_exist_ok=True)
        self._rewrite_local_paths_in_tree(target)

    async def download_file(self, source_path: str, target_path: Path | str):
        source = Path(self._translate_path(source_path))
        target = Path(target_path)
        self._assert_allowed_write_path(target, "download_file")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        source = Path(self._translate_path(source_dir))
        target = Path(target_dir)
        self._assert_allowed_write_path(target, "download_dir")
        shutil.copytree(source, target, dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a host shell command with local-mode path/policy enforcement.

        Behavior:
          - Translates container-style paths in `command` and `cwd` to host paths.
          - Blocks shell profile writes and out-of-scope working directories.
          - Executes via bash, captures stdout/stderr, and enforces timeout.
          - Ignores `user` (local mode runs as host user).
        """
        del user
        translated_command = self._translate_command(command)
        if self.is_shell_profile_write(translated_command):
            return ExecResult(
                return_code=1,
                stdout="",
                stderr=(
                    "Local mode policy violation: writes to shell profile files "
                    "(.bashrc/.zshrc/.profile) are blocked."
                ),
            )

        translated_cwd = self._translate_path(cwd) if cwd else None
        if translated_cwd:
            try:
                self._assert_allowed_write_path(Path(translated_cwd), "exec(cwd)")
            except PermissionError as exc:
                return ExecResult(return_code=1, stdout="", stderr=str(exc))

        merged_env = self._merge_env(env)
        proc_env = None
        if merged_env is not None:
            proc_env = os.environ.copy()
            proc_env.update(merged_env)

        process = await asyncio.create_subprocess_shell(
            translated_command,
            cwd=translated_cwd,
            env=proc_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable="/bin/bash",
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_sec,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return ExecResult(return_code=124, stdout="", stderr="Command timed out")

        return ExecResult(
            return_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )


def is_shell_profile_write(command: str) -> bool:
    """Module-level helper retained for compatibility with older imports."""
    return LocalEnvironment.is_shell_profile_write(command)

