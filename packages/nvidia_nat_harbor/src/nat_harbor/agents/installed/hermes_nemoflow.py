# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental Harbor wrapper for Hermes with NeMo-Flow sidecar enabled."""

from __future__ import annotations

import copy
import json
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.hermes import Hermes
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


@dataclass(frozen=True)
class _ProviderRoute:
    """Provider settings needed to route Hermes model traffic through the sidecar."""

    api_key_envs: tuple[str, ...]
    upstream_base_env: str | None
    default_upstream_base_url: str
    sidecar_route: str
    hermes_provider: str
    cli_provider: str | None
    cli_model: str
    config_model: str


class HermesNeMoFlow(Hermes):
    """Run Hermes inside Harbor with a NeMo-Flow sidecar around the session."""

    _DEFAULT_CONTAINER_NEMO_FLOW_DIR = "/opt/nemo-flow"
    _DEFAULT_ATOF_DIR = "/logs/agent/nemo-flow-atof"
    _DEFAULT_SIDECAR_ATIF_DIR = "/logs/agent/nemo-flow-sidecar-atif"
    _CONVERTED_ATIF_DIR_NAME = "nemo-flow-atof-atif"
    _CONVERTED_ATIF_FILENAME = "trajectory.json"
    _SIDECAR_CANONICAL_ATIF_FILENAME = "trajectory.json"
    _DEFAULT_TASK_DIR = "/testbed"
    _HERMES_HOME = "/tmp/hermes"
    _HERMES_HOOK_EVENTS = (
        "on_session_start",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_llm_call",
        "post_llm_call",
        "pre_api_request",
        "post_api_request",
        "pre_tool_call",
        "post_tool_call",
        "subagent_start",
        "subagent_stop",
    )

    def __init__(
        self,
        *args: Any,
        nemo_flow_repo: str | None = None,
        container_nemo_flow_dir: str = _DEFAULT_CONTAINER_NEMO_FLOW_DIR,
        atof_dir: str = _DEFAULT_ATOF_DIR,
        sidecar_atif_dir: str = _DEFAULT_SIDECAR_ATIF_DIR,
        fail_missing_nemoflow_atof: bool = True,
        fail_missing_nemoflow_atif: bool = False,
        convert_nemoflow_atof: bool = True,
        fail_nemoflow_atof_conversion: bool = True,
        canonicalize_nemoflow_atif: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._nemo_flow_repo = Path(nemo_flow_repo or os.environ.get("NEMO_FLOW_REPO", "external/nemo-flow")).resolve()
        self._container_nemo_flow_dir = container_nemo_flow_dir.rstrip("/")
        self._atof_dir = atof_dir.rstrip("/")
        self._sidecar_atif_dir = sidecar_atif_dir.rstrip("/")
        self._fail_missing_nemoflow_atof = fail_missing_nemoflow_atof
        self._fail_missing_nemoflow_atif = fail_missing_nemoflow_atif
        self._convert_nemoflow_atof = convert_nemoflow_atof
        self._fail_nemoflow_atof_conversion = fail_nemoflow_atof_conversion
        self._canonicalize_nemoflow_atif = canonicalize_nemoflow_atif

    @staticmethod
    def name() -> str:
        return "hermes-nemoflow"

    @property
    def _container_sidecar_bin(self) -> str:
        return f"{self._container_nemo_flow_dir}/target/release/nemo-flow-sidecar"

    def _validate_local_nemo_flow_checkout(self) -> None:
        required = [
            self._nemo_flow_repo / "Cargo.toml",
            self._nemo_flow_repo / "Cargo.lock",
            self._nemo_flow_repo / "crates" / "core" / "Cargo.toml",
            self._nemo_flow_repo / "crates" / "sidecar" / "Cargo.toml",
            self._nemo_flow_repo / "crates" / "sidecar" / "src" / "main.rs",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "NeMo-Flow sidecar checkout is not ready. Missing:\n" + "\n".join(f"- {path}" for path in missing)
            )

    def _prepare_upload_tree(self) -> Path:
        self._validate_local_nemo_flow_checkout()
        setup_dir = self.logs_dir / "setup" / "nemo-flow-sidecar-upload"
        if setup_dir.exists():
            shutil.rmtree(setup_dir)
        setup_dir.mkdir(parents=True, exist_ok=True)

        ignore = shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            ".ruff_cache",
            ".venv",
            ".uv-cache",
            "node_modules",
            "target",
            "__pycache__",
        )
        for filename in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
            source = self._nemo_flow_repo / filename
            if source.exists():
                shutil.copy2(source, setup_dir / filename)
        shutil.copytree(
            self._nemo_flow_repo / "crates",
            setup_dir / "crates",
            ignore=ignore,
            dirs_exist_ok=True,
        )
        return setup_dir

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        upload_tree = self._prepare_upload_tree()

        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y build-essential ca-certificates pkg-config",
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=600,
        )
        await self.exec_as_root(
            environment,
            command=(
                f"mkdir -p {shlex.quote(self._container_nemo_flow_dir)} && "
                f"chmod 777 {shlex.quote(self._container_nemo_flow_dir)}"
            ),
        )
        await environment.upload_dir(
            source_dir=upload_tree,
            target_dir=self._container_nemo_flow_dir,
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v cargo >/dev/null 2>&1; then "
                "curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | "
                "sh -s -- -y --profile minimal; "
                "fi; "
                '. "$HOME/.cargo/env"; '
                f"cd {shlex.quote(self._container_nemo_flow_dir)}; "
                "cargo build -p nemo-flow-sidecar --release; "
                'mkdir -p "$HOME/.local/bin"; '
                f"ln -sf {shlex.quote(self._container_sidecar_bin)} "
                '"$HOME/.local/bin/nemo-flow-sidecar"; '
                "nemo-flow-sidecar --help >/dev/null; "
                "nemo-flow-sidecar run --help | grep -q -- '--atof-dir' || "
                "{ echo 'Error: nemo-flow-sidecar run does not support --atof-dir' >&2; exit 1; }; "
                "nemo-flow-sidecar hook-forward --help | grep -q -- '--atof-dir' || "
                "{ echo 'Error: nemo-flow-sidecar hook-forward does not support --atof-dir' >&2; exit 1; }"
            ),
            timeout_sec=1800,
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, model = self.model_name.split("/", 1)
        route = self._build_provider_route(provider, model)
        env = self._build_run_env(route, instruction)
        config_yaml = self._build_config_yaml_with_sidecar_hooks(route)

        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(self._HERMES_HOME)} "
                f"{shlex.quote(self._atof_dir)} {shlex.quote(self._sidecar_atif_dir)} "
                "/logs/agent/nemo-flow-sidecar && "
                f"cat > {shlex.quote(self._HERMES_HOME)}/config.yaml << 'EOF'\n{config_yaml}EOF"
            ),
            env=env,
            timeout_sec=10,
        )

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            await self.exec_as_agent(environment, command=mcp_command, env=env, timeout_sec=10)

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env, timeout_sec=10)

        run_cmd = self._build_sidecar_run_command(route)
        await self.exec_as_agent(environment, command=run_cmd, env=env)

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)

        atof_path = self.logs_dir / self._atof_dir.removeprefix("/logs/agent/") / "events.jsonl"
        sidecar_dir = self.logs_dir / self._sidecar_atif_dir.removeprefix("/logs/agent/")
        atif_files = self._sidecar_atif_files(sidecar_dir)
        canonical_path = sidecar_dir / self._SIDECAR_CANONICAL_ATIF_FILENAME
        converted_atif_path = self.logs_dir / self._CONVERTED_ATIF_DIR_NAME / self._CONVERTED_ATIF_FILENAME

        if self._fail_missing_nemoflow_atof and not atof_path.exists():
            raise FileNotFoundError(f"Missing NeMo-Flow ATOF JSONL artifact: {atof_path}")

        if self._fail_missing_nemoflow_atif and not atif_files:
            raise FileNotFoundError(f"Missing NeMo-Flow sidecar ATIF artifact under: {sidecar_dir}")

        if self._convert_nemoflow_atof and atof_path.exists():
            try:
                self._convert_atof_to_atif(atof_path, converted_atif_path)
            except Exception as exc:
                self.logger.exception("Failed to convert NeMo-Flow ATOF artifact to ATIF")
                if self._fail_nemoflow_atof_conversion:
                    raise RuntimeError(f"Failed to convert NeMo-Flow ATOF artifact to ATIF: {atof_path}") from exc

        if self._canonicalize_nemoflow_atif and atif_files:
            self._write_canonical_sidecar_atif(atif_files[0], canonical_path)

        if converted_atif_path.exists():
            self._populate_context_tokens_from_atif(context, converted_atif_path)
        elif canonical_path.exists():
            self._populate_context_tokens_from_atif(context, canonical_path)

        metadata = dict(context.metadata or {})
        metadata["nemo_flow_instrumentation"] = "sidecar"
        metadata["nemo_flow_atof_path"] = str(atof_path)
        metadata["nemo_flow_atof_exists"] = atof_path.exists()
        metadata["nemo_flow_converted_atif_path"] = str(converted_atif_path)
        metadata["nemo_flow_converted_atif_exists"] = converted_atif_path.exists()
        metadata["nemo_flow_sidecar_atif_dir"] = str(sidecar_dir)
        metadata["nemo_flow_sidecar_atif_paths"] = [str(path) for path in atif_files]
        metadata["nemo_flow_sidecar_canonical_atif_path"] = str(canonical_path)
        metadata["nemo_flow_sidecar_canonical_atif_exists"] = canonical_path.exists()
        context.metadata = metadata

    def _build_run_env(self, route: _ProviderRoute, instruction: str) -> dict[str, str]:
        env = {
            "HERMES_HOME": self._HERMES_HOME,
            "TERMINAL_ENV": "local",
            "HARBOR_INSTRUCTION": instruction,
            "NEMO_FLOW_ATOF_DIR": self._atof_dir,
            "NEMO_FLOW_ATIF_DIR": self._sidecar_atif_dir,
        }
        api_key_env = self._first_available_env(route.api_key_envs)
        if api_key_env is None:
            raise ValueError(f"No API key found. Set {' or '.join(route.api_key_envs)}.")
        env[api_key_env] = self._get_env(api_key_env) or ""
        if route.hermes_provider == "custom":
            env["OPENAI_API_KEY"] = env[api_key_env]
        return env

    def _build_provider_route(self, provider: str, model: str) -> _ProviderRoute:
        provider = provider.strip().lower()
        if provider == "anthropic":
            return _ProviderRoute(
                api_key_envs=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
                upstream_base_env="ANTHROPIC_BASE_URL",
                default_upstream_base_url="https://api.anthropic.com",
                sidecar_route="anthropic",
                hermes_provider="anthropic",
                cli_provider="anthropic",
                cli_model=model,
                config_model=model,
            )
        if provider == "openai":
            return _ProviderRoute(
                api_key_envs=("OPENAI_API_KEY",),
                upstream_base_env="OPENAI_BASE_URL",
                default_upstream_base_url="https://api.openai.com",
                sidecar_route="openai",
                hermes_provider="custom",
                cli_provider=None,
                cli_model=model,
                config_model=model,
            )
        if provider == "nvidia":
            return _ProviderRoute(
                api_key_envs=("NVIDIA_API_KEY",),
                upstream_base_env="NVIDIA_BASE_URL",
                default_upstream_base_url="https://integrate.api.nvidia.com/v1",
                sidecar_route="openai",
                hermes_provider="custom",
                cli_provider=None,
                cli_model=model,
                config_model=model,
            )
        if provider == "openrouter":
            return _ProviderRoute(
                api_key_envs=("OPENROUTER_API_KEY",),
                upstream_base_env="OPENROUTER_BASE_URL",
                default_upstream_base_url="https://openrouter.ai/api/v1",
                sidecar_route="openai",
                hermes_provider="custom",
                cli_provider=None,
                cli_model=model,
                config_model=model,
            )
        return _ProviderRoute(
            api_key_envs=("OPENROUTER_API_KEY",),
            upstream_base_env="OPENROUTER_BASE_URL",
            default_upstream_base_url="https://openrouter.ai/api/v1",
            sidecar_route="openai",
            hermes_provider="custom",
            cli_provider=None,
            cli_model=self.model_name or model,
            config_model=self.model_name or model,
        )

    def _build_config_yaml_with_sidecar_hooks(self, route: _ProviderRoute) -> str:
        config = yaml.safe_load(self._build_config_yaml(route.config_model)) or {}
        config["model"] = {
            "default": route.config_model,
            "model": route.config_model,
            "provider": route.hermes_provider,
            "base_url": "${NEMO_FLOW_SIDECAR_URL}",
        }
        if route.hermes_provider == "custom":
            config["model"]["api_key"] = "${OPENAI_API_KEY}"
        hooks = copy.deepcopy(config.get("hooks") or {})
        hook_command = self._build_hook_forward_command()
        for event in self._HERMES_HOOK_EVENTS:
            groups = hooks.setdefault(event, [])
            entry = {"command": hook_command, "timeout": 30}
            if entry not in groups:
                groups.append(entry)
        config["hooks"] = hooks
        return yaml.safe_dump(config, default_flow_style=False, sort_keys=False)

    def _build_hook_forward_command(self) -> str:
        return (
            "nemo-flow-sidecar hook-forward hermes "
            f"--atof-dir {shlex.quote(self._atof_dir)} "
            f"--atif-dir {shlex.quote(self._sidecar_atif_dir)} "
            "--gateway-mode passthrough"
        )

    def _build_sidecar_run_command(self, route: _ProviderRoute) -> str:
        args = [
            "nemo-flow-sidecar",
            "run",
            "--agent",
            "hermes",
            "--atof-dir",
            self._atof_dir,
            "--atif-dir",
            self._sidecar_atif_dir,
            "--session-metadata",
            json.dumps({"source": "harbor", "agent": self.name()}, separators=(",", ":")),
        ]
        if route.sidecar_route == "anthropic":
            args.extend(["--anthropic-base-url", self._upstream_base_url(route)])
        else:
            args.extend(["--openai-base-url", self._upstream_base_url(route)])

        child_script = self._build_child_script(route)
        args.extend(["--", "/bin/bash", "-lc", child_script])
        return (
            'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH" && '
            f"{' '.join(shlex.quote(arg) for arg in args)} "
            "2>&1 | stdbuf -oL tee /logs/agent/hermes.txt"
        )

    def _build_child_script(self, route: _ProviderRoute) -> str:
        cli_parts = [
            'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"',
            "set +e",
            self._hermes_chat_command(route),
            "status=$?",
            ("hermes sessions export /logs/agent/hermes-session.jsonl --source cli 2>/dev/null || true"),
            self._synthetic_finalize_command(),
            'exit "$status"',
        ]
        return "; ".join(cli_parts)

    def _hermes_chat_command(self, route: _ProviderRoute) -> str:
        parts = [
            "hermes",
            "--yolo",
            "chat",
            "-q",
            "$HARBOR_INSTRUCTION",
            "-Q",
            "--model",
            route.cli_model,
        ]
        if route.cli_provider:
            parts.extend(["--provider", route.cli_provider])
        toolsets_flag = self._resolved_flags.get("toolsets")
        if toolsets_flag:
            parts.extend(["--toolsets", str(toolsets_flag)])
        return " ".join(
            shlex.quote(part) if part != "$HARBOR_INSTRUCTION" else '"$HARBOR_INSTRUCTION"' for part in parts
        )

    def _synthetic_finalize_command(self) -> str:
        session_id_expr = (
            "import json,pathlib;"
            "p=pathlib.Path('/logs/agent/hermes-session.jsonl');"
            "sid='';"
            "\nfor line in p.read_text().splitlines():"
            "\n    line=line.strip();"
            "\n    if not line: continue"
            "\n    obj=json.loads(line);"
            "\n    sid=str(obj.get('id') or obj.get('session_id') or '');"
            "\n    break"
            "\nprint(sid)"
        )
        payload = (
            'printf \'{"session_id":"%s","hook_event_name":"on_session_finalize","source":"harbor"}\\n\' "$session_id"'
        )
        return (
            f"if ! find {shlex.quote(self._sidecar_atif_dir)} -name '*.atif.json' "
            "-print -quit 2>/dev/null | grep -q .; then "
            f"session_id=$(python3 -c {shlex.quote(session_id_expr)} 2>/dev/null || true); "
            'if [ -n "$session_id" ]; then '
            f"{payload} | {self._build_hook_forward_command()} || true; "
            "fi; "
            "fi"
        )

    def _upstream_base_url(self, route: _ProviderRoute) -> str:
        if route.upstream_base_env:
            value = self._get_env(route.upstream_base_env)
            if value:
                return value
        return route.default_upstream_base_url

    def _first_available_env(self, keys: tuple[str, ...]) -> str | None:
        for key in keys:
            if self._get_env(key):
                return key
        return None

    def _sidecar_atif_files(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            (path for path in directory.glob("*.atif.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _write_canonical_sidecar_atif(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    def _convert_atof_to_atif(self, atof_path: Path, output_path: Path) -> Path:
        try:
            from nat.atof.scripts.atof_to_atif_converter import convert_file
        except ImportError as exc:
            raise RuntimeError("The ATOF-to-ATIF converter is unavailable. Install nvidia-nat-atif[full].") from exc

        trajectory = convert_file(atof_path, output_path)
        steps_count = len(getattr(trajectory, "steps", []) or [])
        self.logger.debug("Wrote ATOF-derived ATIF trajectory to %s with %s steps", output_path, steps_count)
        return output_path

    def _populate_context_tokens_from_atif(self, context: AgentContext, atif_path: Path) -> None:
        try:
            trajectory = json.loads(atif_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        final_metrics = trajectory.get("final_metrics") or {}
        prompt_tokens = final_metrics.get("total_prompt_tokens")
        completion_tokens = final_metrics.get("total_completion_tokens")
        if isinstance(prompt_tokens, int):
            context.n_input_tokens = prompt_tokens
        if isinstance(completion_tokens, int):
            context.n_output_tokens = completion_tokens
