# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import traceback
from pathlib import Path

import pytest
import yaml

from nat.utils.io.yaml_tools import yaml_load
from nat.utils.io.yaml_tools import yaml_loads


@pytest.mark.parametrize("invalid_value", ["*DUMMY_SECRET_ALIAS", "@DUMMY_SECRET_SCANNER", "[DUMMY_SECRET_FLOW"])
def test_yaml_errors_do_not_expose_config(invalid_value: str,
                                          monkeypatch: pytest.MonkeyPatch,
                                          caplog: pytest.LogCaptureFixture):
    monkeypatch.setenv("NAT_TEST_PASSWORD", "DUMMY_PRIVATE_PASSWORD")
    # Environment values are now expanded after parsing; trigger an error in the YAML itself.
    config = f"password: ${{NAT_TEST_PASSWORD}}\ntoken: {invalid_value}\n"

    with caplog.at_level(logging.ERROR, logger="nat.utils.io.yaml_tools"):
        with pytest.raises(ValueError, match="Error loading YAML") as exc_info:
            yaml_loads(config, Path("."))

    rendered_error = "".join(traceback.format_exception(exc_info.value))
    for secret in ("DUMMY_PRIVATE_PASSWORD", invalid_value, invalid_value[1:]):
        assert secret not in caplog.text
        assert secret not in str(exc_info.value)
        assert secret not in rendered_error
    assert "line" in str(exc_info.value)
    assert "column" in str(exc_info.value)


def test_yaml_load_error_retains_safe_location(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    config_path = tmp_path / "config.yml"
    config_path.write_text("workflow:\n  value: @DUMMY_SECRET\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"ScannerError at line 2, column 10") as exc_info:
        yaml_load(config_path)

    assert "ScannerError at line 2, column 10" in caplog.text
    assert "DUMMY_SECRET" not in "".join(traceback.format_exception(exc_info.value))


def test_unmarked_yaml_error_does_not_expose_message(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):

    def fail_load(stream, **kwargs):
        raise yaml.YAMLError("DUMMY_SECRET_IN_PARSER_MESSAGE")

    monkeypatch.setattr(yaml, "load", fail_load)
    with pytest.raises(ValueError, match="Error loading YAML: YAMLError") as exc_info:
        yaml_loads("value: ok", Path("."))

    assert "DUMMY_SECRET_IN_PARSER_MESSAGE" not in caplog.text
    assert "DUMMY_SECRET_IN_PARSER_MESSAGE" not in "".join(traceback.format_exception(exc_info.value))


def test_valid_yaml_interpolation_and_file_loading_are_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NAT_TEST_VALUE", "Hello ${NAT_TEST_INNER}")
    monkeypatch.setenv("NAT_TEST_INNER", "must_not_expand")
    (tmp_path / "prompt.txt").write_text("File ${NAT_TEST_INNER}", encoding="utf-8")
    config = yaml_loads(
        "value: ${NAT_TEST_VALUE}\nprompt: file://prompt.txt\ncount: 42\nenabled: true\n",
        tmp_path,
    )
    assert config == {
        "value": "Hello ${NAT_TEST_INNER}",
        "prompt": "File ${NAT_TEST_INNER}",
        "count": 42,
        "enabled": True,
    }


@pytest.mark.parametrize("value", ["*DUMMY_SECRET_ALIAS", "@DUMMY_SECRET_SCANNER", "[DUMMY_SECRET_FLOW"])
def test_yaml_indicator_env_values_remain_strings(value: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NAT_TEST_TOKEN", value)

    assert yaml_loads("token: ${NAT_TEST_TOKEN}\n", Path(".")) == {"token": value}
