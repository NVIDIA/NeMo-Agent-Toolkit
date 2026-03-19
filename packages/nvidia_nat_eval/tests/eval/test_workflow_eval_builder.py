# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nat.plugins.eval.runtime.builder import WorkflowEvalBuilder


def test_log_evaluator_build_failure_helper_method(caplog):
    """Test the _log_evaluator_build_failure helper method directly."""
    builder = WorkflowEvalBuilder()

    completed_evaluators = ["eval1", "eval2"]
    remaining_evaluators = ["eval3", "eval4"]
    original_error = ValueError("Evaluator build failed")

    builder._log_build_failure_evaluator("failing_evaluator",
                                         completed_evaluators,
                                         remaining_evaluators,
                                         original_error)

    log_text = caplog.text
    assert "Failed to initialize component failing_evaluator (evaluator)" in log_text
    assert "Successfully built components:" in log_text
    assert "- eval1 (evaluator)" in log_text
    assert "- eval2 (evaluator)" in log_text
    assert "Remaining components to build:" in log_text
    assert "- eval3 (evaluator)" in log_text
    assert "- eval4 (evaluator)" in log_text
    assert "Original error:" in log_text


def test_log_evaluator_build_failure_no_completed(caplog):
    """Test evaluator error logging when no evaluators have been successfully built."""
    builder = WorkflowEvalBuilder()

    completed_evaluators = []
    remaining_evaluators = ["eval1", "eval2"]
    original_error = ValueError("First evaluator failed")

    builder._log_build_failure_evaluator("failing_evaluator",
                                         completed_evaluators,
                                         remaining_evaluators,
                                         original_error)

    log_text = caplog.text
    assert "Failed to initialize component failing_evaluator (evaluator)" in log_text
    assert "No components were successfully built before this failure" in log_text
    assert "Remaining components to build:" in log_text
    assert "- eval1 (evaluator)" in log_text
    assert "- eval2 (evaluator)" in log_text
    assert "Original error:" in log_text


def test_log_evaluator_build_failure_no_remaining(caplog):
    """Test evaluator error logging when no evaluators remain to be built."""
    builder = WorkflowEvalBuilder()

    completed_evaluators = ["eval1", "eval2"]
    remaining_evaluators = []
    original_error = ValueError("Last evaluator failed")

    builder._log_build_failure_evaluator("failing_evaluator",
                                         completed_evaluators,
                                         remaining_evaluators,
                                         original_error)

    log_text = caplog.text
    assert "Failed to initialize component failing_evaluator (evaluator)" in log_text
    assert "Successfully built components:" in log_text
    assert "- eval1 (evaluator)" in log_text
    assert "- eval2 (evaluator)" in log_text
    assert "No remaining components to build" in log_text
    assert "Original error:" in log_text
