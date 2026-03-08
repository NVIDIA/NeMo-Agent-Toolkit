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
"""BFCL benchmark evaluator.

Calls BFCL's ast_checker directly in-process to score model outputs.
Supports both AST prompting and FC workflow outputs.
"""

import json
import logging

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem

from .config import BFCLEvaluatorConfig

logger = logging.getLogger(__name__)


def _extract_function_call(raw_output: str) -> str:
    """Extract function call text from model output, stripping markdown and prose."""
    import re

    # Try to extract from markdown code blocks first (python, json, tool_code, or untagged)
    code_blocks = re.findall(r'```(?:python|tool_code|json)?\s*\n?(.*?)\n?```', raw_output, re.DOTALL)
    if code_blocks:
        block = code_blocks[-1].strip()
        # If it's JSON with "name"/"parameters", convert to function call
        converted = _try_convert_json_to_call(block)
        if converted:
            return converted
        # If it's Python code with imports, extract just the function call lines
        if block.startswith("import ") or block.startswith("from "):
            calls = _extract_calls_from_code(block)
            if calls:
                return calls
        return block

    # Try JSON-formatted tool call (without code block)
    converted = _try_convert_json_to_call(raw_output.strip())
    if converted:
        return converted

    # If the whole output looks like Python code with imports, extract calls first
    stripped_output = raw_output.strip()
    if stripped_output.startswith("import ") or stripped_output.startswith("from "):
        calls = _extract_calls_from_code(stripped_output)
        if calls:
            return calls

    # Try to find lines that look like function calls (name(...))
    lines = stripped_output.split('\n')
    func_call_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove common prefixes
        for prefix in ['tools.', '> ', '- ']:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        if re.match(r'^[a-zA-Z_]\w*(?:\.\w+)*\s*\(', stripped):
            func_call_lines.append(stripped)

    if func_call_lines:
        # Strip 'tools.' prefix from extracted calls
        func_call_lines = [re.sub(r'^tools\.', '', c) for c in func_call_lines]
        if len(func_call_lines) == 1:
            return func_call_lines[0]
        return '[' + ', '.join(func_call_lines) + ']'

    # Fallback: return as-is and let the AST parser try
    return raw_output


def _try_convert_json_to_call(text: str) -> str | None:
    """Try to parse JSON tool-call format and convert to Python function call."""
    import json as _json
    try:
        obj = _json.loads(text)
        if isinstance(obj, dict) and "name" in obj:
            name = obj["name"]
            params = obj.get("parameters", obj.get("arguments", {}))
            if isinstance(params, dict):
                param_strs = [f"{k}={repr(v)}" for k, v in params.items()]
                return f"{name}({', '.join(param_strs)})"
    except (_json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_calls_from_code(code: str) -> str | None:
    """Extract function call expressions from Python code (skip imports, assignments)."""
    import re
    lines = code.strip().split('\n')
    calls = []
    for line in lines:
        stripped = line.strip()
        # Skip imports, print, assignments with =
        if stripped.startswith(("import ", "from ", "print(", "#", "def ", "return ")):
            continue
        # Look for variable = func(...) patterns
        assign_match = re.match(r'^[a-zA-Z_]\w*\s*=\s*([a-zA-Z_]\w*(?:\.\w+)*\s*\(.*\))\s*$', stripped)
        if assign_match:
            calls.append(assign_match.group(1))
            continue
        # Look for bare function calls
        if re.match(r'^[a-zA-Z_]\w*(?:\.\w+)*\s*\(', stripped):
            calls.append(stripped)
    if calls:
        if len(calls) == 1:
            return calls[0]
        return '[' + ', '.join(calls) + ']'
    return None


def _evaluate_single(item: EvalInputItem, test_category: str, language: str) -> EvalOutputItem:
    """Evaluate a single BFCL item using ast_checker."""
    from bfcl.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl.model_handler.utils import default_decode_ast_prompting

    if item.output_obj is None:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": "No workflow output (output_obj is None)"},
        )

    # Parse the test entry and ground truth
    try:
        entry = json.loads(item.input_obj) if isinstance(item.input_obj, str) else item.input_obj
        answer = json.loads(item.expected_output_obj) if isinstance(item.expected_output_obj,
                                                                    str) else item.expected_output_obj
    except (json.JSONDecodeError, TypeError) as e:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": f"Failed to parse input/answer: {e}"},
        )

    func_description = entry.get("function", [])
    possible_answer = answer.get("ground_truth", [])
    model_output_raw = _extract_function_call(item.output_obj)

    # Handle irrelevance/relevance tests differently
    if "irrelevance" in test_category:
        # For irrelevance: model should NOT produce a valid function call
        try:
            decoded = default_decode_ast_prompting(model_output_raw, language)
            # If decoding succeeds and produces non-empty output, it's a failure
            if decoded and any(decoded):
                return EvalOutputItem(
                    id=item.id,
                    score=0.0,
                    reasoning={
                        "error": "Model produced function call for irrelevance test", "decoded": str(decoded)
                    },
                )
        except Exception:
            pass  # Decode failure = success for irrelevance
        return EvalOutputItem(id=item.id, score=1.0, reasoning={"status": "correct_irrelevance"})

    if "relevance" in test_category:
        # For relevance: model should produce a valid function call (any)
        try:
            decoded = default_decode_ast_prompting(model_output_raw, language)
            if decoded and any(decoded):
                return EvalOutputItem(id=item.id, score=1.0, reasoning={"status": "correct_relevance"})
        except Exception:
            pass
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={"error": "Model failed to produce function call for relevance test"},
        )

    # Standard AST evaluation: decode and check
    try:
        decoded_output = default_decode_ast_prompting(model_output_raw, language)
    except Exception as e:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={
                "error": f"AST decode failed: {e}", "raw_output": model_output_raw[:500]
            },
        )

    try:
        checker_result = ast_checker(
            func_description,
            decoded_output,
            possible_answer,
            language,
            test_category,
            "nat_benchmark",
        )
    except Exception as e:
        return EvalOutputItem(
            id=item.id,
            score=0.0,
            reasoning={
                "error": f"ast_checker failed: {e}", "decoded": str(decoded_output)[:500]
            },
        )

    is_valid = checker_result.get("valid", False)
    score = 1.0 if is_valid else 0.0
    reasoning = {
        "valid": is_valid,
        "raw_output": model_output_raw[:500],
        "decoded": str(decoded_output)[:500],
    }
    if not is_valid:
        reasoning["error"] = checker_result.get("error", [])
        reasoning["error_type"] = checker_result.get("error_type", "unknown")

    return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)


@register_evaluator(config_type=BFCLEvaluatorConfig)
async def bfcl_evaluator_function(config: BFCLEvaluatorConfig, builder: EvalBuilder):

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        eval_output_items = []

        for item in eval_input.eval_input_items:
            try:
                output_item = _evaluate_single(item, config.test_category, config.language)
            except Exception as e:
                logger.exception("Error evaluating BFCL item %s: %s", item.id, e)
                output_item = EvalOutputItem(
                    id=item.id,
                    score=0.0,
                    reasoning={"error": str(e)},
                )
            eval_output_items.append(output_item)

        scores = [i.score for i in eval_output_items if isinstance(i.score, (int, float))]
        average_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "BFCL evaluation complete: accuracy=%.3f (%d/%d) category=%s",
            average_score,
            sum(1 for s in scores if s == 1.0),
            len(scores),
            config.test_category,
        )

        return EvalOutput(average_score=average_score, eval_output_items=eval_output_items)

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluate_fn,
        description=f"BFCL AST evaluator (category: {config.test_category})",
    )
