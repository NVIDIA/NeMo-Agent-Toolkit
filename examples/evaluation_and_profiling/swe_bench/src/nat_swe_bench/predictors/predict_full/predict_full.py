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

#pylint: disable=line-too-long
# flake8: noqa
import ast
import json
import logging
from pathlib import Path

from nat_swe_bench.config import SweBenchWorkflowConfig
from nat_swe_bench.predictors.predict_abc import SweBenchPredictorBase
from nat_swe_bench.predictors.predictor_registry import register_predictor
from openai import OpenAI

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.swe_bench_model import SWEBenchInput

logger = logging.getLogger(__name__)


def generate_patch(original_code: str, fixed_code: str, file_path: str) -> str:
    """Generate a unified diff patch from original and fixed code."""
    from difflib import unified_diff

    # Split both code versions into lines
    original_lines = original_code.splitlines(keepends=True)
    fixed_lines = fixed_code.splitlines(keepends=True)

    # Generate unified diff
    patch_lines = list(
        unified_diff(original_lines, fixed_lines, fromfile=f'a/{file_path}', tofile=f'b/{file_path}', lineterm=''))

    return ''.join(patch_lines)


def format_output(fixed_code: str, patch: str) -> str:
    """Format the output to include both fixed code and patch."""
    return f"""### Fixed Code ###
{fixed_code}

### Patch ###
{patch}"""


@register_predictor("full")
class SweBenchPredictor(SweBenchPredictorBase):

    def __init__(self, config: SweBenchWorkflowConfig, builder: Builder):
        super().__init__(config, builder)
        self.setup_predictor()
        self.openai_client = OpenAI(api_key=config.predictor.openai_api_key)

    def setup_predictor(self):
        '''Setup git tools'''
        logger.info("Setting up git tools for repository management")
        self.tools = self.builder.get_tools(["git_repo_tool"], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        self.git_tool = self.tools[0]

    def _parse_ast(self, file_path: str):
        """Parse AST of a Python file and extract symbols and imports."""
        logger.info("Starting AST parsing for: %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
                logger.info("Successfully read file: %s", file_path)

            tree = ast.parse(source_code)
            symbols = []
            imports = []

            # Track what we find
            functions_found = 0
            classes_found = 0
            imports_found = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions_found += 1
                    symbol = {
                        "type": "function",
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": getattr(node, 'end_lineno', node.lineno),
                        "docstring": ast.get_docstring(node)
                    }
                    symbols.append(symbol)
                    logger.info("Found function: %s at line %s", node.name, node.lineno)

                elif isinstance(node, ast.ClassDef):
                    classes_found += 1
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    symbol = {
                        "type": "class",
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": getattr(node, 'end_lineno', node.lineno),
                        "methods": methods,
                        "docstring": ast.get_docstring(node)
                    }
                    symbols.append(symbol)
                    logger.info("Found class: %s with %d methods", node.name, len(methods))

                elif isinstance(node, ast.Import):
                    imports_found += 1
                    for alias in node.names:
                        imports.append({"imported_name": alias.name, "imported_as": alias.asname})
                        logger.info("Found import: %s", alias.name)

                elif isinstance(node, ast.ImportFrom):
                    imports_found += 1
                    for alias in node.names:
                        imports.append({
                            "imported_name": alias.name, "imported_as": alias.asname, "origin": node.module
                        })
                        logger.info("Found import from %s: %s", node.module, alias.name)

            logger.info("AST Parsing Summary for %s:", file_path)
            logger.info("- Found %s functions", functions_found)
            logger.info("- Found %s classes", classes_found)
            logger.info("- Found %s imports", imports_found)

            return symbols, imports, source_code

        except Exception as e:
            logger.exception("Error parsing AST for %s: %s", file_path, e, exc_info=True)
            raise

    def _truncate_context(self, prompt: str, max_tokens: int = 2000) -> str:
        """Truncate the context to fit within token limit."""
        import tiktoken

        def num_tokens(text: str) -> int:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

        if num_tokens(prompt) <= max_tokens:
            return prompt

        lines = prompt.splitlines()
        # Keep problem statement, failing tests, and current file
        header_lines = lines[:lines.index("Current Code:") + 1]
        code_lines = []
        dependency_lines = []

        in_current_code = False
        in_dependencies = False

        for line in lines:
            if line == "Current Code:":
                in_current_code = True
                in_dependencies = False
            elif line == "Relevant Dependencies:":
                in_current_code = False
                in_dependencies = True
            elif in_current_code:
                code_lines.append(line)
            elif in_dependencies:
                dependency_lines.append(line)

        # Always keep full current code
        truncated_prompt = "\n".join(header_lines + code_lines)
        current_tokens = num_tokens(truncated_prompt)

        # Add dependencies until we hit token limit
        for line in dependency_lines:
            line_tokens = num_tokens(line)
            if current_tokens + line_tokens < max_tokens:
                truncated_prompt += "\n" + line
                current_tokens += line_tokens
            else:
                break

        return truncated_prompt

    def _gather_dependencies(self, repo_path: Path, target_file: str, max_depth: int = 2):
        """Gather dependencies recursively up to max_depth."""
        logger.info("Gathering dependencies for %s", target_file)
        dependencies = {}
        visited = set()

        def _process_file(file_path: str, depth: int):
            if depth > max_depth or file_path in visited:
                return
            visited.add(file_path)

            try:
                symbols, imports, content = self._parse_ast(file_path)
                if file_path != target_file:  # Don't include target file in dependencies
                    dependencies[file_path] = {'content': content, 'symbols': symbols}

                # Process imports
                for imp in imports:
                    # Try to find the imported file in the repository
                    if imp.get('origin'):
                        module_path = imp['origin'].replace('.', '/')
                        potential_paths = list(repo_path.rglob(f"{module_path}.py"))
                        if potential_paths:
                            _process_file(str(potential_paths[0]), depth + 1)

            except Exception as e:
                logger.exception("Error processing dependency %s: %s", file_path, e, exc_info=True)

        _process_file(target_file, 0)
        logger.info("Found %d dependencies", len(dependencies))
        return dependencies

    def _build_fix_prompt(self, file_path: str, repo_path: Path, failing_tests: list, problem_statement: str) -> str:
        """Build a comprehensive prompt for fix generation."""
        # Parse the target file
        logger.info("Parsing target file for prompt")
        symbols, imports, code_content = self._parse_ast(file_path)

        # Gather dependencies
        dependencies = self._gather_dependencies(repo_path, file_path)

        # Build context sections
        context_details = []
        for symbol in symbols:
            if symbol['type'] == 'function':
                context_details.append(
                    f"Function '{symbol['name']}' (lines {symbol['start_line']}-{symbol['end_line']})")
                if symbol['docstring']:
                    context_details.append(f"Purpose: {symbol['docstring']}")
            elif symbol['type'] == 'class':
                context_details.append(f"Class '{symbol['name']}' (lines {symbol['start_line']}-{symbol['end_line']})")
                if symbol['docstring']:
                    context_details.append(f"Purpose: {symbol['docstring']}")

        # Build import information
        import_details = []
        for imp in imports:
            if imp.get('origin'):
                import_details.append(f"from {imp['origin']} import {imp['imported_name']}")
            else:
                import_details.append(f"import {imp['imported_name']}")

        prompt = f"""
Task: Fix the bug in the following code based on failing tests and context.

Problem Statement: {problem_statement}

Failing Tests: {failing_tests}

File Structure:
{chr(10).join(context_details)}

Imports Used:
{chr(10).join(import_details)}

Current Code:
```python
{code_content}
```

Relevant Dependencies:
{chr(10).join(f'From {file}:' + chr(10) + f'```python{chr(10)}{dep["content"]}{chr(10)}```' for file, dep in dependencies.items())}

Requirements:
1. Generate a patch that fixes the failing tests
2. Maintain existing functionality and compatibility
3. Follow the project's coding style
4. Make minimal changes focused on fixing the bug

Generate the fixed code that addresses the failing tests while maintaining the existing structure and style.
Output only the complete fixed version of the code without any explanations or markdown formatting.
"""
        return prompt

    def _extract_file_from_patch(self, patch_content: str) -> str | None:
        """Extract the file path from the patch content."""
        try:
            lines = patch_content.split('\n')
            for line in lines:
                if line.startswith('diff --git'):
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2][2:]  # Remove 'b/' prefix
            logger.warning("Could not find file path in patch")
            return None
        except Exception as e:
            logger.exception("Error extracting file from patch: %s", e, exc_info=True)
            return None

    async def _generate_fix(self, prompt: str) -> str:
        """Generate fix using OpenAI."""
        try:

            prompt = self._truncate_context(prompt)

            logger.debug("Truncated prompt = %s", prompt)

            logger.info("Generating fix using OpenAI GPT-4")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role":
                        "system",
                    "content":
                        "You are a skilled Python developer. Do not refactor code unnecessarily. Also, when printing output, print only the changes you made and not unchanged code. Generate only the fixed code without any explanations or formatting. "
                }, {
                    "role": "user", "content": prompt
                }],
                temperature=0.2,
                max_tokens=2000)
            logger.info("Fix generated successfully")
            fixed_code = response.choices[0].message.content.strip()
            return fixed_code
        except Exception as e:
            logger.error("Error generating fix with OpenAI: %s", e)
            raise

    async def predict_fn(self, swebench_input: SWEBenchInput) -> str:
        logger.info("Processing instance %s", swebench_input.instance_id)

        try:
            # 1. Setup repository
            repo_name = swebench_input.instance_id.split('-')[0]
            org, repo = repo_name.split('__')
            repo_url = f"https://github.com/{org}/{repo}"

            repo_path = await self.git_tool.arun(
                json.dumps({
                    "operation": "setup", "repo_url": repo_url, "base_commit": swebench_input.base_commit
                }))
            logger.info("Repository setup at %s", repo_path)

            # Extract file path from patch
            target_file = self._extract_file_from_patch(swebench_input.patch)
            if not target_file:
                raise ValueError("Could not extract target file from patch")

            full_file_path = Path(repo_path) / target_file
            if not full_file_path.exists():
                raise FileNotFoundError(f"Target file not found: {full_file_path}")

            # Build prompt with inline analysis
            prompt = self._build_fix_prompt(file_path=str(full_file_path),
                                            repo_path=Path(repo_path),
                                            failing_tests=swebench_input.FAIL_TO_PASS,
                                            problem_statement=swebench_input.problem_statement)

            # logger.info("Generated prompt for fix:")
            # logger.info("=" * 80)
            # logger.info(prompt)
            # logger.info("=" * 80)

            # Generate fix using OpenAI
            fixed_code = await self._generate_fix(prompt)
            logger.info("Generated fix:")
            logger.info("=" * 80)
            logger.info(fixed_code)
            logger.info("=" * 80)

            # Read file content
            with open(full_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Generate patch
            return generate_patch(file_content, fixed_code, target_file)

        except Exception as e:
            logger.exception("Error processing %s: %s", swebench_input.instance_id, e, exc_info=True)
            return f"Error: {str(e)}"
