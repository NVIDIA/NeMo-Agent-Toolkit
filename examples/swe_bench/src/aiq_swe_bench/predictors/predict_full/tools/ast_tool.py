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

import ast
import json
import logging
from pathlib import Path

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AstToolConfig(FunctionBaseConfig, name="ast_tool"):
    """Configuration for AST analysis tool."""
    db_dir: str = "./.workspace/ast_dbs"


def extract_file_from_patch(patch_content: str) -> str | None:
    """Extract the file path from the patch content."""
    try:
        # Typically patches start with "diff --git a/path/to/file b/path/to/file"
        lines = patch_content.split('\n')
        for line in lines:
            if line.startswith('diff --git'):
                # Extract the first file path (after 'a/')
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2][2:]  # Remove 'b/' prefix
        return None
    except Exception as e:
        logger.exception("Error extracting file from patch: %s", e, exc_info=True)
        return None


@register_function(config_type=AstToolConfig)
async def ast_tool(tool_config: AstToolConfig, builder: Builder):
    """AST analysis tool for analyzing Python files."""

    def _analyze_file(file_path: str) -> dict:
        """Analyze a single Python file and return its AST information."""
        logger.info("Analyzing file: %s", file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Extract information
            symbols = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbols.append({
                        'type': 'function',
                        'name': node.name,
                        'start_line': node.lineno,
                        'end_line': getattr(node, 'end_lineno', node.lineno),
                        'docstring': ast.get_docstring(node)
                    })
                    logger.info("Found function: %s", node.name)

                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    symbols.append({
                        'type': 'class',
                        'name': node.name,
                        'start_line': node.lineno,
                        'end_line': getattr(node, 'end_lineno', node.lineno),
                        'methods': methods,
                        'docstring': ast.get_docstring(node)
                    })
                    logger.info("Found class: %s with methods: %s", node.name, methods)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        import_info = {
                            'name': alias.name,
                            'asname': alias.asname,
                            'from_module': node.module if isinstance(node, ast.ImportFrom) else None
                        }
                        imports.append(import_info)
                        logger.info("Found import: %s", import_info)

            return {'file_path': file_path, 'symbols': symbols, 'imports': imports, 'source': source_code}

        except Exception as e:
            logger.exception("Error analyzing file %s %s", file_path, e, exc_info=True)
            return {'file_path': file_path, 'error': str(e)}

    async def ast_operations(args_str: str) -> str:
        """Handle AST analysis operations."""
        args = json.loads(args_str)
        operation = args.get('operation')

        if operation == 'analyze_file':
            logger.info("Starting AST analysis for %s", args.get('file_path'))
            result = _analyze_file(args['file_path'])
            logger.info("AST analysis completed for %s", args.get('file_path'))
            return json.dumps(result)

        if operation == 'analyze_patch':
            patch_content = args.get('patch_content')
            repo_path = args.get('repo_path')

            if not patch_content or not repo_path:
                raise ValueError("Both patch_content and repo_path are required")

            file_path = extract_file_from_patch(patch_content)
            if not file_path:
                raise ValueError("Could not extract file path from patch")

            full_path = Path(repo_path) / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {full_path}")

            logger.info("Analyzing file from patch %s", full_path)
            result = _analyze_file(str(full_path))
            return json.dumps(result)

        raise ValueError(f"Unknown operation: {operation}")

    try:
        yield FunctionInfo.from_fn(ast_operations, description="Tool for analyzing Python code AST")
    finally:
        pass  # No cleanup needed
