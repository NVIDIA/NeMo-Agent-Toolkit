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

import logging
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class CodeExecutionToolConfig(FunctionBaseConfig, name="code_execution"):
    """
    Tool for executing python code in a remotely hosted sandbox environment.
    """
    uri: HttpUrl = Field(default=HttpUrl("http://127.0.0.1:6000"),
                         description="URI for the code execution sandbox server")
    sandbox_type: Literal["local", "piston"] = Field(default="local", description="The type of code execution sandbox")
    timeout: float = Field(default=10.0, description="Number of seconds to wait for a code execution request")
    max_output_characters: int = Field(default=1000, description="Maximum number of characters that can be returned")


@register_function(config_type=CodeExecutionToolConfig)
async def code_execution_tool(config: CodeExecutionToolConfig, builder: Builder):
    from aiq.tool.code_execution.code_sandbox import get_sandbox

    class CodeExecutionInputSchema(BaseModel):
        generated_code: str = Field(description="String containing the code to be executed")

    sandbox = get_sandbox(sandbox_type=config.sandbox_type, uri=config.uri)

    async def _execute_code(generated_code: str) -> dict:
        logger.info("Executing code in the sandbox at %s", config.uri)
        try:
            output = await sandbox.execute_code(
                generated_code=generated_code,
                language="python",
                timeout=config.timeout,
                max_output_characters=config.max_output_characters,
            )
        except Exception as e:
            logger.exception("Error when executing code in the sandbox, %s", e)
            return {"process_status": "error", "stdout": "", "stderr": e}
        return output

    yield FunctionInfo.from_fn(
        fn=_execute_code,
        input_schema=CodeExecutionInputSchema,
        description="""Executes the provied 'generated_code' in a python sandbox environment and returns
        a dictionary containing stdout, stderr, and the execution status, as well as a session_id. The
        session_id can be used to append to code that was previously executed.""")
