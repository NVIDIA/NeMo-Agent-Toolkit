# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PlotChartsWorkflowConfig(FunctionBaseConfig, name="plot_charts"):

    # Add settings
    llm_name: str


@register_function(config_type=PlotChartsWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def simple_workflow(config: PlotChartsWorkflowConfig, builder: Builder):

    import json
    import os

    from dotenv import load_dotenv
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    from .create_plot import DrawPlotAgent

    load_dotenv()
    nvapi_key = os.environ["NVIDIA_API_KEY"]
    llm = ChatNVIDIA(
        # base_url="https://integrate.api.nvidia.com/v1",
        model="meta/llama-3.1-405b-instruct",
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        nvapi_key=nvapi_key)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    plot_agent = DrawPlotAgent(llm)

    # This function will be called with the input message
    async def _response_fn(input_message: str) -> str:
        logger.info("input_message=%s", input_message)
        cur_dir = os.path.abspath('.')
        logger.info("cur_dir=%s", cur_dir)
        data_path = os.path.join(cur_dir, "examples/plot_charts/example_data.json")
        logger.info("data_path=%s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            logger.info("ERROR: Unable to load data from %s", data_path)
            return "Unable to complete the user request."
        out = plot_agent.run(input_message, data)
        logger.info("---" * 10)
        logger.info("plotting agent output: %s", out)
        output_file = out["img_path"]

        return f"Saved output to {output_file}"

    yield _response_fn
