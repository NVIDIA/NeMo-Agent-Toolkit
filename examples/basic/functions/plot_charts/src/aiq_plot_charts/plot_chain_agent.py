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

import json
import logging
import os
import re
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain_core.language_models import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=SyntaxWarning)
import matplotlib.colors as mcolors  # noqa: E402 # pylint: disable=ungrouped-imports, wrong-import-position

load_dotenv()

nvapi_key = os.environ["NVIDIA_API_KEY"]


def find_label_and_data_points(data: dict, llm: LLM | BaseChatModel):
    o = json.dumps(data)
    # construct the system prompt
    prompt_template = """
    ### [INST]
    extract information from the below data_points
    JSON format of input data points with label: {data_points}
    pt_label: one string value, name of the data series
    data_points: a list of numerical values
    Begin!
    [/INST]
    """
    prompt = PromptTemplate(
        input_variables=['data'],
        template=prompt_template,
    )

    # structural output using LMFE
    class PointsLabel(BaseModel):
        pt_label: str = Field(description="look like name usually in string type and usually only one value in it")
        data_points: list = Field(description="something look like data points, usually numbers")

    llm_extract_datapoint_and_label = llm.with_structured_output(PointsLabel)

    # construct the content_creator agent
    content_creator = (prompt | llm_extract_datapoint_and_label)
    out = content_creator.invoke({"data_points": data, "sample_data_point": o})
    logger.info("output from find_label_and_data_points = %s", out)
    return out


def plot_line_chart(x_values: list, y_values: list, chart_name: str, llm: LLM | BaseChatModel, save_fig: bool = True):
    """Draws a line plot with multiple labeled lines overlayed on a single plot.

    Parameters
    ----------
    x_values:
        A list of string or numerical numbers, used in plot on x-asis
    y_values:
        A list of dictionaries usually containing 2 keys : data and label or something similar
        an example of yValues should look similar to the following :
        Example:
        yValues = [
            {
            "data":[152,178,185],
            "label":height,
            },
            {
            "data":[50,72,81],
            "label":weight_in_kgs,
            },
        ]
    llm:
        The LLM or ChatModel to invoke to find and label data points
    chart_name:
        A string with generated chart name
    save_fig:
        to save the plot to PNG or directly plot it
    """
    if not isinstance(x_values, list) or len(x_values) == 0:
        raise ValueError(f"x_values needs to be a non-empty list. Got: {x_values}")

    if not isinstance(y_values, list) or len(y_values) == 0:
        raise ValueError(f"y_values needs to be a non-empty list. Got: {x_values}")
    # if isinstance(x_values, list) and len(x_values) > 0:
    #     x = x_values

    # Create a line plot for each value (fed_acc, fed_pre, fed_recall, fed_f1)
    plt.figure(figsize=(10, 6))

    for y in y_values:
        logger.info("y=%s\n", y)
        if 'label' in y.keys():
            o1 = find_label_and_data_points(y, llm=llm)
            label = o1.pt_label
            y_data_points = o1.data_points
            logger.info("label=%s\n", label)
            logger.info("y_data_points=%s\n", y_data_points)
            sns.lineplot(x=x_values, y=y_data_points, marker='o', color='navy', label=label)

    # Add titles and labels
    plt.title(chart_name)
    plt.xlabel('X Values')
    plt.ylabel('Metrics')
    plt.legend()

    if chart_name is None or len(chart_name.strip()) == 0:
        chart_name = 'test'
    img_path = f'./{chart_name}.png'
    if save_fig:
        plt.savefig(img_path, dpi=100)
        # im = cv2.imread("/home/coder/dev/ai-query-engine/{img_name}.png")
        # cv2.imshow("image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img_path

    # Show plot
    plt.show(block=True)
    return img_path


def plot_bar_plot(x_values: list, y_values: list, chart_name: str, llm: LLM | BaseChatModel, save_fig: bool = True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    x_values:
        A list of string or numerical numbers, used in plot on x-asis
    y_values:
        A list of dictionaries usually containing 2 keys : data and label or something similar
        an example of yValues should look similar to the following :
        Example:
        yValues = [
            {
            "data":[152,178,185],
            "label":height,
            },
            {
            "data":[50,72,81],
            "label":weight_in_kgs,
            },
        ]
    llm:
        The LLM or ChatModel to invoke to find and label data points
    chart_name:
        A string with generated chart name
    save_fig:
        to save the plot to PNG or directly plot it
    """
    total_width = 0.8
    single_width = 1
    colors = list(mcolors.BASE_COLORS.keys())
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(y_values)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []
    x = x_values

    # Iterate over all data
    n_bars = len(y_values)
    for i in range(n_bars):
        y = y_values[i]
        logger.info("y=%s", y)
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        logger.info("x_offset=%s, bar_width=%s, single_width=%s", x_offset, bar_width, single_width)
        o1 = find_label_and_data_points(y, llm)

        _y = o1.data_points
        y_label = o1.pt_label
        logger.info("_y=%s", _y)
        x_off = [int(_x) + x_offset for _x in x]
        b = ax.bar(x_off, _y, width=bar_width * single_width, color=colors[i % len(colors)], label=y_label)
        # Draw legend if we need

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(b[0])
    if chart_name is None or len(chart_name.strip()) == 0:
        chart_name = 'test'
    img_path = f'./{chart_name}.png'
    if save_fig:
        plt.savefig(img_path, dpi=100)
        # im = cv2.imread("/home/coder/dev/ai-query-engine/{img_name}.png")
        # cv2.imshow("image", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img_path

    # Show plot
    plt.show(block=True)
    return img_path


class PlotChartAgents:
    """
    Implementation of the plot agent
    """

    def __init__(self, llm: LLM):
        """
            Initialize the XPlane agent and create appropriate LangGraph workflow
        """
        # self._llm = llm
        self.llm = llm

        # ======================== making vred tool calling agent via lcel =======================
        # making plot agent

        # reference url of graph instructions fetch from:
        # https://github.com/DhruvAtreja/datavisualization_langgraph/blob/main/backend_py/my_agent/graph_instructions.py
        # ============== constructing line graph ==========================
        # construct the system prompt
        # construct the system prompt
        lime_prompt_template = """
        ### [INST]
        JSON format input data : {data}
        {lineGraphIntstruction}

        Given the data points, what would be a good title for this chart/graph/plot?
        Begin!
        [/INST]
        """

        # structural output using LMFE
        class StructureOutput(BaseModel):
            xValues: list = Field(
                description="List of string, integers or float numbers, inside the Json structure, with the key xValues"
            )
            yValues: list = Field(
                description="List of string, integers or float numbers, inside the Json structure, with the key yValues"
            )
            chart_name: str = Field(description="An appropriate short title for this chart")

        llm_with_output_structure = llm.with_structured_output(StructureOutput)
        # sample data to try things out

        # construct the content_creator agent
        """line_prompt = PromptTemplate(
            input_variables=['data'],
            MessagesPlaceholder("chat_history"),
            template=lime_prompt_template,
        )"""
        line_prompt = ChatPromptTemplate.from_messages([
            ("system", lime_prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{data}"),
        ])

        self.line_graph_creator = (line_prompt | llm_with_output_structure)

        # ============== constructing bar graph ==========================
        bar_graph_intstruction = '''
        Where data is: {
            labels: string[]
            values: {data: number[], label: string}[]
        }

        // Examples of usage:
        Each label represents a column on the x axis.
        Each array in values represents a different entity.

        Here we are looking at average income for each month.
        1. data = {
        x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        y: [{data:[21.5, 25.0, 47.5, 64.8, 105.5, 133.2], label: 'Income'}],
        }

        Here we are looking at the performance of american and european players for each series.
        Since there are two entities, we have two arrays in values.
        2. data = {
        xValues: ['series A', 'series B', 'series C'],
        yValues: [{data:[10, 15, 20], label: 'American'}, {data:[20, 25, 30], label: 'European'}],
        }
        '''
        self.bar_instruction = re.compile(bar_graph_intstruction, re.X)
        # construct the system prompt
        bar_prompt = """
        ### [INST]
        JSON format input data : {data}
        {bar_instruction}

        Given the data points, what would be a good title for this chart/graph/plot?
        Begin!
        [/INST]
        """
        """bar_chart_prompt = PromptTemplate(
        input_variables=['data'],
        template=bar_prompt,
        )"""
        bar_chart_prompt = ChatPromptTemplate.from_messages([
            ("system", bar_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{data}"),
        ])
        self.bar_plot_tool_chain = bar_chart_prompt | llm_with_output_structure

        # ============= routeing chain ====================
        # pylint: disable=line-too-long
        ori_route_sys_prompt = """
                Given the  input below, classify it as either being about `bar_chart`, `line_chart` or `general`.
                EXAMPLES:
                ---
                user input query :
                make me a bar chart
                generate a bar chart on the data
                draw bar chart graph for me

                Classification: bar_chart
                ---
                user input query :
                generate a graph about the trend
                make a trend graph
                generate a line chart for me
                draw line chart on this data

                Classification: line_chart
                ---
                general chitchat chain examples:
                tell me a joke
                what is my name
                what is my last query

                Classification: general
                <END OF EXAMPLES>

                Do not respond with more than one word.

                <input>
                {input}
                </input>

                Classification:""".strip()

        # route_sys_prompt_alternative_1 = """
        # Given the  input below, classify it as either being about `bar_chart`, `line_chart` or `general` topic.
        # Just use one of these words as your response.

        # 'bar_chart' - any query related to generate a graph or chart that look like barchart, bar chart
        # 'line_chart' - any questions related to generate a graph or chart that look like lines
        # 'general' - everything else.

        # User query: {input}
        # Classifcation topic:""".strip()
        self.routing_chain = ({
            "input": RunnablePassthrough()
        }
                              | PromptTemplate.from_template(ori_route_sys_prompt)
                              | self.llm
                              | StrOutputParser())

        # ============= general chain for chitchat =================
        system_prompt = """
        "You are an assistant to answer generic chitchat queries from the user "
        "answer concise and short."
        "\n\n"
        """
        history_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.general_chain = history_qa_prompt | self.llm
