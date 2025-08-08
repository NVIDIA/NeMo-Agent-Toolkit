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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class GithubGetFileToolConfig(FunctionBaseConfig, name="github_getfile"):
    """
    Tool that returns the text of a github file using a github url starting with https://github.com and ending
    with a specific file.
    """
    pass


@register_function(config_type=GithubGetFileToolConfig)
async def github_text_from_url(config: GithubGetFileToolConfig, builder: Builder):

    import re

    import requests

    async def _github_text_from_url(url_text: str) -> str:

        # Extract sections of the base github path
        pattern = r"https://github.com/(.*)/blob/(.*)"
        matches = re.findall(pattern, url_text)

        if (len(matches) == 0):
            return ("Invalid github url. Please provide a valid github url. "
                    "Example: 'https://github.com/my_repository/blob/main/file.txt'")

        # Construct raw content path
        raw_url = f"https://raw.githubusercontent.com/{matches[0][0]}/refs/heads/{matches[0][1]}"
        # Grab raw text from github
        try:
            response = requests.get(raw_url, timeout=60)
        except requests.exceptions.Timeout:
            return f"Timeout encountered when retrieving resource: {raw_url}"

        return f"```python\n{response.text}\n```"

    yield FunctionInfo.from_fn(_github_text_from_url,
                               description=("Returns the text of a github file using a github url starting with"
                                            "https://github.com and ending with a specific file."))


class GithubGetFileLinesToolConfig(FunctionBaseConfig, name="github_getfilelines"):
    """
    Tool that returns the text lines of a github file using a github url starting with
    https://github.com and ending with a specific file line references. Examples of line references are
    #L409-L417 and #L166-L171.
    """
    pass


@register_function(config_type=GithubGetFileLinesToolConfig)
async def github_text_lines_from_url(config: GithubGetFileLinesToolConfig, builder: Builder):

    import re

    async def _github_text_lines_from_url(url_text: str) -> str:

        import requests

        # Extract sections of the base github path
        pattern = r"https://github.com/(.*)/blob/(.*)(#L(\d+)-L(\d+))"
        matches = re.findall(pattern, url_text)

        if (len(matches) == 0):
            return ("Invalid github url. Please provide a valid github url with line information. "
                    "Example: 'https://github.com/my_repository/blob/main/file.txt#L409-L417'")

        start_line, end_line = int(matches[0][3]), int(matches[0][4])
        # Construct raw content path
        raw_url = f"https://raw.githubusercontent.com/{matches[0][0]}/refs/heads/{matches[0][1]}"
        # Grab raw text from github
        try:
            response = requests.get(raw_url, timeout=60)
        except requests.exceptions.Timeout:
            return f"Timeout encountered when retrieving resource: {raw_url}"
        # Extract the specified lines
        file_lines = response.text.splitlines()
        selected_lines = file_lines[start_line:end_line]
        joined_selected_lines = "\n".join(selected_lines)

        return f"```python\n{joined_selected_lines}\n```"

    yield FunctionInfo.from_fn(_github_text_lines_from_url,
                               description=("Returns the text lines of a github file using a github url starting with"
                                            "https://github.com and ending with a specific file line references. "
                                            "Examples of line references are #L409-L417 and #L166-L171."))
