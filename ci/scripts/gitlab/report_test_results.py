#!/usr/bin/env python
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

import argparse
import os
import sys
import typing
import xml.etree.ElementTree as ET
from datetime import date

from slack_sdk import WebClient


class ReportMessages(typing.NamedTuple):
    plain_text: list[str]
    blocks: list[dict]
    failure_text: list[str] | None
    failure_blocks: list[dict] | None


def get_testcase_name(testcase: ET.Element) -> str:
    return f"{testcase.attrib.get('classname', 'Unknown')}::{testcase.attrib.get('name', 'Unknown')}"


def parse_junit(junit_file: str) -> dict[str, typing.Any]:
    tree = ET.parse(junit_file)
    root = tree.getroot()

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    failed_tests = []
    for testsuite in root.findall('testsuite'):
        total_tests += int(testsuite.attrib.get('tests', 0))
        num_failures = int(testsuite.attrib.get('failures', 0))
        num_errors = int(testsuite.attrib.get('errors', 0))
        total_failures += num_failures
        total_errors += num_errors
        total_skipped += int(testsuite.attrib.get('skipped', 0))

        if (num_failures + num_errors) > 0:
            for testcase in testsuite.findall('testcase'):
                failure = testcase.find('failure')
                error = testcase.find('error')

                for failed_test_tag in (failure, error):
                    if failed_test_tag is not None:
                        failed_info = {
                            "test_name": get_testcase_name(testcase),
                            "message": failed_test_tag.attrib.get('message', '').strip()
                        }
                        failed_tests.append(failed_info)

    return {
        "num_tests": total_tests,
        "num_failures": total_failures,
        "num_errors": total_errors,
        "num_skipped": total_skipped,
        "failed_tests": failed_tests
    }


def parse_coverage(coverage_file: str) -> str:
    tree = ET.parse(coverage_file)
    root = tree.getroot()
    coverage = root.attrib.get('line-rate', '0')
    return f"{float(coverage) * 100:.2f}%"


def get_error_string(num_errors: int, error_type: str) -> str:
    error_message = f"{error_type}: {num_errors}"
    if num_errors > 0:
        error_message = f"*{error_message}* :x:"
    return error_message


def text_to_block(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def add_text(text: str, blocks: list[dict], plain_text: list[str]) -> None:
    blocks.append(text_to_block(text))
    plain_text.append(text)


def build_messages(junit_data: dict[str, typing.Any], coverage_data: str) -> ReportMessages:
    branch_name = os.environ.get("CI_COMMIT_BRANCH", "unknown")
    num_errors = junit_data['num_errors']
    num_failures = junit_data['num_failures']

    # We need to create both a plain text message and a formatted message with blocks, the plain text message is used
    # for push notifications and accessibility purposes.
    plain_text = []
    blocks = []

    summary_line = f"Nightly CI/CD Test Results for `{branch_name}` - {date.today()}"
    plain_text.append(summary_line + "\n")

    num_errors_and_failures = num_errors + num_failures
    if num_errors_and_failures > 0:
        formatted_summary_line = f"@nat-core-devs :rotating_light: {summary_line}"
    else:
        formatted_summary_line = summary_line

    blocks.append(text_to_block(formatted_summary_line))
    test_results = "\n".join([
        get_error_string(num_failures, "Failures"),
        get_error_string(num_errors, "Errors"),
        f"Skipped: {junit_data['num_skipped']}",
        f"Total Tests: {junit_data['num_tests']}",
        f"Coverage: {coverage_data}"
    ])
    add_text(test_results, blocks, plain_text)

    failure_blocks = None
    failure_text = None
    if num_errors_and_failures > 0:
        failure_blocks = []
        failure_text = []
        add_text(f"*Failed Tests ({num_errors_and_failures}):*", failure_blocks, failure_text)

        failed_tests = junit_data['failed_tests']
        for failed_test in failed_tests:
            test_name = failed_test['test_name']
            message = failed_test['message']
            add_text(f"`{test_name}`\n```\n{message}\n```", failure_blocks, failure_text)
            failure_text.append("---\n")
            failure_blocks.append({"type": "divider"})

        job_url = os.environ.get("CI_JOB_URL")
        if job_url is not None:
            add_text(f"Full details available at: {job_url}", failure_blocks, failure_text)

    return ReportMessages(plain_text=plain_text,
                          blocks=blocks,
                          failure_text=failure_text,
                          failure_blocks=failure_blocks)


def main():
    parser = argparse.ArgumentParser(description='Report test status to slack channel')
    parser.add_argument('junit_file', type=str, help='JUnit XML file to parse')
    parser.add_argument('coverage_file', type=str, help='Coverage report file to parse')

    try:
        slack_token = os.environ["SLACK_TOKEN"]
        slack_channel = os.environ["SLACK_CHANNEL"]
    except KeyError:
        print('Error: Set environment variables SLACK_TOKEN and SLACK_CHANNEL to post to slack.')
        return 1

    args = parser.parse_args()
    junit_data = parse_junit(args.junit_file)
    coverage_data = parse_coverage(args.coverage_file)

    report_messages = build_messages(junit_data, coverage_data)

    client = WebClient(token=slack_token)
    response = client.chat_postMessage(channel=slack_channel,
                                       text="\n".join(report_messages.plain_text),
                                       blocks=report_messages.blocks,
                                       link_names=report_messages.failure_text is not None)

    if report_messages.failure_text is not None:
        # Since potentially a large number of failures could occur, we will post them in a thread to the original
        # message to avoid spamming the channel.
        client.chat_postMessage(channel=slack_channel,
                                text="\n".join(report_messages.failure_text),
                                blocks=report_messages.failure_blocks,
                                thread_ts=response["ts"])

    return 0


if __name__ == '__main__':
    sys.exit(main())
