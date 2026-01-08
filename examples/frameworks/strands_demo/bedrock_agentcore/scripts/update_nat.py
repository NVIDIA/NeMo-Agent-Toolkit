#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import os

import boto3

# Configuration
CONTAINER_IMAGE = 'strands-demo:latest'

AWS_REGION = os.environ['AWS_DEFAULT_REGION']
AWS_ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
IAM_AGENTCORE_ROLE = f'arn:aws:iam::{os.environ.get("AWS_ACCOUNT_ID")}:role/AgentCore_NAT'

RUNTIME_NAME = "strands_demo"

cclient = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)
cresponse = cclient.list_agent_runtimes()

runtime_id = None
for runtime in cresponse['agentRuntimes']:
    if runtime['agentRuntimeName'] == RUNTIME_NAME:
        runtime_id = runtime['agentRuntimeId']
        print(f"Found runtime ID: {runtime_id}")
        break

if runtime_id is None:
    raise RuntimeError(f"No AgentCore runtime named {RUNTIME_NAME!r} found in region {AWS_REGION}")

client = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)

response = client.update_agent_runtime(
    agentRuntimeId=runtime_id,
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': (f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}'
                             f'.amazonaws.com/{CONTAINER_IMAGE}')
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn=IAM_AGENTCORE_ROLE,
    environmentVariables={'AWS_DEFAULT_REGION': AWS_REGION},
)

print("Agent Runtime updated successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
