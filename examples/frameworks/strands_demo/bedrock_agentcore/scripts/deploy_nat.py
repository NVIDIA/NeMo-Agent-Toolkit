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
AWS_REGION = os.environ['AWS_DEFAULT_REGION']
AWS_ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
IAM_AGENTCORE_ROLE = f'arn:aws:iam::{os.environ.get("AWS_ACCOUNT_ID")}:role/AgentCore_NAT'
CONTAINER_IMAGE = 'strands-demo'
AGENT_NAME = 'strands_demo'

client = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)

response = client.create_agent_runtime(
    agentRuntimeName=AGENT_NAME,
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': (f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}'
                             f'.amazonaws.com/{CONTAINER_IMAGE}:latest')
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn=IAM_AGENTCORE_ROLE,
    environmentVariables={'AWS_DEFAULT_REGION': AWS_REGION},
)

print("Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"export AGENT_RUNTIME_ARN={response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
