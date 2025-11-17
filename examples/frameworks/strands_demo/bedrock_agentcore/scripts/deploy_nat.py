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

import boto3

client = boto3.client('bedrock-agentcore-control', region_name='<AWS_REGION>')

response = client.create_agent_runtime(
    agentRuntimeName='strands_demo',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest'
            # '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/nat-test-repo:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    #                                       roleArn='<IAM_AGENTCORE_ROLE>')
    roleArn='<IAM_AGENTCORE_ROLE>')

print("Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
