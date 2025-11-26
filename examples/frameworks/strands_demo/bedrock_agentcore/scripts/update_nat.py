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
"""Update NAT agent runtime with NVIDIA API key from AWS Secrets Manager."""

import json

import boto3


def get_secret(secret_name: str, region_name: str) -> dict[str, str]:
    """Retrieve secret from AWS Secrets Manager."""
    session = boto3.session.Session()
    secrets_client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        get_secret_value_response = secrets_client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        raise RuntimeError(f"Error retrieving secret: {e}") from e

    secret = get_secret_value_response['SecretString']
    return json.loads(secret)


# Configuration
AWS_REGION = '<AWS_REGION>'
AWS_ACCOUNT_ID = '<AWS_ACCOUNT_ID>'
AGENT_RUNTIME_ID = '<AGENT_RUNTIME_ID>'
CONTAINER_IMAGE = 'strands-demo:latest'
IAM_AGENTCORE_ROLE = '<IAM_AGENTCORE_ROLE>'
SECRET_NAME = 'nvidia-api-credentials'

# Fetch NVIDIA API key from Secrets Manager
secrets = get_secret(SECRET_NAME, AWS_REGION)
nvidia_api_key = secrets.get('NVIDIA_API_KEY')

if not nvidia_api_key:
    raise ValueError("NVIDIA_API_KEY not found in secrets")

client = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)

response = client.update_agent_runtime(agentRuntimeId=AGENT_RUNTIME_ID,
                                       agentRuntimeArtifact={
                                           'containerConfiguration': {
                                               'containerUri': (f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}'
                                                                f'.amazonaws.com/{CONTAINER_IMAGE}')
                                           }
                                       },
                                       networkConfiguration={"networkMode": "PUBLIC"},
                                       roleArn=IAM_AGENTCORE_ROLE,
                                       environmentVariables={'NVIDIA_API_KEY': nvidia_api_key})

print("Agent Runtime updated successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
