<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Running Strands with NVIDIA NeMo Agent Toolkit on AWS AgentCore

A comprehensive guide for deploying NVIDIA NeMo Agent toolkit with Strands on AWS AgentCore, including OpenTelemetry instrumentation for monitoring.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker
- git
- git Large File Storage (LFS)
- uv (Python environment with `uv pip install setuptools setuptools-scm`)
- AWS CLI

## Step 1: Setup NeMo Agent Toolkit Environment

Follow the official NeMo Agent toolkit installation guide:

```text
https://docs.nvidia.com/nemo/agent-toolkit/latest/quick-start/installing.html
```

## Step 2: Configure AWS CLI

```bash
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_REGION
unset AWS_DEFAULT_REGION
aws configure
```
Enter your AWS ACCESS KEY, AWS SECRET ACCESS KEY, and REGION for your AWS Account.

**Note**: If you are using temporary AWS credentials (such as those from AWS SSO, assumed roles, or session tokens), these credentials typically expire after one to 12 hours. If you receive authentication errors like `InvalidClientTokenId` or `UnrecognizedClientException`, you need to refresh your credentials by re-authenticating with your credential provider.

After configuring AWS CLI, verify that your credentials are valid:

```bash
aws sts get-caller-identity
```

This command returns your AWS Account ID, User ARN, and User ID if authentication is successful. If you receive an error, check that your credentials are correct and have not expired.

### Setup AWS ENV Variables

```bash
export AWS_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
# export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
# export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
export AWS_DEFAULT_REGION="YOUR_AWS_DEFAULT_REGION"
```
### Set Account for local configuration

## Step 3 Create AWS Secrets Manager entry for NVIDIA_API_KEY
This is needed for storing the API keys needed for running NeMo Agent toolkit workflow.

## Secrets Manager Prerequisites

- AWS CLI installed and configured
- Appropriate IAM permissions to create secrets in AWS Secrets Manager
- Your NVIDIA API key

## Create the Secret

Use the following AWS CLI command to create the secret:

```bash
aws secretsmanager create-secret \
  --name nvidia-api-credentials \
  --description "NVIDIA API credentials for NAT agent runtime" \
  --secret-string '{"NVIDIA_API_KEY":"<YOUR-NVIDIA-API-KEY-HERE>"}' \
  --region $AWS_DEFAULT_REGION
```

Replace `YOUR-NVIDIA-API-KEY-HERE` with your actual NVIDIA GPU Cloud API key. Warning: will throw a "ResourceExistsException" if the secret already exists for this region.

## Verify the Secret

To verify the secret was created successfully:

```bash
aws secretsmanager describe-secret \
  --secret-id nvidia-api-credentials \
  --region $AWS_DEFAULT_REGION
```

## Step 4: Install and Test the Agent Locally

### Install the Example Package

```bash
uv pip install -e examples/frameworks/strands_demo
```

### Make the script executable
```bash
chmod +x examples/frameworks/strands_demo/bedrock_agentcore/scripts/run_nat_no_OTEL.sh
```

### Build the Docker Image

<!-- path-check-skip-begin -->
```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t strands_demo \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --load .
```
<!-- path-check-skip-end -->

### Run the Container Locally
<!-- path-check-skip-begin -->
```bash
docker run \
  -p 8080:8080 \
  -p 6006:6006 \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e AWS_ACCESS_KEY_ID="your-access-key-here" \
  -e AWS_SECRET_ACCESS_KEY="your-secret-key-here" \
  -e AWS_SESSION_TOKEN="your-session-token-here" \
  -e AWS_DEFAULT_REGION="your-region" \
  strands_demo \
  --platform linux/arm64
```
<!-- path-check-skip-end -->

### Test Local Deployment

<!-- path-check-skip-begin -->
```bash
curl -X 'POST' \
  'http://localhost:8080/invocations' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"inputs" : "How do I use the Strands Agents API?"}'
```
<!-- path-check-skip-end -->

**Expected Workflow Output**
The workflow produces a large amount of output, the end of the output should contain something similar to the following:

```console
Workflow Result:
['To answer your question about using the Strands Agents API, I\'ll need to search for the relevant documentation. Let me do that for you.Thank you for providing that information. To get the most relevant information about using the Strands Agents API, I\'ll fetch the content from the "strands_agent_loop" URL, as it seems to be the most relevant to your question about using the API.Based on the information from the Strands Agents documentation, I can provide you with an overview of how to use the Strands Agents API. Here\'s a summary of the key points:\n\n1. Initialization:\n   To use the Strands Agents API, you start by initializing an agent with the necessary components:\n\n   ```python\n   from strands import Agent\n   from strands_tools import calculator\n\n   agent = Agent(\n       tools=[calculator],\n       system_prompt="You are a helpful assistant."\n   )\n   ```\n\n   This sets up the agent with tools (like a calculator in this example) and a system prompt.\n\n2. Processing User Input:\n   You can then use the agent to process user input:\n\n   ```python\n   result = agent("Calculate 25 * 48")\n   ```\n\n3. Agent Loop:\n   The Strands Agents API uses an "agent loop" to process requests. This loop includes:\n   - Receiving user input and context\n   - Processing the input using a language model (LLM)\n   - Deciding whether to use tools to gather information or perform actions\n   - Executing tools and receiving results\n   - Continuing reasoning with new information\n   - Producing a final response or iterating through the loop again\n\n4. Tool Execution:\n   The agent can use tools as part of its processing. When the model decides to use a tool, it will format a request like this:\n\n   ```json\n   {\n     "role": "assistant",\n     "content": [\n       {\n         "toolUse": {\n           "toolUseId": "tool_123",\n           "name": "calculator",\n           "input": {\n             "expression": "25 * 48"\n           }\n         }\n       }\n     ]\n   }\n   ```\n\n   The API then executes the tool and returns the result to the model for further processing.\n\n5. Recursive Processing:\n   The agent loop can continue recursively if more tool executions or multi-step reasoning is required.\n\n6. Completion:\n   The loop completes when the model generates a final text response or when an unhandled exception occurs.\n\nTo effectively use the Strands Agents API, you should:\n- Initialize your agent with appropriate tools and system prompts\n- Design your tools carefully, considering token limits and complexity\n- Handle potential exceptions, such as the MaxTokensReachedException\n\nRemember that the API is designed to support complex, multi-step reasoning and actions with seamless integration of tools and language models. It\'s flexible enough to handle a wide range of tasks and can be customized to your specific needs.']
```



## Step 5: Set Up ECR

If you have not set up the AWS environment in the previous step, do so now.

### Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name strands-demo \
  --region $AWS_DEFAULT_REGION
```

### Authenticate Docker with ECR

```bash
aws ecr get-login-password --region $AWS_DEFAULT_REGION | \
  docker login \
  --username AWS \
  --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
```

### Create AgentCore IAM Role

> **Note:** See Appendix 1 for detailed instructions on creating the AgentCore Runtime Role.

## Step 6: Build and Deploy Agent in AWS AgentCore

### Build and Push Docker Image to ECR

> **Important:** Never pass credentials as build arguments. Use AWS IAM roles and environment variables instead. The example below shows the structure but credentials should be managed securely.

<!-- path-check-skip-begin -->
```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/strands-demo:latest \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --push .
```
<!-- path-check-skip-end -->

### Verify Deployment Script
Verify the REGION, ACCOUNT_ID, and ROLE are correct for your environment

**deploy_nat.py:**

```python

import boto3
import os

# Configuration
AWS_REGION = os.environ['AWS_DEFAULT_REGION']
AWS_ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
IAM_AGENTCORE_ROLE = f'arn:aws:iam::{os.environ.get("AWS_ACCOUNT_ID")}:role/AgentCore_NAT'
CONTAINER_IMAGE = 'strands-demo'
AGENT_NAME = 'strands_demo'

client = boto3.client(
    'bedrock-agentcore-control',
    region_name=AWS_REGION
)

response = client.create_agent_runtime(
    agentRuntimeName=AGENT_NAME,
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': (
                f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}'
                f'.amazonaws.com/{CONTAINER_IMAGE}:latest'
            )
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn=IAM_AGENTCORE_ROLE,
    environmentVariables={
        'AWS_DEFAULT_REGION': AWS_REGION
    },
)

print("Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"export AGENT_RUNTIME_ARN={response['agentRuntimeArn']}")
print(f"Status: {response['status']}")

```

### Deploy the Agent

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/deploy_nat.py
```

**Important:** Record the runtime ID from the output for the next steps. It will look something like: `strands_demo-abc123XYZ`

Copy and Paste the export command from output into your shell for easier configuration.

### Test the Deployment

You can test your agent in AgentCore with the following script:

**test_nat.py:**

```python

import json
import boto3
import os

# Configuration

AWS_REGION = os.environ['AWS_DEFAULT_REGION']
AWS_ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
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


client = boto3.client('bedrock-agentcore', region_name=AWS_REGION)
payload = json.dumps({"inputs": "What is AWS AgentCore?"})

response = client.invoke_agent_runtime(
    agentRuntimeArn=f'arn:aws:bedrock-agentcore:{AWS_REGION}:{AWS_ACCOUNT_ID}:runtime/{runtime_id}',
    payload=payload,
    qualifier="DEFAULT"  # Optional
)
response_body = response['response'].read()
response_data = json.loads(response_body)
print("Agent Response:", response_data)
```


### Run the Test

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/test_nat.py
```

## Step 7: Instrument for OpenTelemetry

### Update `Dockerfile` Environment Variables

For this step you will need your Runtime ID (obtained from Step 6) to update your `Dockerfile`:

NOTE:  If you do not have the runtime ID, you can check the AWS Console or run the following script:
```python
import boto3
import os

# Configuration

AWS_REGION = os.environ['AWS_DEFAULT_REGION']
AWS_ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
RUNTIME_NAME = "strands_demo"

cclient = boto3.client('bedrock-agentcore-control', region_name=AWS_REGION)
cresponse = cclient.list_agent_runtimes()

for runtime in cresponse['agentRuntimes']:
    if runtime['agentRuntimeName'] == RUNTIME_NAME:
        runtime_id = runtime['agentRuntimeId']
        print(f"Found runtime ID: {runtime_id}")
        break
```

You can run it here:
```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/get_agentcore_runtime_id.py
```

Update the following environment variables in the `Dockerfile` with your Runtime ID.

The location of the `Dockerfile` is:
 `./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile`

```dockerfile
ENV OTEL_RESOURCE_ATTRIBUTES=service.name=nat_test_agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/<RUNTIME_ID>

ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/<RUNTIME_ID>,x-aws-log-stream=otel-rt-logs,x-aws-metric-namespace=strands_demo
```

### Enable OpenTelemetry Instrumentation

Comment out the standard entry point:

```dockerfile
# ENTRYPOINT ["sh", "-c", "exec /workspace/examples/frameworks/strands_demo/bedrock_agentcore/scripts/run_nat_no_OTEL.sh"]
```

And uncomment the OpenTelemetry instrumented entry point:

```dockerfile
ENTRYPOINT ["sh", "-c", "exec /workspace/examples/frameworks/strands_demo/bedrock_agentcore/scripts/run_nat_with_OTEL.sh"]
```
Save the updated `Dockerfile`


### ReBuild and Push Docker Image to ECR

> **Important:** Never pass credentials as build arguments. Use AWS IAM roles and environment variables instead. The example below shows the structure but credentials should be managed securely.

<!-- path-check-skip-begin -->
```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/strands-demo:latest \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --push .
```
<!-- path-check-skip-end -->

### Update the Agent with New Version

### Update the Update Script

Since you already have the agent deployed, you will need to run an update (rather than a deploy/create)

**update_nat.py:**

```python
import boto3
import os

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

client = boto3.client(
    'bedrock-agentcore-control',
    region_name=AWS_REGION
)

response = client.update_agent_runtime(
    agentRuntimeId=runtime_id,
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': (
                f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}'
                f'.amazonaws.com/{CONTAINER_IMAGE}'
            )
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn=IAM_AGENTCORE_ROLE,
    environmentVariables={
        'AWS_DEFAULT_REGION': AWS_REGION
    },
)

print("Agent Runtime updated successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")

```

### Run the Update Script

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/update_nat.py
```

### Final Test 

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/test_nat.py
```

> **Note:** If you do not see OpenTelemetry telemetry for your agent after a few test runs, please refer to Appendix 2 to ensure you have enabled OpenTelemetry support in CloudWatch.

## 🎉 Success!

You have successfully set up NeMo Agent toolkit using Strands running on AWS AgentCore with OpenTelemetry monitoring!

---

## Appendices

### Appendix 1: Creating an AWS AgentCore Runtime Role

# Creating an AWS IAM Role for Bedrock AgentCore

This guide provides step-by-step instructions for creating an IAM role using the AWS Management Console that allows AWS Bedrock AgentCore to access necessary AWS services including ECR, CloudWatch Logs, X-Ray, and Bedrock models.

## Overview

### Purpose

This IAM role enables Bedrock AgentCore runtimes to:
- Pull Docker images from Amazon ECR
- Write logs to CloudWatch Logs
- Send traces to AWS X-Ray
- Invoke Bedrock foundation models
- Publish metrics to CloudWatch
- Access workload identity tokens
- Access your NVIDIA_API_KEY from SECRETS MANAGER

### Role Name

We recommend naming this role: `AgentCore_NAT` (or choose your own descriptive name, but you will need to update the scripts with the new role name)

---

## Permission Breakdown

The role includes the following permission sets:

| Permission Set | Purpose |
|---------------|---------|
| **Bedrock Model Access** | Invoke foundation models for AI and ML operations |
| **ECR Access** | Pull container images for runtime deployment |
| **CloudWatch Logs** | Create log groups and streams, and write application logs |
| **X-Ray Tracing** | Send distributed tracing data for observability |
| **CloudWatch Metrics** | Publish custom metrics to CloudWatch |
| **Workload Identity** | Access workload identity tokens for authentication |
| **Secrets Manager** | Access the `secret:nvidia-api-credentials` key in Secrets Manager |

---

## Prerequisites

Before creating the role, ensure you have:

- [ ] Access to the AWS Management Console
- [ ] Appropriate IAM permissions to create roles and policies
- [ ] Your AWS Account ID (you can find this in the top-right corner of the AWS Console)
- [ ] Your target AWS Region

---

## Step-by-Step Instructions

### Step 1: Navigate to IAM

1. Sign in to the [AWS Management Console](https://console.aws.amazon.com/)
2. In the search bar at the top, type **IAM** and select **IAM** from the results
3. In the left sidebar, click **Roles**
4. Click the **Create role** button

### Step 2: Configure Trust Relationship

1. Under **Trusted entity type**, select **Custom trust policy**
2. Delete the default policy in the text editor
3. Copy and paste the following trust policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowAccessToBedrockAgentcore",
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock-agentcore.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

4. Click **Next**

### Step 3: Create Custom Policy

Since we need a custom policy, we'll create it now:

1. Instead of selecting existing policies, click **Create policy** (this opens in a new browser tab)
2. In the new tab, click on the **JSON** tab
3. Delete the default policy in the text editor
4. Copy and paste the following policy:

> **Important:** Before pasting, you need to replace two placeholders:
> - Replace `<AWS_REGION>` with your AWS region (e.g., `us-west-2`, `us-east-1`, `eu-west-1`)
> - Replace `<AWS_ACCOUNT_ID>` with your 12-digit AWS account ID
>
> Your account ID is shown in the top-right corner of the console (click on your username to see it)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockPermissions",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "*"
        },
        {
            "Sid": "ECRImageAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer"
            ],
            "Resource": [
                "arn:aws:ecr:<AWS_REGION>:<AWS_ACCOUNT_ID>:repository/*"
            ]
        },
        {
            "Sid": "ECRTokenAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogStreams",
                "logs:CreateLogGroup"
            ],
            "Resource": [
                "arn:aws:logs:<AWS_REGION>:<AWS_ACCOUNT_ID>:log-group:/aws/bedrock-agentcore/runtimes/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogGroups"
            ],
            "Resource": [
                "arn:aws:logs:<AWS_REGION>:<AWS_ACCOUNT_ID>:log-group:*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:<AWS_REGION>:<AWS_ACCOUNT_ID>:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "xray:PutTraceSegments",
                "xray:PutTelemetryRecords",
                "xray:GetSamplingRules",
                "xray:GetSamplingTargets"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Resource": "*",
            "Action": "cloudwatch:PutMetricData",
            "Condition": {
                "StringEquals": {
                    "cloudwatch:namespace": "bedrock-agentcore"
                }
            }
        },
        {
            "Sid": "GetAgentAccessToken",
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:GetWorkloadAccessToken",
                "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                "bedrock-agentcore:GetWorkloadAccessTokenForUserId"
            ],
            "Resource": [
                "arn:aws:bedrock-agentcore:<AWS_REGION>:<AWS_ACCOUNT_ID>:workload-identity-directory/default",
                "arn:aws:bedrock-agentcore:<AWS_REGION>:<AWS_ACCOUNT_ID>:workload-identity-directory/default/workload-identity/*"
            ]
        },
        {
            "Sid": "SecretsManagerAccess",
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": "arn:aws:secretsmanager:*:*:secret:nvidia-api-credentials"
        }
    ]
}
```

5. Click **Next**

### Step 4: Name the Policy

1. In the **Policy name** field, enter: `AgentCore_NAT_Policy`
2. In the **Description** field, enter: `Permissions for Bedrock AgentCore to access ECR, CloudWatch, X-Ray, and Bedrock models`
3. Scroll down and review the policy summary to ensure all permissions are listed correctly
4. Click **Create policy**

### Step 5: Attach Policy to Role

1. Return to the browser tab where you were creating the role (the "Create role" page)
2. Click the **refresh icon** (🔄) next to the "Filter policies" search box to reload the policy list
3. In the search box, type: `AgentCore_NAT_Policy`
4. Select the checkbox next to **AgentCore_NAT_Policy**
5. Click **Next**

### Step 6: Name and Create the Role

1. In the **Role name** field, enter: `AgentCore_NAT`
2. In the **Description** field, enter: `IAM role for Bedrock AgentCore runtimes to access AWS services`
3. Scroll down to review the configuration:
   - **Trusted entities**: Should show `bedrock-agentcore.amazonaws.com`
   - **Permissions policies**: Should show `AgentCore_NAT_Policy`
4. Click **Create role**

### Step 7: Record the Role ARN

After the role is created, you'll be redirected to the Roles page:

1. In the search box, type: `AgentCore_NAT`
2. Click on the `AgentCore_NAT` role name
3. On the role summary page, locate and copy the **ARN** (Amazon Resource Name)

The ARN will look like this:
```
arn:aws:iam::<AWS_ACCOUNT_ID>:role/AgentCore_NAT
```

**Save this ARN** - you'll need it when deploying your AgentCore runtime!

---

## 🎉 Success!

You have successfully created the IAM role for AWS Bedrock AgentCore. You can now use this role ARN in your AgentCore deployment scripts.

### Appendix 2: Turning on OpenTelemetry Support in CloudWatch

# Enabling Transaction Search in CloudWatch Console

Enable Transaction Search to index and search X-Ray spans as structured logs in CloudWatch.

## Steps

1. Open the [AWS CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)

2. In the left navigation pane, under **Application Signals**, click **Transaction Search**

3. Click **Enable Transaction Search**

4. Select the checkbox to ingest spans as structured logs

5. Enter a percentage of spans to be indexed (start with **1%** for free)

6. Click **Enable** to confirm

---

## Permissions

If you encounter permission errors, you need specific IAM permissions. Refer to the AWS documentation for setup:

📖 [Enable Transaction Search - IAM Permissions](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Enable-TransactionSearch.html)

---

## Notes

- **1% indexing** is available at no additional cost
- You can adjust the indexing percentage later based on your needs
- Higher percentages provide more trace coverage but increase costs
---

## `Dockerfile` Reference

### Complete `Dockerfile`

The `Dockerfile` is organized into the following sections:

1. **Base Image Configuration** - Ubuntu base with Python
2. **Build Dependencies** - Compilers and build tools
3. **Application Setup** - NeMo Agent toolkit package installation
4. **OpenTelemetry Configuration** - Monitoring and observability
5. **Runtime Configuration** - Entry point and environment

<details>
<summary>📄 Click to view complete `Dockerfile`</summary>

<!-- path-check-skip-begin -->
```dockerfile

ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212
ARG PYTHON_VERSION=3.13
# Specified on the command line with --build-arg NAT_VERSION=$(python -m setuptools_scm)
ARG NAT_VERSION

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}

ARG PYTHON_VERSION
ARG NAT_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

# Install compiler [g++, gcc] (currently only needed for thinc indirect dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ gcc curl unzip jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Verify installation
CMD ["aws", "--version"]

# Set working directory
WORKDIR /workspace

# Copy the project into the container
COPY ./ /workspace

# Install the nvidia-nat package and the example package
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \
    test -n "${NAT_VERSION}" || { echo "NAT_VERSION build-arg is required" >&2; exit 1; } && \
    export SETUPTOOLS_SCM_PRETEND_VERSION=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_NVIDIA_NAT=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_NVIDIA_NAT_LANGCHAIN=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_NVIDIA_NAT_TEST=${NAT_VERSION} && \
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NAT_SIMPLE_CALCULATOR=${NAT_VERSION} && \
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \
    uv sync --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install -e '.[telemetry]' --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install --link-mode=copy ./examples/frameworks/strands_demo && \
    uv pip install boto3 aws-opentelemetry-distro && \
    find /workspace/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/.venv -type f -name "*.pyc" -delete && \
    find /workspace/.venv -type f -name "*.pyo" -delete && \
    find /workspace/.venv -name "*.dist-info" -type d -exec rm -rf {}/RECORD {} + 2>/dev/null || true && \
    rm -rf /workspace/.venv/lib/python*/site-packages/pip /workspace/.venv/lib/python*/site-packages/setuptools

# AWS OpenTelemetry Distribution
ENV OTEL_PYTHON_DISTRO=aws_distro
#OTEL_PYTHON_CONFIGURATOR=aws_configurator

# Export Protocol
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
ENV OTEL_TRACES_EXPORTER=otlp

# Enable Agent Observability
ENV AGENT_OBSERVABILITY_ENABLED=true

# Service Identification attributed (gets added to all span logs)
# Example:
# OTEL_RESOURCE_ATTRIBUTES=service.version=1.0,service.name=mcp-calculator,aws.log.group.names=mcp/mcp-calculator-logs
ENV OTEL_RESOURCE_ATTRIBUTES=service.name=nat_test_agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/<AGENTCORE_RUNTIME_ID>

# CloudWatch Integration (ensure the log group and log stream are pre-created and exists)
# Example:
# OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=mcp/mcp-calculator-logs,x-aws-log-stream=default,x-aws-metric-namespace=mcp-calculator
ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/<AGENTCORE_RUNTIME_ID>,x-aws-log-stream=otel-rt-logs,x-aws-metric-namespace=strands_demo

# Remove build dependencies and cleanup (keep ca-certificates, curl, jq, unzip)
RUN apt-mark manual ca-certificates curl jq unzip && \
    apt-get purge -y --auto-remove g++ gcc && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /workspace/.git /workspace/.github /workspace/tests /workspace/docs && \
    find /workspace -type f -name "*.md" -not -path "*/site-packages/*" -delete && \
    find /workspace -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    
# Environment variables for the venv
ENV PATH="/workspace/.venv/bin:$PATH"

# Set the config file environment variable
ENV NAT_CONFIG_FILE=/workspace/examples/frameworks/strands_demo/configs/agentcore_config.yml

# Define the entry point to start the server
ENTRYPOINT ["sh", "-c", "exec /workspace/examples/frameworks/strands_demo/bedrock_agentcore/scripts/run_nat_no_OTEL.sh"]

```
<!-- path-check-skip-end -->
---

## Placeholder Reference

Throughout this guide, replace the following placeholders with your actual values:

| Placeholder | Description | Example |
|------------|-------------|---------|
| `<AWS_ACCOUNT_ID>` | Your AWS account ID | `1234567891011` |
| `<AWS_REGION>` | Your AWS region | `us-west-2`, `us-east-1`, `eu-west-1` |
| `<RUNTIME_ID>` | AgentCore runtime ID | `strands_demo-abc123XYZ` |
| `<NVIDIA_API_KEY>` | Your NVIDIA API key | Retrieve from secrets manager |
| `<AWS_ACCESS_KEY_ID>` | AWS access key | Use IAM roles instead |
| `<AWS_SECRET_ACCESS_KEY>` | AWS secret key | Use IAM roles instead |

### Common AWS Regions

| Region Code | Region Name |
|------------|-------------|
| `us-east-1` | US East (N. Virginia) |
| `us-east-2` | US East (Ohio) |
| `us-west-1` | US West (N. California) |
| `us-west-2` | US West (Oregon) |
| `eu-west-1` | Europe (Ireland) |
| `eu-central-1` | Europe (Frankfurt) |
| `ap-southeast-1` | Asia Pacific (Singapore) |
| `ap-northeast-1` | Asia Pacific (Tokyo) |

---

## Additional Resources

- [NVIDIA NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/1.2/)
- [AWS Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/)
- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [AWS CloudWatch Logs Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [AWS IAM Roles Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html)
- [AWS Regions and Endpoints](https://docs.aws.amazon.com/general/latest/gr/rande.html)