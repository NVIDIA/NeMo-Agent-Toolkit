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
<!-- path-check-skip-file -->

# Running Strands with NAT on AWS AgentCore

A comprehensive guide for deploying NVIDIA NeMo Agent Toolkit (NAT) with Strands on AWS AgentCore, including OpenTelemetry instrumentation for monitoring.

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker
- git
- git Large File Storage (LFS)
- uv
- AWS CLI

## Step 1: Setup NeMo Agent Toolkit Environment

Follow the official NeMo Agent Toolkit installation guide:

```plaintext
https://docs.nvidia.com/nemo/agent-toolkit/1.3/quick-start/installing.html
```

## Step 2: Install and Test the Agent Locally

### Install the Example Package

```bash
uv pip install -e examples/frameworks/strands_demo
```

### Build the Docker Image

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t strands_demo \
  -f examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --load .
```

### Configure AWS CLI

```bash
aws configure
```
Enter your AWS ACCESS KEY, AWS SECRET ACCESS KEY, and REGION for your AWS Account.

### Setup AWS ENV Variables

```bash
    export AWS_ACCESS_KEY_ID=$(aws configure get default.aws_access_key_id)
    export AWS_SECRET_ACCESS_KEY=$(aws configure get default.aws_secret_access_key)
    export AWS_DEFAULT_REGION=$(aws configure get default.region)
```

### Run the Container Locally

> **Note:** The `NVIDIA_API_KEY` is required only when using NVIDIA-hosted NIM endpoints (default configuration). If you are using a self-hosted NVIDIA NIM or model with OAI compatible endpoint and a custom `base_url` specified in your configuration file (such as shown in `sizing_config.yml`), you do not need to set the `NVIDIA_API_KEY` environment variable.

```bash
docker run \
  -p 8080:8080 \
  -p 6006:6006 \
  -e NVIDIA_API_KEY \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  strands_demo \
  --platform linux/arm64
```

### Test Local Deployment

```bash
curl -X 'POST' \
  'http://localhost:8080/invocations' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"inputs" : "How do I use the Strands Agents API?"}'
```

**Expected Workflow Output**
The workflow produces a large amount of output, the end of the output should contain something similar to the following:

```console
Workflow Result:
['To answer your question about using the Strands Agents API, I\'ll need to search for the relevant documentation. Let me do that for you.Thank you for providing that information. To get the most relevant information about using the Strands Agents API, I\'ll fetch the content from the "strands_agent_loop" URL, as it seems to be the most relevant to your question about using the API.Based on the information from the Strands Agents documentation, I can provide you with an overview of how to use the Strands Agents API. Here\'s a summary of the key points:\n\n1. Initialization:\n   To use the Strands Agents API, you start by initializing an agent with the necessary components:\n\n   ```python\n   from strands import Agent\n   from strands_tools import calculator\n\n   agent = Agent(\n       tools=[calculator],\n       system_prompt="You are a helpful assistant."\n   )\n   ```\n\n   This sets up the agent with tools (like a calculator in this example) and a system prompt.\n\n2. Processing User Input:\n   You can then use the agent to process user input:\n\n   ```python\n   result = agent("Calculate 25 * 48")\n   ```\n\n3. Agent Loop:\n   The Strands Agents API uses an "agent loop" to process requests. This loop includes:\n   - Receiving user input and context\n   - Processing the input using a language model (LLM)\n   - Deciding whether to use tools to gather information or perform actions\n   - Executing tools and receiving results\n   - Continuing reasoning with new information\n   - Producing a final response or iterating through the loop again\n\n4. Tool Execution:\n   The agent can use tools as part of its processing. When the model decides to use a tool, it will format a request like this:\n\n   ```json\n   {\n     "role": "assistant",\n     "content": [\n       {\n         "toolUse": {\n           "toolUseId": "tool_123",\n           "name": "calculator",\n           "input": {\n             "expression": "25 * 48"\n           }\n         }\n       }\n     ]\n   }\n   ```\n\n   The API then executes the tool and returns the result to the model for further processing.\n\n5. Recursive Processing:\n   The agent loop can continue recursively if more tool executions or multi-step reasoning is required.\n\n6. Completion:\n   The loop completes when the model generates a final text response or when an unhandled exception occurs.\n\nTo effectively use the Strands Agents API, you should:\n- Initialize your agent with appropriate tools and system prompts\n- Design your tools carefully, considering token limits and complexity\n- Handle potential exceptions, such as the MaxTokensReachedException\n\nRemember that the API is designed to support complex, multi-step reasoning and actions with seamless integration of tools and language models. It\'s flexible enough to handle a wide range of tasks and can be customized to your specific needs.']
```



## Step 3: Set Up AWS Environment

If you have not set up the AWS environment in the previous step, do so now.

### Create ECR Repository

Replace `<AWS_REGION>` with your AWS region (e.g., us-west-2, us-east-1, eu-west-1):

```bash
aws ecr create-repository \
  --repository-name my-strands-demo \
  --region <AWS_REGION>
```

### Authenticate Docker with ECR

Replace `<AWS_ACCOUNT_ID>` with your AWS account ID and `<AWS_REGION>` with your region:

```bash
aws ecr get-login-password --region <AWS_REGION> | \
  docker login \
  --username AWS \
  --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
```

### Create AgentCore IAM Role

> **Note:** See Appendix 1 for detailed instructions on creating the AgentCore Runtime Role.

## Step 4: Build and Deploy Agent in AWS AgentCore

### Build and Push Docker Image to ECR

> **Important:** Never pass credentials as build arguments. Use AWS IAM roles and environment variables instead. The example below shows the structure but credentials should be managed securely.

> **Note:** The `NVIDIA_API_KEY` is required only when using NVIDIA-hosted NIM endpoints (default configuration). If you are using a self-hosted NVIDIA NIM or model with OAI compatible endpoint and a custom `base_url` specified in your configuration file (such as `base_url: <base url to NIM instance>` in `sizing_config.yml`), you do not need to provide the `NVIDIA_API_KEY`.

Replace the following placeholders:
- `<AWS_ACCOUNT_ID>` - Your AWS account ID
- `<AWS_REGION>` - Your AWS region
- `<NVIDIA_API_KEY>` - Your NVIDIA API key for hosted NIM endpoints (use environment variables or secrets manager; not needed for self-hosted NVIDIA NIM or models with custom `base_url`)
- `<AWS_ACCESS_KEY_ID>` - Your AWS access key (use IAM roles instead)
- `<AWS_SECRET_ACCESS_KEY>` - Your AWS secret key (use IAM roles instead)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  --build-arg NVIDIA_API_KEY="<NVIDIA_API_KEY>" \
  --build-arg AWS_ACCESS_KEY_ID="<AWS_ACCESS_KEY_ID>" \
  --build-arg AWS_SECRET_ACCESS_KEY="<AWS_SECRET_ACCESS_KEY>" \
  -t <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest \
  -f examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --push .
```

### Update Deployment Script

Update `examples/frameworks/strands_demo/bedrock_agentcore/scripts/deploy_nat.py` with:
- Your AWS account ID
- Your AWS region
- ECR image URI
- IAM Role ARN

**deploy_nat.py:**

```python
import boto3

client = boto3.client('bedrock-agentcore-control', region_name='<AWS_REGION>')

response = client.create_agent_runtime(
    agentRuntimeName='strands_demo',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn='<IAM_ROLE_ARN>'
)

print(f"Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
```

### Deploy the Agent

```bash
uv run examples/frameworks/strands_demo/bedrock_agentcore/scripts/deploy_nat.py
```

**Important:** Record the runtime ID from the output for the next steps. It will look something like: `strands_demo-abc123XYZ`

### Test the Deployment

Update `examples/frameworks/strands_demo/bedrock_agentcore/scripts/test_nat.py` with:
- Your AWS account ID
- Your AWS region
- Runtime ID from previous step

**test_nat2.py:**

```python
import json
import boto3

client = boto3.client('bedrock-agentcore', region_name='<AWS_REGION>')
payload = json.dumps({"inputs" : "How do I use the Strands Agents API?"})

response = client.invoke_agent_runtime(
    agentRuntimeArn='arn:aws:bedrock-agentcore:<AWS_REGION>:<AWS_ACCOUNT_ID>:runtime/<RUNTIME_ID>',
    payload=payload,
    qualifier="DEFAULT"
)

response_body = response['response'].read()
response_data = json.loads(response_body)
print("Agent Response:", response_data)
```

### Run the Test

```bash
uv run examples/frameworks/strands_demo/bedrock_agentcore/scripts/test_nat.py
```

## Step 5: Instrument for OpenTelemetry

### Update `Dockerfile` Environment Variables

Update the following environment variables in the `Dockerfile` with your Runtime ID (obtained from Step 4):

```dockerfile
ENV OTEL_RESOURCE_ATTRIBUTES=service.name=nat_test_agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/<RUNTIME_ID>

ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/<RUNTIME_ID>,x-aws-log-stream=otel-rt-logs,x-aws-metric-namespace=strands_demo
```

### Enable OpenTelemetry Instrumentation

Comment out the standard entry point:

```dockerfile
# ENTRYPOINT ["sh", "-c", "exec nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
```

And uncomment the OpenTelemetry instrumented entry point:

```dockerfile
ENTRYPOINT ["sh", "-c", "exec opentelemetry-instrument nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
```
Save the updated `Dockerfile`


### ReBuild and Push Docker Image to ECR

> **Important:** Never pass credentials as build arguments. Use AWS IAM roles and environment variables instead. The example below shows the structure but credentials should be managed securely.

> **Note:** The `NVIDIA_API_KEY` is required only when using NVIDIA-hosted NIM endpoints (default configuration). If you are using a self-hosted NVIDIA NIM or model with OAI compatible endpoint and a custom `base_url` specified in your configuration file (such as `base_url: <base url to NIM instance>` in `sizing_config.yml`), you do not need to provide the `NVIDIA_API_KEY`.

Replace the following placeholders:
- `<AWS_ACCOUNT_ID>` - Your AWS account ID
- `<AWS_REGION>` - Your AWS region
- `<NVIDIA_API_KEY>` - Your NVIDIA API key for hosted NIM endpoints (use environment variables or secrets manager; not needed for self-hosted NVIDIA NIM or models with custom `base_url`)
- `<AWS_ACCESS_KEY_ID>` - Your AWS access key (use IAM roles instead)
- `<AWS_SECRET_ACCESS_KEY>` - Your AWS secret key (use IAM roles instead)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  --build-arg NVIDIA_API_KEY \
  --build-arg AWS_ACCESS_KEY_ID \
  --build-arg AWS_SECRET_ACCESS_KEY \
  -t <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest \
  -f examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --push .
```

### Update the Agent with New Version

### Update the Update Script

Update `update_nat2.py` with:
- Your AWS account ID
- Your AWS region
- Runtime ID
- ECR image URI
- IAM Role ARN

**update_nat.py:**

```python
import boto3

client = boto3.client('bedrock-agentcore-control', region_name='<AWS_REGION>')

response = client.update_agent_runtime(
    agentRuntimeId='<RUNTIME_ID>',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '<AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/strands-demo:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn='<IAM_ROLE_ARN>'
)

print(f"Agent Runtime updated successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")
```

### Run the Update Script

```bash
uv run examples/frameworks/strands_demo/bedrock_agentcore/scripts/update_nat.py
```

### Final Test

```bash
uv run examples/frameworks/strands_demo/bedrock_agentcore/scripts/test_nat.py
```

> **Note:** If you do not see OpenTelemetry telemetry for your agent after a few test runs, please refer to Appendix 2 to ensure you have enabled OpenTelemetry support in CloudWatch.

## 🎉 Success!

You have successfully set up NAT using Strands running on AWS AgentCore with OpenTelemetry monitoring!

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

### Role Name

We recommend naming this role: `AgentCore_NAT` (or choose your own descriptive name)

---

## Permission Breakdown

The role includes the following permission sets:

| Permission Set | Purpose |
|---------------|---------|
| **Bedrock Model Access** | Invoke foundation models for AI/ML operations |
| **ECR Access** | Pull container images for runtime deployment |
| **CloudWatch Logs** | Create log groups/streams and write application logs |
| **X-Ray Tracing** | Send distributed tracing data for observability |
| **CloudWatch Metrics** | Publish custom metrics to CloudWatch |
| **Workload Identity** | Access workload identity tokens for authentication |

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
2. Click on the **AgentCore_NAT** role name
3. On the role summary page, locate and copy the **ARN** (Amazon Resource Name)

The ARN will look like this:
```plaintext
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
3. **Application Setup** - NAT package installation
4. **OpenTelemetry Configuration** - Monitoring and observability
5. **Runtime Configuration** - Entry point and environment

<details>
<summary>📄 Click to view complete `Dockerfile`</summary>

```dockerfile
# =============================================================================
# LICENSING AND COPYRIGHT
# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212
ARG PYTHON_VERSION=3.13
# Specified on the command line with --build-arg NAT_VERSION=$(python -m setuptools_scm)
ARG NAT_VERSION

# =============================================================================
# BASE IMAGE
# =============================================================================
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}

ARG PYTHON_VERSION
ARG NAT_VERSION

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# Prevent Python from writing bytecode files
ENV PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================
# Install compilers (required for thinc indirect dependency)
RUN apt-get update && \
    apt-get install -y g++ gcc && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# APPLICATION SETUP
# =============================================================================
WORKDIR /workspace

# Copy project files
COPY ./ /workspace

# Install NAT and example packages
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
    uv pip install --link-mode=copy ./examples/frameworks/strands_demo

# =============================================================================
# OPENTELEMETRY CONFIGURATION
# =============================================================================

# AWS OpenTelemetry Distribution
ENV OTEL_PYTHON_DISTRO=aws_distro
ENV OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
ENV OTEL_TRACES_EXPORTER=otlp
ENV AGENT_OBSERVABILITY_ENABLED=true

# Service Identification
# ⚠️  UPDATE THESE VALUES WITH YOUR RUNTIME ID
# Replace <RUNTIME_ID> with the ID returned from create_agent_runtime
ENV OTEL_RESOURCE_ATTRIBUTES=service.name=nat_test_agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/<RUNTIME_ID>

# CloudWatch Integration
# ⚠️  ENSURE LOG GROUP AND LOG STREAM ARE PRE-CREATED IN CLOUDWATCH
# ⚠️  UPDATE THESE VALUES WITH YOUR RUNTIME ID
# Replace <RUNTIME_ID> with the ID returned from create_agent_runtime
ENV OTEL_EXPORTER_OTLP_LOGS_HEADERS=x-aws-log-group=/aws/bedrock-agentcore/runtimes/<RUNTIME_ID>,x-aws-log-stream=otel-rt-logs,x-aws-metric-namespace=strands_demo

# Install OpenTelemetry dependencies
RUN uv pip install boto3 && \
    uv pip install aws-opentelemetry-distro

# =============================================================================
# RUNTIME CONFIGURATION
# =============================================================================

# Add virtual environment to PATH
ENV PATH="/workspace/.venv/bin:$PATH"

# Set configuration file location
ENV NAT_CONFIG_FILE=/workspace/examples/frameworks/strands_demo/configs/agentcore_config.yml

# =============================================================================
# ENTRYPOINT
# =============================================================================

# Standard entry point (without OpenTelemetry)
ENTRYPOINT ["sh", "-c", "exec nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]

# OpenTelemetry instrumented entry point (recommended for production)
# ENTRYPOINT ["sh", "-c", "exec opentelemetry-instrument nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
```

</details>

### Key Configuration Points

#### 1. Build Arguments
- `NAT_VERSION`: Required at build time via `--build-arg NAT_VERSION=$(python -m setuptools_scm)`
- `PYTHON_VERSION`: Python version (default: 3.13)
- `BASE_IMAGE_TAG`: Ubuntu base image version

#### 2. OpenTelemetry Environment Variables

| Variable | Purpose | Action Required |
|----------|---------|-----------------|
| `OTEL_RESOURCE_ATTRIBUTES` | Service name and log group | ✏️ Update with your runtime ID |
| `OTEL_EXPORTER_OTLP_LOGS_HEADERS` | CloudWatch log configuration | ✏️ Update with your runtime ID |
| `AGENT_OBSERVABILITY_ENABLED` | Enable agent observability | ✅ Set to `true` |

#### 3. Entry Point Options

**Without OpenTelemetry:**
```dockerfile
ENTRYPOINT ["sh", "-c", "exec nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
```

**With OpenTelemetry (Recommended):**
```dockerfile
ENTRYPOINT ["sh", "-c", "exec opentelemetry-instrument nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
```

---

## ⚠️ Security Best Practices

### Credential Management

**NEVER hard-code credentials in your `Dockerfile` or source code.** Always use secure credential management:

| ❌ Never Use | ✅ Use Instead |
|-------------|---------------|
| Hard-coded API keys in `Dockerfile` | AWS Secrets Manager |
| Build-arg for credentials | Environment variables at runtime |
| Embedded passwords | IAM roles for authentication |
| Committed secrets to git | AWS Systems Manager Parameter Store |

### Recommended Approach

**For NVIDIA API Key:**

> **Note:** The NVIDIA API key is only required when using NVIDIA-hosted NIM endpoints. If you are using a self-hosted NVIDIA NIM or model with OAI compatible endpoint and a custom `base_url` in your configuration (such as `base_url: <base url to NIM instance>` in your workflow config), you do not need the NVIDIA API key.

```bash
# Store in AWS Secrets Manager (only if using NVIDIA-hosted endpoints)
aws secretsmanager create-secret \
  --name nvidia-api-key \
  --secret-string "<NVIDIA_API_KEY>" \
  --region <AWS_REGION>

# Reference in ECS/AgentCore configuration
```

**For AWS Credentials:**
- Use IAM roles attached to the AgentCore runtime
- Never use access keys when IAM roles are available
- Enable MFA for sensitive operations

### `Dockerfile` Best Practices

```dockerfile
# ❌ WRONG - Never do this
ENV NVIDIA_API_KEY="nvapi-xxxxx"
ENV AWS_ACCESS_KEY_ID="AKIAxxxxx"

# ✅ CORRECT - Expect credentials from runtime environment
# Let the runtime inject secrets from AWS Secrets Manager
# Or use IAM roles for AWS service authentication
```

### Action Items Before Deployment

- [ ] Remove all hard-coded credentials from code
- [ ] Set up AWS Secrets Manager for API keys
- [ ] Configure IAM roles for AgentCore runtime
- [ ] Enable CloudWatch logging with proper IAM permissions
- [ ] Rotate any exposed credentials immediately
- [ ] Review security group configurations
- [ ] Enable AWS CloudTrail for audit logging

---

## Placeholder Reference

Throughout this guide, replace the following placeholders with your actual values:

| Placeholder | Description | Example |
|------------|-------------|---------|
| `<AWS_ACCOUNT_ID>` | Your AWS account ID | `123456789012` |
| `<AWS_REGION>` | Your AWS region | `us-west-2`, `us-east-1`, `eu-west-1` |
| `<RUNTIME_ID>` | AgentCore runtime ID | `strands_demo-abc123XYZ` |
| `<NVIDIA_API_KEY>` | Your NVIDIA API key (only for hosted NIM endpoints) | Retrieve from secrets manager; not needed for self-hosted NVIDIA NIM or models with custom `base_url` |
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

- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/1.2/)
- [AWS Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/)
- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [AWS CloudWatch Logs Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [AWS IAM Roles Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html)
- [AWS Regions and Endpoints](https://docs.aws.amazon.com/general/latest/gr/rande.html)
