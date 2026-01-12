<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Table of Contents

- [Prerequisites](#prerequisites)
  - [Local Development Tools](#local-development-tools)
  - [AWS Account Requirements](#aws-account-requirements)
  - [IAM Permissions for Deployment](#iam-permissions-for-deployment)
  - [AWS Console Access](#aws-console-access)
  - [Additional Requirements](#additional-requirements)
- [Step 1: Setup NeMo Agent Toolkit Environment](#step-1-setup-nemo-agent-toolkit-environment)
- [Step 2: Configure AWS CLI](#step-2-configure-aws-cli)
  - [Option A: Using Long-Term Credentials](#option-a-using-long-term-credentials)
  - [Option B: Using AWS SSO (Recommended for Organizations)](#option-b-using-aws-sso-recommended-for-organizations)
  - [Verify Your Credentials](#verify-your-credentials)
  - [Setup AWS ENV Variables](#setup-aws-env-variables)
- [Step 3: Create AWS Secrets Manager Entry for NVIDIA_API_KEY](#step-3-create-aws-secrets-manager-entry-for-nvidia_api_key)
  - [Secrets Manager Prerequisites](#secrets-manager-prerequisites)
  - [Create the Secret](#create-the-secret)
  - [Verify the Secret](#verify-the-secret)
- [Step 4: Install and Test the Agent Locally](#step-4-install-and-test-the-agent-locally)
  - [Install the Example Package](#install-the-example-package)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Container Locally](#run-the-container-locally)
  - [Test Local Deployment](#test-local-deployment-arm-and-amd-builds)
- [Step 5: Set Up ECR](#step-5-set-up-ecr)
  - [Create ECR Repository](#create-ecr-repository)
  - [Authenticate Docker with ECR](#authenticate-docker-with-ecr)
- [Step 6: Build and Deploy Agent in AWS AgentCore](#step-6-build-and-deploy-agent-in-aws-agentcore)
  - [Build and Push Docker Image to ECR](#build-and-push-docker-image-to-ecr)
  - [Deploy the Agent](#deploy-the-agent)
  - [Test the Deployment](#test-the-deployment)
- [Step 7: Instrument for OpenTelemetry](#step-7-instrument-for-opentelemetry)
  - [Update `Dockerfile` Environment Variables](#update-dockerfile-environment-variables)
  - [Enable OpenTelemetry Instrumentation](#enable-opentelemetry-instrumentation)
  - [ReBuild and Push Docker Image to ECR](#rebuild-and-push-docker-image-to-ecr)
  - [Update the Agent with New Version](#update-the-agent-with-new-version)
  - [Final Test](#final-test)
- [Troubleshooting](#troubleshooting)
- [Appendices](#appendices)
  - [Appendix 1: Creating an AWS AgentCore Runtime Role](#appendix-1-creating-an-aws-agentcore-runtime-role)
  - [Appendix 2: Turning on OpenTelemetry Support in CloudWatch](#appendix-2-turning-on-opentelemetry-support-in-cloudwatch)
- [`Dockerfile` Reference](#dockerfile-reference)
- [Placeholder Reference](#placeholder-reference)
- [Additional Resources](#additional-resources)

## Prerequisites

Before you begin, ensure you have the following:

### Local Development Tools

- **Docker** - For building and running container images
- **git** - Version control
- **git Large File Storage (LFS)** - For handling large files in the repository
- **uv with Python 3.11-3.13** - Python environment manager. After installing uv, run: `uv pip install setuptools setuptools-scm`
- **AWS CLI v2** - For interacting with AWS services

### AWS Account Requirements

- An active AWS account
- Your 12-digit **AWS Account ID** (visible in the top-right corner of the AWS Console)
- Access to a **supported region**: `us-west-2` or `us-east-1` only

> **Important:** AWS Bedrock AgentCore is only available in specific regions. Using unsupported regions such as `us-west-1` will result in DNS resolution errors.

### IAM Permissions for Deployment

The user or role running this tutorial needs the following IAM permissions:

| Service | Required Permissions | Purpose |
|---------|---------------------|---------|
| **Secrets Manager** | `secretsmanager:CreateSecret`, `secretsmanager:DescribeSecret` | Store NVIDIA API credentials |
| **ECR** | `ecr:CreateRepository`, `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage` | Create repository and push container images |
| **IAM** | `iam:CreateRole`, `iam:CreatePolicy`, `iam:AttachRolePolicy`, `iam:GetRole`, `iam:PassRole` | Create the AgentCore runtime role |
| **Bedrock AgentCore** | `bedrock-agentcore:*`, `bedrock-agentcore-control:*` | Deploy and manage agent runtimes |
| **CloudWatch** | `cloudwatch:PutMetricData`, `logs:*` | Enable observability and Transaction Search |
| **STS** | `sts:GetCallerIdentity` | Verify credentials |

> [!NOTE]
> For a quick start, you can use the `AdministratorAccess` managed policy during initial setup, then scope down permissions for production use.

### AWS Console Access

You will need access to the following AWS Console services:

- **IAM Console** - To create the `AgentCore_NAT` role and policy (see [Appendix 1](#appendix-1-creating-an-aws-agentcore-runtime-role))
- **ECR Console** - To verify repository creation and image uploads
- **Bedrock AgentCore Console** - To view and manage deployed agents
- **CloudWatch Console** - To enable Transaction Search and view logs and traces (see [Appendix 2](#appendix-2-turning-on-opentelemetry-support-in-cloudwatch))
- **Secrets Manager Console** - To manage the NVIDIA API credentials secret

> [!NOTE]
> Detailed instructions for setting up IAM permissions in the AWS console are available in Appendix 1

### Additional Requirements

- **NVIDIA API Key** - Obtain from [NVIDIA NGC](https://ngc.nvidia.com/) or [build.NVIDIA](https://build.nvidia.com). This will be stored in AWS Secrets Manager during setup.

## Step 1: Setup NeMo Agent Toolkit Environment

Follow the official NeMo Agent toolkit [installation guide](https://docs.nvidia.com/nemo/agent-toolkit/latest/quick-start/installing.html)

## Step 2: Configure AWS CLI

### Option A: Using Long-Term Credentials

If you have IAM user credentials, configure them with:

```bash
unset AWS_ACCESS_KEY_ID      # these `unset` commands are non-breaking
unset AWS_SECRET_ACCESS_KEY  # and will help with consistency across
unset AWS_SESSION_TOKEN      # multiple runs. Alternatively,
unset AWS_REGION             # `rm ~/.aws/credentials`  or `rm ~/.aws/config`
unset AWS_DEFAULT_REGION
unset AWS_PROFILE
aws configure
```

> Note: using `aws configure` requires preexisting long- or short-lived access keys for the permitted IAM user.

Enter your AWS ACCESS KEY, AWS SECRET ACCESS KEY, and REGION when prompted.

### Option B: Using AWS SSO (Recommended for Organizations)

If you use AWS SSO, log in with your profile:

```bash
aws sso login --profile your-profile-name
```
> [!NOTE]
> AWS Bedrock AgentCore is available only in specific regions. Use `us-west-2` or `us-east-1`. Other regions such as `us-west-1` are **not supported** and will result in DNS resolution errors.
> Temporary credentials (SSO, assumed roles, session tokens) expire after 1-12 hours. If you receive `InvalidClientTokenId` or `UnrecognizedClientException`, refresh your credentials.

### Verify Your Credentials

```bash
aws sts get-caller-identity
```

This command returns your AWS Account ID, User ARN, and User ID if authentication is successful.

### Setup AWS ENV Variables

```bash
eval $(aws configure export-credentials --format env)
export AWS_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
export AWS_DEFAULT_REGION="us-west-2"  # Use us-west-2 or us-east-1
```

## Step 3: Create AWS Secrets Manager Entry for NVIDIA_API_KEY
This is needed for storing the API keys needed for running NeMo Agent toolkit workflow.

### Secrets Manager Prerequisites

- AWS CLI installed and configured
- Appropriate IAM permissions to create secrets in AWS Secrets Manager
- Your NVIDIA API key

### Create the Secret

Use the following AWS CLI command to create the secret:

```bash
aws secretsmanager create-secret \
  --name nvidia-api-credentials \
  --description "NVIDIA API credentials for NAT agent runtime" \
  --secret-string '{"NVIDIA_API_KEY":"<YOUR-NVIDIA-API-KEY-HERE>"}' \
  --region $AWS_DEFAULT_REGION
```

Replace `<YOUR-NVIDIA-API-KEY-HERE>` with your actual NVIDIA API key.

> [!WARNING]
> This command will throw a `ResourceExistsException` if the secret already exists in this region.

### Verify the Secret

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

### Build the Docker Image

Choose the appropriate build command for your target architecture:

<!-- path-check-skip-begin -->

#### Option A: Build for ARM64 (Apple Silicon, AWS Graviton)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t strands_demo:arm64 \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --load .
```

#### Option B: Build for AMD64 (Intel/AMD x86_64)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t strands_demo:amd64 \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/amd64 \
  --load .
```

> [!NOTE]
> You can build and test both architectures on the same machine. Docker Desktop (macOS/Windows) and Docker with QEMU (Linux) support cross-platform emulation. Emulated builds run slower than native builds.

### Run the Container Locally

<!-- path-check-skip-end -->
Run the following command to view and set Access Key ID, Secret Access Key, and Session Token:

```bash
aws sts get-session-token --duration 3600 --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' --output text
export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID_HERE"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY_HERE"
export AWS_SESSION_TOKEN="YOUR_AWS_SESSION_TOKEN_HERE"
export AWS_DEFAULT_REGION="us-west-2"
```

Run the container using the image you built:

<!-- path-check-skip-begin -->

#### Option A: Run ARM64 Image

```bash
docker run \
  -p 8080:8080 \
  -p 6006:6006 \
  -e NVIDIA_API_KEY \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_SESSION_TOKEN \
  -e AWS_DEFAULT_REGION \
  strands_demo:arm64
```

#### Option B: Run AMD64 Image

```bash
docker run \
  -p 8080:8080 \
  -p 6006:6006 \
  -e NVIDIA_API_KEY \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_SESSION_TOKEN \
  -e AWS_DEFAULT_REGION \
  strands_demo:amd64
```

<!-- path-check-skip-end -->

> [!NOTE]
> The command above passes environment variables from your shell. Ensure they are exported before running. For SSO users, see [Troubleshooting](#troubleshooting) for how to export temporary credentials.

### Test Local Deployment (ARM and AMD builds)

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
The question should be returned with the "value" key in the JSON response. For example:

```text
{"value":"The Strands Agents API is a powerful tool for building autonomous agents that can perform complex tasks. The agent loop is the core concept that enables this, allowing models to reason and act in a recursive cycle. The loop operates on a simple principle: invoke the model, check if it wants to use a tool, execute the tool if so, then invoke the model again with the result. Repeat until the model produces a final response.\n\nTo use the Strands Agents API, you need to understand the agent loop and how it works. The loop has well-defined entry and exit points, and understanding these helps predict agent behavior and handle edge cases. The loop also has a lifecycle, with events emitted at key points that enable observation, metrics collection, and behavior modification.\n\nCommon problems that may arise when using the Strands Agents API include context window exhaustion, inappropriate tool selection, and MaxTokensReachedException. Solutions to these problems include reducing tool output verbosity, simplifying tool schemas, configuring a conversation manager with appropriate strategies, and decomposing large tasks into subtasks.\n\nThe Strands Agents API also provides higher-level patterns that build on top of the agent loop, such as conversation management strategies, hooks for observing and modifying agent behavior, multi-agent architectures, and evaluation frameworks. Understanding the loop deeply makes these advanced patterns more approachable.\n\nIn summary, the Strands Agents API is a powerful tool for building autonomous agents, and understanding the agent loop is key to using it effectively. By following the principles outlined in the documentation, you can build sophisticated agents that can perform complex tasks and achieve your goals."}
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

> [!NOTE]
> This step requires that Appendix 1 was previously followed to properly configure an IAM Role and Policy

## Step 6: Build and Deploy Agent in AWS AgentCore

### Build and Push Docker Image to ECR

> **Important:** Never pass credentials as build arguments. Use AWS IAM roles and environment variables instead. The example below shows the structure but credentials should be managed securely.

Choose the appropriate build command for your target architecture:

<!-- path-check-skip-begin -->

#### Option A: Build and Push for ARM64 (Apple Silicon, AWS Graviton)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/strands-demo:latest \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --push .
```

#### Option B: Build and Push for AMD64 (Intel/AMD x86_64)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/strands-demo:latest \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/amd64 \
  --push .
```

<!-- path-check-skip-end -->

> [!NOTE]
> AWS Graviton instances (ARM64) often provide better price-performance for containerized workloads. AMD64 is widely compatible with traditional EC2 instance types.

### Deploy the Agent

Verify your environment variables are set correctly:

```bash
echo "Account: $AWS_ACCOUNT_ID, Region: $AWS_DEFAULT_REGION"
```

Then run the deployment script:

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/deploy_nat.py
```

> [!WARNING] 
> The script will deploy an ECR instance, which will incur cost. Script source is located at [`scripts/deploy_nat.py`](scripts/deploy_nat.py) if you need to review or modify it.

**Important:** Record the runtime ID from the output for the next steps. It will look something like: `strands_demo-abc123XYZ`

Copy and Paste the export command from output into your shell for easier configuration.

### Test the Deployment

You can test your agent in AgentCore with the following script:

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/verify_nat.py
```

## Step 7: Instrument for OpenTelemetry

### Update `Dockerfile` Environment Variables

For this step you will need your Runtime ID (obtained from Step 6) to update your `Dockerfile`:

NOTE:  If you do not have the runtime ID, you can check the AWS Console or run:

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/get_agentcore_runtime_id.py
```

Update the following environment variables in the `Dockerfile` with your Runtime ID.

The location of the [`Dockerfile`](./Dockerfile) is:
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


### Rebuild and Push Docker Image to ECR

> **Important:** Never pass credentials as build arguments. Use AWS IAM roles and environment variables instead. The example below shows the structure but credentials should be managed securely.

Use the same architecture you chose in Step 6:

<!-- path-check-skip-begin -->

#### Option A: Rebuild and Push for ARM64 (AWS Graviton)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/strands-demo:latest \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/arm64 \
  --push .
```

#### Option B: Rebuild and Push for AMD64 (Intel/AMD x86_64)

```bash
docker build \
  --build-arg NAT_VERSION=$(python -m setuptools_scm) \
  -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/strands-demo:latest \
  -f ./examples/frameworks/strands_demo/bedrock_agentcore/Dockerfile \
  --platform linux/amd64 \
  --push .
```

<!-- path-check-skip-end -->

### Update the Agent with New Version

### Update the Update Script

Since you already have the agent deployed, you will need to run an update (rather than a deploy/create)

[**`update_nat.py`**](scripts/update_nat.py)

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/update_nat.py
```

### Final Test

```bash
uv run ./examples/frameworks/strands_demo/bedrock_agentcore/scripts/verify_nat.py
```

> [!NOTE] 
> If you do not see OpenTelemetry telemetry for your agent after a few test runs, please refer to Appendix 2 to ensure you have enabled OpenTelemetry support in CloudWatch.

## 🎉 Success!

You have successfully set up NeMo Agent toolkit using Strands running on AWS AgentCore with OpenTelemetry monitoring!

---

## Troubleshooting

### "Unable to locate credentials" in Docker

The container cannot access your host AWS credentials. Export them before running:

```bash
# For SSO users: export temporary credentials
eval $(aws configure export-credentials --format env)
```

Then run the Docker container with `-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN`.

### "The security token included in the request is invalid"

Your credentials have expired. Re-authenticate:

```bash
# For SSO
aws sso login --profile your-profile-name

# Then re-export credentials
eval $(aws configure export-credentials --format env)
```

### "Failed to resolve 'bedrock-agentcore-control.REGION.amazonaws.com'"

Bedrock AgentCore is not available in that region. Change to a supported region:

```bash
export AWS_DEFAULT_REGION="us-west-2"  # or us-east-1
```

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
            "Sid": "AllowBedrockAgentCore",
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

1. Instead of selecting existing policies, open IAM > Policies in a new tab and click **Create policy** (this opens in a new browser tab)
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
            "Sid": "CreateServiceLinkedRole",
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "*"
        },
        {
            "Sid": "BedrockAgentCoreControl",
            "Effect": "Allow",
            "Action": [
                "bedrock:*",
                "bedrock-agentcore:*"
            ],
            "Resource": "*"
        },
        {
            "Sid": "PassRoleToAgentCore",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "bedrock-agentcore.amazonaws.com"
                }
            }
        },
        {
            "Sid": "ECRImageAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchCheckLayerAvailability",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage"
            ],
            "Resource": [
                "arn:aws:ecr:<AWS_REGION>:<AWS_ACCOUNT_ID>:repository/*"
            ]
        },
        {
            "Sid": "ECRRepoCreate",
            "Effect": "Allow",
            "Action": [
                "ecr:CreateRepository",
                "ecr:DescribeRepositories",
                "ecr:ListImage"
            ],
            "Resource": "arn:aws:ecr:<AWS_REGION>:<AWS_ACCOUNT_ID>:repository/*"
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
                "secretsmanager:DescribeSecret",
                "secretsmanager:GetSecretValue",
                "secretsmanager:PutSecretValue",
                "secretsmanager:UpdateSecret"
            ],
            "Resource": "arn:aws:secretsmanager:*:*:secret:nvidia-api-credentials*"
        },
        {
            "Sid": "SecretsManagerCreate",
            "Effect": "Allow",
            "Action": [
                "secretsmanager:CreateSecret"
            ],
            "Resource": "*"
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

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

# Install compiler [g++, gcc] (currently only needed for thinc indirect dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ gcc curl unzip jq ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Install AWS CLI v2 (architecture-aware)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
      curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    else \
      curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    fi && \
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
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NAT_STRANDS_DEMO=${NAT_VERSION} && \
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \
    uv sync --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install -e '.[telemetry]' --link-mode=copy --compile-bytecode --python ${PYTHON_VERSION} && \
    uv pip install -e ./examples/frameworks/strands_demo --link-mode=copy && \
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

### Supported AWS Regions for Bedrock AgentCore

> [!NOTE] 
> Bedrock AgentCore is available in limited regions. The following are confirmed to work:

| Region Code | Region Name | AgentCore Support |
|------------|-------------|-------------------|
| `us-east-1` | US East (N. Virginia) | ✅ Supported |
| `us-west-2` | US West (Oregon) | ✅ Supported |
| `us-east-2` | US East (Ohio) | ⚠️ Check availability |
| `eu-west-1` | Europe (Ireland) | ⚠️ Check availability |

Regions like `us-west-1` are **not supported** for Bedrock AgentCore.

---

## Additional Resources

- [NVIDIA NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)
- [AWS Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/)
- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/languages/python/)
- [AWS CloudWatch Logs Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- [AWS IAM Roles Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html)
- [AWS Regions and Endpoints](https://docs.aws.amazon.com/general/latest/gr/rande.html)
