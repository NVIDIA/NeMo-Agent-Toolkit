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

# Integrating AWS Bedrock Models to AIQ Toolkit Workflow

To integrate AWS Bedrock models into your AIQ Toolkit workflow, follow these steps:

1. **Prerequisites**:
   - Set up AWS credentials by configuring `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. For detailed setup instructions, refer to the [AWS Bedrock setup guide](https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html)

1. **Configuration**:
   Add the AWS Bedrock LLM configuration to your workflow config file. Make sure the `region_name` matches the region of your `AWS` account, and the `credentials_profile_name` matches the field in your credential file. Here's an example:

```yaml
llms:
  aws_bedrock_llm:
    _type: aws_bedrock
    model_name: meta.llama3-3-70b-instruct-v1:0
    temperature: 0.0
    max_tokens: 1024
    region_name: us-east-2
    credentials_profile_name: default
```

3. **Usage in Workflow**:
   Reference the AWS Bedrock LLM in your workflow configuration:

```yaml
workflow:
  _type: react_agent
  llm_name: aws_bedrock_llm
  # ... other workflow configurations
```

The AWS Bedrock integration supports various models and configurations, allowing you to leverage AWS's managed LLM services within your AIQ Toolkit workflows.
