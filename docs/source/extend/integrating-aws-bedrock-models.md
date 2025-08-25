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

# AWS Bedrock Integration

The NeMo Agent toolkit supports integration with multiple LLM providers, including AWS Bedrock. This documentation provides a comprehensive guide on how to integrate AWS Bedrock models into your NeMo Agent toolkit workflow. To view the full list of supported LLM providers, run `nat info components -t llm_provider`.


## Configuration

### Prerequisites
Before integrating AWS Bedrock, ensure you have:
- Set up AWS credentials by configuring `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- For detailed setup instructions, refer to the [AWS Bedrock setup guide](https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html)

### Example Configuration
Add the AWS Bedrock LLM configuration to your workflow config file. Make sure the `region_name` matches the region of your AWS account, and the `credentials_profile_name` matches the field in your credential file:

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

### Configurable Options
* `model_name`: The name of the AWS Bedrock model to use (required)
* `temperature`: Controls randomness in the output (0.0 to 1.0, default: 0.0)
* `max_tokens`: Maximum number of tokens to generate (must be > 0, default: 1024)
* `context_size`: Maximum number of tokens for context (must be > 0, default: 1024, required for LlamaIndex)
* `region_name`: AWS region where your Bedrock service is hosted (default: "None")
* `base_url`: Custom Bedrock endpoint URL (default: None, needed if you don't want to use the default us-east-1 endpoint)
* `credentials_profile_name`: AWS credentials profile name from ~/.aws/credentials or ~/.aws/config files (default: None)

## Usage in Workflow
Reference the AWS Bedrock LLM in your workflow configuration:

```yaml
workflow:
  _type: react_agent
  llm_name: aws_bedrock_llm
  # ... other workflow configurations
```
