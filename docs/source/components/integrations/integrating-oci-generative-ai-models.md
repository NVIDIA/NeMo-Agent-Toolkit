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

# NVIDIA NeMo Agent Toolkit OCI Integration

The NeMo Agent Toolkit supports integration with multiple [LLM](../../build-workflows/llms/index.md) providers, including OCI Generative AI. The `oci` provider uses OCI SDK authentication and is designed for OCI Generative AI model and endpoint access. For workflow parity with the AWS Bedrock path, the toolkit also includes a LangChain wrapper built on `langchain-oci`.

To view the full list of supported LLM providers, run `nat info components -t llm_provider`.

## Configuration

### Prerequisites
Before integrating OCI, ensure you have:

- access to OCI Generative AI in the target region
- a valid OCI auth method such as `API_KEY`, `SECURITY_TOKEN`, `INSTANCE_PRINCIPAL`, or `RESOURCE_PRINCIPAL`
- the target compartment OCID
- the target OCI region (defaults to `us-chicago-1`) or a custom endpoint URL

Common deployment patterns include:

- OCI Generative AI regional endpoints
- custom OCI Generative AI endpoints
- OCI-hosted inference for NVIDIA Nemotron used as a live integration target

### Example Configuration
Add the OCI LLM configuration to your workflow config file:

```yaml
llms:
  oci_llm:
    _type: oci
    model_name: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
    region: us-chicago-1
    compartment_id: ocid1.compartment.oc1..example
    auth_type: API_KEY
    auth_profile: DEFAULT
    temperature: 0.0
    max_tokens: 1024
    top_p: 1.0
    request_timeout: 60
```

### Configurable Options
* `model_name`: The name of the OCI-hosted model to use (required)
* `region`: OCI region for the Generative AI service (defaults to `us-chicago-1`). The service endpoint is derived automatically.
* `endpoint`: Optional explicit service endpoint URL. Overrides the region-derived endpoint when set.
* `compartment_id`: OCI compartment OCID
* `auth_type`: OCI SDK auth type
* `auth_profile`: OCI profile name for file-backed auth
* `auth_file_location`: Path to the OCI config file
* `provider`: Optional OCI provider override such as `meta`, `google`, `cohere`, or `openai`
* `temperature`: Controls randomness in the output (0.0 to 1.0)
* `max_tokens`: Maximum number of tokens to generate
* `top_p`: Top-p sampling parameter (0.0 to 1.0)
* `seed`: Optional random seed
* `max_retries`: Maximum number of retries for the request
* `request_timeout`: HTTP request timeout in seconds

### Limitations
* This provider targets OCI Generative AI through the OCI SDK-backed `langchain-oci` path.
* The Responses API is not enabled for this provider in the current release.

## Nemotron On OCI

One strong OCI deployment pattern is NVIDIA Nemotron hosted on OCI and exposed through an OpenAI-compatible route. In that setup, the toolkit can validate live integration behavior against the OCI-hosted Nemotron endpoint while the official provider and LangChain wrapper cover the OCI Generative AI path.

## Usage
Reference the OCI LLM in your configuration:

```yaml
llms:
  oci_llm:
    _type: oci
    model_name: nvidia/Llama-3.1-Nemotron-Nano-8B-v1
    region: us-chicago-1
    compartment_id: ocid1.compartment.oc1..example
    auth_profile: DEFAULT
```

## Troubleshooting
* `401 Unauthorized`: verify the OCI profile, signer, and IAM permissions for Generative AI.
* `404 Not Found`: confirm the regional endpoint or custom endpoint URL is correct.
* `Connection errors`: verify OCI networking and regional endpoint reachability.
* `Tool calling issues`: verify the served model supports tool calling and that the serving stack is configured for it.
