<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Evaluate API Endpoints
The evaluation endpoint can be used to start evaluation jobs on a remote AgentIQ server.

## Evaluation Endpoint Overview
```{mermaid}
graph TD
  A["POST /evaluate"] --> B["Background Job Created"]
  B --> C["GET /evaluate/job/{job_id}"]
  B --> D["GET /evaluate/job/last"]
  B --> E["GET /evaluate/jobs"]
```

## Evaluate Request and Response
The /evaluate endpoint allows you to start an evaluation job. The request is stored for background processing, and the server returns a job ID for tracking the job status.

### Evaluate Request
- **Route**: `/evaluate`
- **Method**: `POST`
- **Description**: Start evaluation. Evaluates the performance and accuracy of the workflow on a dataset.
- HTTP Request Example:
```bash
curl --request POST \
   --url http://localhost:8000/evaluate \
   --header 'Content-Type: application/json' \
   --data '{
    "config_file": "examples/simple/configs/eval_config.yml",
    "expiry_seconds": 600
}' | jq
```
You can optionally pipe the output to `jq` for response formatting.

### Evaluate Request Format
`AIQEvaluateRequest`:
- `config_file`: Path to the evaluation configuration file on the remote server.
- `job_id`: Unique identifier for the evaluation job. If not provided, a new job ID is generated.
- `reps`: Number of repetitions for the evaluation. Defaults to 1.
- `expiry_seconds`: Optional time (in seconds) before the job expires. This is clamped between 600 (10 min) and 86400 (24h). Defaults to 3600 seconds (1 hour).

### Evaluate Response
The evaluation request is stored as a background job in the server and the endpoint returns a job ID and status. Sample response:
```json
{
  "job_id": "882317f0-6149-4b29-872b-9c8018d64784",
  "status": "submitted"
}
```

### Evaluate Response Format
`AIQEvaluateResponse`:
- `job_id`: Unique identifier for the evaluation job.
- `status`: Status of the evaluation job. Possible values are:
**Possible `status` values**:
- `submitted` – The job has been submitted and is waiting to be processed.
- `running` – The job is currently being processed.
- `success` – The job has completed successfully.
- `failure` – The job has failed.
- `interrupted` – The job was interrupted before completion.
- `not_found` – The job ID was not found.


## Evaluate Job Status
### Job Status by ID
A submitted job's status can be checked using the job ID. The status endpoint is defined as follows:
- **Route**: `/evaluate/job/{job_id}`
- **Method**: `GET`
- **Description**: Get the status of a submitted evaluation job using the job ID.
- HTTP Request Example:
```bash
curl --request GET \
   --url http://localhost:8000/evaluate/job/882317f0-6149-4b29-872b-9c8018d64784 | jq
```

### Evaluate Job Status Response
The response contains the status of the job, including the job ID, status, and any error messages if applicable. Sample response:
```json
{
  "job_id": "882317f0-6149-4b29-872b-9c8018d64784",
  "status": "success",
  "config_file": "examples/simple/configs/eval_config.yml",
  "error": null,
  "output_path": ".tmp/aiq/examples/simple/jobs/882317f0-6149-4b29-872b-9c8018d64784",
  "created_at": "2025-04-11T17:33:38.018904Z",
  "updated_at": "2025-04-11T17:34:40.359080Z",
  "expires_at": "2025-04-11T17:44:40.359080Z"
}
```

### Job Status: Last Submitted Job
The last job status can be checked using the following endpoint:
- **Route**: `/evaluate/job/last`
- **Method**: `GET`
- **Description**: Get the status of the last submitted evaluation job.
- HTTP Request Example:
```bash
curl --request GET \
   --url http://localhost:8000/evaluate/job/last | jq
```

### Status of all jobs
The status of all jobs can be checked using the following endpoint:
- **Route**: `/evaluate/jobs`
- **Method**: `GET`
- **Description**: Get the status of all submitted evaluation jobs.
- HTTP Request Example:
```bash
curl --request GET \
   --url http://localhost:8000/evaluate/jobs | jq
```

#### Sample Response
```bash
[
  {
    "job_id": "df6fddd7-2adf-45dd-a105-8559a7569ec9",
    "status": "success",
    "config_file": "examples/simple/configs/eval_config.yml",
    "error": null,
    "output_path": ".tmp/aiq/examples/simple/jobs/df6fddd7-2adf-45dd-a105-8559a7569ec9",
    "created_at": "2025-04-11T17:33:16.711636Z",
    "updated_at": "2025-04-11T17:34:24.753742Z",
    "expires_at": "2025-04-11T17:44:24.753742Z"
  },
  ...
]
```

## Output Storage
A separate output directory is created for each job. The output directory contains the evaluation results, including the evaluation metrics and any generated files. The `jobs/{job-id}` is appended to the `eval.general.output.dir` configuration parameter in the evaluation configuration file to maintain the results of each job. If upload to remote storage is enabled, `jobs/{job-id}` is similarly appended to the `eval.general.output.remote_dir` configuration parameter in the evaluation configuration file.

### Output Directory Cleanup
As the results are maintained per-job, output directory cleanup is recommended. This can be done by enabling `eval.general.output.cleanup` in the evaluation configuration file. If this configuration is enabled, the server removes the entire contents of the output directory at the start of each job. This way only the last job's results are kept in the output directory.

### Job Expiry
You can also configure the expiry timer per-job using the `expiry_seconds` parameter in the `AIQEvaluateRequest`. The server will automatically clean up expired jobs based on this timer. The default expiry value is 3600 seconds (1 hour). The expiration time is clamped between 600 (10 min) and 86400 (24h).

This cleanup includes both the job metadata and the contents of the output directory. The most recently finished job is always preserved, even if expired. Similarly, active jobs, `["submitted", "running"]`, are exempt from cleanup.
