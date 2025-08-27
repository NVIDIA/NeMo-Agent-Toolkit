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

# Report Tool for NVIDIA NeMo Agent Toolkit

And example tool in the NeMo Agent toolkit that makes use of an Object Store to retrieve data.

## Table of Contents

- [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
  - [Choose an Object Store](#choose-an-object-store)
    - [Setting up MinIO](#setting-up-minio)
    - [Setting up MySQL](#setting-up-mysql)
    - [Setting up Redis](#setting-up-redis)
  - [Loading Mock Data](#loading-mock-data)
- [NeMo Agent Toolkit File Server](#nemo-agent-toolkit-file-server)
  - [Using the Object Store Backed File Server (Optional)](#using-the-object-store-backed-file-server-optional)
- [Run the Workflow](#run-the-workflow)
  - [Get User Report](#get-user-report)
  - [Put User Report](#put-user-report)
  - [Update User Report](#update-user-report)
  - [Delete User Report](#delete-user-report)

## Key Features

- **Object Store Integration:** Demonstrates comprehensive integration with object storage systems including AWS S3 and MinIO for storing and retrieving user report data.
- **Multi-Database Support:** Shows support for object stores (S3-compatible), relational databases (MySQL), and key-value stores (Redis) for flexible data storage architectures.
- **File Server Backend:** Provides a complete file server implementation with object store backing, supporting REST API operations for upload, download, update, and delete.
- **Real-Time Report Management:** Enables dynamic creation, retrieval, and management of user reports through natural language interfaces with automatic timestamp handling.
- **Mock Data Pipeline:** Includes complete setup scripts and mock data for testing object store workflows without requiring production data sources.

## Installation and Setup
If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit, and follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key.

### Install this Workflow

From the root directory of the NeMo Agent toolkit repository, run the following commands:

```bash
uv pip install -e examples/object_store/user_report
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Choose an Object Store

You must choose an object store to use for this example. The in-memory object store is useful for transient use cases, but is not particularly useful for this example due to the lack of persistence.

#### Setting up MinIO

If you want to run this example in a local setup without creating a bucket in AWS, you can set up MinIO in your local machine. MinIO is an object storage system and acts as drop-in replacement for AWS S3.

You can use the [docker-compose.minio.yml](../../deploy/docker-compose.minio.yml) file to start a MinIO server in a local docker container.

```bash
docker compose -f examples/deploy/docker-compose.minio.yml up -d
```

> [!NOTE]
> This is not a secure configuration and should not be used in production systems.

#### Setting up MySQL

If you want to use a MySQL server, you can use the [docker-compose.mysql.yml](../../deploy/docker-compose.mysql.yml) file to start a MySQL server in a local docker container.

You should first specify the `MYSQL_ROOT_PASSWORD` environment variable.

```bash
export MYSQL_ROOT_PASSWORD=<password>
```

Then start the MySQL server.

```bash
docker compose -f examples/deploy/docker-compose.mysql.yml up -d
```

> [!NOTE]
> This is not a secure configuration and should not be used in production systems.

#### Setting up Redis

If you want to use a Redis server, you can use the [docker-compose.redis.yml](../../deploy/docker-compose.redis.yml) file to start a Redis server in a local docker container.

```bash
docker compose -f examples/deploy/docker-compose.redis.yml up -d
```

> [!NOTE]
> This is not a secure configuration and should not be used in production systems.

### Loading Mock Data

This example uses mock data to demonstrate the functionality of the object store. Mock data can be loaded to the object store by running the following commands based on the object store selected.

```bash
# Load mock data to MinIO
nat object-store \
  s3 --endpoint-url http://127.0.0.1:9000 --access-key minioadmin --secret-key minioadmin my-bucket \
  upload ./examples/object_store/user_report/data/object_store/

# Load mock data to MySQL
nat object-store \
  mysql --host 127.0.0.1 --username root --password ${MYSQL_ROOT_PASSWORD} --port 3306 my-bucket \
  upload ./examples/object_store/user_report/data/object_store/

# Load mock data to Redis
nat object-store \
  redis --host 127.0.0.1 --port 6379 --db 0 my-bucket \
  upload ./examples/object_store/user_report/data/object_store/
```

There are additional command-line arguments that can be used to specify authentication credentials for some object stores.

## NeMo Agent Toolkit File Server

By adding the `object_store` field in the `general.front_end` block of the configuration, clients directly download and
upload files to the connected object store. An example configuration looks like:

```yaml
general:
  front_end:
    object_store: my_object_store
    ...

object_stores:
  my_object_store:
  ...
```

You can start the file server by running the following command with the appropriate configuration file:
```bash
nat serve --config_file examples/object_store/user_report/configs/config_s3.yml
```

The above command will use the S3-compatible object store. Other configuration files are available in the `configs` directory for the different object stores.

> [!NOTE]
> The only way to populate the in-memory object store is through `nat serve` followed by the appropriate `PUT` or `POST` request. All subsequent interactions must be done through the REST API rather than through `nat run`.

### Using the Object Store Backed File Server (Optional)

- Download an object: `curl -X GET http://<hostname>:<port>/static/{file_path} -o {filename}`
- Upload an object: `curl -X POST http://<hostname>:<port>/static/{file_path} --data-binary @{filename}`
- Upsert an object: `curl -X PUT http://<hostname>:<port>/static/{file_path} --data-binary @{filename}`
- Delete an object: `curl -X DELETE http://<hostname>:<port>/static/{file_path}`

If any of the loading scripts were run and the files are in the object store, example commands are:

- Get an object: `curl -X GET http://localhost:8000/static/reports/67890/latest.json`
- Delete an object: `curl -X DELETE http://localhost:8000/static/reports/67890/latest.json`

## Run the Workflow

For each of the following examples, a command is provided to run the workflow with the specified input. Run the following command from the root of the NeMo Agent toolkit repo to execute the workflow.

You have three options for running the workflow:
1. Using the S3-compatible object store (`config_s3.yml`)
2. Using the MySQL object store (`config_mysql.yml`)
3. Using the Redis object store (`config_redis.yml`)

The configuration file used in the examples below is `config_s3.yml` which uses an S3-compatible object store.
You can change the configuration file by changing the `--config_file` argument to `config_mysql.yml` for the MySQL server
or `config_redis.yml` for the Redis server.

### Get User Report
```
nat run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 67890"
```

**Expected Workflow Output**
```console
<snipped for brevity>

[AGENT]
Calling tools: get_user_report
Tool's input: {"user_id": "67890", "date": null}

<snipped for brevity>

Workflow Result:
['The latest report of user 67890 is:\n\n{\n    "user_id": "67890",\n    "timestamp": "2025-04-21T15:40:00Z",\n    "system": {\n      "os": "macOS 14.1",\n      "cpu_usage": "43%",\n      "memory_usage": "8.1 GB / 16 GB",\n      "disk_space": "230 GB free of 512 GB"\n    },\n    "network": {\n      "latency_ms": 95,\n      "packet_loss": "0%",\n      "vpn_connected": true\n    },\n    "errors": [],\n    "recommendations": [\n      "System operating normally",\n      "No action required"\n    ]\n}']
```

In the case of a non-existent report, the workflow will return an error message.

```
nat run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 12345"
```

**Expected Workflow Output**
```console
<snipped for brevity>

Workflow Result:
['The report for user 12345 is not available.']
```

### Put User Report
```bash
nat run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Create a latest report for user 6789 with the following JSON contents:
    {
        "recommendations": [
            "Update graphics driver",
            "Check for overheating hardware",
            "Enable automatic crash reporting"
        ]
    }
'
```

**Expected Workflow Output**
```console
<snipped for brevity>

[AGENT]
Calling tools: put_user_report
Tool's input: {"report": "{\n    \"recommendations\": [\n        \"Update graphics driver\",\n        \"Check for overheating hardware\",\n        \"Enable automatic crash reporting\"\n    ]\n}", "user_id": "6789", "date": null}
Tool's response:
User report for 678901 with date latest added successfully

<snipped for brevity>

Workflow Result:
['The latest report for user 6789 has been created with the provided JSON contents.']
```

If you attempt to put a report for a user and date that already exists, the workflow will return an error message. Rerunning the workflow should produce the following output:

**Expected Workflow Output**
```console
<snipped for brevity>

[AGENT]
Calling tools: put_user_report
Tool's input: {"report": "{\"recommendations\": [\"Update graphics driver\", \"Check for overheating hardware\", \"Enable automatic crash reporting\"]}", "user_id": "6789", "date": null}
Tool's response:
User report for 6789 with date latest already exists

<snipped for brevity>

Workflow Result:
['The report for user 6789 with date "latest" already exists and cannot be replaced.']
```

### Update User Report
```bash
nat run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Update the latest report for user 6789 with the following JSON contents:
    {
        "recommendations": [
            "Update graphics driver",
            "Check for overheating hardware",
            "Reboot the system"
        ]
    }
'
```

**Expected Workflow Output**
```console
<snipped for brevity>

[AGENT]
Calling tools: update_user_report
Tool's input: {"report": "{\"recommendations\": [\"Update graphics driver\", \"Check for overheating hardware\", \"Reboot the system\"]}", "user_id": "6789", "date": null}
Tool's response:
User report for 6789 with date latest updated

<snipped for brevity>

Workflow Result:
['The latest report for user 6789 has been updated with the provided JSON contents.']
```

### Delete User Report
```bash
nat run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Delete the latest report for user 6789'
```

**Expected Workflow Output**
```console
<snipped for brevity>

[AGENT]
Calling tools: delete_user_report
Tool's input: {"user_id": "6789", "date": null}
Tool's response:
User report for 6789 with date latest deleted

<snipped for brevity>

Workflow Result:
['The latest report for user 6789 has been successfully deleted.']
```

If you attempt to delete a report that does not exist, the workflow will return an error message. Rerunning the workflow should produce the following output:

**Expected Workflow Output**
```console
<snipped for brevity>

[AGENT]
Calling tools: delete_user_report
Tool's input: {"user_id": "6789", "date": null}
Tool's response:
Tool call failed after all retry attempts. Last error: No object found with key: /reports/6789/latest.json. An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.

<snipped for brevity>

Workflow Result:
['The report for user 6789 does not exist, so it cannot be deleted.']
```
