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
  - [Choose an Object Store](#choose-an-object-store)
    - [Setting up MinIO](#setting-up-minio)
    - [Setting up the MySQL Server](#setting-up-the-mysql-server)
  - [Loading Mock Data](#loading-mock-data)
    - [Load Mock Data to MiniIO](#load-mock-data-to-miniio)
    - [Load Mock Data to MySQL Server](#load-mock-data-to-mysql-server)
- [NeMo Agent Toolkit File Server](#nemo-agent-toolkit-file-server)
  - [Using the Object Store Backed File Server](#using-the-object-store-backed-file-server)
- [Run the Workflow](#run-the-workflow)
  - [Example 1](#example-1)
  - [Example 2](#example-2)
  - [Example 3](#example-3)
  - [Example 4 ](#example-4)

## Key Features

- **Object Store Integration:** Demonstrates comprehensive integration with object storage systems including AWS S3 and MinIO for storing and retrieving user report data.
- **Multi-Database Support:** Shows support for both object stores (S3/MinIO) and relational databases (MySQL) for flexible data storage architectures.
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

### Choose an Object Store

You must choose an object store to use for this example. It is recommended to use MinIO or MySQL.
The in-memory object store is useful for transient use cases, but is not particularly useful for this example due to the lack of persistence.

#### Setting up MinIO
If you want to run this example in a local setup without creating a bucket in AWS, you can set up MinIO in your local machine. MinIO is an object storage system and acts as drop-in replacement for AWS S3.

You can use the [docker-compose.minio.yml](../../deploy/docker-compose.minio.yml) file to start a MinIO server in a local docker container.

```bash
docker compose -f examples/deploy/docker-compose.minio.yml up -d
```

> [!NOTE]
> This is not a secure configuration and should not to be used in production systems.

#### Setting up the MySQL Server

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
> This is not a secure configuration and should not to be used in production systems.

### Loading Mock Data

This example uses mock data to demonstrate the functionality of the object store. You can load the mock data to the object store by running the following commands based on the object store you chose.

#### Load Mock Data to MiniIO
To load mock data to minIO, use the `upload_to_minio.py` script in this directory. For this example, we will load the mock user reports in the `data/object_store` directory.

```bash
cd examples/object_store/user_report/
./upload_to_minio.py data/object_store myminio my-bucket
cd -
```

#### Load Mock Data to MySQL Server
To load mock data to the MySQL server, use the `upload_to_mysql.py` script in this directory. For this example, we will load the mock user reports in the `data/object_store` directory.

```bash
cd examples/object_store/user_report/
./upload_to_mysql.py data/object_store my-bucket
cd -
```


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

You can start the server by running the following command.

```bash
aiq serve --config_file examples/object_store/user_report/configs/config_s3.yml
```

Other configuration files are available in the `configs` directory for the different object stores.

The only way to populate the in-memory object store is through `aiq serve` followed by the appropriate `PUT` or `POST` request.

### Using the Object Store Backed File Server

- Download an object: `curl -X GET http://<hostname>:<port>/static/{file_path}`
- Upload an object: `curl -X POST http://<hostname>:<port>/static/{file_path}`
- Upsert an object: `curl -X PUT http://<hostname>:<port>/static/{file_path}`
- Delete an object: `curl -X DELETE http://<hostname>:<port>/static/{file_path}`

If any of the loading scripts were run and the files are in the object store, example commands are:

- Get an object: `curl -X GET http://localhost:8000/static/reports/67890/latest.json`
- Delete an object: `curl -X DELETE http://localhost:8000/static/reports/67890/latest.json`


## Run the Workflow

You have two options for running the workflow:
1. Using the object store (`config_s3.yml`)
2. Using the MySQL server (`config_mysql.yml`)

The configuration file used in the examples below is `config_s3.yml` which uses an S3 object store.
You can change the configuration file by changing the `--config_file` argument to `config_mysql.yml` for the MySQL server.

### Example 1

```bash
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 67890"
```

**Expected Workflow Output**
```console
The latest report for user 67890 is as follows:
- Timestamp: 2025-04-21T15:40:00Z
- System:
  - OS: macOS 14.1
  - CPU Usage: 43%
  - Memory Usage: 8.1 GB / 16 GB
  - Disk Space: 230 GB free of 512 GB
- Network:
  - Latency: 95 ms
  - Packet Loss: 0%
  - VPN Connected: True
- Errors: None
- Recommendations: System operating normally, No action required.
```

### Example 2
```bash
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 12345 on April 15th 2025"
```

**Expected Workflow Output**
```console
The latest report for user 12345 on April 15th, 2025, is as follows:

- **System Information:**
  - OS: Windows 11
  - CPU Usage: 82%
  - Memory Usage: 6.3 GB / 8 GB
  - Disk Space: 120 GB free of 500 GB

- **Network Information:**
  - Latency: 240 ms
  - Packet Loss: 0.5%
  - VPN Connected: False

- **Errors:**
  - Timestamp: 2025-04-15T10:21:59Z
  - Message: "App crash detected: \'PhotoEditorPro.exe\' exited unexpectedly"
  - Severity: High

- **Recommendations:**
  - Update graphics driver
  - Check for overheating hardware
  - Enable automatic crash reporting
```

### Example 3
```bash
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Create a latest report for user 6789 with the following JSON contents:
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
The latest report for user 6789 has been successfully created with the specified recommendations.
```

### Example 4
```bash
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Get the latest report for user 6789'
```

**Expected Workflow Output**
```console
The latest report for user 6789 includes the following recommendations:
1. Update graphics driver
2. Check for overheating hardware
3. Enable automatic crash reporting
```
