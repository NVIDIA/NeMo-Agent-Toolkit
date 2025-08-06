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
  - [Setting up MinIO (Optional)](#setting-up-minio-optional)
  - [Setting up the MySQL Server (Optional)](#setting-up-the-mysql-server-optional)
- [NeMo Agent Toolkit File Server](#nemo-agent-toolkit-file-server)
  - [Using the Object Store Backed File Server](#using-the-object-store-backed-file-server)
- [Run the Workflow](#run-the-workflow)
  - [Example 1](#example-1)
  - [Example 2](#example-2)
  - [Example 3](#example-3)
  - [Example 4 (Continued from Example 3)](#example-4-continued-from-example-3)

## Key Features

- **Object Store Integration:** Demonstrates comprehensive integration with object storage systems including AWS S3 and MinIO for storing and retrieving user report data.
- **Multi-Database Support:** Shows support for both object stores (S3-compatible) and relational databases (MySQL) for flexible data storage architectures.
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

### Setting up MinIO (Optional)
If you want to run this example in a local setup without creating a bucket in AWS, you can set up MinIO in your local machine. MinIO is an object storage system and acts as drop-in replacement for AWS S3.

For the up-to-date installation instructions of MinIO, see [MinIO Page](https://github.com/minio/minio) and MinIO client see [MinIO Client Page](https://github.com/minio/mc)

#### macOS
To install MinIO on your macOS machine, run the following commands:
```bash
brew install minio/stable/mc
mc --help
mc alias set myminio http://localhost:9000 minioadmin minioadmin

brew install minio/stable/minio
```

#### Linux
To install MinIO on your Linux machine, run the following commands:
```bash
curl https://dl.min.io/client/mc/release/linux-amd64/mc \
  --create-dirs \
  -o $HOME/minio-binaries/mc

chmod +x $HOME/minio-binaries/mc
export PATH=$PATH:$HOME/minio-binaries/
mc --help
mc alias set myminio http://localhost:9000 minioadmin minioadmin

wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio_20250422221226.0.0_amd64.deb -O minio.deb
sudo dpkg -i minio.deb
```

### Start the MinIO Server
To start the MinIO server, run the following command:
```bash
minio server ~/.minio
```

### Useful MinIO Commands

List buckets:
```bash
mc ls myminio
```

List all files in a bucket:
```bash
mc ls --recursive myminio/my-bucket
```

### Load Mock Data to MiniIO
To load mock data to minIO, use the `upload_to_minio.sh` script in this directory. For this example, we will load the mock user reports in the `data/object_store` directory.

```bash
cd examples/object_store/user_report/
./upload_to_minio.sh data/object_store myminio my-bucket
```

### Setting up the MySQL Server (Optional)

#### Linux (Ubuntu)

1. Install MySQL Server:
```bash
sudo apt update
sudo apt install mysql-server
```

2. Verify installation:
```
sudo systemctl status mysql
```

Make sure that the service is `active (running)`.

3. The default installation of the MySQL server allows root access only if you’re the system user "root" (socket-based authentication). To be able to connect using the root user and password, run the following command:
```
sudo mysql
```

4. Inside the MySQL console, run the following command (you can choose any password but make sure it matches the one used in the config):
```
ALTER USER 'root'@'localhost'
  IDENTIFIED WITH mysql_native_password BY 'my_password';
FLUSH PRIVILEGES;
quit
```

Note: This is not a secure configuration and should not to be used in production systems.

5. Back in the terminal:
```bash
sudo service mysql restart
```

### Load Mock Data to MySQL Server
To load mock data to the MySQL server:

1. Update the MYSQL configuration:
```bash
sudo tee /etc/mysql/my.cnf > /dev/null <<EOF
[mysqld]
secure_file_priv=""
EOF
```

2. Append this rule to MySQL's AppArmor profile local override:
```bash
echo "/tmp/** r," | sudo tee -a /etc/apparmor.d/local/usr.sbin.mysqld
```

3. Reload the AppArmor policy:
```bash
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.mysqld
```

4. Restart the MySQL server:
```bash
sudo systemctl restart mysql
```

5. Use the `upload_to_mysql.sh` script in this directory. For this example, we will load the mock user reports in the `data/object_store` directory.

```bash
cd examples/object_store/user_report/
./upload_to_mysql.sh root my_password data/object_store my-bucket
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

You can start the server by running:
```bash
aiq serve --config_file examples/object_store/user_report/configs/config_s3.yml
```

### Using the Object Store Backed File Server

- Download an object: `curl -X GET http://<hostname>:<port>/static/{file_path}`
- Upload an object: `curl -X POST http://<hostname>:<port>/static/{file_path}`
- Upsert an object: `curl -X PUT http://<hostname>:<port>/static/{file_path}`
- Delete an object: `curl -X DELETE http://<hostname>:<port>/static/{file_path}`

If any of the loading scripts were run and the files are in the object store, example commands are:

- Get an object: `curl -X GET http://localhost:8000/static/reports/67890/latest.json`
- Delete an object: `curl -X DELETE http://localhost:8000/static/reports/67890/latest.json`


## Run the Workflow

Run the following command from the root of the NeMo Agent toolkit repo to execute this workflow with the specified input:

### Example 1
```
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
```
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

### Example 4 (Continued from Example 3)
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
