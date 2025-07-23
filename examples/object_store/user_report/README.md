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

And example tool in the AIQ toolkit that makes use of an Object Store to retrieve data.

## Installation and Setup
If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit, and follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key.

### Install this Workflow:

From the root directory of the AIQ toolkit repository, run the following commands:

```bash
uv pip install -e examples/object_store/user_report
```

### Setting up MinIO (Optional)
If you want to run this example in a local setup without creating a bucket in AWS, you can set up MinIO in your local machine. MinIO is an object storage system and acts as drop-in replacement for AWS S3.

For the up-to-date installation instructions of MinIO, see [MinIO Page](https://github.com/minio/minio) and MinIO client see [MinIO Client Page](https://github.com/minio/mc)

#### MacOS
To install MinIO on your MacOS machine, run the following commands:
```
brew install minio/stable/mc
mc --help
mc alias set myminio http://localhost:9000 minioadmin minioadmin

brew install minio/stable/minio
```

#### Linux
To install MinIO on your Linux machine, run the following commands:
```
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
```
minio server ~/.minio
```

### Useful MinIO Commands

List buckets:
```
mc ls myminio
```

List all files in a bucket:
```
mc ls --recursive myminio/my-bucket
```

### Load Mock Data to MiniIO
To load mock data to minIO, use the `upload_to_minio.sh` script in this directory. For this example, we will load the mock user reports in the `data/object_store` directory.

```
cd examples/object_store/user_report/
./upload_to_minio.sh data/object_store myminio my-bucket
```

### Setting up the MySQL Server (Optional)

#### Linux (Ubuntu)

1. Install MySQL Server:
```
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
```
sudo service mysql restart
```

### Load Mock Data to MySQL Server
To load mock data to the MySQL server:

1. Update the MYSQL configuration:
```
sudo tee /etc/mysql/my.cnf > /dev/null <<EOF
[mysqld]
secure_file_priv=""
EOF
```

2. Append this rule to MySQL's AppArmor profile local override:
````
echo "/tmp/** r," | sudo tee -a /etc/apparmor.d/local/usr.sbin.mysqld
```

3. Reload the AppArmor policy:
```
sudo apparmor_parser -r /etc/apparmor.d/usr.sbin.mysqld
```

4. Restart the MySQL server:
```
sudo systemctl restart mysql
```

5. Use the `upload_to_mysql.sh` script in this directory. For this example, we will load the mock user reports in the `data/object_store` directory.

```
cd examples/object_store/user_report/
./upload_to_mysql.sh root my_password data/object_store my-bucket
```

## NeMo Agent Toolkit File Server

By adding the `object_store` field in the `general.front_end` block of the configuration, clients directly download and
upload files to the connected object store. An example configuration looks like:

```
general:
  front_end:
    object_store: my_object_store
    ...

object_stores:
  my_object_store:
  ...
```

You can start the server by running:
```
aiq serve --config_file examples/object_store/user_report/configs/config_s3.yml
```

### Using the Object Store Backed File Server

- Downloading an object: `curl -X GET http://<hostname>:<port>/static/{file_path}`
- Uploading an object: `curl -X POST http://<hostname>:<port>/static/{file_path}`
- Upserting an object: `curl -X PUT http://<hostname>:<port>/static/{file_path}`
- Deleting an object: `curl -X DELETE http://<hostname>:<port>/static/{file_path}`

If any of the loading scripts were run and the files are in the object store, example commands are:

- Getting an object: `curl -X GET http://localhost:8000/static/reports/67890/latest.json`
- Deleting an object: `curl -X DELETE http://localhost:8000/static/reports/67890/latest.json`


## Run the Workflow

Run the following command from the root of the AIQ toolkit repo to execute this workflow with the specified input:

### Example 1
```
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 67890"
```

**Expected Output**
```console
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 67890"
2025-07-02 14:21:55,362 - aiq.runtime.loader - WARNING - Loading module 'aiq.agent.register' from entry point 'aiq_agents' took a long time (201.046228 ms). Ensure all imports are inside your registered functions.
2025-07-02 14:21:55,643 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/object_store/user_report/configs/config_s3.yml'
2025-07-02 14:21:55,644 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 1
Number of Retrievers: 0

2025-07-02 14:21:56,879 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Give me the latest report of user 67890
Agent's thoughts:
Thought: I need to fetch the latest report for user 67890.
Action: get_user_report
Action Input: {"user_id": "67890", "date": null}
------------------------------
2025-07-02 14:21:56,886 - aiq_user_report.register - INFO - Fetching report from /reports/67890/latest.json
2025-07-02 14:21:56,935 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: get_user_report
Tool's input: {"user_id": "67890", "date": null}
Tool's response:
{'user_id': '67890', 'timestamp': '2025-04-21T15:40:00Z', 'system': {'os': 'macOS 14.1', 'cpu_usage': '43%', 'memory_usage': '8.1 GB / 16 GB', 'disk_space': '230 GB free of 512 GB'}, 'network': {'latency_ms': 95, 'packet_loss': '0%', 'vpn_connected': True}, 'errors': [], 'recommendations': ['System operating normally', 'No action required']}
------------------------------
2025-07-02 14:21:58,901 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Give me the latest report of user 67890
Agent's thoughts:
Thought: I now have the latest report for user 67890.

Final Answer: The latest report for user 67890 is as follows:
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
------------------------------
2025-07-02 14:21:58,905 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['The latest report for user 67890 is as follows:\n- Timestamp: 2025-04-21T15:40:00Z\n- System:\n  - OS: macOS 14.1\n  - CPU Usage: 43%\n  - Memory Usage: 8.1 GB / 16 GB\n  - Disk Space: 230 GB free of 512 GB\n- Network:\n  - Latency: 95 ms\n  - Packet Loss: 0%\n  - VPN Connected: True\n- Errors: None\n- Recommendations: System operating normally, No action required.']
--------------------------------------------------
```

### Example 2
```
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 12345 on April 15th 2025"
```

**Expected Output**
```console
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input "Give me the latest report of user 12345 on April 15th 2025"
2025-07-02 14:25:20,994 - aiq.runtime.loader - WARNING - Loading module 'aiq.agent.register' from entry point 'aiq_agents' took a long time (193.936348 ms). Ensure all imports are inside your registered functions.
2025-07-02 14:25:21,269 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/object_store/user_report/configs/config_s3.yml'
2025-07-02 14:25:21,271 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 1
Number of Retrievers: 0

2025-07-02 14:25:22,486 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Give me the latest report of user 12345 on April 15th 2025
Agent's thoughts:
Thought: I need to fetch the user diagnostic report for user ID 12345 on April 15th, 2025.
Action: get_user_report
Action Input: {"user_id": "12345", "date": "2025-04-15"}
------------------------------
2025-07-02 14:25:22,490 - aiq_user_report.register - INFO - Fetching report from /reports/12345/2025-04-15.json
2025-07-02 14:25:22,539 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: get_user_report
Tool's input: {"user_id": "12345", "date": "2025-04-15"}
Tool's response:
{'user_id': '12345', 'timestamp': '2025-04-15T10:22:30Z', 'system': {'os': 'Windows 11', 'cpu_usage': '82%', 'memory_usage': '6.3 GB / 8 GB', 'disk_space': '120 GB free of 500 GB'}, 'network': {'latency_ms': 240, 'packet_loss': '0.5%', 'vpn_connected': False}, 'errors': [{'timestamp': '2025-04-15T10:21:59Z', 'message': "App crash detected: 'PhotoEditorPro.exe' exited unexpectedly", 'severity': 'high'}], 'recommendations': ['Update graphics driver', 'Check for overheating hardware', 'Enable automatic crash reporting']}
------------------------------
2025-07-02 14:25:25,463 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Give me the latest report of user 12345 on April 15th 2025
Agent's thoughts:
Thought: I now have the latest report for user 12345 on April 15th, 2025.

Final Answer: The latest report for user 12345 on April 15th, 2025, is as follows:

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
  - Message: "App crash detected: 'PhotoEditorPro.exe' exited unexpectedly"
  - Severity: High

- **Recommendations:**
  - Update graphics driver
  - Check for overheating hardware
  - Enable automatic crash reporting
------------------------------
2025-07-02 14:25:25,466 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['The latest report for user 12345 on April 15th, 2025, is as follows:\n\n- **System Information:**\n  - OS: Windows 11\n  - CPU Usage: 82%\n  - Memory Usage: 6.3 GB / 8 GB\n  - Disk Space: 120 GB free of 500 GB\n\n- **Network Information:**\n  - Latency: 240 ms\n  - Packet Loss: 0.5%\n  - VPN Connected: False\n\n- **Errors:**\n  - Timestamp: 2025-04-15T10:21:59Z\n  - Message: "App crash detected: \'PhotoEditorPro.exe\' exited unexpectedly"\n  - Severity: High\n\n- **Recommendations:**\n  - Update graphics driver\n  - Check for overheating hardware\n  - Enable automatic crash reporting']
--------------------------------------------------
```


### Example 3
```
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

**Expected Output**
```console
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Create a latest report for user 6789 with the following JSON contents:
    {
        "recommendations": [
            "Update graphics driver",
            "Check for overheating hardware",
            "Enable automatic crash reporting"
        ]
    }
'
2025-07-02 14:28:18,977 - aiq.runtime.loader - WARNING - Loading module 'aiq.agent.register' from entry point 'aiq_agents' took a long time (209.315062 ms). Ensure all imports are inside your registered functions.
2025-07-02 14:28:19,258 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/object_store/user_report/configs/config_s3.yml'
2025-07-02 14:28:19,259 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 1
Number of Retrievers: 0

2025-07-02 14:28:21,202 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Create a latest report for user 6789 with the following JSON contents:
    {
        "recommendations": [
            "Update graphics driver",
            "Check for overheating hardware",
            "Enable automatic crash reporting"
        ]
    }

Agent's thoughts:
Thought: I need to use the `put_user_report` tool to create a latest report for user 6789 with the provided JSON contents.
Action: put_user_report
Action Input: {"report": "{\"recommendations\": [\"Update graphics driver\", \"Check for overheating hardware\", \"Enable automatic crash reporting\"]}", "user_id": "6789", "date": null}
------------------------------
2025-07-02 14:28:21,209 - aiq_user_report.register - INFO - Fetching report from /reports/6789/latest.json
2025-07-02 14:28:21,260 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: put_user_report
Tool's input: {"report": "{\"recommendations\": [\"Update graphics driver\", \"Check for overheating hardware\", \"Enable automatic crash reporting\"]}", "user_id": "6789", "date": null}
Tool's response:
None
------------------------------
2025-07-02 14:28:22,258 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Create a latest report for user 6789 with the following JSON contents:
    {
        "recommendations": [
            "Update graphics driver",
            "Check for overheating hardware",
            "Enable automatic crash reporting"
        ]
    }

Agent's thoughts:
Thought: The empty response indicates that the report was successfully created for user 6789.

Final Answer: The latest report for user 6789 has been successfully created with the specified recommendations.
------------------------------
2025-07-02 14:28:22,262 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['The latest report for user 6789 has been successfully created with the specified recommendations.']
--------------------------------------------------
```

### Example 4 (Continued from Example 3)
```
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Get the latest report for user 6789'
```

**Expected Output**
```console
aiq run --config_file examples/object_store/user_report/configs/config_s3.yml --input 'Get the latest report for user 6789'
2025-07-02 14:29:37,531 - aiq.runtime.loader - WARNING - Loading module 'aiq.agent.register' from entry point 'aiq_agents' took a long time (197.992086 ms). Ensure all imports are inside your registered functions.
2025-07-02 14:29:37,808 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/object_store/user_report/configs/config_s3.yml'
2025-07-02 14:29:37,810 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0
Number of Object Stores: 1
Number of Retrievers: 0

2025-07-02 14:29:39,099 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Get the latest report for user 6789
Agent's thoughts:
Thought: I need to fetch the latest report for user 6789 using the appropriate tool.
Action: get_user_report
Action Input: {"user_id": "6789", "date": null}
------------------------------
2025-07-02 14:29:39,106 - aiq_user_report.register - INFO - Fetching report from /reports/6789/latest.json
2025-07-02 14:29:39,156 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: get_user_report
Tool's input: {"user_id": "6789", "date": null}
Tool's response:
{'recommendations': ['Update graphics driver', 'Check for overheating hardware', 'Enable automatic crash reporting']}
------------------------------
2025-07-02 14:29:40,345 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: Get the latest report for user 6789
Agent's thoughts:
Thought: I have obtained the latest report for user 6789.
Final Answer: The latest report for user 6789 includes the following recommendations:
1. Update graphics driver
2. Check for overheating hardware
3. Enable automatic crash reporting
------------------------------
2025-07-02 14:29:40,349 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['The latest report for user 6789 includes the following recommendations:\n1. Update graphics driver\n2. Check for overheating hardware\n3. Enable automatic crash reporting']
--------------------------------------------------
```
