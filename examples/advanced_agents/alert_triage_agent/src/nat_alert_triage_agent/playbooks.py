# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example playbook to monitor host performance metrics including CPU, memory and disk I/O
# This playbook runs commands to collect system performance data and check for threshold violations
# NOTE: This is just an example implementation of Linux monitoring commands.
# Users should implement their own monitoring commands specific to their environment and
# infrastructure setup.
HOST_PERFORMANCE_CHECK_PLAYBOOK = [{
    "name":
        "Monitor system performance",
    "hosts":
        "all",
    "tasks": [
        # CPU usage collection
        {
            "name": "Collect CPU usage data",
            "ansible.builtin.shell": {
                "cmd": "mpstat -P ALL 1 1"
            },
            "register": "cpu_usage",
        },
        {
            "name": "CPU usage data", "debug": {
                "msg": "{{ cpu_usage.stdout }}"
            }
        },

        # Memory usage collection
        {
            "name": "Collect memory usage data",
            "ansible.builtin.shell": {
                "cmd": "free -m"
            },
            "register": "memory_usage",
        },
        {
            "name": "memory usage data", "debug": {
                "msg": "{{ memory_usage.stdout }}"
            }
        },

        # Disk I/O collection
        {
            "name": "Collect disk I/O statistics",
            "ansible.builtin.shell": {
                "cmd": "iostat -dx"
            },
            "register": "disk_io_stats",
        },
        {
            "name": "disk I/O statistics", "debug": {
                "msg": "{{ disk_io_stats.stdout }}"
            }
        },

        # High CPU usage check (threshold: 80%)
        {
            "name": "Check for high CPU usage",
            "ansible.builtin.shell": {
                "cmd": "mpstat 1 1 | awk '/Average/ && $NF > 80 {exit 1}'"
            },
            "register": "cpu_check",
            "failed_when": "cpu_check.rc == 1",
            "ignore_errors": True,
        },
        {
            "name": "CPU usage check", "debug": {
                "msg": "{{ cpu_check.stdout }}"
            }
        },

        # High memory usage check (threshold: 80%)
        {
            "name": "Check for high memory usage",
            "ansible.builtin.shell": {
                "cmd": "free -m | awk '/Mem:/ {if ($3/$2 * 100.0 > 80.0) exit 1}'"
            },
            "register": "memory_check",
            "failed_when": "memory_check.rc == 1",
            "ignore_errors": True,
        },
        {
            "name": "memory usage check", "debug": {
                "msg": "{{ memory_check.stdout }}"
            }
        },

        # High disk I/O wait check (threshold: 10%)
        {
            "name": "Check for high disk I/O wait",
            "ansible.builtin.shell": {
                "cmd": "iostat -dx 1 1 | awk '/^Device:/ {getline; if ($10 > 10.0) exit 1}'"
            },
            "register": "disk_io_check",
            "failed_when": "disk_io_check.rc == 1",
            "ignore_errors": True,
        },
        {
            "name": "disk I/O wait check", "debug": {
                "msg": "{{ disk_io_check.stdout }}"
            }
        },

        # Alert notifications for threshold violations
        {
            "name": "Notify admin of high CPU usage",
            "ansible.builtin.debug": {
                "msg": "High CPU usage detected on {{ instance_name }}"
            },
            "when": "cpu_check.rc == 1",
        },
        {
            "name": "Notify admin of high CPU usage", "debug": {
                "msg": "{{ cpu_check.stdout }}"
            }
        },
        {
            "name": "Notify admin of high memory usage",
            "ansible.builtin.debug": {
                "msg": "High memory usage detected on {{ instance_name }}"
            },
            "when": "memory_check.rc == 1",
        },
        {
            "name": "Notify admin of high disk I/O wait",
            "ansible.builtin.debug": {
                "msg": "High disk I/O wait detected on {{ instance_name }}"
            },
            "when": "disk_io_check.rc == 1",
        },
    ],
}]

# Example playbook to check critical service status on a host
# This playbook runs commands to verify if key services related to alert monitoring are running properly
# NOTE: In this example, we check the Telegraf service, but users should modify the commands to check
# whatever services are critical for monitoring and alerting in their environment
MONITOR_PROCESS_CHECK_PLAYBOOK = [{
    "name":
        "Monitor Telegraf process",  # Playbook name
    "hosts":
        "all",
    "tasks": [
        {
            "name": "ps telegraf process",  # Task to check if Telegraf process is running
            "ansible.builtin.shell": {
                "cmd": "ps -ef | grep telegraf"
            },  # List processes and filter for telegraf
            "register": "ps_usage",  # Store output in ps_usage variable
        },
        {
            "name": "ps telegraf process", "debug": {
                "msg": "{{ ps_usage.stdout }}"
            }
        },  # Print process list output
        {
            "name": "systemctl status telegraf",  # Task to check Telegraf service status
            "ansible.builtin.shell": {
                "cmd": "systemctl status telegraf"
            },  # Get service status from systemd
            "register": "systemctl_usage",  # Store output in systemctl_usage variable
        },
        {
            "name": "systemctl status telegraf", "debug": {
                "msg": "{{ systemctl_usage.stdout }}"
            }
        },  # Print service status
    ],
}]
