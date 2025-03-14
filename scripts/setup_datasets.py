# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess

import click


def run_command(command):
    """
    Subprocess wrapper
    """
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, text=True, check=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command} {e}")


@click.group()
def cli():
    """
    CLI to manage dataset directories
    """
    pass


@cli.command()
@click.option("--nfs-server-ip",
              envvar="NFS_SERVER_IP",
              required=True,
              help="The IP address of the NFS server. Can also be set as the NFS_SERVER_IP environment variable.")
def mount(nfs_server_ip):
    """
    Mount the NFS share for aiq datasets.
    """
    remote_path = f"{nfs_server_ip}:/public/datasets/aiq"
    mount_point = "/mnt/nfs/aiq"

    # Install NFS common tools
    run_command("sudo apt -y update")
    run_command("sudo apt -y install nfs-common")

    # Create the mount point
    run_command(f"sudo mkdir -p {mount_point}")
    # Mount the NFS share
    run_command(f"sudo mount -v -t nfs -o nfsvers=3 {remote_path} {mount_point}")

    # Print only the AgentIQ mount
    print("\nAgentIQ mount details:")
    run_command(f"mount | grep {mount_point}")
    print("NFS mount completed successfully")


@cli.command()
def unmount():
    """
    Unmount the NFS share from the mount point.
    """
    mount_point = "/mnt/nfs/aiq"
    run_command(f"sudo umount {mount_point}")
    print("NFS unmount completed successfully")


if __name__ == "__main__":
    cli()
