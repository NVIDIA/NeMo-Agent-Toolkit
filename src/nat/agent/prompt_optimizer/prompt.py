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
# flake8: noqa W291

mutator_prompt = """

## CORE DIRECTIVES
- **Preserve the original objective and task.** Do not change what the prompt is meant to accomplish.  
- **Keep the intent intact.** The improved prompt must solve the same problem as the original.  
- **Do not invent new goals.** Only improve clarity, structure, constraints, and usability.  
- **Do not drop critical instructions.** Everything essential from the original prompt must remain.  
- **Return only the mutated prompt text.** No rationale, no diffs, no explanations.  
- **Be Creative within bounds.** You may rephrase, reorganize, and enhance, but not alter meaning.
- **DO NOT use curly braces in your prompt** for anything other than existing variables in the prompt as the string
will be treated as an f-string.
- **Examples are a good idea** if the original prompt lacks them. They help clarify expected output.

---

## IMPROVEMENT HINTS
When modifying, apply these principles:
1. **Clarity & Precision** – remove vague language, strengthen directives.  
2. **Structure & Flow** – order sections as: *Objective → Constraints → Tools → Steps → Output Schema → Examples*.  
3. **Schema Adherence** – enforce a single canonical output schema (JSON/XML) with `schema_version`.  
4. **Tool Governance** – clarify when/how tools are used, their inputs/outputs, and fallback behavior.  
5. **Error Handling** – specify behavior if tools fail or inputs are insufficient.  
6. **Budget Awareness** – minimize verbosity, respect token/latency limits.  
7. **Safety** – include refusals for unsafe requests, enforce compliance with rules.  
8. **Consistency** – avoid format drift; always maintain the same schema.  
9. **Integrity** – confirm the task, objective, and intent are preserved.  

---

## MUTATION OPERATORS
You may:
- **Tighten** (remove fluff, redundancies)  
- **Reorder** (improve logical flow)  
- **Constrain** (add explicit rules/limits)  
- **Harden** (improve error handling/fallbacks)  
- **Defuse** (replace ambiguous verbs with measurable actions)  
- **Format-lock** (wrap outputs in JSON/XML fenced blocks)
- **Example-ify** (add examples if missing or weak)  

---

## INPUT
Here is the prompt to mutate:
{original_prompt}

## OBJECTIVE
The prompt must acheive the following objective:
{objective}

The modified prompt is: \n

"""