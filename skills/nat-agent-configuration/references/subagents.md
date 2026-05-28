# Sub-Agents: Composing Agents as Tools

Composing one agent as a tool inside another. Companion: [`subagent-patterns.md`](subagent-patterns.md) has four detailed composition patterns.

A powerful pattern in NeMo Agent Toolkit is defining an agent under `functions:` and then referencing it as a tool for another agent. This enables hierarchical multi-agent systems where a top-level agent delegates to specialized sub-agents.

**Key principle:** Any agent defined under `functions:` can be referenced by name in another agent's `tool_names`, `branches`, or `tool_list`.

Four detailed composition patterns (Router with Sub-Agents, Reasoning wrapping, Parallel Fan-Out, Sequential Pipeline) are documented in [subagent-patterns.md](subagent-patterns.md). These patterns have not yet been adopted in internal projects but demonstrate the full composability of NeMo Agent Toolkit agents.
