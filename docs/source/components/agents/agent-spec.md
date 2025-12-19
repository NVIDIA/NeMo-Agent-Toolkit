# Agent Spec Workflow (Milestone 1)

This workflow allows running an [Agent Spec] configuration inside NeMo Agent Toolkit by converting it to a LangGraph component via the Agent Spec → LangGraph adapter.

## Install

- Install optional extra:

```bash
pip install 'nvidia-nat[agentspec]'
```

## Example configuration

```yaml
workflow:
  _type: agent_spec
  description: Agent Spec workflow
  agentspec_path: path/to/agent_spec.yaml  # or agentspec_yaml / agentspec_json
  tool_names: [pretty_formatting]
  max_history: 15
  verbose: true
```

Exactly one of `agentspec_yaml`, `agentspec_json`, or `agentspec_path` must be provided.

## Notes and limitations

- Tools: NAT tools provided in `tool_names` are exposed to the adapter `tool_registry` by name. If the Agent Spec also defines tools, the registries are merged; duplicate names are overwritten by NAT tools.
- I/O: Inputs are standard `ChatRequest` messages; the workflow returns a `ChatResponse`.
- Streaming: Non‑streaming response in M1.
- Checkpointing: Not wired in M1.

[Agent Spec]: https://github.com/oracle/agent-spec
