# Synap Memory Plugin for NeMo Agent Toolkit

[Synap](https://maximem.ai) is a managed memory layer for AI agents. This example demonstrates how to use the `SynapMemoryEditor` — a drop-in `MemoryEditor` implementation — with NeMo Agent Toolkit workflows.

## Installation

```bash
pip install maximem-synap-nemo-agent-toolkit
```

Get an API key at [synap.maximem.ai](https://synap.maximem.ai) and set `SYNAP_API_KEY` in your environment.

## YAML Configuration

Add `synap` to the `memory:` block in your NAT workflow YAML:

```yaml
memory:
  synap:
    _type: synap_memory
    customer_id: "acme_corp"
    mode: "accurate"

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: synap
  llm_name: nim_llm
```

## Programmatic Usage

```python
from maximem_synap import MaximemSynapSDK
from synap_nemo_agent_toolkit import SynapMemoryEditor
from nat.memory.models import MemoryItem

sdk = MaximemSynapSDK(api_key="sk-...")
await sdk.initialize()

editor = SynapMemoryEditor(sdk=sdk, customer_id="acme_corp")

# Write
await editor.add_items([
    MemoryItem(user_id="user_123", memory="User prefers concise answers.")
])

# Search
items = await editor.search("communication style", top_k=5, user_id="user_123")
for item in items:
    print(item.memory)
```

Memory is scoped to `user_id` and `customer_id`, ensuring strict isolation in multi-tenant applications.

## More Resources

- [Synap Documentation](https://docs.maximem.ai)
- [NeMo Agent Toolkit Integration Guide](https://docs.maximem.ai/integrations/nemo-agent-toolkit)
- [Dashboard](https://synap.maximem.ai)
- [PyPI: maximem-synap-nemo-agent-toolkit](https://pypi.org/project/maximem-synap-nemo-agent-toolkit/)
