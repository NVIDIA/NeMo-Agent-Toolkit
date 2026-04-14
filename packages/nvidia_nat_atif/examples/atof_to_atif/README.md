# ATOF-to-ATIF Examples

End-to-end examples exercising the ATOF v0.2 reference implementation:
regenerate three canonical event streams, then convert each to an ATIF
trajectory. Each scenario demonstrates one of the three profile declaration
modes introduced by `StreamHeaderEvent` (spec §5).

## Scripts

- `generate_examples.py` — produces `output/exmpNN_atof.jsonl` for each
  scenario using the v0.2 public API (`DefaultLlmV1`, `DefaultToolV1`,
  `StreamHeaderEvent`, `ScopeStartEvent`, `ScopeEndEvent`, `write_jsonl`).
- `convert_to_atif.py` — reads each regenerated JSONL, runs the ATOF→ATIF
  converter (`nat.atof.scripts.atof_to_atif_converter.convert_file`), and
  writes `output/exmpNN_atif.json` as a formatted ATIF `Trajectory`.

## The three profile modes

Per ATOF spec §4.4, a profile's schema may be declared in one of three
modes. `StreamHeaderEvent.profile_mode_default` advertises the stream-wide
default; individual profiles may override via `$mode`.

### Header mode (EXMP-01)

A `StreamHeaderEvent` at the top of the stream declares a schema registry
mapping schema IDs → full JSON Schema bodies. Each subsequent profile
references a schema by string `$schema` ID (e.g., `"default/llm.v1"`).
Consumers validate by looking up the ID in the registry.

**When to use:** many profiles share a small set of schemas; registry
centralization keeps the stream compact.

### Inline mode (EXMP-02)

Each profile carries its full JSON Schema inline — `$schema` is a dict
containing `$id` and the full schema body. The `StreamHeaderEvent`
advertises `profile_mode_default: "inline"` and an empty `schemas` registry.
Profiles are self-describing and consumers validate without a lookup.

**When to use:** heterogeneous vendor schemas, one-off streams, or when
recipients cannot pre-load a schema registry.

### Mixed mode (EXMP-03)

The stream-level default is `header` (schemas centralized in the registry),
but individual profiles override via `$mode: "inline"` and carry inline
schemas for that single event. Useful for introducing a schema variant
without modifying the central registry.

**When to use:** most events follow the registry, but a few need per-event
overrides (e.g., staging a new schema version).

## Running

```bash
cd NeMo-Agent-Toolkit/packages/nvidia_nat_atif/examples/atof_to_atif
python generate_examples.py
python convert_to_atif.py
# Outputs in output/
```

## Event counts

| Scenario | Events | ATIF steps | Mode   | Workflow                                          |
| -------- | ------ | ---------- | ------ | ------------------------------------------------- |
| EXMP-01  | 9      | 5          | header | Calculator (simple LLM → tool → LLM)              |
| EXMP-02  | 13     | 5          | inline | Nested weather lookup (LLM → weather(temp) → LLM) |
| EXMP-03  | 15     | 5          | mixed  | Branching search + summarize                      |

Each event count includes exactly one `StreamHeaderEvent` at the head of
the stream; the remaining events are the same v0.1 structural events
preserved per D-24.

## See also

- `../../atof-event-format.md` — canonical v0.2 spec (see §4 Profile
  Contract Protocol, §5 Stream Header Event, §6 Reference Profile
  Implementations)
- `../../src/nat/atof/profiles.py` — `DefaultLlmV1` and `DefaultToolV1`
  reference profile implementations
- `../../src/nat/atof/scripts/atof_to_atif_converter.py` — the converter
  that translates ATOF streams into ATIF trajectories
