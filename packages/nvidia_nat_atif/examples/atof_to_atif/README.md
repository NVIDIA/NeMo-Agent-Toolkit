# ATOF-to-ATIF Examples

End-to-end examples exercising the ATOF v0.1 reference implementation:
regenerate three canonical event streams, then convert each to an ATIF
trajectory. Each scenario demonstrates one of the **three producer
enrichment tiers** (spec §1.1).

## Scripts

- `generate_examples.py` — produces `output/exmpNN_atof.jsonl` for each
  scenario using the v0.1 public API (`StreamHeaderEvent`, `ScopeStartEvent`,
  `ScopeEndEvent`, `ErrorInfo`, `write_jsonl`).
- `convert_to_atif.py` — reads each regenerated JSONL, runs the ATOF→ATIF
  converter (`nat.atof.scripts.atof_to_atif_converter.convert_file`), and
  writes `output/exmpNN_atif.json` as a formatted ATIF `Trajectory`.

## The three producer tiers

ATOF supports progressive enrichment at the producer's discretion. Tier 1
must always work — a consumer that doesn't understand higher tiers MUST
preserve the event verbatim and fall back to opaque pass-through.

### EXMP-01 — tier-2 semantic-tagged (basic)

Single calculator tool call. The producer knows **the kind of work**
(LLM call, tool invocation) and populates `profile` with scope-type-specific
keys (`profile.model_name` for llm, `profile.tool_call_id` for tool — see
spec §4.4) but does not have a schema for the LLM payload shape. The
`StreamHeader` is a minimal manifest with empty `schemas`.

**When to use:** native producers that classify events at the hook site
but don't decode provider-specific request/response shapes.

### EXMP-02 — tier-2 with error recovery

A web-search tool times out (`status: "error"` + `ErrorInfo`); the parent
agent catches the failure and reports `status: "ok"` with a graceful
output message. Demonstrates spec §5.2-5.3 — each scope reports its own
terminal status; parents may catch child errors.

**When to use:** showcase status semantics, error propagation, and
parent-side recovery patterns.

### EXMP-03 — tier-3 schema-annotated

Same calculator workflow as EXMP-01, but the producer declares a schema
(`openai/chat-completions.v1`) on each LLM event and attaches structured
`annotated_request` / `annotated_response` payloads. The `StreamHeader`
declares the schema in its registry with an inline `$schema` body —
priority-2 fallback for any consumer that doesn't have it locally.

**When to use:** producers wrapping a known provider API; consumers that
want structured access to messages, params, tool defs, usage metrics
without bespoke per-provider parsing.

See `../../atof-schema-profiles.md` §6 for the full 4-priority schema
resolution protocol.

## Running

```bash
cd NeMo-Agent-Toolkit/packages/nvidia_nat_atif/examples/atof_to_atif
python generate_examples.py
python convert_to_atif.py
# Outputs in output/
```

## Event counts

| Scenario | Events | ATIF steps | Tier | Workflow                                        |
| -------- | ------ | ---------- | ---- | ----------------------------------------------- |
| EXMP-01  | 9      | 5          | 2    | Calculator: agent → llm → tool → llm → agent    |
| EXMP-02  | 7      | 3          | 2    | Search: agent → llm → tool (timeout) → agent    |
| EXMP-03  | 9      | 5          | 3    | Calculator with OpenAI schema annotations       |

Each event count includes exactly one `StreamHeaderEvent` at the head
of the stream (spec §3.4 — MUST be first when present).

## See also

- `../../atof-event-format.md` — canonical v0.1 spec
- `../../atof-schema-profiles.md` — schema identifiers, canonical registry,
  4-priority schema resolution protocol (§6)
- `../../atof-to-atif-converter.md` — normative ATOF → ATIF mapping
- `../../src/nat/atof/scripts/atof_to_atif_converter.py` — reference
  converter implementation
