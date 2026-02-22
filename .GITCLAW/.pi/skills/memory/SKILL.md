---
name: memory
description: Search and recall information from past sessions and memory logs. Use when you need to remember something, reference past conversations, ask "what did we talk about", "remember when", or need historical context.
---

# Memory

You don't have to start every session blank. Past context lives in files you can search.

## Where Memory Lives

- **state/memory.log** — Append-only log of important facts, preferences, and decisions
- **state/sessions/*.jsonl** — Full conversation transcripts per session

## When to Search

- User says "remember when..." or "you said..." or "we talked about..."
- User references a preference, decision, or fact you should know
- You need context from a previous session
- Before asking the user something they may have already told you

## Quick Searches

```bash
# Search memory log
rg -i "search term" state/memory.log

# Recent memories
tail -30 state/memory.log

# Search all sessions
rg -i "search term" state/sessions/

# Search with context (2 lines before/after)
rg -i -C 2 "search term" state/memory.log state/sessions/
```

## Writing to Memory

When you learn something worth keeping:

```bash
echo "[$(date -u '+%Y-%m-%d %H:%M')] Memory entry here." >> state/memory.log
```

Keep entries atomic — one fact per line. Future you will grep this.

---

## Session Log Deep Queries

Session files are JSONL. Each line:

```json
{
  "type": "message",
  "timestamp": "2026-02-05T05:16:04.856Z",
  "message": {
    "role": "user" | "assistant" | "toolResult",
    "content": [
      { "type": "text", "text": "..." },
      { "type": "thinking", "thinking": "..." },
      { "type": "toolCall", "name": "bash", "arguments": {...} }
    ]
  }
}
```

### List sessions by date

```bash
for f in state/sessions/*.jsonl; do
  [ -f "$f" ] || continue
  ts=$(head -1 "$f" | jq -r '.timestamp // empty' 2>/dev/null)
  echo "$(echo "$ts" | cut -dT -f1) $(basename $f)"
done | sort -r
```

### Extract user messages

```bash
jq -r 'select(.message.role == "user") | .message.content[]? | select(.type == "text") | .text' state/sessions/<file>.jsonl
```

### Extract assistant responses

```bash
jq -r 'select(.message.role == "assistant") | .message.content[]? | select(.type == "text") | .text' state/sessions/<file>.jsonl
```

### Conversation overview (skip thinking/tools)

```bash
jq -r 'select(.message.role == "user" or .message.role == "assistant") | .message.content[]? | select(.type == "text") | .text' state/sessions/<file>.jsonl | head -100
```

### Tool usage stats

```bash
jq -r '.message.content[]? | select(.type == "toolCall") | .name' state/sessions/<file>.jsonl | sort | uniq -c | sort -rn
```

## Tips

- Sessions are append-only JSONL (one JSON per line)
- Large sessions can be several MB — use `head`/`tail` for sampling
- Filter `type=="text"` to skip thinking blocks and tool calls
- Use `jq -r` for raw output without JSON escaping
