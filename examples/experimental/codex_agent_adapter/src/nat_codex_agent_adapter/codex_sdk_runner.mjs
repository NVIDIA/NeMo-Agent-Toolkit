#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from "node:module";
import process from "node:process";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);

function compactObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const result = {};
  for (const [key, nestedValue] of Object.entries(value)) {
    if (nestedValue !== null && nestedValue !== undefined) {
      result[key] = compactObject(nestedValue);
    }
  }
  return result;
}

function describeItem(item) {
  if (!item || typeof item !== "object") {
    return "";
  }
  if (item.type === "command_execution") {
    return `${item.type} status=${item.status ?? "unknown"} command=${JSON.stringify(item.command ?? "")}`;
  }
  if (item.type === "agent_message") {
    return `${item.type} chars=${typeof item.text === "string" ? item.text.length : 0}`;
  }
  if (item.type === "reasoning") {
    return `${item.type} chars=${typeof item.text === "string" ? item.text.length : 0}`;
  }
  if (item.type === "error") {
    return `${item.type} message=${JSON.stringify(item.message ?? "")}`;
  }
  return `${item.type ?? "item"} status=${item.status ?? "unknown"}`;
}

function logEvent(event) {
  if (!event || typeof event !== "object") {
    return;
  }
  if (event.type === "thread.started") {
    process.stderr.write(`codex sdk event: thread.started id=${event.thread_id}\n`);
  } else if (event.type === "turn.started" || event.type === "turn.completed") {
    process.stderr.write(`codex sdk event: ${event.type}\n`);
  } else if (event.type === "turn.failed") {
    process.stderr.write(`codex sdk event: turn.failed ${event.error?.message ?? ""}\n`);
  } else if (event.type === "error") {
    process.stderr.write(`codex sdk event: error ${event.message ?? ""}\n`);
  } else if (event.item) {
    process.stderr.write(`codex sdk event: ${event.type} ${describeItem(event.item)}\n`);
  }
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function importSdk(packageName, moduleSearchPaths) {
  try {
    return await import(packageName);
  } catch (importError) {
    try {
      const resolved = require.resolve(packageName, { paths: moduleSearchPaths });
      return await import(pathToFileURL(resolved).href);
    } catch {
      throw new Error(
        `Missing ${packageName}. Run \`npm install\` from examples/experimental/codex_agent_adapter, ` +
          "or set workflow.node_package_directory to the directory containing node_modules.",
        { cause: importError },
      );
    }
  }
}

async function main() {
  const input = JSON.parse(await readStdin());
  const { Codex } = await importSdk("@openai/codex-sdk", input.moduleSearchPaths ?? []);

  const codex = new Codex(compactObject(input.codexOptions ?? {}));
  const threadOptions = compactObject(input.threadOptions ?? {});
  const thread = input.threadId
    ? codex.resumeThread(input.threadId, threadOptions)
    : codex.startThread(threadOptions);

  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort(new Error(`Codex SDK runner timed out after ${input.timeoutMs} ms`));
  }, input.timeoutMs);

  const items = [];
  let finalResponse = "";
  let usage = null;
  let threadId = input.threadId ?? null;
  try {
    const { events } = await thread.runStreamed(input.prompt, { signal: controller.signal });
    for await (const event of events) {
      logEvent(event);
      if (event.type === "thread.started") {
        threadId = event.thread_id;
      } else if (event.type === "item.completed") {
        items.push(event.item);
        if (event.item?.type === "agent_message") {
          finalResponse = event.item.text ?? "";
        }
      } else if (event.type === "turn.completed") {
        usage = event.usage;
      } else if (event.type === "turn.failed") {
        throw new Error(event.error?.message ?? "Codex turn failed");
      } else if (event.type === "error") {
        throw new Error(event.message ?? "Codex event stream failed");
      }
    }
  } finally {
    clearTimeout(timer);
  }

  process.stdout.write(
    JSON.stringify({
      text: finalResponse,
      threadId: thread.id ?? threadId,
      usage,
      items,
    }),
  );
}

main().catch((error) => {
  process.stderr.write(`${error?.stack ?? String(error)}\n`);
  process.exit(1);
});
