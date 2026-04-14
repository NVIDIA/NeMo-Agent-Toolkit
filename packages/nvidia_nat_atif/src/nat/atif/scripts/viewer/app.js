/* ATIF Trajectory Viewer — rendering logic */

function esc(s) {
  if (!s) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function formatTS(ts) {
  if (!ts) return "";
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", {
      hour12: false, hour: "2-digit", minute: "2-digit",
      second: "2-digit", fractionalSecondDigits: 3,
    });
  } catch (e) {
    return ts;
  }
}

function truncate(text, max) {
  if (!text || text.length <= max) return { short: text || "", full: text || "", truncated: false };
  return { short: text.slice(0, max) + "...", full: text, truncated: true };
}

let _toggleId = 0;

function renderCollapsible(label, contentHtml, startOpen) {
  const id = _toggleId++;
  return `<div class="collapsible-section">
    <div class="collapsible-header" onclick="toggleCollapsible(${id})">
      <span class="tool-arrow${startOpen ? " open" : ""}" id="ca${id}">&#9654;</span> ${esc(label)}
    </div>
    <div class="collapsible-body${startOpen ? " open" : ""}" id="cb${id}">${contentHtml}</div>
  </div>`;
}

function renderJSON(obj) {
  return `<pre>${esc(JSON.stringify(obj, null, 2))}</pre>`;
}

function renderExtraFields(extra, label) {
  if (!extra || typeof extra !== "object" || !Object.keys(extra).length) return "";
  const filtered = {};
  for (const [k, v] of Object.entries(extra)) {
    if (k === "subagent_trajectories") continue;
    filtered[k] = v;
  }
  if (!Object.keys(filtered).length) return "";
  return `<div class="extra-section">${renderCollapsible(label || "Extra", renderJSON(filtered), false)}</div>`;
}

function renderMetricsSummary(fm) {
  if (!fm) return "";
  const items = [];
  if (fm.total_prompt_tokens)
    items.push(`<div class="metric-item"><span class="metric-value">${fm.total_prompt_tokens.toLocaleString()}</span><span class="metric-label">Prompt Tokens</span></div>`);
  if (fm.total_completion_tokens)
    items.push(`<div class="metric-item"><span class="metric-value">${fm.total_completion_tokens.toLocaleString()}</span><span class="metric-label">Completion Tokens</span></div>`);
  if (fm.total_cached_tokens)
    items.push(`<div class="metric-item"><span class="metric-value">${fm.total_cached_tokens.toLocaleString()}</span><span class="metric-label">Cached Tokens</span></div>`);
  if (fm.total_steps)
    items.push(`<div class="metric-item"><span class="metric-value">${fm.total_steps}</span><span class="metric-label">Agent Steps</span></div>`);
  if (fm.total_cost_usd)
    items.push(`<div class="metric-item"><span class="metric-value">$${fm.total_cost_usd.toFixed(4)}</span><span class="metric-label">Cost</span></div>`);
  if (!items.length && !fm.extra) return "";
  let html = items.length ? `<div class="metrics-summary">${items.join("")}</div>` : "";
  if (fm.extra) html += renderExtraFields(fm.extra, "Metrics Extra");
  return html;
}

function renderTokenMetrics(m) {
  if (!m) return "";
  const parts = [];
  if (m.prompt_tokens) parts.push(`<span>Prompt: ${m.prompt_tokens}</span>`);
  if (m.completion_tokens) parts.push(`<span>Completion: ${m.completion_tokens}</span>`);
  if (m.cached_tokens) parts.push(`<span>Cached: ${m.cached_tokens}</span>`);
  if (m.cost_usd) parts.push(`<span>Cost: $${m.cost_usd.toFixed(4)}</span>`);
  if (!parts.length && !m.extra) return "";
  let html = parts.length ? `<div class="token-metrics">${parts.join("")}</div>` : "";
  if (m.extra) {
    const extraParts = [];
    for (const [k, v] of Object.entries(m.extra)) extraParts.push(`<span>${esc(k)}: ${v}</span>`);
    if (extraParts.length) html += `<div class="token-metrics">${extraParts.join("")}</div>`;
  }
  return html;
}

function findObservation(obs, callId) {
  if (!obs || !obs.results) return null;
  return obs.results.find(r => r.source_call_id === callId) || null;
}

function renderToolCalls(toolCalls, observation) {
  if (!toolCalls || !toolCalls.length) return "";
  let html = '<div class="tool-calls">';
  for (const tc of toolCalls) {
    const tid = _toggleId++;
    const obs = findObservation(observation, tc.tool_call_id);
    const argsStr = tc.arguments ? JSON.stringify(tc.arguments, null, 2) : "{}";
    const obsContent = obs
      ? (typeof obs.content === "string" ? obs.content : JSON.stringify(obs.content, null, 2))
      : "";
    const hasRef = obs && obs.subagent_trajectory_ref && obs.subagent_trajectory_ref.length;
    html += `<div class="tool-card">
      <div class="tool-header" onclick="toggleTool(${tid})">
        <span class="tool-arrow" id="ta${tid}">&#9654;</span>
        <span class="tool-name">${esc(tc.function_name)}</span>
        <span style="color:#8b949e;font-size:11px">${esc(tc.tool_call_id)}</span>
      </div>
      <div class="tool-body" id="tb${tid}">
        <div class="tool-section-label">Arguments</div>
        <pre>${esc(argsStr)}</pre>
        <div class="tool-section-label">Observation</div>
        <pre>${esc(obsContent) || '<span style="color:#484f58">(empty)</span>'}</pre>
        ${hasRef
          ? `<div class="tool-section-label">Subagent Trajectory Ref</div>
             <pre>${esc(JSON.stringify(obs.subagent_trajectory_ref, null, 2))}</pre>`
          : ""}
      </div>
    </div>`;
  }

  if (observation && observation.results) {
    const unmatched = observation.results.filter(
      r => !r.source_call_id || !toolCalls.find(tc => tc.tool_call_id === r.source_call_id)
    );
    for (const r of unmatched) {
      const tid = _toggleId++;
      const content = typeof r.content === "string" ? r.content : JSON.stringify(r.content, null, 2);
      html += `<div class="tool-card">
        <div class="tool-header" onclick="toggleTool(${tid})">
          <span class="tool-arrow" id="ta${tid}">&#9654;</span>
          <span class="tool-name" style="color:#8b949e">Observation (no tool_call)</span>
        </div>
        <div class="tool-body" id="tb${tid}">
          <pre>${esc(content) || '<span style="color:#484f58">(empty)</span>'}</pre>
        </div>
      </div>`;
    }
  }

  html += "</div>";
  return html;
}

function renderStep(step) {
  const src = step.source || "agent";
  const badge = `<span class="badge badge-${src}">${src}</span>`;
  const model = step.model_name ? `<span class="model-tag">${esc(step.model_name)}</span>` : "";
  const effort = step.reasoning_effort
    ? `<span class="effort-tag">effort: ${esc(String(step.reasoning_effort))}</span>` : "";
  const copied = step.is_copied_context ? `<span class="copied-tag">copied context</span>` : "";
  const ts = `<span class="timestamp">${formatTS(step.timestamp)}</span>`;
  const msg = typeof step.message === "string" ? step.message : JSON.stringify(step.message, null, 2);
  const t = truncate(msg, 500);
  const mid = _toggleId++;

  let msgHtml = `<pre class="message-text" id="msg${mid}">${esc(t.short)}</pre>`;
  if (t.truncated) {
    msgHtml += `<button class="message-toggle" onclick="toggleMsg(${mid},this)"
      data-full="${btoa(unescape(encodeURIComponent(t.full)))}"
      data-short="${btoa(unescape(encodeURIComponent(t.short)))}">Show more</button>`;
  }

  let reasoningHtml = "";
  if (step.reasoning_content) {
    const rt = truncate(step.reasoning_content, 300);
    const rid = _toggleId++;
    reasoningHtml = `<div class="reasoning-block">
      <div class="reasoning-label">Reasoning</div>
      <pre class="message-text" id="msg${rid}"
           style="background:transparent;border:none;padding:0;max-height:250px">${esc(rt.short)}</pre>
      ${rt.truncated
        ? `<button class="message-toggle" onclick="toggleMsg(${rid},this)"
            data-full="${btoa(unescape(encodeURIComponent(rt.full)))}"
            data-short="${btoa(unescape(encodeURIComponent(rt.short)))}">Show more</button>`
        : ""}
    </div>`;
  }

  return `<div class="step source-${src}">
    <div class="step-card">
      <div class="step-header">
        <span class="step-num">#${step.step_id}</span>
        ${badge}${model}${effort}${copied}${ts}
      </div>
      <div class="step-message">${msgHtml}</div>
      ${reasoningHtml}
      ${renderTokenMetrics(step.metrics)}
      ${renderToolCalls(step.tool_calls, step.observation)}
      ${renderExtraFields(step.extra, "Step Extra")}
    </div>
  </div>`;
}

function renderTrajectory(traj, depth) {
  depth = depth || 0;
  const agent = traj.agent || {};
  let html = "";

  if (depth === 0) {
    html += `<div class="header">
      <h1>ATIF Trajectory Viewer</h1>
      <div class="header-meta">
        <span><strong>Agent:</strong> ${esc(agent.name || "unknown")} v${esc(agent.version || "?")}</span>
        <span><strong>Model:</strong> ${esc(agent.model_name || "N/A")}</span>
        <span><strong>Schema:</strong> ${esc(traj.schema_version || "")}</span>
        <span><strong>Session:</strong> ${esc(traj.session_id || "")}</span>
        <span><strong>Steps:</strong> ${(traj.steps || []).length}</span>
        ${traj.continued_trajectory_ref
          ? `<span><strong>Continued:</strong> ${esc(traj.continued_trajectory_ref)}</span>` : ""}
      </div>
      ${traj.notes ? `<div class="header-notes"><strong>Notes:</strong> ${esc(traj.notes)}</div>` : ""}
      ${renderMetricsSummary(traj.final_metrics)}
      ${agent.tool_definitions
        ? renderCollapsible(
            "Tool Definitions (" + agent.tool_definitions.length + ")",
            renderJSON(agent.tool_definitions), false)
        : ""}
      ${agent.extra ? renderCollapsible("Agent Extra", renderJSON(agent.extra), false) : ""}
      ${renderExtraFields(traj.extra, "Trajectory Extra")}
    </div>`;
  } else {
    const cAgent = traj.agent || {};
    html += `<div style="margin-bottom:12px;font-size:13px;color:#8b949e">
      <strong>Session:</strong> ${esc(traj.session_id || "")}
      &middot; <strong>Agent:</strong> ${esc(cAgent.name || "unknown")}
      &middot; <strong>Model:</strong> ${esc(cAgent.model_name || "N/A")}
      &middot; <strong>Steps:</strong> ${(traj.steps || []).length}
    </div>`;
  }

  html += '<div class="timeline">';
  for (const step of (traj.steps || [])) {
    html += renderStep(step);
  }
  html += "</div>";

  const subs = (traj.extra || {}).subagent_trajectories;
  if (subs && Object.keys(subs).length) {
    for (const [sid, child] of Object.entries(subs)) {
      const stid = _toggleId++;
      const childAgent = child.agent || {};
      html += `<div class="subagent-section">
        <div class="subagent-header" onclick="toggleSubagent(${stid})">
          <span class="tool-arrow" id="sa${stid}">&#9654;</span>
          Subagent: ${esc(childAgent.name || sid)} (${(child.steps || []).length} steps)
        </div>
        <div class="subagent-body" id="sb${stid}">
          ${renderTrajectory(child, depth + 1)}
        </div>
      </div>`;
    }
  }

  return html;
}

/* ---- toggle helpers ---- */

function toggleTool(id) {
  document.getElementById("tb" + id).classList.toggle("open");
  document.getElementById("ta" + id).classList.toggle("open");
}

function toggleSubagent(id) {
  document.getElementById("sb" + id).classList.toggle("open");
  document.getElementById("sa" + id).classList.toggle("open");
}

function toggleCollapsible(id) {
  document.getElementById("cb" + id).classList.toggle("open");
  document.getElementById("ca" + id).classList.toggle("open");
}

function toggleMsg(id, btn) {
  const el = document.getElementById("msg" + id);
  const isShort = btn.textContent === "Show more";
  const raw = isShort ? atob(btn.dataset.full) : atob(btn.dataset.short);
  el.textContent = decodeURIComponent(escape(raw));
  btn.textContent = isShort ? "Show less" : "Show more";
}

/* ---- render ---- */
document.getElementById("app").innerHTML = renderTrajectory(DATA, 0);
