"use strict";

/* ════════════════════════════════════════════════════════════════
   Markdown-lite renderer — self-contained, no libraries.
   XSS-safe by construction: every piece of raw text is HTML-escaped
   FIRST; the formatting transforms below only ever run on
   already-escaped text and only insert fixed tag names.
   ════════════════════════════════════════════════════════════════ */
function escapeHTML(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderInline(s) {
  s = escapeHTML(s);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*([^\s*](?:[^*]*[^\s*])?)\*/g, "<em>$1</em>");
  return s;
}

function renderMarkdown(raw) {
  const lines = raw.split("\n");
  let out = "", inCode = false, codeLines = [], codeLang = "", inUl = false, inOl = false;
  function flushCode() {
    out += '<div class="codeblock">'
      + (codeLang ? '<span class="code-lang">' + escapeHTML(codeLang) + "</span>" : "")
      + "<pre><code>" + escapeHTML(codeLines.join("\n")) + "</code></pre></div>";
    inCode = false; codeLines = []; codeLang = "";
  }
  function closeList() {
    if (inUl) { out += "</ul>"; inUl = false; }
    if (inOl) { out += "</ol>"; inOl = false; }
  }
  for (const line of lines) {
    if (!inCode && line.startsWith("```")) {
      closeList();
      inCode = true;
      codeLines = [];
      codeLang = line.slice(3).trim();
      continue;
    }
    if (inCode) {
      if (line.startsWith("```")) flushCode();
      else codeLines.push(line);
      continue;
    }
    const hm = line.match(/^(#{1,3}) (.+)/);
    if (hm) { closeList(); out += `<h${hm[1].length}>${renderInline(hm[2])}</h${hm[1].length}>`; continue; }
    const ulm = line.match(/^[*\-] (.+)/);
    if (ulm) { if (inOl) { out += "</ol>"; inOl = false; } if (!inUl) { out += "<ul>"; inUl = true; } out += `<li>${renderInline(ulm[1])}</li>`; continue; }
    const olm = line.match(/^\d+\. (.+)/);
    if (olm) { if (inUl) { out += "</ul>"; inUl = false; } if (!inOl) { out += "<ol>"; inOl = true; } out += `<li>${renderInline(olm[1])}</li>`; continue; }
    closeList();
    if (line.trim() === "") out += "<br>";
    else out += renderInline(line) + "<br>";
  }
  if (inCode) flushCode();
  closeList();
  return out.replace(/^(<br>)+/, "").replace(/(<br>)+$/, "");
}

/* ── Clipboard helpers ── */
function bindCopy(btn, getText) {
  const label = btn.textContent;
  btn.addEventListener("click", () => {
    if (!navigator.clipboard || !navigator.clipboard.writeText) return;
    navigator.clipboard.writeText(getText()).then(() => {
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = label; }, 1400);
    }).catch(() => {});
  });
}

function attachCopyButton(el) {
  if (el.querySelector(":scope > .copy-btn")) return;
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.type = "button";
  btn.textContent = "Copy";
  btn.setAttribute("aria-label", "Copy message to clipboard");
  bindCopy(btn, () => el.dataset.raw || "");
  el.appendChild(btn);
}

function addCodeCopyButtons(container) {
  container.querySelectorAll(".codeblock").forEach((block) => {
    const code = block.querySelector("code");
    if (!code || block.querySelector(".code-copy")) return;
    const btn = document.createElement("button");
    btn.className = "code-copy";
    btn.type = "button";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy code to clipboard");
    bindCopy(btn, () => code.textContent);
    block.appendChild(btn);
  });
}

function prettyJSON(v) {
  if (typeof v !== "string") {
    try { return JSON.stringify(v, null, 2); } catch (_) { return String(v); }
  }
  try { return JSON.stringify(JSON.parse(v), null, 2); } catch (_) { return v; }
}

(function initChat() {
  const form          = document.getElementById("form");
  const promptEl      = document.getElementById("prompt");
  const messagesEl    = document.getElementById("messages");
  const emptyEl       = document.getElementById("empty");
  const emptyModelEl  = document.getElementById("emptyModel");
  const statusEl      = document.getElementById("status");
  const statusTextEl  = document.getElementById("statusText");
  const sendEl        = document.getElementById("send");
  const clearEl       = document.getElementById("clear");
  const maxTokensEl   = document.getElementById("maxTokens");
  const temperatureEl = document.getElementById("temperature");
  const tempValueEl   = document.getElementById("tempValue");
  const scrollEl      = document.getElementById("scroll");
  const modelSelectEl = document.getElementById("modelSelect");
  const modelNameEl   = document.getElementById("modelName");

  const history = [];
  let busy = false;
  let controller = null;

  /* ── Scroll anchoring: only auto-scroll while the user is at the
     bottom; scrolling up pauses following, scrolling back resumes. ── */
  let followStream = true;
  function isNearBottom() {
    return scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 60;
  }
  scrollEl.addEventListener("scroll", () => { followStream = isNearBottom(); }, { passive: true });
  function scrollToBottom(force) {
    if (force || followStream) scrollEl.scrollTop = scrollEl.scrollHeight;
  }

  function setStatus(text) { statusTextEl.textContent = text; }

  function setBusy(b) {
    busy = b;
    modelSelectEl.disabled = b;
    sendEl.textContent = b ? "Stop" : "Send";
    sendEl.classList.toggle("stop", b);
    sendEl.setAttribute("aria-label", b ? "Stop generating" : "Send message");
    statusEl.classList.toggle("busy", b);
    setStatus(b ? "Thinking…" : "Ready");
  }

  /* ── Model discovery & hot-swap ── */
  async function loadModelList() {
    try {
      const res = await fetch("/models");
      if (!res.ok) return;
      const data = await res.json();
      const models = data.models || [];
      if (models.length === 0) return;
      modelSelectEl.innerHTML = "";
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.path;
        const label = m.name || m.id;
        const arch = m.architecture ? ` [${m.architecture}]` : "";
        const size = m.size_gb ? ` — ${m.size_gb.toFixed(1)} GB` : "";
        const unsupported = !m.supported ? " (unsupported)" : "";
        opt.textContent = label + arch + size + unsupported;
        if (m.loaded) {
          opt.selected = true;
          opt.dataset.loaded = "true";
        }
        if (!m.supported) opt.style.color = "var(--muted)";
        modelSelectEl.appendChild(opt);
      }
    } catch (_) {}
  }

  function setModelName(name) {
    if (!name) return;
    modelNameEl.textContent = name;
    if (emptyModelEl) emptyModelEl.textContent = name;
  }

  modelSelectEl.addEventListener("change", async () => {
    const path = modelSelectEl.value;
    if (!path || busy) return;
    const prev = [...modelSelectEl.options].find((o) => o.dataset.loaded === "true")?.value || "";
    modelSelectEl.disabled = true;
    setStatus("Loading model…");
    statusEl.classList.add("busy");
    try {
      const res = await fetch("/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path })
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err || "HTTP " + res.status);
      }
      const data = await res.json();
      history.length = 0;
      messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
      emptyEl.hidden = false;
      setModelName(data.model);
      setStatus("Ready — " + (data.model || "model loaded"));
      [...modelSelectEl.options].forEach((o) => delete o.dataset.loaded);
      const loaded = [...modelSelectEl.options].find((o) => o.value === path);
      if (loaded) loaded.dataset.loaded = "true";
    } catch (err) {
      setStatus("Error loading model");
      addMessage("error", "Failed to load model: " + err.message);
      if (prev) modelSelectEl.value = prev;
    } finally {
      statusEl.classList.remove("busy");
      modelSelectEl.disabled = false;
    }
  });

  /* ── Message DOM ── */
  function addMessage(role, text) {
    emptyEl.hidden = true;
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.dataset.raw = text;
    const content = document.createElement("div");
    content.className = "content";
    if (role === "assistant" && !text) {
      content.innerHTML = '<span class="dots" aria-label="Waiting for response"><span>•</span><span>•</span><span>•</span></span>';
    } else {
      content.textContent = text;
    }
    el.appendChild(content);
    if (role === "user" && text) attachCopyButton(el);
    messagesEl.appendChild(el);
    scrollToBottom(role === "user");
    return el;
  }

  /* Finalizes an assistant bubble: reasoning block, markdown content,
     tool-call cards, and a subtle stats line. */
  function finalizeAssistant(el, result) {
    const content = el.querySelector(".content");
    content.classList.remove("streaming");
    el.dataset.raw = result.answer || "";

    if (result.reasoning) {
      const details = document.createElement("details");
      details.className = "reasoning";
      const summary = document.createElement("summary");
      summary.textContent = "Reasoning";
      const body = document.createElement("div");
      body.className = "reasoning-body";
      body.textContent = result.reasoning;
      details.appendChild(summary);
      details.appendChild(body);
      el.insertBefore(details, content);
    }

    const toolCalls = result.toolCalls || [];
    if (result.answer) {
      content.innerHTML = renderMarkdown(result.answer);
      addCodeCopyButtons(content);
    } else if (toolCalls.length) {
      content.remove();
    } else {
      content.textContent = "(empty response)";
      content.classList.add("muted");
    }

    if (toolCalls.length) {
      const wrap = document.createElement("div");
      wrap.className = "tool-calls";
      for (const call of toolCalls) {
        const fn = (call && call.function) || {};
        const card = document.createElement("div");
        card.className = "tool-call";
        const head = document.createElement("div");
        head.className = "tool-head";
        const badge = document.createElement("span");
        badge.className = "tool-badge";
        badge.textContent = "tool call";
        const name = document.createElement("span");
        name.className = "tool-name";
        name.textContent = fn.name || "(unnamed function)";
        head.appendChild(badge);
        head.appendChild(name);
        const args = document.createElement("pre");
        args.className = "tool-args";
        args.textContent = prettyJSON(fn.arguments != null ? fn.arguments : "{}");
        card.appendChild(head);
        card.appendChild(args);
        wrap.appendChild(card);
      }
      el.appendChild(wrap);
    }

    const parts = [];
    const u = result.usage;
    if (u && typeof u.completion_tokens === "number") {
      if (result.decodeMS > 150 && u.completion_tokens > 0) {
        parts.push((u.completion_tokens / (result.decodeMS / 1000)).toFixed(1) + " tok/s");
      }
      parts.push(u.completion_tokens + " tokens");
      if (typeof u.prompt_tokens === "number") parts.push(u.prompt_tokens + " prompt");
    }
    if (result.finishReason && result.finishReason !== "stop") parts.push(result.finishReason);
    if (parts.length) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = parts.join(" · ");
      el.appendChild(meta);
    }

    if (result.answer) attachCopyButton(el);
    scrollToBottom(false);
  }

  /* ── SSE stream reader ──
     Collects content deltas plus everything the terminal chunk carries:
     finish_reason and usage live on choices[0]; reasoning_content and
     tool_calls arrive in the final delta. */
  async function readChatStream(response, onToken) {
    if (!response.body) throw new Error("Streaming response has no body");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const out = { answer: "", reasoning: "", toolCalls: null, usage: null, finishReason: "" };

    function applyChunk(payload) {
      if (!payload || payload === "[DONE]") return;
      const event = JSON.parse(payload);
      if (event.error) throw new Error(event.error);
      const choice = event.choices && event.choices[0];
      if (event.usage) out.usage = event.usage;
      if (!choice) return;
      if (choice.finish_reason) out.finishReason = choice.finish_reason;
      if (choice.usage) out.usage = choice.usage;
      const delta = choice.delta || {};
      if (delta.reasoning_content) out.reasoning += delta.reasoning_content;
      if (delta.tool_calls) out.toolCalls = (out.toolCalls || []).concat(delta.tool_calls);
      if (delta.content) {
        out.answer += delta.content;
        onToken(out.answer);
      }
    }

    function applyEventBlock(block) {
      const data = block.split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) applyChunk(data);
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const event of events) applyEventBlock(event);
    }
    buffer += decoder.decode();
    if (buffer.trim()) applyEventBlock(buffer);
    return out;
  }

  /* ── Composer behaviour ── */
  temperatureEl.addEventListener("input", () => {
    tempValueEl.textContent = Number(temperatureEl.value).toFixed(2);
  });
  tempValueEl.textContent = Number(temperatureEl.value).toFixed(2);

  promptEl.addEventListener("input", () => {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + "px";
  });

  // Enter sends, Shift+Enter inserts a newline (Ctrl/Cmd+Enter also sends).
  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  sendEl.addEventListener("click", (e) => {
    if (busy) {
      e.preventDefault();
      if (controller) controller.abort();
    }
  });

  clearEl.addEventListener("click", () => {
    if (busy) return;
    history.length = 0;
    messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
    emptyEl.hidden = false;
    promptEl.focus();
  });

  loadModelList();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const text = promptEl.value.trim();
    if (!text) return;

    promptEl.value = "";
    promptEl.style.height = "";
    const userMessage = { role: "user", content: text };
    addMessage("user", text);
    followStream = true;
    const assistantEl = addMessage("assistant", "");
    const contentEl = assistantEl.querySelector(".content");
    setBusy(true);
    controller = new AbortController();

    const startedAt = performance.now();
    let firstTokenAt = 0;
    let latest = "";
    let renderPending = false;
    // Streamed chunks are painted as plain text at most once per animation
    // frame (re-rendering the whole Markdown tree per token was the observed
    // bottleneck); full Markdown rendering happens once, after the stream.
    function onToken(answer) {
      latest = answer;
      assistantEl.dataset.raw = answer;
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
        setStatus("Generating…");
        contentEl.classList.add("streaming");
      }
      if (renderPending) return;
      renderPending = true;
      requestAnimationFrame(() => {
        renderPending = false;
        contentEl.textContent = latest;
        scrollToBottom(false);
      });
    }

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          messages: history.concat(userMessage),
          stream: true,
          stream_options: { include_usage: true },
          max_tokens: Number(maxTokensEl.value) || 512,
          temperature: Number(temperatureEl.value)
        })
      });

      if (!response.ok) {
        const errText = await response.text();
        let msg = "HTTP " + response.status;
        try { msg = JSON.parse(errText).error || msg; } catch (_) {}
        throw new Error(msg);
      }

      let result;
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/event-stream")) {
        result = await readChatStream(response, onToken);
      } else {
        const data = await response.json();
        const choice = (data.choices && data.choices[0]) || {};
        const message = choice.message || {};
        result = {
          answer: message.content || "",
          reasoning: message.reasoning_content || "",
          toolCalls: message.tool_calls || null,
          usage: data.usage || null,
          finishReason: choice.finish_reason || ""
        };
      }
      result.decodeMS = (firstTokenAt ? performance.now() - firstTokenAt : performance.now() - startedAt);
      finalizeAssistant(assistantEl, result);
      const assistantMessage = { role: "assistant", content: result.answer };
      if (result.toolCalls && result.toolCalls.length) assistantMessage.tool_calls = result.toolCalls;
      history.push(userMessage, assistantMessage);
    } catch (err) {
      const partial = assistantEl.dataset.raw || "";
      if (err.name === "AbortError") {
        if (partial) {
          finalizeAssistant(assistantEl, {
            answer: partial, reasoning: "", toolCalls: null, usage: null,
            finishReason: "stopped", decodeMS: 0
          });
          history.push(userMessage, { role: "assistant", content: partial });
        } else {
          assistantEl.remove();
          if (!messagesEl.querySelector(".msg")) emptyEl.hidden = false;
        }
      } else {
        if (partial) {
          finalizeAssistant(assistantEl, {
            answer: partial, reasoning: "", toolCalls: null, usage: null,
            finishReason: "interrupted", decodeMS: 0
          });
          history.push(userMessage, { role: "assistant", content: partial });
        } else {
          assistantEl.remove();
        }
        addMessage("error", "Error: " + err.message);
      }
    } finally {
      controller = null;
      setBusy(false);
      scrollToBottom(false);
      promptEl.focus();
    }
  });
}());
