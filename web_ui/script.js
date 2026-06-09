"use strict";

function renderMarkdown(raw) {
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*([^\s*](?:[^*]*[^\s*])?)\*/g, "<em>$1</em>");
    return s;
  }
  const lines = raw.split("\n");
  let out = "", inCode = false, codeLines = [], inUl = false, inOl = false;
  function closeList() {
    if (inUl) { out += "</ul>"; inUl = false; }
    if (inOl) { out += "</ol>"; inOl = false; }
  }
  for (const line of lines) {
    if (!inCode && line.startsWith("```")) { closeList(); inCode = true; codeLines = []; continue; }
    if (inCode) {
      if (line.startsWith("```")) { out += "<pre><code>" + esc(codeLines.join("\n")) + "</code></pre>"; inCode = false; codeLines = []; }
      else { codeLines.push(line); }
      continue;
    }
    const hm = line.match(/^(#{1,3}) (.+)/);
    if (hm) { closeList(); out += `<h${hm[1].length}>${inline(hm[2])}</h${hm[1].length}>`; continue; }
    const ulm = line.match(/^[*\-] (.+)/);
    if (ulm) { if (inOl) { out += "</ol>"; inOl = false; } if (!inUl) { out += "<ul>"; inUl = true; } out += `<li>${inline(ulm[1])}</li>`; continue; }
    const olm = line.match(/^\d+\. (.+)/);
    if (olm) { if (inUl) { out += "</ul>"; inUl = false; } if (!inOl) { out += "<ol>"; inOl = true; } out += `<li>${inline(olm[1])}</li>`; continue; }
    closeList();
    if (line.trim() === "") { out += "<br>"; } else { out += inline(line) + "<br>"; }
  }
  if (inCode) { out += "<pre><code>" + esc(codeLines.join("\n")) + "</code></pre>"; }
  closeList();
  return out.replace(/^(<br>)+/, "").replace(/(<br>)+$/, "");
}

function attachCopyButton(el) {
  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.type = "button";
  btn.textContent = "Copy";
  btn.setAttribute("aria-label", "Copy message to clipboard");
  btn.addEventListener("click", () => {
    navigator.clipboard.writeText(el.dataset.raw || "").then(() => {
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = "Copy"; }, 1500);
    }).catch(() => {});
  });
  el.appendChild(btn);
}

(function initChat() {
  const form           = document.getElementById("form");
  const promptEl       = document.getElementById("prompt");
  const messagesEl     = document.getElementById("messages");
  const emptyEl        = document.getElementById("empty");
  const statusEl       = document.getElementById("status");
  const sendEl         = document.getElementById("send");
  const clearEl        = document.getElementById("clear");
  const maxTokensEl    = document.getElementById("maxTokens");
  const temperatureEl  = document.getElementById("temperature");
  const tempValueEl    = document.getElementById("tempValue");
  const scrollEl       = document.getElementById("scroll");
  const modelSelectEl  = document.getElementById("modelSelect");

  const history = [];
  let busy = false;

  function setBusy(b) {
    busy = b;
    sendEl.disabled = b;
    modelSelectEl.disabled = b;
    statusEl.textContent = b ? "Generating…" : "Ready";
  }

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
        const unsupported = !m.supported ? " ⚠" : "";
        opt.textContent = label + arch + size + unsupported;
        if (m.loaded) opt.selected = true;
        if (!m.supported) opt.style.color = "var(--muted)";
        modelSelectEl.appendChild(opt);
      }
    } catch (_) {}
  }

  modelSelectEl.addEventListener("change", async () => {
    const path = modelSelectEl.value;
    if (!path || busy) return;
    const prev = [...modelSelectEl.options].find((o) => o.dataset.loaded === "true")?.value || "";
    modelSelectEl.disabled = true;
    statusEl.textContent = "Loading model…";
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
      statusEl.textContent = "Ready — " + (data.model || "model loaded");
      [...modelSelectEl.options].forEach((o) => delete o.dataset.loaded);
      const loaded = [...modelSelectEl.options].find((o) => o.value === path);
      if (loaded) loaded.dataset.loaded = "true";
    } catch (err) {
      statusEl.textContent = "Error loading model";
      addMessage("error", "Failed to load model: " + err.message);
      if (prev) modelSelectEl.value = prev;
    } finally {
      modelSelectEl.disabled = false;
    }
  });

  function addMessage(role, text) {
    emptyEl.hidden = true;
    const el = document.createElement("div");
    el.className = "msg " + role;
    el.dataset.raw = text;
    if (role === "assistant") {
      el.innerHTML = text
        ? renderMarkdown(text)
        : '<span class="dots"><span>•</span><span>•</span><span>•</span></span>';
    } else if (role === "error") {
      el.textContent = text;
    } else {
      el.textContent = text;
    }
    if (role !== "error" && text) attachCopyButton(el);
    messagesEl.appendChild(el);
    scrollEl.scrollTop = scrollEl.scrollHeight;
    return el;
  }

  async function readChatStream(response, assistantEl) {
    if (!response.body) throw new Error("Streaming response has no body");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";

    function applyChunk(payload) {
      if (!payload || payload === "[DONE]") return;
      const event = JSON.parse(payload);
      if (event.error) throw new Error(event.error);
      const choice = event.choices?.[0];
      const content = choice?.delta?.content || "";
      if (!content) return;
      answer += content;
      assistantEl.dataset.raw = answer;
      assistantEl.innerHTML = renderMarkdown(answer);
      scrollEl.scrollTop = scrollEl.scrollHeight;
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const event of events) {
        const data = event.split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data) applyChunk(data);
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const data = buffer.split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) applyChunk(data);
    }
    return answer;
  }

  temperatureEl.addEventListener("input", () => {
    tempValueEl.textContent = Number(temperatureEl.value).toFixed(2);
  });

  promptEl.addEventListener("input", () => {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + "px";
  });

  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) form.requestSubmit();
  });

  clearEl.addEventListener("click", () => {
    history.length = 0;
    messagesEl.querySelectorAll(".msg").forEach((n) => n.remove());
    emptyEl.hidden = false;
  });

  loadModelList();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = promptEl.value.trim();
    if (!text || busy) return;

    promptEl.value = "";
    promptEl.style.height = "";
    const userMessage = { role: "user", content: text };
    addMessage("user", text);
    const assistantEl = addMessage("assistant", "");
    setBusy(true);

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.concat(userMessage),
          stream: true,
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

      let answer = "";
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/event-stream")) {
        answer = await readChatStream(response, assistantEl);
      } else {
        const result = await response.json();
        answer = result.choices?.[0]?.message?.content || "";
      }
      assistantEl.dataset.raw = answer;
      assistantEl.innerHTML = renderMarkdown(answer);
      attachCopyButton(assistantEl);
      history.push(userMessage, { role: "assistant", content: answer });
    } catch (err) {
      assistantEl.remove();
      addMessage("error", "Error: " + err.message);
    } finally {
      setBusy(false);
      scrollEl.scrollTop = scrollEl.scrollHeight;
      promptEl.focus();
    }
  });
}());
