"use strict";

const TICK = String.fromCharCode(96);
const FENCE = TICK.repeat(3);

function escapeHTML(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function renderInline(s) {
  s = escapeHTML(s);
  const inlineCode = new RegExp(TICK + "([^" + TICK + "]+)" + TICK, "g");
  s = s.replace(inlineCode, "<code>$1</code>");
  s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return s.replace(/\*([^\s*](?:[^*]*[^\s*])?)\*/g, "<em>$1</em>");
}

function renderMarkdown(raw) {
  const lines = String(raw || "").split("\n");
  let out = "", inCode = false, codeLines = [], codeLang = "", inUL = false, inOL = false;
  const closeLists = () => {
    if (inUL) { out += "</ul>"; inUL = false; }
    if (inOL) { out += "</ol>"; inOL = false; }
  };
  const flushCode = () => {
    out += '<div class="codeblock">' + (codeLang ? '<span class="code-lang">' + escapeHTML(codeLang) + "</span>" : "")
      + "<pre><code>" + escapeHTML(codeLines.join("\n")) + "</code></pre></div>";
    inCode = false;
    codeLines = [];
    codeLang = "";
  };
  for (const line of lines) {
    if (!inCode && line.startsWith(FENCE)) {
      closeLists();
      inCode = true;
      codeLang = line.slice(FENCE.length).trim();
      continue;
    }
    if (inCode) {
      if (line.startsWith(FENCE)) flushCode();
      else codeLines.push(line);
      continue;
    }
    const heading = line.match(/^(#{1,3}) (.+)/);
    if (heading) {
      closeLists();
      out += "<h" + heading[1].length + ">" + renderInline(heading[2]) + "</h" + heading[1].length + ">";
      continue;
    }
    const unordered = line.match(/^[*-] (.+)/);
    if (unordered) {
      if (inOL) { out += "</ol>"; inOL = false; }
      if (!inUL) { out += "<ul>"; inUL = true; }
      out += "<li>" + renderInline(unordered[1]) + "</li>";
      continue;
    }
    const ordered = line.match(/^\d+\. (.+)/);
    if (ordered) {
      if (inUL) { out += "</ul>"; inUL = false; }
      if (!inOL) { out += "<ol>"; inOL = true; }
      out += "<li>" + renderInline(ordered[1]) + "</li>";
      continue;
    }
    closeLists();
    out += line.trim() ? renderInline(line) + "<br>" : "<br>";
  }
  if (inCode) flushCode();
  closeLists();
  return out.replace(/^(<br>)+/, "").replace(/(<br>)+$/, "");
}

function notify(text, kind) {
  document.dispatchEvent(new CustomEvent("gopherllm:notice", { detail: { text: String(text), kind } }));
}

function bindCopy(button, getText) {
  const label = button.textContent;
  button.addEventListener("click", () => {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      notify("Clipboard access is unavailable in this browser.", "error");
      return;
    }
    navigator.clipboard.writeText(getText()).then(() => {
      button.textContent = "Copied";
      notify("Copied to clipboard", "success");
      setTimeout(() => { button.textContent = label; }, 1400);
    }).catch(() => notify("Could not copy to the clipboard.", "error"));
  });
}

function attachCopyButton(el) {
  if (el.querySelector(":scope > .copy-btn")) return;
  const button = document.createElement("button");
  button.className = "copy-btn";
  button.type = "button";
  button.textContent = "Copy";
  button.setAttribute("aria-label", "Copy message to clipboard");
  bindCopy(button, () => el.dataset.raw || "");
  el.appendChild(button);
}

function addCodeCopyButtons(container) {
  container.querySelectorAll(".codeblock").forEach((block) => {
    if (block.querySelector(".code-copy")) return;
    const code = block.querySelector("code");
    if (!code) return;
    const button = document.createElement("button");
    button.className = "code-copy";
    button.type = "button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code to clipboard");
    bindCopy(button, () => code.textContent);
    block.appendChild(button);
  });
}

function prettyJSON(value) {
  if (typeof value !== "string") {
    try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
  }
  try { return JSON.stringify(JSON.parse(value), null, 2); } catch (_) { return value; }
}

function splitThinkText(raw) {
  const openTag = "<think>";
  const closeTag = "</think>";
  let rest = raw || "";
  let answer = "";
  const reasoning = [];
  let hasThink = false;
  let isThinking = false;
  const firstClose = rest.indexOf(closeTag);
  const firstOpen = rest.indexOf(openTag);
  if (firstClose >= 0 && (firstOpen < 0 || firstClose < firstOpen)) {
    hasThink = true;
    reasoning.push(rest.slice(0, firstClose));
    rest = rest.slice(firstClose + closeTag.length);
  }
  while (true) {
    const openAt = rest.indexOf(openTag);
    if (openAt < 0) {
      answer += rest;
      break;
    }
    hasThink = true;
    answer += rest.slice(0, openAt);
    rest = rest.slice(openAt + openTag.length);
    const closeAt = rest.indexOf(closeTag);
    if (closeAt < 0) {
      reasoning.push(rest);
      isThinking = true;
      break;
    }
    reasoning.push(rest.slice(0, closeAt));
    rest = rest.slice(closeAt + closeTag.length);
  }
  return { answer: answer.trim(), reasoning: reasoning.join("\n\n").trim(), hasThink, isThinking };
}

/* IndexedDB is the primary local workspace store. localStorage is a fallback
   for browsers where IndexedDB is unavailable. */
const DB_NAME = "gopherllm-chat";
const STORE_NAME = "workspace";
const STORE_KEY = "state";
const FALLBACK_KEY = "gopherllm-chat-fallback-v1";
let dbPromise;

function openDB() {
  if (dbPromise) return dbPromise;
  if (!("indexedDB" in window)) return Promise.reject(new Error("IndexedDB unavailable"));
  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Could not open chat storage"));
  });
  return dbPromise;
}

function fallbackRead() {
  try {
    const data = localStorage.getItem(FALLBACK_KEY);
    return data ? JSON.parse(data) : null;
  } catch (_) {
    return null;
  }
}

function fallbackWrite(value) {
  try {
    localStorage.setItem(FALLBACK_KEY, JSON.stringify(value));
    return true;
  } catch (_) {
    return false;
  }
}

async function readWorkspace() {
  try {
    const db = await openDB();
    return await new Promise((resolve, reject) => {
      const request = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(STORE_KEY);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("Could not read chat storage"));
    });
  } catch (_) {
    return fallbackRead();
  }
}

async function writeWorkspace(value) {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put(value, STORE_KEY);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error || new Error("Could not save chat storage"));
      tx.onabort = () => reject(tx.error || new Error("Could not save chat storage"));
    });
    return true;
  } catch (_) {
    return fallbackWrite(value);
  }
}

const PERSONAS = {
  general: "You are a helpful, accurate assistant. Be clear, direct, and honest about uncertainty.",
  code: "You are a careful software engineer. Explain trade-offs, provide safe code, and call out assumptions and edge cases.",
  writer: "You are a thoughtful writing partner. Preserve the user's voice, make concrete edits, and explain major changes briefly.",
  translator: "You are a precise translator. Preserve meaning, tone, formatting, names, and technical terms. Return only the translation unless asked otherwise."
};

function makeID() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  return "chat-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}

function boundedNumber(value, fallback, min, max, integer) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  const bounded = Math.min(max, Math.max(min, number));
  return integer ? Math.round(bounded) : bounded;
}

function cleanSettings(value, defaults) {
  value = value && typeof value === "object" ? value : {};
  return {
    maxTokens: boundedNumber(value.maxTokens, defaults.maxTokens, 1, 4096, true),
    temperature: boundedNumber(value.temperature, defaults.temperature, 0, 2, false),
    topP: boundedNumber(value.topP, defaults.topP, 0, 1, false),
    topK: boundedNumber(value.topK, defaults.topK, 0, 200, true),
    minP: boundedNumber(value.minP, defaults.minP, 0, 1, false),
    repeatPenalty: boundedNumber(value.repeatPenalty, defaults.repeatPenalty, 0, 3, false),
    seed: value.seed == null ? "" : String(value.seed).slice(0, 20),
    stopSequences: typeof value.stopSequences === "string" ? value.stopSequences.slice(0, 2000) : ""
  };
}

function cleanMessage(value) {
  if (!value || typeof value !== "object" || (value.role !== "user" && value.role !== "assistant") || typeof value.content !== "string") return null;
  return {
    role: value.role,
    content: value.content.slice(0, 1000000),
    reasoning: typeof value.reasoning === "string" ? value.reasoning.slice(0, 1000000) : "",
    tool_calls: Array.isArray(value.tool_calls) ? value.tool_calls : null,
    usage: value.usage && typeof value.usage === "object" ? value.usage : null,
    finishReason: typeof value.finishReason === "string" ? value.finishReason : ""
  };
}

function titleFor(text) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  return !compact ? "New chat" : (compact.length > 54 ? compact.slice(0, 53).trimEnd() + "…" : compact);
}

function newChat(defaults) {
  const now = Date.now();
  return {
    id: makeID(), title: "New chat", titleManual: false, createdAt: now, updatedAt: now,
    model: "", persona: "custom", systemPrompt: "", draft: "", settings: cleanSettings({}, defaults), messages: []
  };
}

function cleanChat(value, defaults) {
  if (!value || typeof value !== "object") return null;
  const messages = Array.isArray(value.messages) ? value.messages.map(cleanMessage).filter(Boolean) : [];
  const firstUser = messages.find((message) => message.role === "user");
  return {
    id: typeof value.id === "string" && value.id ? value.id.slice(0, 160) : makeID(),
    title: typeof value.title === "string" && value.title.trim() ? value.title.trim().slice(0, 160) : titleFor(firstUser && firstUser.content),
    titleManual: value.titleManual === true,
    createdAt: Number.isFinite(value.createdAt) ? value.createdAt : Date.now(),
    updatedAt: Number.isFinite(value.updatedAt) ? value.updatedAt : Date.now(),
    model: typeof value.model === "string" ? value.model.slice(0, 500) : "",
    persona: Object.prototype.hasOwnProperty.call(PERSONAS, value.persona) ? value.persona : "custom",
    systemPrompt: typeof value.systemPrompt === "string" ? value.systemPrompt.slice(0, 100000) : "",
    draft: typeof value.draft === "string" ? value.draft.slice(0, 100000) : "",
    settings: cleanSettings(value.settings, defaults),
    messages
  };
}

function download(filename, type, content) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function toMarkdown(chat) {
  const lines = ["# " + chat.title, ""];
  if (chat.systemPrompt.trim()) lines.push("## Instructions", "", chat.systemPrompt.trim(), "");
  for (const message of chat.messages) {
    lines.push(message.role === "user" ? "## You" : "## Assistant", "", message.content || "(no text)", "");
  }
  return lines.join("\n");
}

(async function initChat() {
  const $ = (id) => document.getElementById(id);
  const form = $("form");
  const promptEl = $("prompt");
  const messagesEl = $("messages");
  const emptyEl = $("empty");
  const statusEl = $("status");
  const statusTextEl = $("statusText");
  const sendEl = $("send");
  const sendLabelEl = sendEl.querySelector(".send-label");
  const maxTokensEl = $("maxTokens");
  const temperatureEl = $("temperature");
  const tempValueEl = $("tempValue");
  const topPEl = $("topP");
  const topKEl = $("topK");
  const minPEl = $("minP");
  const repeatPenaltyEl = $("repeatPenalty");
  const seedEl = $("seed");
  const stopSequencesEl = $("stopSequences");
  const personaEl = $("personaSelect");
  const systemPromptEl = $("systemPrompt");
  const settingsEl = $("settings");
  const settingsCloseEl = $("settingsClose");
  const skillsNoteEl = $("skillsNote");
  const autoTuneEffortEl = $("autoTuneEffort");
  const autoTuneRunEl = $("autoTuneRun");
  const autoTuneStatusEl = $("autoTuneStatus");
  const settingsToggleEl = $("settingsToggle");
  const scrollEl = $("scroll");
  const jumpLatestEl = $("jumpLatest");
  const contextEstimateEl = $("contextEstimate");
  const modelSelectEl = $("modelSelect");
  const modelNameEl = $("modelName");
  const chatTitleEl = $("chatTitle");
  const chatListEl = $("chatList");
  const chatSearchEl = $("chatSearch");
  const sidebarEl = $("sidebar");
  const sidebarToggleEl = $("sidebarToggle");
  const sidebarScrimEl = $("sidebarScrim");
  const newChatEl = $("newChat");
  const clearEl = $("clear");
  const renameChatEl = $("renameChat");
  const attachTextEl = $("attachText");
  const textFileInputEl = $("textFileInput");
  const exportChatsEl = $("exportChats");
  const exportMarkdownEl = $("exportMarkdown");
  const importChatsEl = $("importChats");
  const importInputEl = $("importInput");
  const themeSelectEl = $("themeSelect");
  const editStateEl = $("editState");
  const cancelEditEl = $("cancelEdit");
  const toastEl = $("toast");

  const defaults = {
    maxTokens: boundedNumber(maxTokensEl.value, 512, 1, 4096, true),
    temperature: boundedNumber(temperatureEl.value, .7, 0, 2, false),
    topP: boundedNumber(topPEl.value, .9, 0, 1, false),
    topK: boundedNumber(topKEl.value, 40, 0, 200, true),
    minP: boundedNumber(minPEl.value, 0, 0, 1, false),
    repeatPenalty: boundedNumber(repeatPenaltyEl.value, 1.1, 0, 3, false)
  };
  const MAX_CHATS = 100;
  let chats = [];
  let activeID = "";
  let preferences = { theme: "system" };
  let busy = false;
  let tuning = false;
  let loadingModel = false;
  let controller = null;
  let editingIndex = null;
  let draftBeforeEdit = "";
  let followStream = true;
  let contextLimit = 0;
  let saveTimer = null;
  let saveQueue = Promise.resolve();
  let toastTimer = null;

  function activeChat() {
    return chats.find((chat) => chat.id === activeID) || null;
  }

  function workspace() {
    return { format: "gopherllm-chat-workspace", version: 1, activeID, preferences, conversations: chats };
  }

  function save() {
    const snapshot = workspace();
    saveQueue = saveQueue.catch(() => {}).then(async () => {
      if (!await writeWorkspace(snapshot)) showToast("Chat history could not be saved in this browser.", "error");
    });
    return saveQueue;
  }

  function saveSoon() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 350);
  }

  function touch(chat) {
    if (chat) chat.updatedAt = Date.now();
  }

  function showToast(text, kind) {
    clearTimeout(toastTimer);
    toastEl.textContent = text;
    toastEl.className = "toast" + (kind ? " toast-" + kind : "");
    toastEl.hidden = false;
    toastTimer = setTimeout(() => { toastEl.hidden = true; }, 2600);
  }
  document.addEventListener("gopherllm:notice", (event) => showToast(event.detail.text, event.detail.kind));

  function setStatus(text) {
    statusTextEl.textContent = text;
  }

  function applyTheme(theme) {
    const value = theme === "light" || theme === "dark" ? theme : "system";
    preferences.theme = value;
    if (value === "system") delete document.body.dataset.theme;
    else document.body.dataset.theme = value;
    themeSelectEl.value = value;
  }

  function setSidebar(open) {
    sidebarEl.classList.toggle("is-open", open);
    sidebarScrimEl.hidden = !open;
    sidebarToggleEl.setAttribute("aria-expanded", String(open));
    sidebarToggleEl.setAttribute("aria-label", open ? "Close chat history" : "Open chat history");
  }

  const timeFormat = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
  const weekdayFormat = new Intl.DateTimeFormat(undefined, { weekday: "short" });
  const dateFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
  function stamp(value) {
    const date = new Date(value);
    const age = Date.now() - value;
    if (age < 86400000) return timeFormat.format(date);
    if (age < 604800000) return weekdayFormat.format(date);
    return dateFormat.format(date);
  }

  function renderChatList() {
    const needle = chatSearchEl.value.trim().toLowerCase();
    const visible = chats.slice().sort((a, b) => b.updatedAt - a.updatedAt).filter((chat) => {
      const text = chat.title + "\n" + chat.systemPrompt + "\n" + chat.messages.slice(-8).map((message) => message.content).join("\n");
      return !needle || text.toLowerCase().includes(needle);
    });
    chatListEl.replaceChildren();
    if (!visible.length) {
      const empty = document.createElement("p");
      empty.className = "chat-empty";
      empty.textContent = needle ? "No chats match your search." : "No saved chats yet.";
      chatListEl.appendChild(empty);
      return;
    }
    for (const chat of visible) {
      const row = document.createElement("div");
      row.className = "chat-row" + (chat.id === activeID ? " active" : "");
      const select = document.createElement("button");
      select.type = "button";
      select.className = "chat-select";
      select.setAttribute("aria-current", chat.id === activeID ? "page" : "false");
      const title = document.createElement("span");
      title.className = "chat-row-title";
      title.textContent = chat.title;
      const meta = document.createElement("span");
      meta.className = "chat-row-meta";
      meta.textContent = (chat.messages.length ? chat.messages.length + " messages" : "Empty chat") + " · " + stamp(chat.updatedAt);
      select.append(title, meta);
      select.addEventListener("click", () => openChat(chat.id));
      const menu = document.createElement("button");
      menu.type = "button";
      menu.className = "chat-menu";
      menu.textContent = "⋯";
      menu.title = "Rename or delete chat";
      menu.setAttribute("aria-label", "Rename or delete " + chat.title);
      menu.addEventListener("click", () => manageChat(chat.id));
      row.append(select, menu);
      chatListEl.appendChild(row);
    }
  }

  function resizePrompt() {
    promptEl.style.height = "auto";
    promptEl.style.height = Math.min(promptEl.scrollHeight, 200) + "px";
  }

  /* Caches the char count of everything except the live draft (system prompt +
     message history), so typing in the composer doesn't re-walk the whole
     conversation on every keystroke. Invalidated by message count / system
     prompt identity, which covers pushes, truncation (edit/retry), and edits
     to the instructions field. */
  const historyCharsCache = new Map();
  function historyChars(chat) {
    const cached = historyCharsCache.get(chat.id);
    if (cached && cached.count === chat.messages.length && cached.systemPrompt === chat.systemPrompt) return cached.chars;
    let chars = chat.systemPrompt.length;
    for (const message of chat.messages) chars += (message.content || "").length + 1;
    historyCharsCache.set(chat.id, { count: chat.messages.length, systemPrompt: chat.systemPrompt, chars });
    return chars;
  }

  function tokenEstimate(chat) {
    const chars = historyChars(chat) + chat.draft.trim().length;
    return chars ? Math.max(1, Math.ceil(chars / 4)) : 0;
  }

  function updateComposer(storeDraft) {
    const chat = activeChat();
    if (!chat) return;
    if (storeDraft) {
      chat.draft = promptEl.value;
      saveSoon();
    }
    const count = chat.messages.length;
    const estimate = tokenEstimate(chat);
    const context = contextLimit ? " / " + contextLimit + " context" : "";
    const warning = contextLimit && estimate >= contextLimit * .85 ? " · near context limit" : "";
    contextEstimateEl.textContent = count + " message" + (count === 1 ? "" : "s") + " · ~" + estimate + " input tokens" + context + warning;
    if (!busy) {
      sendEl.disabled = tuning || !promptEl.value.trim();
      sendLabelEl.textContent = editingIndex === null ? "Send" : "Save & retry";
    }
  }

  function syncControls(withDraft) {
    const chat = activeChat();
    if (!chat) return;
    chatTitleEl.textContent = chat.title;
    chatTitleEl.title = chat.title;
    if (chat.model) {
      modelNameEl.textContent = chat.model;
      modelNameEl.title = chat.model;
    }
    if (withDraft) {
      promptEl.value = chat.draft;
      resizePrompt();
    }
    maxTokensEl.value = chat.settings.maxTokens;
    temperatureEl.value = chat.settings.temperature;
    tempValueEl.textContent = Number(chat.settings.temperature).toFixed(2);
    topPEl.value = Number(chat.settings.topP).toFixed(2);
    topKEl.value = chat.settings.topK;
    minPEl.value = Number(chat.settings.minP).toFixed(2);
    repeatPenaltyEl.value = Number(chat.settings.repeatPenalty).toFixed(2);
    seedEl.value = chat.settings.seed;
    stopSequencesEl.value = chat.settings.stopSequences;
    personaEl.value = Object.prototype.hasOwnProperty.call(PERSONAS, chat.persona) ? chat.persona : "custom";
    systemPromptEl.value = chat.systemPrompt;
    updateComposer(false);
  }

  function setBusy(value) {
    busy = value;
    modelSelectEl.disabled = value;
    newChatEl.disabled = value;
    renameChatEl.disabled = value;
    attachTextEl.disabled = value;
    exportChatsEl.disabled = value;
    exportMarkdownEl.disabled = value;
    if (value) {
      sendEl.disabled = false;
      sendLabelEl.textContent = "Stop";
    } else {
      updateComposer(false);
    }
    sendEl.classList.toggle("stop", value);
    sendEl.setAttribute("aria-label", value ? "Stop generating" : "Send message");
    statusEl.classList.toggle("busy", value);
    setStatus(value ? "Thinking…" : "Ready");
  }

  function addMessage(role, text) {
    emptyEl.hidden = true;
    const el = document.createElement("article");
    el.className = "msg " + role;
    if (role === "error") el.setAttribute("role", "alert");
    el.dataset.raw = text || "";
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
    return el;
  }

  function upsertReasoning(el, text, thinking) {
    let details = el.querySelector(":scope > .reasoning");
    if (!text && !thinking) {
      if (details) details.remove();
      return;
    }
    if (!details) {
      details = document.createElement("details");
      details.className = "reasoning";
      const summary = document.createElement("summary");
      const body = document.createElement("div");
      body.className = "reasoning-body";
      details.append(summary, body);
      el.insertBefore(details, el.querySelector(".content"));
    }
    details.classList.toggle("reasoning-live", thinking);
    details.querySelector("summary").textContent = thinking ? "Thinking…" : "Reasoning";
    details.querySelector(".reasoning-body").textContent = text;
  }

  function finalizeAssistant(el, result) {
    const content = el.querySelector(".content");
    content.classList.remove("streaming");
    const parsed = splitThinkText(result.answer || "");
    result.answer = parsed.hasThink ? parsed.answer : (result.answer || "");
    result.reasoning = result.reasoning || parsed.reasoning;
    el.dataset.raw = result.answer;
    upsertReasoning(el, result.reasoning, false);
    const calls = result.toolCalls || [];
    if (result.answer) {
      content.innerHTML = renderMarkdown(result.answer);
      addCodeCopyButtons(content);
    } else if (calls.length) {
      content.remove();
    } else {
      content.textContent = "(empty response)";
      content.classList.add("muted");
    }
    if (calls.length) {
      const wrap = document.createElement("div");
      wrap.className = "tool-calls";
      for (const call of calls) {
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
        const args = document.createElement("pre");
        args.className = "tool-args";
        args.textContent = prettyJSON(fn.arguments != null ? fn.arguments : "{}");
        head.append(badge, name);
        card.append(head, args);
        wrap.appendChild(card);
      }
      el.appendChild(wrap);
    }
    const parts = [];
    if (result.usage && typeof result.usage.completion_tokens === "number") {
      if (result.decodeMS > 150 && result.usage.completion_tokens > 0) parts.push((result.usage.completion_tokens / (result.decodeMS / 1000)).toFixed(1) + " tok/s");
      parts.push(result.usage.completion_tokens + " tokens");
      if (typeof result.usage.prompt_tokens === "number") parts.push(result.usage.prompt_tokens + " prompt");
    }
    if (result.finishReason && result.finishReason !== "stop") parts.push(result.finishReason);
    if (parts.length) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = parts.join(" · ");
      el.appendChild(meta);
    }
    if (result.answer) attachCopyButton(el);
  }

  function addActions(el, message, index) {
    if (message.role !== "user" && message.role !== "assistant") return;
    const wrap = document.createElement("div");
    wrap.className = "message-actions";
    const action = document.createElement("button");
    action.className = "message-action";
    action.type = "button";
    if (message.role === "user") {
      action.textContent = "Edit";
      action.setAttribute("aria-label", "Edit this message");
      action.addEventListener("click", () => editMessage(index));
    } else {
      action.textContent = "Retry";
      action.setAttribute("aria-label", "Regenerate this answer");
      action.addEventListener("click", () => retryMessage(index));
    }
    wrap.appendChild(action);
    el.appendChild(wrap);
  }

  function addRetryAction(el) {
    const wrap = document.createElement("div");
    wrap.className = "message-actions";
    const action = document.createElement("button");
    action.className = "message-action";
    action.type = "button";
    action.textContent = "Retry";
    action.addEventListener("click", () => { if (!busy) generate(); });
    wrap.appendChild(action);
    el.appendChild(wrap);
  }

  function renderConversation(scrollEnd) {
    const chat = activeChat();
    if (!chat) return;
    messagesEl.querySelectorAll(".msg").forEach((el) => el.remove());
    emptyEl.hidden = chat.messages.length > 0;
    chat.messages.forEach((message, index) => {
      const el = addMessage(message.role, message.content);
      if (message.role === "assistant") {
        finalizeAssistant(el, {
          answer: message.content, reasoning: message.reasoning, toolCalls: message.tool_calls,
          usage: message.usage, finishReason: message.finishReason, decodeMS: 0
        });
      }
      addActions(el, message, index);
    });
    if (scrollEnd) {
      followStream = true;
      scrollEl.scrollTop = scrollEl.scrollHeight;
    }
    jumpLatestEl.hidden = true;
    updateComposer(false);
  }

  function renderWorkspace(withDraft) {
    syncControls(withDraft);
    renderChatList();
    renderConversation(true);
  }

  function createChat() {
    if (busy) return;
    const chat = newChat(defaults);
    chat.model = modelNameEl.textContent || "";
    chats.unshift(chat);
    chats = chats.slice(0, MAX_CHATS);
    activeID = chat.id;
    editingIndex = null;
    editStateEl.hidden = true;
    renderWorkspace(true);
    save();
    setSidebar(false);
    promptEl.focus();
  }

  function openChat(id) {
    if (busy || id === activeID) {
      setSidebar(false);
      return;
    }
    if (!chats.some((chat) => chat.id === id)) return;
    activeID = id;
    editingIndex = null;
    editStateEl.hidden = true;
    renderWorkspace(true);
    save();
    setSidebar(false);
    promptEl.focus();
  }

  function renameChat(id) {
    const chat = chats.find((item) => item.id === id);
    if (!chat || busy) return;
    const title = window.prompt("Name this chat", chat.title);
    if (title === null) return;
    if (!title.trim()) {
      showToast("A chat name cannot be empty.", "error");
      return;
    }
    chat.title = title.trim().slice(0, 160);
    chat.titleManual = true;
    touch(chat);
    renderWorkspace(false);
    save();
  }

  function deleteChat(id) {
    const chat = chats.find((item) => item.id === id);
    if (!chat || busy) return;
    if (!window.confirm('Delete "' + chat.title + '" from this browser? This cannot be undone.')) return;
    chats = chats.filter((item) => item.id !== id);
    historyCharsCache.delete(id);
    if (!chats.length) {
      const replacement = newChat(defaults);
      replacement.model = modelNameEl.textContent || "";
      chats = [replacement];
    }
    if (activeID === id) activeID = chats.slice().sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
    editingIndex = null;
    editStateEl.hidden = true;
    renderWorkspace(true);
    save();
    showToast("Chat deleted", "success");
  }

  function manageChat(id) {
    const chat = chats.find((item) => item.id === id);
    if (!chat || busy) return;
    const choice = window.prompt('Enter a new name, or type DELETE to remove "' + chat.title + '".', chat.title);
    if (choice === null) return;
    if (choice.trim().toUpperCase() === "DELETE") {
      deleteChat(id);
      return;
    }
    if (!choice.trim()) {
      showToast("Type DELETE to remove a chat.", "error");
      return;
    }
    chat.title = choice.trim().slice(0, 160);
    chat.titleManual = true;
    touch(chat);
    renderWorkspace(false);
    save();
  }

  function cancelEdit() {
    const chat = activeChat();
    if (editingIndex !== null && chat) {
      chat.draft = draftBeforeEdit;
      promptEl.value = draftBeforeEdit;
      resizePrompt();
    }
    editingIndex = null;
    draftBeforeEdit = "";
    editStateEl.hidden = true;
    updateComposer(false);
    saveSoon();
  }

  function editMessage(index) {
    const chat = activeChat();
    const message = chat && chat.messages[index];
    if (!message || message.role !== "user" || busy) return;
    draftBeforeEdit = chat.draft;
    editingIndex = index;
    promptEl.value = message.content;
    chat.draft = message.content;
    editStateEl.hidden = false;
    resizePrompt();
    updateComposer(true);
    promptEl.focus();
  }

  function isNearBottom() {
    return scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 60;
  }

  function scrollToBottom(force) {
    if (force || followStream) {
      scrollEl.scrollTop = scrollEl.scrollHeight;
      jumpLatestEl.hidden = true;
    }
  }

  async function readStream(response, onToken) {
    if (!response.body) throw new Error("Streaming response has no body");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const out = { answer: "", reasoning: "", toolCalls: null, usage: null, finishReason: "" };
    const applyChunk = (payload) => {
      if (!payload || payload === "[DONE]") return;
      const event = JSON.parse(payload);
      if (event.error) throw new Error(event.error);
      if (event.usage) out.usage = event.usage;
      const choice = event.choices && event.choices[0];
      if (!choice) return;
      if (choice.finish_reason) out.finishReason = choice.finish_reason;
      if (choice.usage) out.usage = choice.usage;
      const delta = choice.delta || {};
      if (delta.reasoning_content) {
        out.reasoning += delta.reasoning_content;
        onToken(out.answer, out.reasoning, true);
      }
      if (delta.tool_calls) out.toolCalls = (out.toolCalls || []).concat(delta.tool_calls);
      if (delta.content) {
        out.answer += delta.content;
        onToken(out.answer, out.reasoning, false);
      }
    };
    const applyBlock = (block) => {
      const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
      if (data) applyChunk(data);
    };
    while (true) {
      const packet = await reader.read();
      if (packet.done) break;
      buffer += decoder.decode(packet.value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      blocks.forEach(applyBlock);
    }
    buffer += decoder.decode();
    if (buffer.trim()) applyBlock(buffer);
    return out;
  }

  function requestFor(chat) {
    const settings = chat.settings;
    const seed = /^\d+$/.test(settings.seed || "") ? Number(settings.seed) : undefined;
    const stop = (settings.stopSequences || "").split(",").map((s) => s.trim()).filter(Boolean);
    return {
      messages: chat.messages.map((message) => {
        const out = { role: message.role, content: message.content };
        if (message.role === "assistant" && message.tool_calls && message.tool_calls.length) out.tool_calls = message.tool_calls;
        return out;
      }),
      stream: true,
      stream_options: { include_usage: true },
      max_tokens: settings.maxTokens,
      temperature: settings.temperature,
      top_p: settings.topP,
      top_k: settings.topK,
      min_p: settings.minP,
      repeat_penalty: settings.repeatPenalty,
      seed,
      system_prompt: chat.systemPrompt.trim() || undefined,
      stop: stop.length ? stop : undefined
    };
  }

  async function generate() {
    const chat = activeChat();
    if (!chat || !chat.messages.some((message) => message.role === "user")) return;
    const assistantEl = addMessage("assistant", "");
    followStream = true;
    setBusy(true);
    controller = new AbortController();
    const startedAt = performance.now();
    let firstTokenAt = 0;
    let latest = "";
    let reasoning = "";
    let streamFinished = false;
    let pending = false;
    const onToken = (answer, nextReasoning, thinking) => {
      const parsed = splitThinkText(answer);
      latest = parsed.answer;
      reasoning = nextReasoning || parsed.reasoning;
      assistantEl.dataset.raw = latest;
      if (!firstTokenAt) {
        firstTokenAt = performance.now();
        assistantEl.querySelector(".content").classList.add("streaming");
      }
      setStatus(thinking || parsed.isThinking ? "Thinking…" : "Generating…");
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (streamFinished) return;
        const content = assistantEl.querySelector(".content");
        if (content) content.textContent = latest;
        upsertReasoning(assistantEl, reasoning, thinking || parsed.isThinking);
        scrollToBottom(false);
      });
    };
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(requestFor(chat))
      });
      if (!response.ok) {
        const body = await response.text();
        let message = "HTTP " + response.status;
        try {
          const json = JSON.parse(body);
          message = (json.error && (json.error.message || json.error)) || message;
        } catch (_) {}
        throw new Error(message);
      }
      let result;
      if ((response.headers.get("content-type") || "").includes("text/event-stream")) {
        result = await readStream(response, onToken);
      } else {
        const data = await response.json();
        const choice = (data.choices && data.choices[0]) || {};
        const message = choice.message || {};
        result = {
          answer: message.content || "", reasoning: message.reasoning_content || "",
          toolCalls: message.tool_calls || null, usage: data.usage || null, finishReason: choice.finish_reason || ""
        };
      }
      streamFinished = true;
      result.decodeMS = firstTokenAt ? performance.now() - firstTokenAt : performance.now() - startedAt;
      finalizeAssistant(assistantEl, result);
      const stored = {
        role: "assistant", content: result.answer || "", reasoning: result.reasoning || "",
        tool_calls: result.toolCalls || null, usage: result.usage || null, finishReason: result.finishReason || ""
      };
      chat.messages.push(stored);
      addActions(assistantEl, stored, chat.messages.length - 1);
      touch(chat);
      renderChatList();
      save();
      setStatus("Ready");
    } catch (error) {
      streamFinished = true;
      const partial = assistantEl.dataset.raw || "";
      if (error && error.name === "AbortError") {
        if (partial || reasoning) {
          const stored = { role: "assistant", content: partial, reasoning, tool_calls: null, usage: null, finishReason: "stopped" };
          finalizeAssistant(assistantEl, { answer: partial, reasoning, toolCalls: null, usage: null, finishReason: "stopped", decodeMS: 0 });
          chat.messages.push(stored);
          addActions(assistantEl, stored, chat.messages.length - 1);
          touch(chat);
          save();
        } else {
          assistantEl.remove();
        }
        showToast("Generation stopped");
      } else {
        assistantEl.remove();
        const errorEl = addMessage("error", "Error: " + (error && error.message ? error.message : "Request failed"));
        addRetryAction(errorEl);
        showToast("The answer could not be generated. Retry the last message.", "error");
      }
    } finally {
      controller = null;
      setBusy(false);
      renderChatList();
      scrollToBottom(false);
      promptEl.focus();
    }
  }

  async function submitPrompt() {
    if (busy || tuning) return;
    const text = promptEl.value.trim();
    const chat = activeChat();
    if (!text || !chat) return;
    if (editingIndex !== null) {
      chat.messages = chat.messages.slice(0, editingIndex);
      cancelEdit();
    }
    chat.messages.push({ role: "user", content: text, reasoning: "", tool_calls: null, usage: null, finishReason: "" });
    if (!chat.titleManual && chat.messages.filter((message) => message.role === "user").length === 1) {
      chat.title = titleFor(text);
      chatTitleEl.textContent = chat.title;
      chatTitleEl.title = chat.title;
    }
    chat.draft = "";
    touch(chat);
    promptEl.value = "";
    resizePrompt();
    renderConversation(true);
    renderChatList();
    save();
    await generate();
  }

  function retryMessage(index) {
    const chat = activeChat();
    if (!chat || busy || tuning || !chat.messages[index] || chat.messages[index].role !== "assistant") return;
    if (!chat.messages.slice(0, index).some((message) => message.role === "user")) return;
    chat.messages = chat.messages.slice(0, index);
    touch(chat);
    renderConversation(true);
    renderChatList();
    save();
    generate();
  }

  function updateSettings() {
    const chat = activeChat();
    if (!chat) return;
    chat.settings = cleanSettings({
      maxTokens: maxTokensEl.value, temperature: temperatureEl.value, topP: topPEl.value,
      topK: topKEl.value, minP: minPEl.value, repeatPenalty: repeatPenaltyEl.value, seed: seedEl.value.trim(),
      stopSequences: stopSequencesEl.value
    }, defaults);
    chat.systemPrompt = systemPromptEl.value.slice(0, 100000);
    chat.persona = Object.prototype.hasOwnProperty.call(PERSONAS, personaEl.value) ? personaEl.value : "custom";
    tempValueEl.textContent = Number(chat.settings.temperature).toFixed(2);
    touch(chat);
    updateComposer(false);
    saveSoon();
  }

  function setModelName(name) {
    if (!name) return;
    modelNameEl.textContent = name;
    modelNameEl.title = name;
    const chat = activeChat();
    if (chat) {
      chat.model = name;
      saveSoon();
    }
  }

  async function loadModels() {
    try {
      const response = await fetch("/models");
      if (!response.ok) return;
      const data = await response.json();
      if (!data.models || !data.models.length) return;
      modelSelectEl.replaceChildren();
      data.models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        const context = model.context_length ? " · " + (model.context_length >= 1000 ? Math.round(model.context_length / 1000) + "K" : model.context_length) + " ctx" : "";
        option.textContent = (model.name || model.id) + (model.architecture ? " [" + model.architecture + "]" : "") + context + (model.size_gb ? " — " + model.size_gb.toFixed(1) + " GB" : "") + (!model.supported ? " (unsupported)" : "");
        if (model.loaded) {
          option.selected = true;
          option.dataset.loaded = "true";
          contextLimit = Number(model.context_length) || 0;
          setModelName(model.name || model.id);
        }
        if (!model.supported) option.style.color = "var(--muted)";
        modelSelectEl.appendChild(option);
      });
      updateComposer(false);
    } catch (_) {
      setStatus("Offline");
    }
  }

  async function loadSkills() {
    try {
      const response = await fetch("/v1/skills");
      if (!response.ok) return;
      const data = await response.json();
      const list = Array.isArray(data.skills) ? data.skills.filter((skill) => skill && skill.name) : [];
      if (!list.length) return;
      skillsNoteEl.textContent = "Skills available to the model: " + list.map((skill) => skill.name).join(", ");
      skillsNoteEl.title = list.map((skill) => skill.name + (skill.description ? " — " + skill.description : "")).join("\n");
      skillsNoteEl.hidden = false;
    } catch (_) {
      /* Skills are an optional server feature; leave the note hidden. */
    }
  }

  function formatAutoTuneStatus(data) {
    if (!data || !data.result) return "Not tuned for this model and machine yet.";
    const r = data.result;
    const bits = ["threads=" + r.threads, "q8=" + (r.q8_activations ? "on" : "off"), "kv-f16=" + (r.kv_cache_f16 ? "on" : "off"), "prefill-chunk=" + r.prefill_chunk];
    const gains = [];
    if (r.baseline_decode_ms > 0 && r.tuned_decode_ms > 0) gains.push((r.baseline_decode_ms / r.tuned_decode_ms).toFixed(2) + "x decode");
    if (r.baseline_prefill_tokens_per_second > 0 && r.tuned_prefill_tokens_per_second > 0) {
      gains.push((r.tuned_prefill_tokens_per_second / r.baseline_prefill_tokens_per_second).toFixed(2) + "x prefill");
    }
    let text = (data.active ? "Active: " : "Measured previously, not applied this session: ") + bits.join(" ");
    if (gains.length) text += " · " + gains.join(", ") + " faster";
    return text;
  }

  async function loadAutoTuneStatus() {
    try {
      const response = await fetch("/autotune");
      if (!response.ok) throw new Error("HTTP " + response.status);
      autoTuneStatusEl.textContent = formatAutoTuneStatus(await response.json());
    } catch (_) {
      autoTuneStatusEl.textContent = "Could not check tuning status.";
    }
  }

  modelSelectEl.addEventListener("change", async () => {
    const model = modelSelectEl.value;
    if (!model || busy || tuning || loadingModel) return;
    loadingModel = true;
    const previous = Array.from(modelSelectEl.options).find((option) => option.dataset.loaded === "true");
    modelSelectEl.disabled = true;
    autoTuneRunEl.disabled = true;
    autoTuneEffortEl.disabled = true;
    statusEl.classList.add("busy");
    setStatus("Loading model…");
    try {
      const response = await fetch("/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model })
      });
      if (!response.ok) throw new Error((await response.text()) || "HTTP " + response.status);
      const data = await response.json();
      Array.from(modelSelectEl.options).forEach((option) => { delete option.dataset.loaded; });
      const loaded = Array.from(modelSelectEl.options).find((option) => option.value === model);
      if (loaded) loaded.dataset.loaded = "true";
      contextLimit = Number(data.context_length) || 0;
      setModelName(data.model || model);
      updateComposer(false);
      renderChatList();
      save();
      setStatus("Ready");
      showToast("Model loaded. Your chats were kept.", "success");
      loadAutoTuneStatus();
    } catch (error) {
      if (previous) modelSelectEl.value = previous.value;
      setStatus("Error loading model");
      addMessage("error", "Failed to load model: " + (error.message || error));
    } finally {
      loadingModel = false;
      statusEl.classList.remove("busy");
      modelSelectEl.disabled = false;
      autoTuneRunEl.disabled = false;
      autoTuneEffortEl.disabled = false;
    }
  });

  autoTuneRunEl.addEventListener("click", async () => {
    if (busy || tuning || loadingModel) return;
    tuning = true;
    autoTuneRunEl.disabled = true;
    autoTuneEffortEl.disabled = true;
    const modelSelectWasDisabled = modelSelectEl.disabled;
    modelSelectEl.disabled = true;
    updateComposer(false);
    statusEl.classList.add("busy");
    const effort = autoTuneEffortEl.value;
    const eta = effort === "quick" ? "quick, ~10-20s" : effort === "thorough" ? "thorough, several minutes" : "balanced, ~1-2 min";
    setStatus("Tuning (" + eta + ")…");
    autoTuneStatusEl.textContent = "Tuning now (" + eta + ") — generation is paused until this finishes.";
    try {
      const response = await fetch("/autotune/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ effort })
      });
      if (!response.ok) throw new Error((await response.text()) || "HTTP " + response.status);
      const data = await response.json();
      autoTuneStatusEl.textContent = formatAutoTuneStatus({ active: true, result: data.result });
      showToast(data.cached ? "Applied a previously measured tuning." : "Auto-tuning complete.", "success");
    } catch (error) {
      autoTuneStatusEl.textContent = "Auto-tuning failed: " + (error.message || error);
      showToast("Auto-tuning failed: " + (error.message || error), "error");
    } finally {
      tuning = false;
      autoTuneRunEl.disabled = false;
      autoTuneEffortEl.disabled = false;
      modelSelectEl.disabled = modelSelectWasDisabled;
      statusEl.classList.remove("busy");
      setStatus("Ready");
      updateComposer(false);
    }
  });

  scrollEl.addEventListener("scroll", () => {
    followStream = isNearBottom();
    jumpLatestEl.hidden = followStream || !messagesEl.querySelector(".msg");
  }, { passive: true });
  jumpLatestEl.addEventListener("click", () => {
    followStream = true;
    scrollToBottom(true);
  });
  promptEl.addEventListener("input", () => {
    resizePrompt();
    updateComposer(true);
  });
  promptEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey || !event.shiftKey) && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
    if (event.key === "Escape" && editingIndex !== null) {
      event.preventDefault();
      cancelEdit();
    }
  });
  sendEl.addEventListener("click", (event) => {
    if (busy) {
      event.preventDefault();
      if (controller) controller.abort();
    }
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrompt();
  });

  newChatEl.addEventListener("click", createChat);
  clearEl.addEventListener("click", createChat);
  renameChatEl.addEventListener("click", () => renameChat(activeID));
  cancelEditEl.addEventListener("click", cancelEdit);
  chatSearchEl.addEventListener("input", renderChatList);
  sidebarToggleEl.addEventListener("click", () => setSidebar(!sidebarEl.classList.contains("is-open")));
  sidebarScrimEl.addEventListener("click", () => setSidebar(false));
  function openSettings() {
    settingsEl.hidden = false;
    settingsToggleEl.setAttribute("aria-expanded", "true");
    settingsCloseEl.focus();
  }
  function closeSettings() {
    if (settingsEl.hidden) return;
    settingsEl.hidden = true;
    settingsToggleEl.setAttribute("aria-expanded", "false");
    settingsToggleEl.focus();
  }
  settingsToggleEl.addEventListener("click", () => {
    if (settingsEl.hidden) openSettings();
    else closeSettings();
  });
  settingsCloseEl.addEventListener("click", closeSettings);
  settingsEl.addEventListener("click", (event) => {
    if (event.target === settingsEl) closeSettings();
  });
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      promptEl.value = button.dataset.prompt || "";
      resizePrompt();
      updateComposer(true);
      promptEl.focus();
    });
  });
  [maxTokensEl, temperatureEl, topPEl, topKEl, minPEl, repeatPenaltyEl, seedEl, stopSequencesEl].forEach((control) => {
    control.addEventListener("input", updateSettings);
    control.addEventListener("change", updateSettings);
  });
  personaEl.addEventListener("change", () => {
    if (Object.prototype.hasOwnProperty.call(PERSONAS, personaEl.value)) systemPromptEl.value = PERSONAS[personaEl.value];
    updateSettings();
  });
  systemPromptEl.addEventListener("input", () => {
    if (!Object.keys(PERSONAS).some((key) => PERSONAS[key] === systemPromptEl.value)) personaEl.value = "custom";
    updateSettings();
  });
  systemPromptEl.addEventListener("change", updateSettings);

  attachTextEl.addEventListener("click", () => textFileInputEl.click());
  textFileInputEl.addEventListener("change", async () => {
    const file = textFileInputEl.files && textFileInputEl.files[0];
    textFileInputEl.value = "";
    if (!file || busy) return;
    if (file.size > 500000) {
      showToast("Text files are limited to 500 KB.", "error");
      return;
    }
    try {
      const text = await file.text();
      const prefix = promptEl.value.trim() ? promptEl.value.replace(/\s*$/, "") + "\n\n" : "";
      promptEl.value = prefix + "Please use the following text from " + file.name + ":\n\n" + FENCE + "text\n" + text + "\n" + FENCE;
      resizePrompt();
      updateComposer(true);
      promptEl.focus();
      showToast("Added " + file.name + " as local text.", "success");
    } catch (_) {
      showToast("Could not read that text file.", "error");
    }
  });

  exportChatsEl.addEventListener("click", () => {
    download("gopherllm-chats-" + new Date().toISOString().slice(0, 10) + ".json", "application/json", JSON.stringify(Object.assign(workspace(), { exportedAt: new Date().toISOString() }), null, 2));
    showToast("Saved chat archive", "success");
  });
  exportMarkdownEl.addEventListener("click", () => {
    const chat = activeChat();
    if (!chat) return;
    const name = chat.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "chat";
    download(name + ".md", "text/markdown;charset=utf-8", toMarkdown(chat));
    showToast("Saved this chat as Markdown", "success");
  });
  importChatsEl.addEventListener("click", () => importInputEl.click());
  importInputEl.addEventListener("change", async () => {
    const file = importInputEl.files && importInputEl.files[0];
    importInputEl.value = "";
    if (!file || busy) return;
    try {
      const parsed = JSON.parse(await file.text());
      const raw = Array.isArray(parsed) ? parsed : parsed && parsed.conversations;
      if (!Array.isArray(raw)) throw new Error("invalid");
      const known = new Set(chats.map((chat) => chat.id));
      const incoming = raw.map((value) => cleanChat(value, defaults)).filter(Boolean);
      incoming.forEach((chat) => {
        while (known.has(chat.id)) chat.id = makeID();
        known.add(chat.id);
      });
      if (!incoming.length) throw new Error("empty");
      chats = incoming.concat(chats).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CHATS);
      activeID = incoming[0].id;
      editingIndex = null;
      editStateEl.hidden = true;
      renderWorkspace(true);
      save();
      showToast(incoming.length + " chat" + (incoming.length === 1 ? "" : "s") + " imported", "success");
    } catch (_) {
      showToast("That file is not a valid GopherLLM chat archive.", "error");
    }
  });
  themeSelectEl.addEventListener("change", () => {
    applyTheme(themeSelectEl.value);
    save();
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      setSidebar(true);
      chatSearchEl.focus();
    }
    if (event.key === "Escape" && sidebarEl.classList.contains("is-open")) setSidebar(false);
    if (event.key === "Escape" && !settingsEl.hidden) closeSettings();
  });
  window.addEventListener("beforeunload", save);

  const stored = await readWorkspace();
  if (stored && typeof stored === "object") {
    const seen = new Set();
    chats = (Array.isArray(stored.conversations) ? stored.conversations : []).map((value) => cleanChat(value, defaults)).filter((chat) => {
      if (!chat || seen.has(chat.id)) return false;
      seen.add(chat.id);
      return true;
    }).slice(0, MAX_CHATS);
    if (stored.preferences && typeof stored.preferences === "object") preferences = { theme: stored.preferences.theme };
    if (typeof stored.activeID === "string" && chats.some((chat) => chat.id === stored.activeID)) activeID = stored.activeID;
  }
  if (!chats.length) {
    const chat = newChat(defaults);
    chat.model = modelNameEl.textContent || "";
    chats = [chat];
  }
  if (!activeID) activeID = chats.slice().sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
  applyTheme(preferences.theme);
  renderWorkspace(true);
  loadModels();
  loadSkills();
  loadAutoTuneStatus();
}());
