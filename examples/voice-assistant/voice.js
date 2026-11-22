"use strict";

const get = (id) => document.getElementById(id);
const record = get("record"), file = get("file"), input = get("text");
const handsfree = get("handsfree"), speech = get("speak"), stop = get("stop");
const conversation = get("conversation");
const Audio = window.AudioContext || window.webkitAudioContext;
const OfflineAudio = window.OfflineAudioContext || window.webkitOfflineAudioContext;
const canRecord = Boolean(Audio && window.AudioWorkletNode && navigator.mediaDevices?.getUserMedia);
const pcmLimit = 30 * 16000 * 2;
const streamType = "application/x-ndjson";
let state = "idle", epoch = 0, history = [], capture, pending, stopPlayback;
const voiceOptions = fetch("/options").then((r) => r.json()).catch(() => ({}));

function status(text) { get("status").textContent = text; }
function spokenText(text) { return text.replace(/\*\*|__|`/g, "").trim(); }
function update(mode) {
  state = mode;
  record.textContent = mode === "recording" ? "Stop & send" : "Start microphone";
  record.disabled = !canRecord || (mode !== "idle" && mode !== "recording");
  record.setAttribute("aria-pressed", String(mode === "recording"));
  file.disabled = !OfflineAudio || mode !== "idle";
  input.disabled = get("send").disabled = mode !== "idle";
  stop.disabled = mode === "idle";
}
update("idle");
if (!canRecord) status("Microphone capture is unavailable here. Type a message or upload audio.");

function addTurn(role, text) {
  const tile = document.createElement("article");
  tile.className = role;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "Gopher";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  tile.append(label, paragraph);
  conversation.append(tile);
  conversation.scrollTop = conversation.scrollHeight;
  return paragraph;
}

function wavFromPCM(blocks, bytes) {
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const label = (at, text) => [...text].forEach((ch, i) => view.setUint8(at + i, ch.charCodeAt(0)));
  label(0, "RIFF"); view.setUint32(4, 36 + bytes, true); label(8, "WAVE");
  label(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  label(36, "data"); view.setUint32(40, bytes, true);
  return new Blob([header, ...blocks], { type: "audio/wav" });
}

async function prepareWAV(blob) {
  if (!blob.size || blob.size > 12 * 1024 * 1024) throw new Error("Choose a nonempty audio file under 12 MB.");
  // decodeAudioData resamples to its context's rate. No playback context or
  // second render pass is required for an uploaded file.
  const context = new OfflineAudio(1, 1, 16000);
  const audio = await context.decodeAudioData(await blob.arrayBuffer());
  const count = audio.length;
  if (!count || 2 * count > pcmLimit) throw new Error("Choose an audio clip of up to 30 seconds.");
  const data = new ArrayBuffer(2 * count);
  const view = new DataView(data);
  const channels = Array.from({ length: audio.numberOfChannels }, (_, i) => audio.getChannelData(i));
  for (let i = 0; i < count; i++) {
    let sample = 0;
    for (const channel of channels) sample += channel[i];
    const value = Math.max(-1, Math.min(1, sample / channels.length));
    view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return wavFromPCM([data], data.byteLength);
}

function endCapture(job) {
  for (const track of job.stream?.getTracks() || []) track.stop();
  if (job.node) {
    job.node.port.onmessage = null;
    job.node.disconnect();
    job.node.port.close();
    job.node = undefined;
  }
  job.source?.disconnect();
  job.source = undefined;
  if (job.context.state !== "closed") void job.context.close().catch(() => {});
  if (capture === job) capture = undefined;
}

function finishCapture(job) {
  if (job.closing) return;
  job.closing = true;
  update("working");
  status("Preparing microphone audio…");
  job.node.port.postMessage("stop");
}

function onPCM(job, data) {
  if (data === "stopped") {
    endCapture(job);
    if (job.bytes) void submit({ wav: wavFromPCM(job.blocks, job.bytes), fromMic: true });
    else { update("idle"); status("No audio was recorded."); }
    return;
  }
  if (!(data instanceof ArrayBuffer)) return;
  const frame = data.byteLength <= pcmLimit - job.bytes ? data : data.slice(0, pcmLimit - job.bytes);
  if (frame.byteLength) job.blocks.push(frame);
  job.bytes += frame.byteLength;
  if (job.closing) return;
  const view = new DataView(frame);
  let power = 0, count = 0;
  for (let i = 0; i < view.byteLength; i += 8) {
    const sample = view.getInt16(i, true);
    power += sample * sample;
    count++;
  }
  if (count && power / count > 524 * 524) {
    job.voiced++;
    job.quiet = 0;
  } else {
    job.quiet++;
  }
  if (job.bytes >= pcmLimit || (job.voiced >= 2 && job.quiet >= 4)) finishCapture(job);
  else if (!job.voiced && job.bytes >= 9 * 32000) {
    endCapture(job);
    update("idle");
    status("No speech detected.");
  }
}

async function listen() {
  if (!canRecord || state !== "idle" || document.hidden) return;
  update("waiting");
  status("Opening microphone…");
  let job;
  try {
    job = { context: new Audio(), blocks: [], bytes: 0, voiced: 0, quiet: 0 };
    capture = job;
    await job.context.resume();
    if (capture !== job) return;
    // Request permission before waiting for the worklet code. Treat the
    // permission result as stale if the user stopped or left the page.
    job.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    if (capture !== job) { endCapture(job); return; }
    await job.context.audioWorklet.addModule("/audio-worklet.js");
    if (capture !== job) return;
    job.source = job.context.createMediaStreamSource(job.stream);
    job.node = new AudioWorkletNode(job.context, "voxtral-pcm");
    job.node.port.onmessage = ({ data }) => {
      if (capture === job) onPCM(job, data);
    };
    for (const track of job.stream.getAudioTracks()) {
      track.addEventListener("ended", () => {
        if (capture === job) cancelActivity("Microphone disconnected.");
      });
    }
    job.source.connect(job.node);
    job.node.connect(job.context.destination);
    update("recording");
    status("Listening…");
  } catch (error) {
    if (capture === job) {
      if (job) endCapture(job);
      update("idle");
      status(error.message);
    }
  }
}

async function playReply(text, signal) {
  if (!speech.checked) return "";
  text = spokenText(text);
  if (!text) return "";
  const options = await voiceOptions;
  if (!speech.checked) return "";
  signal.throwIfAborted();
  if (options.tts) {
    const response = await fetch("/speak", { method: "POST", body: text, signal });
    if (!response.ok) throw new Error("Local speech synthesis failed.");
    const audio = await response.blob();
    signal.throwIfAborted();
    if (!speech.checked) return "";
    const url = URL.createObjectURL(audio);
    const player = new window.Audio(url);
    try {
      await waitForPlayback(signal, (done, fail) => {
        player.onended = done;
        player.onerror = () => fail(new Error("Could not play the generated audio."));
        player.play().catch(fail);
        return () => player.pause();
      });
    } finally {
      URL.revokeObjectURL(url);
    }
    return "";
  }
  const voices = window.speechSynthesis?.getVoices().filter((v) => v.localService) || [];
  const locale = navigator.language.split("-")[0];
  const voice = voices.find((v) => v.lang.split("-")[0] === locale) || voices[0];
  if (!voice) return " · No local browser voice available";
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  await waitForPlayback(signal, (done, fail) => {
    utterance.onend = done;
    utterance.onerror = () => fail(new Error("Could not play the browser voice."));
    window.speechSynthesis.speak(utterance);
    return () => window.speechSynthesis.cancel();
  });
  return "";
}

async function waitForPlayback(signal, start) {
  signal.throwIfAborted();
  let stopThis;
  try {
    await new Promise((resolve, reject) => {
      const pause = start(resolve, reject);
      stopThis = () => { resolve(); pause(); };
      stopPlayback = stopThis;
      signal.addEventListener("abort", stopThis, { once: true });
    });
  } finally {
    if (stopThis) signal.removeEventListener("abort", stopThis);
    if (stopPlayback === stopThis) stopPlayback = undefined;
  }
}

async function readTurn(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) throw new Error("The reply was interrupted.");
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const event = JSON.parse(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        if (event.type === "error") throw new Error(event.text);
        if (event.type === "done") return event.result;
        onEvent(event);
      }
    }
  } finally {
    void reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function submit({ text, blob, wav, fromMic }) {
  const id = epoch;
  const request = new AbortController();
  pending = request;
  update("working");
  const audio = Boolean(blob || wav);
  status(audio ? "Transcribing audio…" : "Generating a reply…");
  let followUp = false, user, reply, delta = "", frame = 0, committed = false;
  const render = () => {
    frame = 0;
    reply.textContent = spokenText(delta);
    conversation.scrollTop = conversation.scrollHeight;
  };
  try {
    const form = new FormData();
    const recent = history.slice(-8);
    while (new TextEncoder().encode(JSON.stringify(recent)).length > 24000) recent.splice(0, 2);
    form.set("history", JSON.stringify(recent));
    if (audio) form.set("audio", wav || await prepareWAV(blob), "utterance.wav");
    else form.set("text", text);
    const response = await fetch("/turn", { method: "POST", headers: { Accept: streamType }, body: form, signal: request.signal });
    if (!response.ok) throw new Error((await response.text()).trim());
    const result = await readTurn(response, (event) => {
      if (id !== epoch) return;
      if (event.type === "input") {
        user = addTurn("user", event.text);
        status("Generating a reply…");
      } else if (event.type === "delta") {
        if (!reply) {
          reply = addTurn("assistant", "");
          reply.parentElement.setAttribute("aria-busy", "true");
        }
        delta += event.text;
        if (!frame) frame = requestAnimationFrame(render);
      }
    });
    if (id !== epoch) return;
    cancelAnimationFrame(frame);
    user ??= addTurn("user", result.input);
    reply ??= addTurn("assistant", "");
    reply.textContent = spokenText(result.text);
    reply.parentElement.removeAttribute("aria-busy");
    history.push({ role: "user", text: result.input }, { role: "assistant", text: result.text });
    history = history.slice(-8);
    committed = true;
    if (text && input.value.trim() === text) input.value = "";
    const timing = audio ? `Speech ${result.stt_ms} ms · Reply ${result.reply_ms} ms` : `Reply ${result.reply_ms} ms`;
    status(timing);
    const note = await playReply(result.text, request.signal);
    if (id !== epoch) return;
    status(timing + note);
    followUp = fromMic && handsfree.checked;
  } catch (error) {
    if (id === epoch) status(error.message);
  } finally {
    cancelAnimationFrame(frame);
    if (!committed) {
      user?.parentElement.remove();
      reply?.parentElement.remove();
    }
    if (pending === request) pending = undefined;
    if (id === epoch) update("idle");
  }
  if (followUp && id === epoch && !document.hidden) void listen();
}

function cancelActivity(message) {
  epoch++;
  pending?.abort();
  if (capture) endCapture(capture);
  update("idle");
  status(message);
}

record.onclick = () => {
  if (state === "recording") finishCapture(capture);
  else void listen();
};
file.onchange = () => {
  const blob = file.files[0];
  file.value = "";
  if (blob && state === "idle") void submit({ blob });
};
get("type").onsubmit = (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (text && state === "idle") void submit({ text });
};
stop.onclick = () => cancelActivity("Stopped.");
speech.onchange = () => { if (!speech.checked) stopPlayback?.(); };
get("reset").onclick = () => {
  cancelActivity("Ready");
  history = [];
  input.value = "";
  conversation.replaceChildren();
};
document.addEventListener("visibilitychange", () => {
  if (document.hidden && state !== "idle") cancelActivity("Paused while the tab was hidden.");
});
window.addEventListener("pagehide", () => cancelActivity("Ready"));
