"use strict";

const get = (id) => document.getElementById(id);
const record = get("record"), file = get("file"), input = get("text");
const handsfree = get("handsfree"), speech = get("speak");
const conversation = get("conversation");
const Audio = window.AudioContext || window.webkitAudioContext;
const OfflineAudio = window.OfflineAudioContext || window.webkitOfflineAudioContext;
let state = "idle", epoch = 0, history = [], capture, pending, playing, speechDone;
let nativeVoice = false;
fetch("/options").then((r) => r.json()).then((v) => { nativeVoice = v.tts; }).catch(() => {});

function status(text) { get("status").textContent = text; }
function spokenText(text) { return text.replace(/\*\*|__|`/g, "").trim(); }
function update(mode) {
  state = mode;
  record.textContent = mode === "recording" ? "Stop & send" : "Start microphone";
  record.disabled = !Audio || !OfflineAudio || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder ||
    (mode !== "idle" && mode !== "recording");
  record.setAttribute("aria-pressed", String(mode === "recording"));
  file.disabled = input.disabled = get("send").disabled = mode !== "idle";
}
update("idle");

function addTurn(role, text) {
  history.push({ role, text });
  const tile = document.createElement("article");
  tile.className = role;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "Gopher";
  const paragraph = document.createElement("p");
  paragraph.textContent = role === "assistant" ? spokenText(text) : text;
  tile.append(label, paragraph);
  conversation.append(tile);
  conversation.scrollTop = conversation.scrollHeight;
}

async function prepareWAV(blob) {
  if (!blob.size || blob.size > 12 * 1024 * 1024) throw new Error("Choose a nonempty audio file under 12 MB.");
  const context = new Audio();
  let audio;
  try { audio = await context.decodeAudioData(await blob.arrayBuffer()); }
  finally { await context.close(); }
  const count = Math.floor(audio.duration * 16000);
  if (!count || count > 30 * 16000) throw new Error("Choose an audio clip of up to 30 seconds.");
  const offline = new OfflineAudio(1, count, 16000);
  const mono = offline.createBuffer(1, audio.length, audio.sampleRate);
  const samples = mono.getChannelData(0);
  for (let channel = 0; channel < audio.numberOfChannels; channel++) {
    const source = audio.getChannelData(channel);
    for (let i = 0; i < source.length; i++) samples[i] += source[i] / audio.numberOfChannels;
  }
  const player = offline.createBufferSource();
  player.buffer = mono;
  player.connect(offline.destination);
  player.start();
  const output = (await offline.startRendering()).getChannelData(0);
  const buffer = new ArrayBuffer(44 + 2 * count);
  const view = new DataView(buffer);
  const label = (at, text) => [...text].forEach((ch, i) => view.setUint8(at + i, ch.charCodeAt(0)));
  label(0, "RIFF"); view.setUint32(4, buffer.byteLength - 8, true); label(8, "WAVE");
  label(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  label(36, "data"); view.setUint32(40, 2 * count, true);
  for (let i = 0; i < count; i++) {
    const value = Math.max(-1, Math.min(1, output[i]));
    view.setInt16(44 + i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function endCapture(job) {
  if (!job || job.closed) return;
  job.closed = true;
  clearInterval(job.timer);
  job.stream.getTracks().forEach((track) => track.stop());
  job.context.close().catch(() => {});
  if (capture === job) capture = undefined;
}

async function listen() {
  if (state !== "idle") return;
  const id = epoch;
  update("waiting");
  status("Opening microphone…");
  let stream, context;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true }, video: false });
    if (id !== epoch) { stream.getTracks().forEach((track) => track.stop()); return; }
    context = new Audio();
    await context.resume();
    if (id !== epoch) {
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
      return;
    }
    const analyzer = context.createAnalyser();
    const source = context.createMediaStreamSource(stream);
    source.connect(analyzer);
    const meter = new Float32Array(analyzer.fftSize);
    const media = new MediaRecorder(stream);
    const chunks = [];
    const job = { media, context, stream, source };
    capture = job;
    media.ondataavailable = (e) => chunks.push(e.data);
    media.onstop = () => {
      endCapture(job);
      if (id === epoch) void submit({ blob: new Blob(chunks, { type: media.mimeType }), fromMic: true });
    };
    media.onerror = () => {
      endCapture(job);
      if (id === epoch) { update("idle"); status("Microphone recording failed."); }
    };
    media.start(250);
    update("recording");
    status("Listening…");
    const began = performance.now();
    let lastSpeech = 0, voiced = 0;
    job.timer = setInterval(() => {
      if (media.state !== "recording") return;
      const now = performance.now();
      analyzer.getFloatTimeDomainData(meter);
      let level = 0;
      for (const sample of meter) level += sample * sample;
      if (Math.sqrt(level / meter.length) > 0.016) { voiced++; lastSpeech = now; }
      if (now - began > 27000 || (voiced >= 3 && now - lastSpeech > 850)) {
        media.stop();
      } else if (voiced === 0 && now - began > 9000) {
        media.onstop = null;
        media.stop();
        endCapture(job);
        update("idle");
        status("No speech detected.");
      }
    }, 100);
  } catch (error) {
    stream?.getTracks().forEach((track) => track.stop());
    context?.close().catch(() => {});
    if (id === epoch) { update("idle"); status(error.message); }
  }
}

async function playReply(text, signal) {
  if (!speech.checked) return "";
  text = spokenText(text);
  if (!text) return "";
  if (nativeVoice) {
    const response = await fetch("/speak", { method: "POST", body: text, signal });
    if (!response.ok) throw new Error("Local speech synthesis failed.");
    const url = URL.createObjectURL(await response.blob());
    try {
      const player = new window.Audio(url);
      playing = player;
      await new Promise((resolve, reject) => {
        speechDone = resolve;
        player.onended = resolve;
        player.onerror = () => reject(new Error("Could not play the generated audio."));
        player.play().catch(reject);
      });
    } finally {
      playing = speechDone = undefined;
      URL.revokeObjectURL(url);
    }
    return "";
  }
  // Browser voices are filtered to local services. Prefer a voice for the
  // page's language, or the browser's own locale on the English demo page.
  const voices = window.speechSynthesis?.getVoices().filter((v) => v.localService) || [];
  const locale = navigator.language.split("-")[0];
  const voice = voices.find((v) => v.lang.split("-")[0] === locale) || voices[0];
  if (!voice) return " · No local browser voice available";
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  await new Promise((resolve, reject) => {
    speechDone = resolve;
    utterance.onend = resolve;
    utterance.onerror = () => reject(new Error("Could not play the browser voice."));
    window.speechSynthesis.speak(utterance);
  }).finally(() => { speechDone = undefined; });
  return "";
}

async function submit({ text, blob, fromMic }) {
  const id = epoch;
  const request = new AbortController();
  pending = request;
  update("working");
  status(blob ? "Preparing and transcribing audio…" : "Generating a reply…");
  let followUp = false;
  try {
    const form = new FormData();
    // Keep whole pairs and bound server/model input for long conversations.
    const recent = history.slice(-8);
    while (new TextEncoder().encode(JSON.stringify(recent)).length > 24000) recent.splice(0, 2);
    form.set("history", JSON.stringify(recent));
    if (blob) form.set("audio", await prepareWAV(blob), "utterance.wav");
    else form.set("text", text);
    const response = await fetch("/turn", { method: "POST", body: form, signal: request.signal });
    if (!response.ok) throw new Error((await response.text()).trim());
    const result = await response.json();
    if (id !== epoch) return;
    addTurn("user", result.input);
    addTurn("assistant", result.text);
    const timing = blob ? `Speech ${result.stt_ms} ms · Reply ${result.reply_ms} ms` : `Reply ${result.reply_ms} ms`;
    status(timing);
    const note = await playReply(result.text, request.signal);
    if (id !== epoch) return;
    status(timing + note);
    followUp = fromMic && handsfree.checked;
  } catch (error) {
    if (id === epoch) status(error.message);
  } finally {
    if (pending === request) pending = undefined;
    if (id === epoch) update("idle");
  }
  if (followUp && id === epoch) void listen();
}

record.onclick = () => {
  if (state === "recording" && capture?.media.state === "recording") capture.media.stop();
  else if (state === "idle") void listen();
};
file.onchange = () => {
  const blob = file.files[0];
  file.value = "";
  if (blob && state === "idle") void submit({ blob });
};
get("type").onsubmit = (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (text && state === "idle") { input.value = ""; void submit({ text }); }
};
get("reset").onclick = () => {
  epoch++;
  pending?.abort();
  if (capture) {
    capture.media.onstop = null;
    if (capture.media.state !== "inactive") capture.media.stop();
    endCapture(capture);
  }
  playing?.pause();
  speechDone?.();
  window.speechSynthesis?.cancel();
  history = [];
  conversation.replaceChildren();
  update("idle");
  status("Ready");
};
