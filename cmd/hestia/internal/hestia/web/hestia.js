'use strict';

const form = document.getElementById('command');
const textField = document.getElementById('text');
const send = document.getElementById('send');
const mic = document.getElementById('mic');
const listen = document.getElementById('listen');
const micStatus = document.getElementById('micStatus');
const listenHelp = document.getElementById('listenHelp');
const searchHelp = document.getElementById('searchHelp');
const speak = document.getElementById('speak');
const errorPanel = document.getElementById('error');
const turns = document.getElementById('turns');
const AudioContext = window.AudioContext || window.webkitAudioContext;
const canRecord = Boolean(AudioContext && window.AudioWorkletNode && navigator.mediaDevices?.getUserMedia);
const synth = window.speechSynthesis;

let capabilities = {};
let conversation;
let capture;
let playback;

function setError(message = '') {
  errorPanel.textContent = message;
  errorPanel.hidden = !message;
}

async function requestJSON(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body.error || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function ensureConversation() {
  conversation ??= requestJSON('/api/conversations', {method: 'POST'})
    .then(body => body.conversation_id)
    .catch(error => {
      conversation = null;
      throw error;
    });
  return conversation;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function sourceURL(raw) {
  try {
    const url = new URL(raw);
    if ((url.protocol === 'https:' || url.protocol === 'http:') && !url.username && !url.password) {
      return url.href;
    }
  } catch {
    // Invalid URLs are omitted from the result.
  }
  return '';
}

function renderTurn(turn) {
  const item = element('article', '', 'turn');
  if (turn.InputText) item.append(element('p', `„${turn.InputText}“`));
  if (turn.Result) item.append(element('p', turn.Result));
  if (Array.isArray(turn.sources)) {
    const sources = element('ul');
    for (const source of turn.sources) {
      const href = sourceURL(source.url);
      if (!href) continue;
      const li = element('li');
      const link = element('a', source.title || new URL(href).hostname);
      link.href = href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      li.append(link);
      if (source.snippet) li.append(element('p', source.snippet));
      sources.append(li);
    }
    if (sources.childElementCount) item.append(sources);
  }
  if (turn.Status === 'clarifying') {
    const choices = element('div', '', 'actions');
    for (const device of turn.Candidates || []) {
      const choice = element('button', `${device.Name} (${device.Room})`);
      choice.addEventListener('click', () => void confirm(turn.ID, device.ID, choices));
      choices.append(choice);
    }
    if (turn.Proposal?.Scope === 'group') {
      const choice = element('button', 'Bestätigen');
      choice.addEventListener('click', () => void confirm(turn.ID, '', choices));
      choices.append(choice);
    }
    item.append(choices);
  }
  turns.prepend(item);
}

async function confirm(turnID, deviceID, choices) {
  for (const button of choices.children) button.disabled = true;
  setError();
  try {
    const options = {method: 'POST'};
    if (deviceID) {
      options.headers = {'Content-Type': 'application/json'};
      options.body = JSON.stringify({device_id: deviceID});
    }
    showTurn(await requestJSON(`/api/turns/${encodeURIComponent(turnID)}/confirm`, options));
    choices.remove();
  } catch (error) {
    setError(error.message);
    for (const button of choices.children) button.disabled = false;
  }
}

function showTurn(turn) {
  renderTurn(turn);
  readAloud(turn.Result);
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const text = textField.value.trim();
  if (!text || send.disabled) return;
  send.disabled = true;
  setError();
  try {
    const conv = await ensureConversation();
    showTurn(await requestJSON(`/api/conversations/${encodeURIComponent(conv)}/turns`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}),
    }));
    if (textField.value.trim() === text) textField.value = '';
  } catch (error) {
    setError(error.message);
  } finally {
    send.disabled = false;
  }
});

function localVoice() {
  return synth?.getVoices().find(voice => voice.localService && /^de(?:[-_]|$)/i.test(voice.lang));
}

function updateVoiceOption() {
  const option = document.getElementById('speechLabel');
  option.hidden = !localVoice();
}

function readAloud(text) {
  if (!text || !speak.checked || capture?.mode === 'manual' || document.hidden) return;
  const voice = localVoice();
  if (!voice) return;
  stopPlayback();
  const utterance = new SpeechSynthesisUtterance(text.slice(0, 1600));
  utterance.voice = voice;
  utterance.lang = voice.lang;
  playback = utterance;
  if (capture?.mode === 'listen') {
    clearUtterance(capture);
    setTracksEnabled(capture, false);
    micStatus.textContent = 'Mikrofon pausiert während der Ausgabe.';
  }
  utterance.onend = utterance.onerror = () => finishPlayback(utterance);
  try {
    synth.speak(utterance);
  } catch {
    finishPlayback(utterance);
  }
}

function finishPlayback(utterance) {
  if (playback !== utterance) return;
  playback = null;
  if (capture?.mode === 'listen') {
    capture.cooldown = 2; // Discard audio from the speaker's tail.
    setTracksEnabled(capture, true);
    micStatus.textContent = capture.waiting ? 'Prüfe Aktivierungswort…' : listeningStatus();
  }
}

function stopPlayback() {
  if (!playback) return;
  const utterance = playback;
  synth.cancel();
  finishPlayback(utterance);
}

const silenceFrames = 3; // 200 ms per frame from the shared worklet.
const maxListenFrames = 40;
const maxManualFrames = 150;
const pcmHeaders = {'Content-Type': 'application/octet-stream'};

function listeningStatus() {
  return `Höre auf „${capabilities.listen_word}“.`;
}

function updateCaptureButtons() {
  const available = capabilities.speech_enabled && canRecord;
  mic.disabled = !available || capture?.mode === 'listen' || Boolean(capture?.closing);
  listen.disabled = !available || !capabilities.listen_word || capture?.mode === 'manual';
  mic.textContent = capture?.mode === 'manual' ? (capture.node ? 'Aufnahme abschließen' : 'Abbrechen') : 'Aufnehmen';
  listen.textContent = capture?.mode === 'listen' ? 'Zuhören stoppen' : 'Zuhören aktivieren';
  listen.setAttribute('aria-pressed', String(capture?.mode === 'listen'));
}

function clearUtterance(job) {
  job.frames = [];
  job.previous = null;
  job.silence = 0;
  job.overflowing = false;
}

function setTracksEnabled(job, enabled) {
  for (const track of job.media?.getAudioTracks() || []) track.enabled = enabled;
}

function releaseMic(job) {
  if (job.node) {
    job.node.port.onmessage = null;
    job.node.disconnect();
    job.node.port.close();
  }
  job.source?.disconnect();
  for (const track of job.media?.getTracks() || []) track.stop();
  if (job.context && job.context.state !== 'closed') void job.context.close().catch(() => {});
}

function cancelTurn(turnID) {
  if (turnID) {
    void fetch(`/api/turns/${encodeURIComponent(turnID)}/cancel`, {method: 'POST', keepalive: true}).catch(() => {});
  }
}

function stopCapture(job, status) {
  if (capture !== job) return;
  capture = null;
  job.aborter.abort();
  releaseMic(job);
  clearUtterance(job);
  job.queue.length = 0;
  cancelTurn(job.turnID);
  updateCaptureButtons();
  micStatus.textContent = status;
}

async function startCapture(mode) {
  if (capture || !canRecord) return;
  const job = {mode, aborter: new AbortController(), queue: [], count: 0, cooldown: 0};
  clearUtterance(job);
  capture = job;
  updateCaptureButtons();
  micStatus.textContent = 'Mikrofon wird angefragt…';
  try {
    job.context = new AudioContext();
    await job.context.resume();
    if (capture !== job) return;
    job.media = await navigator.mediaDevices.getUserMedia({
      audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true},
    });
    if (capture !== job) {
      releaseMic(job);
      return;
    }
    await job.context.audioWorklet.addModule('/audio-worklet.js');
    const conv = await ensureConversation();
    if (capture !== job) return;
    job.turnsPath = `/api/conversations/${encodeURIComponent(conv)}/turns`;
    if (mode === 'manual') {
      const turn = await requestJSON(`${job.turnsPath}/voice`, {method: 'POST'});
      if (capture !== job) {
        cancelTurn(turn.ID);
        return;
      }
      job.turnID = turn.ID;
    }
    job.source = job.context.createMediaStreamSource(job.media);
    job.node = new AudioWorkletNode(job.context, 'voxtral-pcm');
    job.node.port.onmessage = ({data}) => {
      if (capture !== job) return;
      if (data === 'stopped') {
        void finishManual(job);
      } else if (data instanceof ArrayBuffer) {
        if (mode === 'listen') receiveListenFrame(job, data);
        else receiveManualFrame(job, data);
      }
    };
    for (const track of job.media.getAudioTracks()) {
      track.addEventListener('ended', () => stopCapture(job, 'Mikrofonverbindung beendet.'));
    }
    if (mode === 'listen' && playback) setTracksEnabled(job, false);
    job.source.connect(job.node);
    job.node.connect(job.context.destination);
    updateCaptureButtons();
    micStatus.textContent = mode === 'manual' ? 'Nehme auf…'
      : playback ? 'Mikrofon pausiert während der Ausgabe.' : listeningStatus();
  } catch (error) {
    const reason = error.status === 409 ? 'Es läuft bereits eine Aufnahme.'
      : error.name === 'NotAllowedError' ? 'Mikrofonzugriff verweigert.' : error.message;
    stopCapture(job, reason);
  }
}

function speechLike(pcm) {
  const view = new DataView(pcm);
  let sum = 0;
  let count = 0;
  for (let i = 0; i + 1 < view.byteLength; i += 8) {
    const sample = view.getInt16(i, true);
    sum += sample * sample;
    count++;
  }
  // RMS amplitude is only a local prefilter. The recognition model checks the word.
  return count > 0 && sum / count >= 400 * 400;
}

function receiveListenFrame(job, frame) {
  if (job.waiting || playback) return;
  if (job.cooldown > 0) {
    job.cooldown--;
    return;
  }
  const voiced = speechLike(frame);
  if (job.overflowing) {
    job.silence = voiced ? 0 : job.silence + 1;
    if (job.silence >= silenceFrames) {
      clearUtterance(job);
      micStatus.textContent = listeningStatus();
    }
    return;
  }
  if (!job.frames.length) {
    if (!voiced) {
      job.previous = frame;
      return;
    }
    if (job.previous) job.frames.push(job.previous);
    job.previous = null;
  }
  job.frames.push(frame);
  job.silence = voiced ? 0 : job.silence + 1;
  if (job.silence >= silenceFrames) {
    const pcm = new Uint8Array(job.frames.reduce((bytes, frame) => bytes + frame.byteLength, 0));
    let offset = 0;
    for (const frame of job.frames) {
      pcm.set(new Uint8Array(frame), offset);
      offset += frame.byteLength;
    }
    clearUtterance(job);
    void checkListenWord(job, pcm);
  } else if (job.frames.length >= maxListenFrames) {
    clearUtterance(job);
    job.overflowing = true;
    micStatus.textContent = 'Zu lange Aufnahme. Bitte kurz pausieren und den Auftrag erneut sagen.';
  }
}

async function checkListenWord(job, pcm) {
  job.waiting = true;
  micStatus.textContent = 'Prüfe Aktivierungswort…';
  try {
    const outcome = await requestJSON(`${job.turnsPath}/listen`, {
      method: 'POST', headers: pcmHeaders, body: pcm, signal: job.aborter.signal,
    });
    if (capture !== job) return;
    if (outcome.matched && outcome.turn) showTurn(outcome.turn);
    else if (outcome.matched) micStatus.textContent = 'Bitte Aktivierungswort und Auftrag zusammen sagen.';
  } catch (error) {
    if (capture !== job) return;
    if (error.status === 409) micStatus.textContent = 'Belegt. Bitte gleich erneut sprechen.';
    else stopCapture(job, error.message);
  } finally {
    job.waiting = false;
    if (capture === job && micStatus.textContent === 'Prüfe Aktivierungswort…') {
      micStatus.textContent = listeningStatus();
    }
  }
}

function receiveManualFrame(job, frame) {
  if (job.count >= maxManualFrames) return;
  job.queue.push(frame);
  if (++job.count >= maxManualFrames) stopManual(job);
  if (!job.draining) job.draining = drainManual(job);
}

async function drainManual(job) {
  try {
    while (capture === job && job.queue.length) {
      const turn = await requestJSON(`/api/turns/${encodeURIComponent(job.turnID)}/audio`, {
        method: 'POST', headers: pcmHeaders, body: job.queue.shift(), signal: job.aborter.signal,
      });
      if (capture === job && !job.closing && turn.InputText) {
        micStatus.textContent = `Nehme auf… „${turn.InputText}“`;
      }
    }
  } catch (error) {
    stopCapture(job, error.message);
  } finally {
    job.draining = null;
  }
}

function stopManual(job) {
  if (job.closing) return;
  job.closing = true;
  updateCaptureButtons();
  micStatus.textContent = 'Verarbeite…';
  job.node.port.postMessage('stop');
}

async function finishManual(job) {
  releaseMic(job);
  await job.draining;
  if (capture !== job) return;
  try {
    const turn = await requestJSON(`/api/turns/${encodeURIComponent(job.turnID)}/finish`, {
      method: 'POST', signal: job.aborter.signal,
    });
    if (capture !== job) return;
    capture = null;
    updateCaptureButtons();
    micStatus.textContent = 'Mikrofon aus.';
    showTurn(turn);
  } catch (error) {
    stopCapture(job, error.message);
  }
}

mic.addEventListener('click', () => {
  if (!capture) {
    stopPlayback();
    void startCapture('manual');
  } else if (capture.node) stopManual(capture);
  else stopCapture(capture, 'Mikrofon aus.');
});
listen.addEventListener('click', () => {
  if (capture) stopCapture(capture, 'Mikrofon aus.');
  else void startCapture('listen');
});
function pausePage() {
  if (capture) stopCapture(capture, 'Mikrofon aus. Aktiviere es bei Bedarf erneut.');
  stopPlayback();
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) pausePage();
});
window.addEventListener('pagehide', pausePage);
synth?.addEventListener('voiceschanged', updateVoiceOption);
speak.addEventListener('change', () => {
  if (!speak.checked) stopPlayback();
});
updateVoiceOption();

requestJSON('/api/capabilities').then(result => {
  capabilities = result;
  updateCaptureButtons();
  if (!result.speech_enabled) micStatus.textContent = 'Kein Sprachmodell geladen.';
  else if (!canRecord) micStatus.textContent = 'Dieser Browser unterstützt hier keine Mikrofonaufnahme.';
  else micStatus.textContent = 'Mikrofon aus.';
  if (result.listen_word && result.speech_enabled && canRecord) {
    listenHelp.textContent = `Sage „${result.listen_word}, …“ und deinen Auftrag in einem Satz. Das Mikrofon bleibt bei sichtbarer Seite an. Kandidaten werden auf dem Hestia-Server geprüft.`;
    listenHelp.hidden = false;
  }
  const hints = [];
  if (result.wiki_enabled) hints.push('„Wikipedia: Suchbegriff“');
  if (result.web_enabled) hints.push('„Websuche: Suchbegriff“');
  if (hints.length) {
    searchHelp.textContent = `Online suchen mit ${hints.join(' oder ')}. Nur der Suchbegriff geht an den eingeschalteten Suchdienst.`;
    searchHelp.hidden = false;
  }
}).catch(error => {
  micStatus.textContent = 'Sprachfunktionen konnten nicht geladen werden.';
  setError(error.message);
});
