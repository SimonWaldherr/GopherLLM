/* Local Voxtral transcription controls. Audio is converted in the browser;
   only 16 kHz mono WAV is sent to this server, never to a speech provider. */
(function (global) {
  "use strict";
  const MAX_SECONDS = 30;
  const MAX_FILE_BYTES = 25 * 1024 * 1024;
  const AUDIO_DEVICE_KEY = "gopherllm.audio-input-device";
  const SYSTEM_AUDIO_VALUE = "__system_audio__";

  // System/tab audio has no dedicated capture API: getDisplayMedia() is the
  // Screen Capture API repurposed for it, requires a video track even though
  // only audio is wanted (the spec makes user agents reject audio-only
  // requests), and only Chromium (Chrome/Edge/Opera) actually delivers an
  // audio track from it -- Firefox has never implemented the audio option
  // and Safari silently ignores it, both resolving with zero audio tracks
  // and no error rather than throwing. echoCancellation/noiseSuppression/
  // autoGainControl default on and are tuned for a voice mic, not arbitrary
  // system audio (music, a video's mixed track), so they're turned off here.
  async function captureSystemAudio() {
    const GetDisplayMedia = navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia;
    if (!GetDisplayMedia) throw new Error("This browser cannot share system or tab audio.");
    const display = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: { systemAudio: "include", echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    // Never rendered; discard immediately so it's never attached to a sink
    // and stops counting toward the browser's active-share indicator early.
    display.getVideoTracks().forEach(t => t.stop());
    const audioTracks = display.getAudioTracks();
    if (!audioTracks.length) {
      display.getTracks().forEach(t => t.stop());
      throw new Error('No audio was shared. Enable "Share audio" / "Share tab audio" in the picker, and share a tab or screen that\'s playing sound. System audio capture only works in Chrome, Edge, or Opera today.');
    }
    return new MediaStream(audioTracks);
  }

  function encodeWAV(samples) {
    const bytes = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(bytes);
    const label = (offset, text) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); };
    label(0, "RIFF"); view.setUint32(4, bytes.byteLength - 8, true); label(8, "WAVE");
    label(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    label(36, "data"); view.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      if (!Number.isFinite(samples[i])) throw new Error("Audio contains invalid samples.");
      const sample = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
    }
    return new Blob([bytes], { type: "audio/wav" });
  }

  async function toWAV(file, recording) {
    if (!file.size) throw new Error("The audio file is empty.");
    if (file.size > MAX_FILE_BYTES) throw new Error("Choose an audio file smaller than 25 MiB.");
    const AudioContext = global.AudioContext || global.webkitAudioContext;
    const OfflineAudioContext = global.OfflineAudioContext || global.webkitOfflineAudioContext;
    if (!AudioContext || !OfflineAudioContext) throw new Error("This browser cannot decode audio. Try a recent browser.");
    const context = new AudioContext();
    let decoded;
    try {
      decoded = await context.decodeAudioData(await file.arrayBuffer());
    } catch (_) {
      throw new Error("This audio format cannot be decoded by your browser. Try WAV, MP3, or another supported format.");
    } finally {
      await context.close();
    }
    if (!recording && decoded.duration > MAX_SECONDS) throw new Error("Choose a clip of 30 seconds or less.");
    const count = Math.min(Math.floor(decoded.duration * 16000), MAX_SECONDS * 16000);
    if (!count) throw new Error("The audio has no samples.");
    const offline = new OfflineAudioContext(1, count, 16000);
    // Explicit arithmetic channel averaging matches the CLI's mono conversion.
    const mono = offline.createBuffer(1, decoded.length, decoded.sampleRate);
    const target = mono.getChannelData(0);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const channel = decoded.getChannelData(c);
      for (let i = 0; i < target.length; i++) target[i] += channel[i] / decoded.numberOfChannels;
    }
    const source = offline.createBufferSource();
    source.buffer = mono;
    source.connect(offline.destination);
    source.start();
    return encodeWAV((await offline.startRendering()).getChannelData(0));
  }

  function init(options) {
    const $ = id => document.getElementById(id);
    const toggle = $("audioToggle"), panel = $("audioPanel");
    if (!toggle || !panel) return null;
    const model = $("audioModel"), refresh = $("audioRefresh"), upload = $("audioUpload");
    const file = $("audioFile"), record = $("audioRecord"), live = $("audioLive"), cancel = $("audioCancel");
    const status = $("audioStatus"), result = $("audioResult"), insert = $("audioInsert");
    const levelRow = $("audioLevelRow"), levelFill = $("audioLevelFill");
    const inputDevice = $("audioInputDevice");
    let available = true, working = false, recorder = null, stream = null;
    let abort = null, timer = null, ticker = null, sequence = 0;
    let catalogSequence = 0;
    let liveSession = null;
    const message = (text, error = false) => {
      status.textContent = text;
      status.classList.toggle("is-error", error);
    };
    // A silent or near-silent capture (wrong input device, muted mic,
    // aggressive browser noise suppression) correctly transcribes to
    // nothing, which otherwise looks identical to a real bug: "Listening…"
    // shows, no error appears, but text never comes. Surfacing the
    // worklet's own peak-level reports makes that failure mode visible.
    const updateLevel = peak => {
      if (!levelFill) return;
      const pct = Math.max(0, Math.min(100, Math.round((peak || 0) * 140)));
      levelFill.style.width = pct + "%";
      levelFill.classList.toggle("is-quiet", pct < 4);
      if (levelRow) levelRow.querySelector(".audio-level")?.setAttribute("aria-valuenow", String(pct));
    };
    // Device labels are blank until the origin has been granted microphone
    // access at least once (browsers hide them from unauthenticated callers
    // as a fingerprinting/privacy guard), so the first listing before any
    // permission grant falls back to generic "Microphone N" names; callers
    // re-list after a successful getUserMedia to pick up the real labels.
    async function loadInputDevices() {
      if (!inputDevice || !navigator.mediaDevices?.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const inputs = devices.filter(d => d.kind === "audioinput");
        const previous = inputDevice.value;
        inputDevice.replaceChildren();
        const def = document.createElement("option");
        def.value = ""; def.textContent = "Default microphone";
        inputDevice.appendChild(def);
        // Offered whenever getDisplayMedia exists at all, even though only
        // Chromium browsers actually deliver an audio track from it --
        // there's no reliable capability check short of trying the capture,
        // so unsupported browsers are caught by captureSystemAudio's own
        // zero-audio-track error instead of being guessed at here.
        if (navigator.mediaDevices.getDisplayMedia) {
          const sys = document.createElement("option");
          sys.value = SYSTEM_AUDIO_VALUE; sys.textContent = "System audio (share a tab or screen)";
          inputDevice.appendChild(sys);
        }
        inputs.forEach((d, i) => {
          const option = document.createElement("option");
          option.value = d.deviceId;
          option.textContent = d.label || "Microphone " + (i + 1);
          inputDevice.appendChild(option);
        });
        const stored = previous || readStoredDevice();
        const validStored = stored === SYSTEM_AUDIO_VALUE ? !!navigator.mediaDevices.getDisplayMedia : inputs.some(d => d.deviceId === stored);
        if (stored && validStored) inputDevice.value = stored;
        inputDevice.hidden = inputs.length === 0 && !navigator.mediaDevices.getDisplayMedia;
      } catch (_) {
        // enumerateDevices needs a secure context; leave the default-only
        // dropdown in place rather than surfacing an error for this.
      }
    }
    function readStoredDevice() {
      try {
        return localStorage.getItem(AUDIO_DEVICE_KEY) || "";
      } catch (_) {
        return "";
      }
    }
    function audioConstraints() {
      const id = inputDevice && inputDevice.value;
      return id ? { deviceId: { ideal: id } } : true;
    }
    function sync() {
      toggle.hidden = !available;
      if (!available) panel.hidden = true;
      model.disabled = working;
      refresh.disabled = working;
      if (inputDevice) inputDevice.disabled = working;
      upload.disabled = working || !model.value;
      record.disabled = (working && !recorder) || !model.value || !navigator.mediaDevices?.getUserMedia || !global.MediaRecorder;
      record.textContent = recorder ? "Stop & transcribe" : "Record microphone";
      record.setAttribute("aria-pressed", String(!!recorder));
      if (live) {
        live.disabled = (working && !liveSession) || !!liveSession?.stopping || !model.value || !navigator.mediaDevices?.getUserMedia || !global.AudioWorkletNode;
        live.textContent = liveSession ? "Stop live transcription" : "Start live transcription";
        live.setAttribute("aria-pressed", String(!!liveSession));
      }
      cancel.hidden = !working;
      insert.disabled = working || !result.value.trim();
      panel.setAttribute("aria-busy", String(working && !recorder));
    }
    function stopTracks() {
      clearTimeout(timer); clearInterval(ticker);
      timer = ticker = null;
      if (stream) stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    function cancelJob() {
      sequence++;
      if (abort) abort.abort();
      abort = null;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      recorder = null;
      const job = liveSession;
      liveSession = null;
      if (job) { job.abort.abort(); releaseLive(job); }
      stopTracks();
      working = false;
      message("Cancelled.");
      sync();
    }
    function stopLiveCapture(job) {
      if (!job) return;
      clearTimeout(job.stopTimer);
      if (job.node) { job.node.disconnect(); job.node.port.onmessage = null; }
      if (job.source) job.source.disconnect();
      if (job.context) job.context.close().catch(() => {});
      if (job.media) job.media.getTracks().forEach(track => track.stop());
      job.node = job.source = job.context = job.media = null;
      if (levelRow) levelRow.hidden = true;
      updateLevel(0);
    }
    function releaseLive(job) {
      stopLiveCapture(job);
      if (job.id) options.fetch("/v1/audio/realtime/sessions/" + encodeURIComponent(job.id), {method:"DELETE", keepalive:true}).catch(() => {});
    }
    function finishLive(job, error) {
      if (job !== liveSession) return;
      liveSession = null;
      job.abort.abort(); releaseLive(job);
      working = false; sync();
      message(error || (result.value.trim() ? "Transcript ready. Review it, then use it in your message." : "No speech was recognized."), !!error);
    }
    async function flushLive(job) {
      if (job !== liveSession || job.sending || !job.id) return;
      if (!job.queue.length && !job.final) return;
      job.sending = true;
      try {
        while (job === liveSession && (job.queue.length || job.final)) {
          // Amortize inference overhead if capture outruns a request. Send
          // already queued PCM together, without waiting for a larger block.
          let bytes = 0, count = 0;
          while (count < job.queue.length && bytes + job.queue[count].byteLength <= 64000) bytes += job.queue[count++].byteLength;
          const body = new ArrayBuffer(bytes), view = new Uint8Array(body);
          let offset = 0;
          for (const block of job.queue.splice(0, count)) { view.set(new Uint8Array(block), offset); offset += block.byteLength; }
          job.queuedBytes -= body.byteLength;
          const final = job.final && !job.queue.length;
          const response = await options.fetch("/v1/audio/realtime/sessions/" + encodeURIComponent(job.id) + (final ? "?final=1" : ""), {
            method:"POST", headers:{"Content-Type":"application/octet-stream"}, body, signal:job.abort.signal
          });
          if (!response.ok) throw new Error(await response.text() || "Live transcription failed.");
          const data = await response.json();
          if (job !== liveSession) return;
          result.value = String(data.text || "").trim();
          result.hidden = !result.value; insert.hidden = !result.value;
          if (final) { finishLive(job); return; }
          const lag = job.queuedBytes / 32000;
          message(job.stopping ? "Finishing the last words…" : lag > 1
            ? "Listening… Server is " + lag.toFixed(1) + " seconds behind."
            : "Listening… Text appears as you speak.");
        }
      } catch (error) {
        if (job === liveSession) finishLive(job, error.message);
      } finally {
        job.sending = false;
      }
    }
    function endLive(job) {
      if (!job || job !== liveSession || job.stopping) return;
      job.stopping = true; sync();
      message("Finishing the last words…");
      if (!job.node) { finishLive(job); return; }
      // The worklet first emits its partial block and then the stop marker.
      job.node.port.postMessage("stop");
      job.stopTimer = setTimeout(() => {
        if (job === liveSession && !job.final) finishLive(job, "Microphone capture stopped responding.");
      }, 3000);
    }
    async function startLive() {
      if (working || !model.value || !available) return;
      const AudioContext = global.AudioContext || global.webkitAudioContext;
      if (!AudioContext || !global.AudioWorkletNode) { message("Live capture requires a browser with AudioWorklet support.", true); return; }
      const usingSystemAudio = inputDevice && inputDevice.value === SYSTEM_AUDIO_VALUE;
      const job = {id:"", abort:new AbortController(), queue:[], queuedBytes:0, sending:false, final:false, stopping:false};
      liveSession = job; working = true; ++sequence;
      result.value = ""; result.hidden = true; insert.hidden = true;
      message(usingSystemAudio ? "Waiting for you to pick what to share, then loading Voxtral…" : "Preparing microphone and loading Voxtral…"); sync();
      try {
        // Create/resume during the user gesture so autoplay policy cannot
        // suspend the audio graph while model loading takes place.
        job.context = new AudioContext({sampleRate:16000});
        await job.context.resume();
        if (job !== liveSession) return;
        job.media = usingSystemAudio ? await captureSystemAudio() : await navigator.mediaDevices.getUserMedia({audio:audioConstraints()});
        if (!usingSystemAudio) loadInputDevices();
        if (job !== liveSession) { releaseLive(job); return; }
        // Covers both the browser's native "Stop sharing" control (system
        // audio) and a microphone track ending unexpectedly (device removed).
        job.media.getTracks().forEach(t => t.addEventListener("ended", () => endLive(job)));
        await job.context.audioWorklet.addModule("/audio-worklet.js");
        if (job !== liveSession) return;
        const response = await options.fetch("/v1/audio/realtime/sessions", {
          method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({model:model.value}), signal:job.abort.signal
        });
        if (!response.ok) throw new Error(await response.text() || "Cannot start live transcription.");
        const session = await response.json(); job.id = session.id;
        if (job !== liveSession) { releaseLive(job); return; }
        job.source = job.context.createMediaStreamSource(job.media);
        job.node = new global.AudioWorkletNode(job.context, "voxtral-pcm");
        job.node.port.onmessage = event => {
          if (job !== liveSession) return;
          if (event.data === "stopped") {
            stopLiveCapture(job); job.final = true; flushLive(job); return;
          }
          if (event.data && typeof event.data === "object" && "level" in event.data) {
            updateLevel(event.data.level);
            return;
          }
          job.queue.push(event.data); job.queuedBytes += event.data.byteLength;
          if (job.queuedBytes > 320000) { finishLive(job, "Server cannot keep up: more than 10 seconds of audio queued. Try a faster inference backend."); return; }
          flushLive(job);
        };
        job.source.connect(job.node); job.node.connect(job.context.destination);
        if (levelRow) levelRow.hidden = false;
        message(usingSystemAudio ? "Listening to shared audio… Text appears as it plays." : "Listening… Text appears as you speak."); sync();
      } catch (error) {
        if (job === liveSession) {
          finishLive(job, error.name === "NotAllowedError"
            ? (usingSystemAudio ? "Screen/audio sharing was cancelled or denied." : "Microphone permission was denied.")
            : error.message);
        } else releaseLive(job);
      }
    }
    async function loadModels() {
      if (!available || working) return;
      const id = ++catalogSequence;
      refresh.disabled = true;
      try {
        const response = await options.fetch("/models/audio", { cache: "no-store" });
        if (!response.ok) throw new Error(await response.text() || "Cannot load audio models.");
        const data = await response.json();
        if (id !== catalogSequence || !available) return;
        const selected = model.value;
        model.replaceChildren();
        for (const entry of data.models || []) {
          const option = document.createElement("option");
          option.value = entry.id; option.textContent = entry.name || entry.id;
          model.appendChild(option);
        }
        if (Array.from(model.options).some(option => option.value === selected)) model.value = selected;
        message(model.value
          ? "Start live transcription, upload audio, or record a clip up to 30 seconds. Voxtral runs locally on this server."
          : "No Voxtral model found. Place a Voxtral Realtime GGUF in the server's model directory, then refresh.");
      } catch (error) {
        if (id === catalogSequence) message(error.message, true);
      } finally {
        if (id === catalogSequence) sync();
      }
    }
    async function transcribe(blob, recording, id, selected) {
      try {
        message("Preparing audio…");
        const wav = await toWAV(blob, recording);
        if (id !== sequence || !available) return;
        const form = new FormData();
        form.append("model", selected);
        form.append("file", wav, "audio.wav");
        abort = new AbortController();
        message("Transcribing with Voxtral… The model may take a while to load.");
        const response = await options.fetch("/v1/audio/transcriptions", { method: "POST", body: form, signal: abort.signal });
        if (!response.ok) throw new Error(await response.text() || "Transcription failed.");
        const data = await response.json();
        if (id !== sequence || !available) return;
        result.value = String(data.text || "").trim();
        result.hidden = !result.value;
        insert.hidden = !result.value;
        message(result.value ? "Transcript ready. Review it, then use it in your message." : "No speech was recognized. Try a clearer recording.");
      } catch (error) {
        if (id === sequence && error.name !== "AbortError") message(error.message, true);
      } finally {
        if (id === sequence) { abort = null; working = false; sync(); }
      }
    }
    toggle.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      toggle.setAttribute("aria-expanded", String(!panel.hidden));
      if (!panel.hidden && !working) { loadModels(); loadInputDevices(); }
    });
    refresh.addEventListener("click", loadModels);
    model.addEventListener("change", sync);
    if (inputDevice) {
      inputDevice.addEventListener("change", () => {
        try { localStorage.setItem(AUDIO_DEVICE_KEY, inputDevice.value); } catch (_) { /* per-tab only */ }
      });
    }
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", () => loadInputDevices());
    }
    upload.addEventListener("click", () => file.click());
    file.addEventListener("change", () => {
      const blob = file.files[0]; file.value = "";
      if (!blob || working || !available || !model.value) return;
      working = true; const id = ++sequence; sync();
      transcribe(blob, false, id, model.value);
    });
    record.addEventListener("click", async () => {
      if (recorder) { if (recorder.state !== "inactive") recorder.stop(); return; }
      if (working || !available || !model.value) return;
      const usingSystemAudio = inputDevice && inputDevice.value === SYSTEM_AUDIO_VALUE;
      working = true; const id = ++sequence; const selected = model.value;
      message(usingSystemAudio ? "Waiting for you to pick what to share…" : "Waiting for microphone permission…"); sync();
      try {
        const media = usingSystemAudio ? await captureSystemAudio() : await navigator.mediaDevices.getUserMedia({ audio: audioConstraints() });
        if (!usingSystemAudio) loadInputDevices();
        if (id !== sequence || !available) { media.getTracks().forEach(track => track.stop()); return; }
        stream = media;
        // Covers the browser's native "Stop sharing" control (system audio)
        // or a microphone device disappearing mid-recording.
        media.getTracks().forEach(t => t.addEventListener("ended", () => {
          if (id !== sequence || recorder?.state === "inactive") return;
          if (recorder) recorder.stop();
        }));
        const chunks = [];
        const capture = new MediaRecorder(media);
        recorder = capture;
        capture.addEventListener("dataavailable", event => { if (event.data.size) chunks.push(event.data); });
        capture.addEventListener("error", () => {
          if (id !== sequence) return;
          cancelJob(); message((usingSystemAudio ? "Audio sharing" : "Microphone recording") + " failed. Try uploading an audio file.", true);
        });
        capture.addEventListener("stop", () => {
          if (id !== sequence) return;
          recorder = null; stopTracks(); sync();
          transcribe(new Blob(chunks, { type: capture.mimeType }), true, id, selected);
        });
        capture.start(250);
        const started = Date.now();
        const update = () => message("Recording… " + Math.min(MAX_SECONDS, Math.floor((Date.now() - started) / 1000)) + " / 30 seconds");
        update(); ticker = setInterval(update, 500);
        timer = setTimeout(() => { if (capture.state !== "inactive") capture.stop(); }, MAX_SECONDS * 1000);
        sync();
      } catch (error) {
        if (id !== sequence) return;
        stopTracks(); recorder = null; working = false;
        message(error.name === "NotAllowedError"
          ? (usingSystemAudio ? "Screen/audio sharing was cancelled or denied." : "Microphone permission was denied. Allow access or upload an audio file.")
          : error.message, true);
        sync();
      }
    });
    if (live) live.addEventListener("click", () => liveSession ? endLive(liveSession) : startLive());
    cancel.addEventListener("click", cancelJob);
    insert.addEventListener("click", () => {
      if (!result.value.trim() || working) return;
      if (options.insert(result.value.trim())) {
        result.value = ""; result.hidden = true; insert.hidden = true;
        message("Transcript added to your message. You can edit it before sending."); sync();
      } else message("Wait for the current response to finish before inserting the transcript.");
    });
    global.addEventListener("pagehide", cancelJob);
    sync();
    return { setAvailable(value) {
      available = !!value;
      if (!available) { catalogSequence++; if (working) cancelJob(); toggle.setAttribute("aria-expanded", "false"); }
      sync();
    } };
  }
  global.GopherLLMAudio = { init, encodeWAV, toWAV };
})(globalThis);
