// GopherLLM in-browser demo. No third-party code: only this project's own
// wasm module (via wasm_exec.js, which ships with the Go toolchain itself)
// and the browser's native File/FileReader/WebGPU APIs.

const gpuBadge = document.getElementById("gpuBadge");
const textModelFile = document.getElementById("textModelFile");
const visionModelFile = document.getElementById("visionModelFile");
const loadButton = document.getElementById("loadButton");
const loadStatus = document.getElementById("loadStatus");
const loadPanel = document.getElementById("loadPanel");
const chatPanel = document.getElementById("chatPanel");
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const promptInput = document.getElementById("promptInput");
const sendButton = document.getElementById("sendButton");
const stopButton = document.getElementById("stopButton");
const attachLabel = document.getElementById("attachLabel");
const imageFile = document.getElementById("imageFile");
const webcamButton = document.getElementById("webcamButton");
const screenButton = document.getElementById("screenButton");
const captureError = document.getElementById("captureError");
const imagePreviewRow = document.getElementById("imagePreviewRow");
const imagePreview = document.getElementById("imagePreview");
const clearImageButton = document.getElementById("clearImageButton");
const captureModal = document.getElementById("captureModal");
const captureVideo = document.getElementById("captureVideo");
const captureCancelButton = document.getElementById("captureCancelButton");
const captureConfirmButton = document.getElementById("captureConfirmButton");

let pendingImageDataURL = null; // data:image/...;base64,XXXX, or null
let history = []; // [{role, content, imageDataURL?}]

function setLoadStatus(text, cls) {
  loadStatus.textContent = text;
  loadStatus.className = "status" + (cls ? " " + cls : "");
}

function setBadge(text, cls) {
  gpuBadge.textContent = text;
  gpuBadge.className = "badge " + cls;
}

async function waitForBridge(timeoutMs) {
  const start = Date.now();
  while (typeof window.gopherllm_loadModelWithVision !== "function") {
    if (Date.now() - start > timeoutMs) throw new Error("wasm bridge functions never registered");
    await new Promise((r) => setTimeout(r, 10));
  }
}

function readFileAsUint8Array(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result));
    reader.onerror = () => reject(reader.error || new Error("file read failed"));
    reader.readAsArrayBuffer(file);
  });
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("file read failed"));
    reader.readAsDataURL(file);
  });
}

function renderMessages() {
  messagesEl.innerHTML = "";
  for (const m of history) {
    const div = document.createElement("div");
    div.className = "msg msg-" + m.role;
    if (m.imageDataURL) {
      const img = document.createElement("img");
      img.src = m.imageDataURL;
      div.appendChild(img);
    }
    const textNode = document.createElement("span");
    textNode.textContent = m.content;
    div.appendChild(textNode);
    messagesEl.appendChild(div);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateAttachVisibility() {
  const hasVision = window.gopherllm_hasVision ? window.gopherllm_hasVision() : false;
  attachLabel.classList.toggle("hidden", !hasVision);
  webcamButton.classList.toggle("hidden", !hasVision);
  screenButton.classList.toggle("hidden", !hasVision);
}

function showCaptureError(err) {
  captureError.textContent = err && err.message ? err.message : String(err);
  captureError.classList.remove("hidden");
}
function clearCaptureError() {
  captureError.classList.add("hidden");
}

function acceptCapturedImage(dataURL) {
  pendingImageDataURL = dataURL;
  imagePreview.src = dataURL;
  imagePreviewRow.classList.remove("hidden");
  clearCaptureError();
}

// The capture modal shows a live preview (webcam or screen share) with
// Capture/Cancel buttons, rather than grabbing a frame blind the instant
// permission is granted -- lets you see what you're about to send, retry a
// bad angle, or back out, the same as any normal camera app.
let activeCaptureStream = null;

function stopActiveCaptureStream() {
  if (activeCaptureStream) {
    activeCaptureStream.getTracks().forEach((t) => t.stop());
    activeCaptureStream = null;
  }
}

async function openCaptureModal(stream) {
  activeCaptureStream = stream;
  // If the user stops sharing from the browser's own "Stop sharing" bar
  // (screen share) or unplugs/disables the camera mid-preview, the modal
  // should close itself rather than sit there showing a frozen last frame.
  stream.getVideoTracks().forEach((t) => {
    t.addEventListener("ended", () => {
      if (captureModal.open) closeCaptureModal();
    });
  });
  captureVideo.srcObject = stream;
  await captureVideo.play();
  captureModal.showModal();
}

function closeCaptureModal() {
  if (captureModal.open) captureModal.close();
  captureVideo.srcObject = null;
  stopActiveCaptureStream();
}

async function main() {
  setBadge("loading wasm…", "badge-pending");
  const go = new Go();
  const resp = await fetch("/bin/gopherllm.wasm");
  if (!resp.ok) throw new Error("fetching gopherllm.wasm: HTTP " + resp.status);
  const { instance } = await WebAssembly.instantiateStreaming(resp, go.importObject);
  go.run(instance);
  await waitForBridge(2000);

  const gpu = await window.gopherllm_webgpuStatus();
  if (gpu === "available") {
    setBadge("⚡ WebGPU accelerated", "badge-ok");
  } else {
    setBadge("🐌 CPU only (WebGPU unavailable)", "badge-warn");
  }

  textModelFile.addEventListener("change", () => {
    loadButton.disabled = !textModelFile.files.length;
  });

  loadButton.addEventListener("click", async () => {
    if (!textModelFile.files.length) return;
    loadButton.disabled = true;
    try {
      setLoadStatus("reading text model file…");
      const textBytes = await readFileAsUint8Array(textModelFile.files[0]);

      let visionBytes = null;
      if (visionModelFile.files.length) {
        setLoadStatus("reading vision projector file…");
        visionBytes = await readFileAsUint8Array(visionModelFile.files[0]);
      }

      setLoadStatus("loading into GopherLLM (this can take a while for large models)…");
      if (visionBytes) {
        await window.gopherllm_loadModelWithVision(textBytes, visionBytes);
      } else {
        await window.gopherllm_loadModelWithVision(textBytes);
      }

      setLoadStatus("model loaded.", "ok");
      updateAttachVisibility();
      loadPanel.classList.add("hidden");
      chatPanel.classList.remove("hidden");
      promptInput.focus();
    } catch (err) {
      setLoadStatus("failed to load: " + (err && err.message ? err.message : String(err)), "error");
      loadButton.disabled = false;
    }
  });

  attachLabel.addEventListener("click", () => imageFile.click());
  imageFile.addEventListener("change", async () => {
    if (!imageFile.files.length) return;
    pendingImageDataURL = await readFileAsDataURL(imageFile.files[0]);
    imagePreview.src = pendingImageDataURL;
    imagePreviewRow.classList.remove("hidden");
  });
  clearImageButton.addEventListener("click", () => {
    pendingImageDataURL = null;
    imageFile.value = "";
    imagePreviewRow.classList.add("hidden");
  });

  webcamButton.addEventListener("click", async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("camera access is unavailable (requires HTTPS or localhost)");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      await openCaptureModal(stream);
    } catch (err) {
      showCaptureError(err);
    }
  });

  screenButton.addEventListener("click", async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
        throw new Error("screen capture is unavailable (requires HTTPS or localhost)");
      }
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
      await openCaptureModal(stream);
    } catch (err) {
      showCaptureError(err);
    }
  });

  captureCancelButton.addEventListener("click", () => {
    closeCaptureModal();
  });

  captureConfirmButton.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = captureVideo.videoWidth;
    canvas.height = captureVideo.videoHeight;
    canvas.getContext("2d").drawImage(captureVideo, 0, 0);
    const dataURL = canvas.toDataURL("image/png");
    closeCaptureModal();
    acceptCapturedImage(dataURL);
  });

  // Covers Escape (native <dialog> behavior) in addition to the Cancel
  // button above.
  captureModal.addEventListener("close", () => {
    stopActiveCaptureStream();
  });

  stopButton.addEventListener("click", () => {
    window.gopherllm_stopGeneration();
  });

  chatForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const content = promptInput.value.trim();
    if (!content || window.gopherllm_isGenerating()) return;

    const userMsg = { role: "user", content, imageDataURL: pendingImageDataURL };
    history.push(userMsg);
    renderMessages();

    const wireMessages = history.map((m) => {
      const wire = { role: m.role, content: m.content };
      if (m.imageDataURL) {
        const comma = m.imageDataURL.indexOf(",");
        wire.images = [m.imageDataURL.slice(comma + 1)];
      }
      return wire;
    });

    promptInput.value = "";
    pendingImageDataURL = null;
    imageFile.value = "";
    imagePreviewRow.classList.add("hidden");

    const assistantMsg = { role: "assistant", content: "" };
    history.push(assistantMsg);
    renderMessages();

    sendButton.disabled = true;
    stopButton.classList.remove("hidden");
    try {
      await window.gopherllm_generate(
        JSON.stringify(wireMessages),
        JSON.stringify({ maxTokens: 512, temperature: 0.7, topP: 0.9, topK: 40, repeatPenalty: 1.1 }),
        (tok) => {
          assistantMsg.content += tok;
          renderMessages();
          return true;
        }
      );
    } catch (err) {
      assistantMsg.content += "\n[error: " + (err && err.message ? err.message : String(err)) + "]";
      renderMessages();
    } finally {
      sendButton.disabled = false;
      stopButton.classList.add("hidden");
    }
  });
}

main().catch((err) => {
  console.error(err);
  setBadge("error: " + (err && err.message ? err.message : String(err)), "badge-warn");
});
