// Lazy loader + thin promise wrapper around the GopherLLM WASM/WebGPU
// runtime (cmd/gopherllm-wasm), for the production chat UI's "run in this
// browser tab" inference mode. Only fetched by script.js when the user
// actually switches to browser mode -- server-mode users never load this
// file, wasm_exec.js, or the (multi-MB) gopherllm.wasm binary.
//
// No third-party code: wasm_exec.js ships with the Go toolchain itself
// (copied into bin/ by `make wasm-build`), and gopherllm.wasm is this
// project's own engine compiled with GOOS=js GOARCH=wasm.
(function () {
  "use strict";

  let goInstancePromise = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector('script[src="' + src + '"]');
      if (existing) {
        if (existing.dataset.loaded === "true") { resolve(); return; }
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () => reject(new Error("failed to load " + src)));
        return;
      }
      const el = document.createElement("script");
      el.src = src;
      el.addEventListener("load", () => { el.dataset.loaded = "true"; resolve(); });
      el.addEventListener("error", () => reject(new Error("failed to load " + src)));
      document.head.appendChild(el);
    });
  }

  async function waitForBridge(timeoutMs) {
    const start = Date.now();
    while (typeof window.gopherllm_loadModelWithVision !== "function") {
      if (Date.now() - start > timeoutMs) throw new Error("wasm bridge functions never registered");
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  // Idempotent: the first caller pays for fetching wasm_exec.js and
  // instantiating gopherllm.wasm; every later call (or concurrent call)
  // shares that same promise instead of re-running go.run() a second time,
  // which would spin up a second Go runtime instance in the same page.
  function ensureLoaded() {
    if (!goInstancePromise) {
      goInstancePromise = (async () => {
        if (typeof window.Go !== "function") {
          await loadScript("/wasm/wasm_exec.js");
        }
        const go = new window.Go();
        const resp = await fetch("/wasm/gopherllm.wasm");
        if (!resp.ok) throw new Error("fetching gopherllm.wasm: HTTP " + resp.status);
        const { instance } = await WebAssembly.instantiateStreaming(resp, go.importObject);
        go.run(instance);
        await waitForBridge(4000);
      })().catch((error) => {
        // A transient fetch, CSP, or compile failure must not poison every
        // later retry in this tab with the same rejected Promise.
        goInstancePromise = null;
        throw error;
      });
    }
    return goInstancePromise;
  }

  async function loadModel(textBytes, visionBytes) {
    await ensureLoaded();
    // The Go bridge copies these views synchronously before returning its
    // Promise. Drop the JS-side references immediately afterwards so model
    // loading does not retain a second full copy while Go parses the GGUF.
    const pending = visionBytes
      ? window.gopherllm_loadModelWithVision(textBytes, visionBytes)
      : window.gopherllm_loadModelWithVision(textBytes);
    textBytes = null;
    visionBytes = null;
    await pending;
  }

  async function hasVision() {
    await ensureLoaded();
    return window.gopherllm_hasVision();
  }

  async function isModelLoaded() {
    if (!goInstancePromise) return false;
    await ensureLoaded();
    return window.gopherllm_isModelLoaded();
  }

  async function webgpuStatus() {
    await ensureLoaded();
    return window.gopherllm_webgpuStatus();
  }

  // messages: [{role, content, images?: [base64,...]}]
  // options: {maxTokens, temperature, topP, topK, minP, repeatPenalty, systemPrompt}
  // onToken(text: string) => bool|undefined -- returning false stops generation early.
  async function generate(messages, options, onToken) {
    await ensureLoaded();
    return window.gopherllm_generate(JSON.stringify(messages), JSON.stringify(options || {}), onToken);
  }

  function stopGeneration() {
    if (typeof window.gopherllm_stopGeneration === "function") window.gopherllm_stopGeneration();
  }

  function isGenerating() {
    return typeof window.gopherllm_isGenerating === "function" && window.gopherllm_isGenerating();
  }

  window.GopherLLMBrowser = {
    ensureLoaded,
    loadModel,
    hasVision,
    isModelLoaded,
    webgpuStatus,
    generate,
    stopGeneration,
    isGenerating
  };
})();
