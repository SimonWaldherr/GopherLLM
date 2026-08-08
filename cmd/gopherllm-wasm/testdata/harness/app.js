// Phase A verification harness: load the wasm module, run one greedy
// generation against a tiny synthetic mistral3 GGUF, and check the result
// against the known-correct output from a native (non-wasm) run of the same
// bytes with the same (deterministic, temperature-0) sampler settings. See
// wasm_fixture_test.go / TestWasmFixtureGoldenOutputPrint for how that
// reference text was produced.
const GOLDEN_TEXT = "id<s>gbje";

function log(msg) {
  console.log(msg);
  document.getElementById("log").textContent += msg + "\n";
}

function setStatus(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = cls;
}

// wasm_exec.js's Go.run registers window.gopherllm_* synchronously before
// its returned promise's first await (Go's main() runs registerCallbacks()
// then blocks in select{}), but poll briefly anyway rather than assume it.
async function waitForBridge(timeoutMs) {
  const start = Date.now();
  while (typeof window.gopherllm_loadModel !== "function") {
    if (Date.now() - start > timeoutMs) {
      throw new Error("gopherllm_loadModel was never registered by the wasm module");
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function main() {
  log("fetching gopherllm.wasm...");
  const go = new Go();
  const resp = await fetch("/bin/gopherllm.wasm");
  if (!resp.ok) throw new Error("fetching gopherllm.wasm: HTTP " + resp.status);
  const { instance } = await WebAssembly.instantiateStreaming(resp, go.importObject);

  // Deliberately not awaited: main() calls select{} and never returns, so
  // go.run()'s promise never resolves. Its body runs synchronously up to
  // that point, which is what registers window.gopherllm_*.
  go.run(instance);
  await waitForBridge(2000);
  log("wasm module running, bridge functions registered");

  const modelResp = await fetch("tiny-model.gguf");
  if (!modelResp.ok) throw new Error("fetching tiny-model.gguf: HTTP " + modelResp.status);
  const bytes = new Uint8Array(await modelResp.arrayBuffer());
  log("fetched tiny-model.gguf (" + bytes.length + " bytes), loading...");

  await window.gopherllm_loadModel(bytes);
  log("model loaded");

  const messages = JSON.stringify([{ role: "user", content: "Hello" }]);
  const options = JSON.stringify({
    maxTokens: 8,
    temperature: 0,
    topP: 1,
    topK: 0,
    minP: 0,
    repeatPenalty: 1.1,
  });

  let streamed = "";
  const text = await window.gopherllm_generate(messages, options, (tok) => {
    streamed += tok;
    document.getElementById("output").textContent = streamed;
    return true;
  });
  log("generation finished: " + JSON.stringify(text));
  document.getElementById("output").textContent = text;

  if (text === GOLDEN_TEXT) {
    setStatus("PASS: wasm output matches native golden output", "pass");
  } else {
    setStatus(
      "MISMATCH: expected " + JSON.stringify(GOLDEN_TEXT) + " got " + JSON.stringify(text),
      "fail"
    );
  }
}

main().catch((err) => {
  console.error(err);
  log("FATAL: " + (err && err.message ? err.message : String(err)));
  setStatus("ERROR: " + (err && err.message ? err.message : String(err)), "fail");
});
