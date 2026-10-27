(() => {
  "use strict";
  const byID = (id) => document.getElementById(id);
  const form = byID("imageForm"), status = byID("imageStatus");
  const generate = byID("generate"), cancel = byID("cancel");
  const keepAlive = byID("keepAlive"), unload = byID("unload");
  let controller = null;
  function busy(value, cancellable = true) {
    generate.disabled = unload.disabled = keepAlive.disabled = value;
    cancel.hidden = !value || !cancellable;
    form.setAttribute("aria-busy", String(value));
  }
  unload.addEventListener("click", async () => {
    if (controller) return;
    controller = new AbortController();
    busy(true, false);
    try {
      const response = await fetch("/v1/images/unload", {method: "POST", signal: controller.signal});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || "Could not unload the model.");
      keepAlive.checked = false;
      status.textContent = "Model unloaded.";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      controller = null;
      busy(false);
    }
  });
  cancel.addEventListener("click", () => { if (controller) controller.abort(); });
  window.addEventListener("pagehide", () => { if (controller) controller.abort(); });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (controller || !form.reportValidity()) return;
    const prompt = byID("prompt").value.trim();
    if (!prompt) { status.textContent = "Describe the image first."; return; }
    const request = {prompt, size: byID("size").value, seed: Number(byID("seed").value), keep_alive: keepAlive.checked};
    controller = new AbortController();
    busy(true);
    const started = performance.now();
    status.textContent = "Loading the model and generating your image…";
    try {
      const response = await fetch("/v1/images/generations", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(request), signal: controller.signal
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || "Generation failed (HTTP " + response.status + ").");
      const data = result.data?.[0]?.b64_json;
      if (typeof data !== "string" || !data.startsWith("iVBORw0KGgo")) throw new Error("The server did not return a PNG image.");
      const url = "data:image/png;base64," + data;
      byID("result").src = url;
      byID("result").alt = request.prompt;
      byID("result").hidden = false;
      byID("empty").hidden = true;
      byID("download").href = url;
      byID("download").hidden = false;
      status.textContent = "Image ready in " + ((performance.now() - started) / 1000).toFixed(1) + " s · seed " + request.seed + (request.keep_alive ? ". Model kept loaded." : ". Model unloaded.");
    } catch (error) {
      status.textContent = error.name === "AbortError" ? "Cancelled. The server is releasing the model; retry shortly." : error.message;
    } finally {
      controller = null;
      busy(false);
    }
  });
})();
