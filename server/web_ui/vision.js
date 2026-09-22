/* Local YOLO object detection for the chat UI. The image never leaves this
   server: the browser posts it to /v1/vision/detections, draws the boxes,
   and turns the regions the user picks into a crop (or a numbered collage
   of crops) that replaces the attached image, so the vision LLM spends its
   image tokens on what matters at full resolution instead of on a
   downscaled whole photo. The same helpers drive the live camera's
   detector gate in script.js. */
(function (global) {
  "use strict";
  // Longest edge of the frame sent to the detector. YOLO letterboxes to 640
  // anyway; 1280 keeps small objects legible and the upload small.
  const DETECT_EDGE = 1280;
  // Longest edge kept for cropping. Crops come from this canvas, so a photo
  // straight off a phone keeps its detail where the user points.
  const SOURCE_EDGE = 4096;
  // Context around each box: models describe an object better when they
  // can see where it stands, and detector boxes are often a little tight.
  const CROP_PAD = 0.15;
  // Tiny crops are upscaled to this edge so the vision encoder gets more
  // than a handful of patches to look at.
  const MIN_CROP_EDGE = 256;
  const MAX_SELECTED = 4;
  // Above this many boxes, labels only show on hover, focus or selection.
  const CROWDED = 8;
  const COLLAGE_ROW = 448;
  const COLLAGE_WIDTH = 1344;
  const COLLAGE_GAP = 8;
  const MODEL_KEY = "gopherllm.detect-model";

  function labelOf(det) {
    return det && det.label ? String(det.label) : "class " + (det && Number.isFinite(det.class_id) ? det.class_id : "?");
  }

  function percent(value) {
    return Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 100) + "%";
  }

  // The crop rectangle for one detection box: padded on every side,
  // at least minSide pixels, clamped to the image, in whole pixels.
  function cropRect(box, width, height, pad = CROP_PAD, minSide = 32) {
    const bw = Math.max(1, box.x_max - box.x_min), bh = Math.max(1, box.y_max - box.y_min);
    const cx = (box.x_min + box.x_max) / 2, cy = (box.y_min + box.y_max) / 2;
    const w = Math.min(width, Math.max(minSide, bw * (1 + 2 * pad)));
    const h = Math.min(height, Math.max(minSide, bh * (1 + 2 * pad)));
    const x = Math.round(Math.max(0, Math.min(width - w, cx - w / 2)));
    const y = Math.round(Math.max(0, Math.min(height - h, cy - h / 2)));
    return { x, y, w: Math.max(1, Math.min(width - x, Math.round(w))), h: Math.max(1, Math.min(height - y, Math.round(h))) };
  }

  // Where a box sits in the frame, in words a model can repeat back.
  function describePosition(box, width, height) {
    const cx = (box.x_min + box.x_max) / 2 / Math.max(1, width);
    const cy = (box.y_min + box.y_max) / 2 / Math.max(1, height);
    const v = cy < 1 / 3 ? "top" : cy > 2 / 3 ? "bottom" : "middle";
    const h = cx < 1 / 3 ? "left" : cx > 2 / 3 ? "right" : "centre";
    return v === "middle" && h === "centre" ? "centre" : v + " " + h;
  }

  // The text sent alongside a crop. It tells the model what it is looking
  // at and that the detector is a hint, not ground truth: a vision LLM is a
  // good second opinion on a YOLO false positive, but only if asked.
  function describeSelection(selected, width, height, source) {
    const what = source === "live" ? "the current live frame" : "a larger " + width + "×" + height + " image";
    if (!selected.length) return "";
    if (selected.length === 1) {
      const d = selected[0];
      return "The attached image is a crop of " + what + ". A local object detector (YOLO) marked this region as \"" +
        labelOf(d) + "\" (" + percent(d.confidence) + " confidence), in the " + describePosition(d.box, width, height) +
        " of the original. The detector can be wrong: if the crop does not actually show a " + labelOf(d) + ", say so.";
    }
    return "The attached image is a collage of " + selected.length + " numbered crops from " + what +
      ", chosen from local object detector (YOLO) results:\n" +
      selected.map((d, i) => (i + 1) + ". " + labelOf(d) + " (" + percent(d.confidence) + "), " + describePosition(d.box, width, height)).join("\n") +
      "\nRefer to the crops by their number. The detector can be wrong: say so if a crop does not show what it is labelled as.";
  }

  // Tile positions for a collage of crops. Crops share a row height
  // (justified rows, up to three per row) instead of sitting in square
  // cells: the vision encoder is charged per pixel area, so black padding
  // around tall person crops would cost tokens and show nothing.
  function collageLayout(sizes, rowHeight = COLLAGE_ROW, maxWidth = COLLAGE_WIDTH, gap = COLLAGE_GAP) {
    const rows = [];
    const perRow = sizes.length === 4 ? 2 : 3;
    for (let i = 0; i < sizes.length; i += perRow) rows.push(sizes.slice(i, i + perRow).map((s, j) => ({ s, index: i + j })));
    const tiles = new Array(sizes.length);
    let y = 0, width = 0;
    rows.forEach((row) => {
      const aspect = row.reduce((sum, t) => sum + t.s.w / t.s.h, 0);
      const h = Math.max(1, Math.round(Math.min(rowHeight, (maxWidth - gap * (row.length - 1)) / aspect)));
      let x = 0;
      row.forEach((t) => {
        const w = Math.max(1, Math.round(t.s.w / t.s.h * h));
        tiles[t.index] = { x, y, w, h };
        x += w + gap;
      });
      width = Math.max(width, x - gap);
      y += h + gap;
    });
    return { width, height: Math.max(1, y - gap), tiles };
  }

  function parseClassList(text) {
    return new Set(String(text || "").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean));
  }

  function iou(a, b) {
    const w = Math.max(0, Math.min(a.x_max, b.x_max) - Math.max(a.x_min, b.x_min));
    const h = Math.max(0, Math.min(a.y_max, b.y_max) - Math.max(a.y_min, b.y_min));
    const inter = w * h;
    const union = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - inter;
    return union > 0 ? inter / union : 0;
  }

  // Whether the set of detections differs enough from the last one the LLM
  // saw to be worth asking it again: a label appeared, disappeared or
  // changed count, or some object moved so far that its box no longer
  // overlaps its old position by minIoU.
  function sceneChanged(previous, next, minIoU = 0.5) {
    if (!previous) return true;
    const count = (list) => list.reduce((m, d) => m.set(labelOf(d), (m.get(labelOf(d)) || 0) + 1), new Map());
    const a = count(previous), b = count(next);
    if (a.size !== b.size) return true;
    for (const [label, n] of b) if (a.get(label) !== n) return true;
    return next.some((d) => !previous.some((p) => labelOf(p) === labelOf(d) && iou(p.box, d.box) >= minIoU));
  }

  // The detections worth sending: the most confident first, at most max.
  function strongest(dets, max = MAX_SELECTED) {
    return dets.slice().sort((a, b) => b.confidence - a.confidence).slice(0, max);
  }

  function scaleDetections(dets, factor) {
    return dets.map((d) => Object.assign({}, d, { box: {
      x_min: d.box.x_min * factor, y_min: d.box.y_min * factor, x_max: d.box.x_max * factor, y_max: d.box.y_max * factor
    } }));
  }

  // Per-label counts, most frequent first: [["person", 12], ["bicycle", 4]].
  function countByLabel(dets) {
    const counts = new Map();
    dets.forEach((d) => counts.set(labelOf(d), (counts.get(labelOf(d)) || 0) + 1));
    return Array.from(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }

  // One line for sending a whole image to a chat model with the detector's
  // findings as context.
  function summarizeDetections(dets) {
    if (!dets.length) return "A local object detector (YOLO) found no objects in the attached image.";
    return "A local object detector (YOLO) found in the attached image: " +
      countByLabel(dets).map(([label, n]) => n + " × " + label).join(", ") +
      ". Detector labels can be wrong; rely on what is actually visible.";
  }

  // Turns per-frame counts into appearance/departure events for a live
  // feed. A change is reported only after the same new counts were seen in
  // confirmFrames consecutive frames, so one flickering low-confidence box
  // does not fill the log with "person appeared / person left" pairs.
  function createCountTracker(confirmFrames = 2) {
    let stable = new Map(), pending = null, hits = 0;
    const same = (a, b) => a.size === b.size && Array.from(a).every(([k, v]) => b.get(k) === v);
    return function update(dets) {
      const next = new Map(countByLabel(dets));
      if (same(next, stable)) { pending = null; return []; }
      if (pending && same(next, pending)) hits++;
      else { pending = next; hits = 1; }
      if (hits < confirmFrames) return [];
      const events = [];
      new Set([...stable.keys(), ...next.keys()]).forEach((label) => {
        const from = stable.get(label) || 0, to = next.get(label) || 0;
        if (from !== to) events.push({ label, from, to });
      });
      stable = next;
      pending = null;
      return events.sort((a, b) => a.label.localeCompare(b.label));
    };
  }

  function describeEvent(e) {
    if (!e.from) return e.label + " appeared" + (e.to > 1 ? " (" + e.to + ")" : "");
    if (!e.to) return e.label + " left";
    return e.label + ": " + e.from + " → " + e.to;
  }

  /* ---- canvas helpers (browser only) ---- */

  function loadImage(dataURL) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("The image could not be decoded."));
      img.src = dataURL;
    });
  }

  // Draws a decoded image (or a video frame) onto a canvas no larger than
  // maxEdge. Drawing an <img> applies its EXIF orientation, so boxes found
  // on this canvas line up with what the user sees in the page; sending the
  // raw JPEG bytes instead would let the server detect on the unrotated
  // sensor image of a portrait phone photo.
  function toCanvas(source, sourceWidth, sourceHeight, maxEdge) {
    const scale = Math.min(1, maxEdge / Math.max(sourceWidth, sourceHeight, 1));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));
    canvas.getContext("2d", { alpha: false }).drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  // Encodes a canvas for upload. Browsers throttle the asynchronous
  // encoders (canvas.toBlob, OffscreenCanvas.convertToBlob) for pages in
  // the background to about one callback per second, which would stall a
  // live detection loop the moment the user switches tabs -- exactly when
  // "beep when a person appears" matters. A hidden page therefore encodes
  // synchronously (a few ms for a detector-sized frame); a visible one keeps
  // the encode off the main thread.
  function canvasBlob(canvas, type = "image/jpeg", quality = 0.9) {
    if (global.document && document.visibilityState === "hidden") {
      const url = canvas.toDataURL(type, quality);
      const bin = atob(url.slice(url.indexOf(",") + 1));
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return Promise.resolve(new Blob([bytes], { type }));
    }
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Encoding the image failed.")), type, quality);
    });
  }

  // A JPEG data URL of the selected regions: one crop, or a numbered
  // collage of up to MAX_SELECTED crops so the chat still sends one image.
  function selectionDataURL(canvas, selected) {
    const rects = selected.map((d) => cropRect(d.box, canvas.width, canvas.height));
    const out = document.createElement("canvas");
    const ctx = out.getContext("2d", { alpha: false });
    if (rects.length === 1) {
      const r = rects[0];
      const up = Math.min(4, Math.max(1, MIN_CROP_EDGE / Math.max(r.w, r.h)));
      out.width = Math.round(r.w * up);
      out.height = Math.round(r.h * up);
      ctx.drawImage(canvas, r.x, r.y, r.w, r.h, 0, 0, out.width, out.height);
      return out.toDataURL("image/jpeg", 0.9);
    }
    const layout = collageLayout(rects);
    out.width = layout.width;
    out.height = layout.height;
    ctx.fillStyle = "#1c1c1c";
    ctx.fillRect(0, 0, out.width, out.height);
    layout.tiles.forEach((t, i) => {
      const r = rects[i];
      ctx.drawImage(canvas, r.x, r.y, r.w, r.h, t.x, t.y, t.w, t.h);
      const radius = 18, cx = t.x + radius + 6, cy = t.y + radius + 6;
      ctx.fillStyle = "rgba(0,0,0,0.72)";
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 22px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), cx, cy + 1);
    });
    return out.toDataURL("image/jpeg", 0.9);
  }

  // Draws detection boxes onto a transparent overlay canvas laid over a
  // frame shown at the overlay's size. fit "cover" matches a video with
  // object-fit: cover (centred crop); "fill" a frame scaled to the overlay.
  function drawOverlay(overlay, dets, frameWidth, frameHeight, fit = "fill") {
    const ratio = global.devicePixelRatio || 1;
    const cw = Math.round(overlay.clientWidth * ratio), ch = Math.round(overlay.clientHeight * ratio);
    if (overlay.width !== cw) overlay.width = cw;
    if (overlay.height !== ch) overlay.height = ch;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, cw, ch);
    if (!frameWidth || !frameHeight) return;
    const sx = fit === "cover" ? Math.max(cw / frameWidth, ch / frameHeight) : cw / frameWidth;
    const sy = fit === "cover" ? sx : ch / frameHeight;
    const ox = fit === "cover" ? (cw - frameWidth * sx) / 2 : 0, oy = fit === "cover" ? (ch - frameHeight * sy) / 2 : 0;
    paintBoxes(ctx, dets, (x, y) => [ox + x * sx, oy + y * sy], ratio);
  }

  // A copy of the frame with its detections burned in, for saving.
  function annotatedCanvas(frame, dets) {
    const out = document.createElement("canvas");
    out.width = frame.width;
    out.height = frame.height;
    const ctx = out.getContext("2d");
    ctx.drawImage(frame, 0, 0);
    paintBoxes(ctx, dets, (x, y) => [x, y], Math.max(1, Math.round(Math.max(frame.width, frame.height) / 800)));
    return out;
  }

  const BOX_COLORS = ["#2f9e6b", "#e8590c", "#1c7ed6", "#c2255c", "#7048e8", "#e8a200", "#0c8599", "#5c940d"];

  function colorFor(label) {
    let h = 0;
    for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return BOX_COLORS[h % BOX_COLORS.length];
  }

  function paintBoxes(ctx, dets, map, scale) {
    ctx.lineWidth = 2 * scale;
    ctx.font = "700 " + Math.round(11 * scale) + "px ui-monospace, monospace";
    ctx.textBaseline = "bottom";
    dets.forEach((d) => {
      const [x1, y1] = map(d.box.x_min, d.box.y_min), [x2, y2] = map(d.box.x_max, d.box.y_max);
      const color = colorFor(labelOf(d));
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const text = labelOf(d) + " " + percent(d.confidence);
      const tw = ctx.measureText(text).width + 8 * scale, th = 16 * scale;
      const ty = y1 - th < 0 ? y1 + th : y1;
      ctx.fillStyle = color;
      ctx.fillRect(x1 - ctx.lineWidth / 2, ty - th, tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x1 + 3 * scale, ty - 2 * scale);
    });
  }

  // Clickable, percentage-positioned box buttons over an image of
  // width×height pixels (see .detect-box). Largest first, so a small box (a
  // bicycle inside its rider's box) is on top and stays clickable.
  function renderBoxButtons(container, dets, width, height, selected, onToggle) {
    container.replaceChildren();
    container.classList.toggle("is-crowded", dets.length > CROWDED);
    container.classList.toggle("has-selection", selected.size > 0);
    if (!width || !height) return;
    const area = (d) => (d.box.x_max - d.box.x_min) * (d.box.y_max - d.box.y_min);
    dets.map((d, i) => i).sort((a, b) => area(dets[b]) - area(dets[a])).forEach((i) => {
      const d = dets[i];
      const b = document.createElement("button");
      b.type = "button";
      b.className = "detect-box" + (selected.has(i) ? " is-selected" : "");
      b.style.left = (100 * d.box.x_min / width) + "%";
      b.style.top = (100 * d.box.y_min / height) + "%";
      b.style.width = (100 * (d.box.x_max - d.box.x_min) / width) + "%";
      b.style.height = (100 * (d.box.y_max - d.box.y_min) / height) + "%";
      b.style.setProperty("--detect-color", colorFor(labelOf(d)));
      b.setAttribute("aria-pressed", String(selected.has(i)));
      b.setAttribute("aria-label", labelOf(d) + " " + percent(d.confidence));
      const tag = document.createElement("span");
      tag.className = "detect-box-label";
      tag.textContent = labelOf(d) + " " + percent(d.confidence);
      b.appendChild(tag);
      b.addEventListener("click", () => onToggle(i));
      container.appendChild(b);
    });
  }

  /* ---- server calls ---- */

  async function listModels(fetchFn) {
    const response = await fetchFn("/models/detection", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text() || "Listing detection models failed.");
    const data = await response.json();
    return Array.isArray(data.models) ? data.models : [];
  }

  async function detect(fetchFn, model, blob, opts = {}) {
    const form = new FormData();
    form.append("model", model);
    form.append("file", blob, "image.jpg");
    if (opts.conf) form.append("conf", String(opts.conf));
    if (opts.classes && opts.classes.size) form.append("classes", Array.from(opts.classes).join(","));
    const response = await fetchFn("/v1/vision/detections", { method: "POST", body: form, signal: opts.signal });
    if (!response.ok) throw new Error((await response.text()).trim() || "Object detection failed.");
    const data = await response.json();
    return Array.isArray(data.detections) ? data.detections : [];
  }

  // Fetches a stock model through the admin-controlled download route and
  // waits for its NDJSON stream to finish.
  async function downloadStock(fetchFn, entry, onProgress) {
    const response = await fetchFn("/models/download", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ ref: entry.download_ref, files: [entry.download_file] })
    });
    if (!response.ok || !response.body) throw new Error((await response.text()).trim() || "Download failed.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffered.indexOf("\n")) >= 0) {
        const line = buffered.slice(0, newline).trim();
        buffered = buffered.slice(newline + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.status === "error") throw new Error(event.error || "Download failed.");
        if (event.status === "success") return;
        if (onProgress && event.total) onProgress(event.completed / event.total);
      }
    }
    throw new Error("The download ended without completing.");
  }

  function preferredModel(models) {
    let saved = "";
    try { saved = global.localStorage.getItem(MODEL_KEY) || ""; } catch (_) { /* private mode */ }
    const usable = models.filter((m) => m.available);
    return (usable.find((m) => m.id === saved) || usable[0] || {}).id || "";
  }

  function rememberModel(id) {
    try { global.localStorage.setItem(MODEL_KEY, id); } catch (_) { /* private mode */ }
  }

  function fillModelSelect(select, models, placeholder) {
    const current = select.value;
    select.replaceChildren();
    if (placeholder !== undefined) {
      const off = document.createElement("option");
      off.value = "";
      off.textContent = placeholder;
      select.appendChild(off);
    }
    models.filter((m) => m.available).forEach((m) => {
      const option = document.createElement("option");
      option.value = m.id;
      option.textContent = m.id;
      select.appendChild(option);
    });
    if (Array.from(select.options).some((o) => o.value === current)) select.value = current;
  }

  /* ---- composer panel ---- */

  // options: fetch, onSelection(dataURL|null, context) -- called with the
  // crop to send (null restores the original attachment).
  function init(options) {
    const $ = (id) => document.getElementById(id);
    const button = $("detectObjectsButton");
    const panel = $("detectPanel");
    const modelSelect = $("detectModel");
    const status = $("detectStatus");
    const stage = $("detectStage");
    const image = $("detectImage");
    const boxes = $("detectBoxes");
    const useWhole = $("detectUseWhole");
    const done = $("detectDone");
    const download = $("detectDownload");
    if (!button || !panel) return null;

    let available = false;
    let sourceURL = null;
    let canvas = null;
    let detections = [];
    let selected = new Set();
    let models = [];
    let sequence = 0;

    function setStatus(text, isError) {
      status.textContent = text;
      status.classList.toggle("is-error", !!isError);
    }

    function sync() {
      button.hidden = !available || !sourceURL;
      if (button.hidden) panel.hidden = true;
      button.setAttribute("aria-expanded", String(!panel.hidden));
    }

    async function refreshModels() {
      models = await listModels(options.fetch);
      fillModelSelect(modelSelect, models);
      modelSelect.value = modelSelect.value || preferredModel(models);
      const stock = models.find((m) => m.source === "stock" && !m.available);
      download.hidden = models.some((m) => m.available) || !stock;
      if (!download.hidden) download.textContent = "Download " + stock.id;
      return modelSelect.value;
    }

    function renderBoxes() {
      renderBoxButtons(boxes, detections, canvas ? canvas.width : 0, canvas ? canvas.height : 0, selected, toggle);
    }

    function emitSelection() {
      const picked = Array.from(selected).sort((a, b) => a - b).map((i) => detections[i]);
      if (!picked.length) {
        options.onSelection(null, "");
        return;
      }
      options.onSelection(selectionDataURL(canvas, picked), describeSelection(picked, canvas.width, canvas.height, "upload"));
    }

    function toggle(index) {
      if (selected.has(index)) selected.delete(index);
      else if (selected.size < MAX_SELECTED) selected.add(index);
      else { setStatus("At most " + MAX_SELECTED + " regions can be sent at once.", true); return; }
      renderBoxes();
      emitSelection();
      setStatus(selected.size
        ? "Sending " + (selected.size === 1 ? "this region" : selected.size + " regions as a numbered collage") + " instead of the whole image."
        : detections.length + " object" + (detections.length === 1 ? "" : "s") + " found. Click boxes to send only those regions.", false);
    }

    async function run() {
      const seq = ++sequence;
      detections = [];
      selected = new Set();
      renderBoxes();
      emitSelection();
      try {
        setStatus("Preparing image…", false);
        if (!canvas) {
          const img = await loadImage(sourceURL);
          canvas = toCanvas(img, img.naturalWidth, img.naturalHeight, SOURCE_EDGE);
        }
        const model = modelSelect.value || await refreshModels();
        if (!model) {
          setStatus(download.hidden ? "No YOLO model found. Place a .pt or .safetensors checkpoint in the server's model directory." : "No YOLO model is downloaded yet.", true);
          return;
        }
        setStatus("Detecting objects…", false);
        const small = toCanvas(canvas, canvas.width, canvas.height, DETECT_EDGE);
        const found = await detect(options.fetch, model, await canvasBlob(small, "image/jpeg", 0.9));
        if (seq !== sequence) return;
        detections = scaleDetections(found, canvas.width / small.width);
        renderBoxes();
        setStatus(detections.length
          ? detections.length + " object" + (detections.length === 1 ? "" : "s") + " found. Click boxes to send only those regions."
          : "No objects found. The whole image will be sent.", false);
      } catch (err) {
        if (seq === sequence) setStatus(err && err.message ? err.message : String(err), true);
      }
    }

    button.addEventListener("click", async () => {
      if (!panel.hidden) { panel.hidden = true; sync(); return; }
      panel.hidden = false;
      image.src = sourceURL;
      sync();
      try { await refreshModels(); } catch (err) { setStatus(err.message || String(err), true); return; }
      if (!detections.length) run();
    });
    modelSelect.addEventListener("change", () => { rememberModel(modelSelect.value); run(); });
    useWhole.addEventListener("click", () => { selected = new Set(); renderBoxes(); emitSelection(); setStatus("The whole image will be sent.", false); });
    done.addEventListener("click", () => { panel.hidden = true; sync(); });
    download.addEventListener("click", async () => {
      const stock = models.find((m) => m.source === "stock" && !m.available);
      if (!stock) return;
      download.disabled = true;
      try {
        await downloadStock(options.fetch, stock, (p) => setStatus("Downloading " + stock.id + "… " + Math.round(p * 100) + "%", false));
        rememberModel(stock.id);
        await refreshModels();
        run();
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        download.disabled = false;
      }
    });
    stage.addEventListener("keydown", (event) => { if (event.key === "Escape") done.click(); });

    return {
      setAvailable(value) { available = !!value; sync(); },
      // A new attachment (or null when it is removed). Crops the user made
      // earlier belong to the old image and are discarded.
      setSource(dataURL) {
        sequence++;
        sourceURL = dataURL;
        canvas = null;
        detections = [];
        selected = new Set();
        renderBoxes();
        panel.hidden = true;
        sync();
      },
      refreshModels
    };
  }

  /* ---- detection workbench (no language model) ---- */

  // The standalone "Object detection" dialog: YOLO on an image, or live on
  // a camera or screen share, with counts, an appearance log and exports.
  // Nothing here needs a chat model. options:
  //   fetch             same-origin fetch (admin token aware)
  //   canSendToChat()   whether a vision chat model could take the result
  //   sendToChat(originalURL, imageURL, context)
  //   alert(text)       optional audible/visual alert hook
  function initWorkbench(options) {
    const $ = (id) => document.getElementById(id);
    const el = {
      modelSelect: $("yoloModel"), download: $("yoloDownload"), conf: $("yoloConf"), confValue: $("yoloConfValue"),
      classes: $("yoloClasses"), pick: $("yoloPickImage"), file: $("yoloFileInput"), camera: $("yoloCamera"),
      screen: $("yoloScreen"), stop: $("yoloStop"), stage: $("yoloStage"), empty: $("yoloEmpty"), frame: $("yoloFrame"),
      image: $("yoloImage"), video: $("yoloVideo"), boxes: $("yoloBoxes"), overlay: $("yoloOverlay"),
      status: $("yoloStatus"), counts: $("yoloCounts"), table: $("yoloTable"), events: $("yoloEvents"),
      alert: $("yoloAlert"), copy: $("yoloCopyJSON"), save: $("yoloSaveImage"), send: $("yoloSendToChat"), stats: $("yoloStats")
    };
    if (!el.stage) return null;

    let models = [];
    let still = null;        // { url, canvas } for an image
    let detections = [];     // in still.canvas / lastFrame coordinates
    let selected = new Set();
    let stream = null, liveEpoch = 0, lastFrame = null, tracker = null, controller = null;
    let sequence = 0;

    const setStatus = (text, isError) => { el.status.textContent = text; el.status.classList.toggle("is-error", !!isError); };
    const plural = (n, word) => n + " " + word + (n === 1 ? "" : "s");
    const conf = () => Number(el.conf.value) / 100;
    const frameCanvas = () => (still ? still.canvas : lastFrame);

    async function refreshModels() {
      models = await listModels(options.fetch);
      fillModelSelect(el.modelSelect, models);
      el.modelSelect.value = el.modelSelect.value || preferredModel(models);
      const stock = models.find((m) => m.source === "stock" && !m.available);
      el.download.hidden = models.some((m) => m.available) || !stock;
      if (!el.download.hidden) el.download.textContent = "Download " + stock.id;
      if (!el.modelSelect.value) setStatus(el.download.hidden
        ? "No YOLO model found. Place a .pt or .safetensors checkpoint in the server's model directory."
        : "No YOLO model is downloaded yet.", el.download.hidden);
      return el.modelSelect.value;
    }

    function syncActions() {
      const have = Boolean(frameCanvas());
      el.copy.disabled = !have;
      el.save.disabled = !have;
      el.send.hidden = !(options.canSendToChat && options.canSendToChat());
      el.send.disabled = !have;
      el.send.textContent = selected.size ? "Ask the chat model about " + plural(selected.size, "region") : "Ask the chat model about this image";
      el.stop.hidden = !stream;
      el.camera.disabled = el.screen.disabled = Boolean(stream);
      el.pick.disabled = Boolean(stream);
    }

    function renderCounts(dets) {
      el.counts.replaceChildren();
      countByLabel(dets).forEach(([label, n]) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "yolo-count";
        chip.style.setProperty("--detect-color", colorFor(label));
        chip.title = "Show only " + label;
        chip.textContent = label + " × " + n;
        chip.addEventListener("click", () => { el.classes.value = label; onFilterChange(); });
        el.counts.appendChild(chip);
      });
    }

    function renderTable() {
      const body = el.table.tBodies[0];
      body.replaceChildren();
      el.table.hidden = Boolean(stream) || !detections.length;
      if (el.table.hidden) return;
      detections.slice(0, 200).forEach((d, i) => {
        const row = body.insertRow();
        row.className = selected.has(i) ? "is-selected" : "";
        row.tabIndex = 0;
        [String(i + 1), labelOf(d), percent(d.confidence),
          Math.round(d.box.x_min) + "," + Math.round(d.box.y_min) + " " + Math.round(d.box.x_max - d.box.x_min) + "×" + Math.round(d.box.y_max - d.box.y_min)]
          .forEach((text) => { row.insertCell().textContent = text; });
        row.addEventListener("click", () => toggle(i));
        row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(i); } });
      });
    }

    function renderStill() {
      renderBoxButtons(el.boxes, detections, still ? still.canvas.width : 0, still ? still.canvas.height : 0, selected, toggle);
      renderCounts(detections);
      renderTable();
      syncActions();
    }

    function toggle(i) {
      if (selected.has(i)) selected.delete(i);
      else if (selected.size < MAX_SELECTED) selected.add(i);
      else { setStatus("At most " + MAX_SELECTED + " regions can be handed to the chat at once.", true); return; }
      renderStill();
    }

    function showStage(kind) {
      el.empty.hidden = kind !== "empty";
      el.frame.hidden = kind === "empty";
      el.image.hidden = kind !== "still";
      el.boxes.hidden = kind !== "still";
      el.video.hidden = kind !== "live";
      el.overlay.hidden = kind !== "live";
      el.events.hidden = kind !== "live";
      el.alert.closest("label").hidden = kind !== "live";
    }

    async function detectStill() {
      if (!still) return;
      const seq = ++sequence;
      const model = el.modelSelect.value || await refreshModels();
      if (!model) return;
      setStatus("Detecting objects…", false);
      try {
        const small = toCanvas(still.canvas, still.canvas.width, still.canvas.height, DETECT_EDGE);
        const t0 = performance.now();
        const found = await detect(options.fetch, model, await canvasBlob(small, "image/jpeg", 0.9), { conf: conf(), classes: parseClassList(el.classes.value) });
        if (seq !== sequence) return;
        detections = scaleDetections(found, still.canvas.width / small.width);
        selected = new Set();
        el.stats.textContent = Math.round(performance.now() - t0) + " ms · " + still.canvas.width + "×" + still.canvas.height;
        setStatus(detections.length ? plural(detections.length, "object") + " found." : "No objects found at this confidence.", false);
        renderStill();
      } catch (err) {
        if (seq === sequence) setStatus(err.message || String(err), true);
      }
    }

    async function openImage(file) {
      if (!file || !/^image\//.test(file.type)) { setStatus("Choose a JPEG or PNG image.", true); return; }
      await stopLive();
      const url = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
      const img = await loadImage(url);
      still = { url, canvas: toCanvas(img, img.naturalWidth, img.naturalHeight, SOURCE_EDGE) };
      detections = [];
      selected = new Set();
      el.image.src = url;
      showStage("still");
      renderStill();
      await detectStill();
    }

    async function startLive(kind) {
      const model = el.modelSelect.value || await refreshModels();
      if (!model) return;
      try {
        stream = kind === "screen"
          ? await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false })
          : await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      } catch (err) {
        setStatus((kind === "screen" ? "Screen sharing" : "The camera") + " is not available: " + (err.message || err.name), true);
        return;
      }
      stream.getVideoTracks().forEach((t) => t.addEventListener("ended", () => stopLive()));
      still = null;
      detections = [];
      selected = new Set();
      tracker = createCountTracker(2);
      el.events.replaceChildren();
      el.video.srcObject = stream;
      await el.video.play().catch(() => {});
      showStage("live");
      renderCounts([]);
      renderTable();
      syncActions();
      setStatus("Live detection running. No language model is involved.", false);
      liveLoop(++liveEpoch);
    }

    async function liveLoop(epoch) {
      let fps = 0;
      while (stream && epoch === liveEpoch) {
        const video = el.video;
        if (!video.videoWidth) { await sleep(100); continue; }
        const t0 = performance.now();
        try {
          const frame = toCanvas(video, video.videoWidth, video.videoHeight, DETECT_EDGE);
          controller = new AbortController();
          const dets = await detect(options.fetch, el.modelSelect.value, await canvasBlob(frame, "image/jpeg", 0.8),
            { conf: conf(), classes: parseClassList(el.classes.value), signal: controller.signal });
          if (epoch !== liveEpoch) return;
          lastFrame = frame;
          detections = dets;
          drawOverlay(el.overlay, dets, frame.width, frame.height, "fill");
          renderCounts(dets);
          logEvents(tracker(dets));
          const ms = performance.now() - t0;
          fps = fps ? fps * 0.8 + 200 / ms : 1000 / ms;
          el.stats.textContent = Math.round(ms) + " ms · " + fps.toFixed(1) + " fps";
          syncActions();
        } catch (err) {
          if (err && err.name === "AbortError") return;
          setStatus(err.message || String(err), true);
          await sleep(1000);
        }
      }
    }

    function logEvents(events) {
      if (!events.length) return;
      const time = new Date().toLocaleTimeString();
      events.forEach((e) => {
        const item = document.createElement("li");
        const text = describeEvent(e);
        item.textContent = time + "  " + text;
        el.events.prepend(item);
        if (e.to > e.from && el.alert.checked && options.alert) options.alert(text);
      });
      while (el.events.children.length > 100) el.events.lastChild.remove();
    }

    async function stopLive() {
      liveEpoch++;
      if (controller) controller.abort();
      controller = null;
      if (stream) stream.getTracks().forEach((t) => t.stop());
      stream = null;
      el.video.srcObject = null;
      const ctx = el.overlay.getContext("2d");
      ctx.clearRect(0, 0, el.overlay.width, el.overlay.height);
      if (!still) {
        // Keep the last frame and its boxes as a still, so it can still be
        // exported or handed to the chat after stopping.
        if (lastFrame) {
          still = { url: lastFrame.toDataURL("image/jpeg", 0.92), canvas: lastFrame };
          el.image.src = still.url;
          showStage("still");
          renderStill();
          setStatus("Stopped. The last frame is kept below.", false);
        } else {
          showStage("empty");
        }
      }
      lastFrame = null;
      syncActions();
    }

    function payload() {
      const frame = frameCanvas();
      return JSON.stringify({ model: el.modelSelect.value, width: frame.width, height: frame.height, detections }, null, 2);
    }

    let filterTimer = null;
    function onFilterChange() {
      el.confValue.textContent = el.conf.value + "%";
      clearTimeout(filterTimer);
      if (!stream) filterTimer = setTimeout(detectStill, 250);
    }

    el.pick.addEventListener("click", () => el.file.click());
    el.file.addEventListener("change", () => { const f = el.file.files[0]; el.file.value = ""; openImage(f).catch((e) => setStatus(e.message || String(e), true)); });
    el.stage.addEventListener("dragover", (e) => { e.preventDefault(); el.stage.classList.add("is-drop"); });
    el.stage.addEventListener("dragleave", () => el.stage.classList.remove("is-drop"));
    el.stage.addEventListener("drop", (e) => {
      e.preventDefault();
      el.stage.classList.remove("is-drop");
      const f = e.dataTransfer && e.dataTransfer.files[0];
      if (f) openImage(f).catch((err) => setStatus(err.message || String(err), true));
    });
    el.camera.addEventListener("click", () => startLive("camera"));
    el.screen.addEventListener("click", () => startLive("screen"));
    el.stop.addEventListener("click", () => stopLive());
    el.modelSelect.addEventListener("change", () => { rememberModel(el.modelSelect.value); if (!stream) detectStill(); });
    el.conf.addEventListener("input", onFilterChange);
    el.classes.addEventListener("input", onFilterChange);
    el.download.addEventListener("click", async () => {
      const stock = models.find((m) => m.source === "stock" && !m.available);
      if (!stock) return;
      el.download.disabled = true;
      try {
        await downloadStock(options.fetch, stock, (p) => setStatus("Downloading " + stock.id + "… " + Math.round(p * 100) + "%", false));
        rememberModel(stock.id);
        await refreshModels();
        setStatus(stock.id + " is ready.", false);
        detectStill();
      } catch (err) {
        setStatus(err.message || String(err), true);
      } finally {
        el.download.disabled = false;
      }
    });
    el.copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(payload());
        setStatus("Detections copied as JSON.", false);
      } catch (_) {
        setStatus("The clipboard is not available in this browser.", true);
      }
    });
    el.save.addEventListener("click", async () => {
      const frame = frameCanvas();
      if (!frame) return;
      const blob = await canvasBlob(annotatedCanvas(frame, detections), "image/png");
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "detections.png";
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
    el.send.addEventListener("click", () => {
      const frame = frameCanvas();
      if (!frame || !options.sendToChat) return;
      const original = still ? still.url : frame.toDataURL("image/jpeg", 0.92);
      const picked = Array.from(selected).sort((a, b) => a - b).map((i) => detections[i]);
      if (picked.length) options.sendToChat(original, selectionDataURL(frame, picked), describeSelection(picked, frame.width, frame.height, "upload"));
      else options.sendToChat(original, original, summarizeDetections(detections));
    });

    return {
      async open() {
        showStage(still ? "still" : "empty");
        syncActions();
        try { await refreshModels(); } catch (err) { setStatus(err.message || String(err), true); }
      },
      close() { return stopLive(); }
    };
  }

  function sleep(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

  global.GopherLLMVision = {
    init, initWorkbench, listModels, detect, fillModelSelect, preferredModel, rememberModel,
    toCanvas, canvasBlob, selectionDataURL, drawOverlay, annotatedCanvas,
    cropRect, describePosition, describeSelection, collageLayout, parseClassList, iou, sceneChanged, strongest, scaleDetections,
    countByLabel, summarizeDetections, createCountTracker, describeEvent,
    DETECT_EDGE, MAX_SELECTED
  };
})(globalThis);
