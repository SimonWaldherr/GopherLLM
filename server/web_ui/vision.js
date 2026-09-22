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

  function canvasBlob(canvas, type = "image/jpeg", quality = 0.9) {
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
      boxes.replaceChildren();
      boxes.classList.toggle("is-crowded", detections.length > CROWDED);
      boxes.classList.toggle("has-selection", selected.size > 0);
      if (!canvas) return;
      // Largest first, so a small box (a bicycle inside its rider's box)
      // is drawn on top and stays clickable.
      const area = (d) => (d.box.x_max - d.box.x_min) * (d.box.y_max - d.box.y_min);
      const order = detections.map((d, i) => i).sort((a, b) => area(detections[b]) - area(detections[a]));
      order.forEach((i) => {
        const d = detections[i];
        const b = document.createElement("button");
        b.type = "button";
        b.className = "detect-box" + (selected.has(i) ? " is-selected" : "");
        b.style.left = (100 * d.box.x_min / canvas.width) + "%";
        b.style.top = (100 * d.box.y_min / canvas.height) + "%";
        b.style.width = (100 * (d.box.x_max - d.box.x_min) / canvas.width) + "%";
        b.style.height = (100 * (d.box.y_max - d.box.y_min) / canvas.height) + "%";
        b.setAttribute("aria-pressed", String(selected.has(i)));
        b.setAttribute("aria-label", labelOf(d) + " " + percent(d.confidence));
        const tag = document.createElement("span");
        tag.className = "detect-box-label";
        tag.textContent = labelOf(d) + " " + percent(d.confidence);
        b.appendChild(tag);
        b.addEventListener("click", () => toggle(i));
        boxes.appendChild(b);
      });
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

  global.GopherLLMVision = {
    init, listModels, detect, fillModelSelect, preferredModel, rememberModel,
    toCanvas, canvasBlob, selectionDataURL,
    cropRect, describePosition, describeSelection, collageLayout, parseClassList, iou, sceneChanged, strongest, scaleDetections,
    DETECT_EDGE, MAX_SELECTED
  };
})(globalThis);
