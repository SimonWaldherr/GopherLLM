"use strict";
const $ = (id) => document.getElementById(id);
let data = null,
  results = [],
  selected = -1,
  page = 0,
  busy = false,
  controller = null,
  ministral = false,
  revision = 0;
let modelState = null;
const pageSize = 50;
const text = (tag, value) => {
  const node = document.createElement(tag);
  node.textContent = value;
  return node;
};
const separator = () =>
  $("delimiter").value === "tab" ? "\t" : $("delimiter").value;
const status = (value) => {
  $("status").textContent = value;
};
async function api(path, body, signal) {
  const response = await fetch("/api/" + path, {
    method: "POST",
    body: body instanceof FormData ? body : JSON.stringify(body),
    headers:
      body instanceof FormData ? {} : { "Content-Type": "application/json" },
    signal,
  });
  if (!response.ok) {
    const error = new Error(await response.text());
    error.status = response.status;
    throw error;
  }
  return response.headers.get("Content-Type")?.includes("application/json")
    ? response.json()
    : response.blob();
}
function question() {
  const q = { type: $("type").value, instructions: $("task").value.trim() };
  if (!q.instructions) throw new Error("Bitte eine Frage eingeben.");
  if (q.type !== "noul") q.criteria = JSON.parse($("criteria").value);
  return q;
}
function state(i) {
  const row = data.rows[i],
    col = $("column").value;
  if (col !== "") return row[Number(col)];
  return Object.fromEntries(data.headers.map((h, n) => [h, row[n]]));
}
function value(answer) {
  if (answer.choice !== undefined) return answer.choice;
  if (answer.score !== undefined) return String(answer.score);
  if (answer.noul !== undefined) return String(answer.noul);
  throw new Error("Modell lieferte kein Ergebnis.");
}
function current(i) {
  return results[i]?.override ?? (results[i] ? value(results[i].answer) : "");
}
function setBusy(value) {
  busy = value;
  for (const id of [
    "import",
    "file",
    "delimiter",
    "noHeader",
    "column",
    "task",
    "type",
    "criteria",
    "resultColumn",
    "preview",
    "run",
    "saveCorrection",
    "resetCorrection",
    "layaModel",
    "assistantModel",
    "loadLaya",
    "loadAssistant",
    "refreshModels",
  ])
    $(id).disabled = value;
  $("preview").disabled = value || !modelState?.laya;
  $("run").disabled = value || !modelState?.laya;
  $("suggest").disabled = value || !ministral;
  $("explain").disabled = value || !ministral || !results[selected];
  $("cancel").hidden = !value;
  $("export").disabled = value || !results.some(Boolean);
}
function resetResults(
  message = "Aufgabe geändert. Vorhandene Ergebnisse wurden zurückgesetzt.",
) {
  revision++;
  results = [];
  selected = -1;
  page = 0;
  $("detail").hidden = true;
  $("explanation").textContent = "";
  $("progress").value = 0;
  render();
  status(message);
}
function render() {
  const done = results.filter(Boolean).length,
    corrected = results.filter((r) => r?.override !== undefined).length;
  $("summary").textContent = data
    ? data.rows.length +
      " Datensätze · " +
      done +
      " verarbeitet · " +
      corrected +
      " korrigiert"
    : "Noch keine Datei geladen.";
  $("thead").replaceChildren();
  $("tbody").replaceChildren();
  if (!data) return;
  const head = document.createElement("tr");
  for (const h of [
    "Zeile",
    ...data.headers,
    $("resultColumn").value || "Ergebnis",
    "Modellwahrscheinlichkeit",
    "Status",
  ])
    head.append(text("th", h));
  $("thead").append(head);
  const threshold = Number($("threshold").value) / 100,
    filter = $("filter").value;
  const indices = data.rows
    .map((_, i) => i)
    .filter(
      (i) =>
        filter === "all" ||
        (filter === "pending" && !results[i]) ||
        (filter === "corrected" && results[i]?.override !== undefined) ||
        (filter === "uncertain" &&
          results[i] &&
          results[i].answer.answer_confidence < threshold),
    );
  page = Math.min(page, Math.max(0, Math.ceil(indices.length / pageSize) - 1));
  for (const i of indices.slice(page * pageSize, (page + 1) * pageSize)) {
    const tr = document.createElement("tr");
    if (i === selected) tr.className = "selected";
    const cell = document.createElement("td"),
      open = text("button", String(i + 1));
    open.setAttribute("aria-label", "Datensatz " + (i + 1) + " öffnen");
    open.onclick = () => showDetail(i);
    cell.append(open);
    tr.append(cell);
    for (const v of data.rows[i])
      tr.append(text("td", v.length > 160 ? v.slice(0, 160) + " …" : v));
    tr.append(text("td", current(i) || "—"));
    const probability = text(
      "td",
      results[i]
        ? (100 * results[i].answer.answer_confidence).toFixed(1) + " %"
        : "—",
    );
    if (results[i]?.answer.answer_confidence < threshold)
      probability.className = "uncertain";
    tr.append(
      probability,
      text(
        "td",
        !results[i]
          ? "Offen"
          : results[i].override !== undefined
            ? "Korrigiert"
            : "Modellvorschlag",
      ),
    );
    $("tbody").append(tr);
  }
  $("page").textContent = indices.length
    ? page * pageSize +
      1 +
      "–" +
      Math.min(indices.length, (page + 1) * pageSize) +
      " von " +
      indices.length
    : "Keine passenden Datensätze";
  $("prev").disabled = page === 0;
  $("next").disabled = (page + 1) * pageSize >= indices.length;
  $("export").disabled = busy || !done;
}
function showDetail(i) {
  selected = i;
  $("detail").hidden = false;
  $("detailTitle").textContent = "Datensatz " + (i + 1);
  $("original").textContent = JSON.stringify(
    Object.fromEntries(data.headers.map((h, n) => [h, data.rows[i][n]])),
    null,
    2,
  );
  $("modelAnswer").textContent = results[i]
    ? value(results[i].answer)
    : "Noch nicht verarbeitet";
  $("probabilities").replaceChildren();
  for (const [label, p] of Object.entries(
    results[i]?.answer.probabilities || {},
  ).sort((a, b) => b[1] - a[1]))
    $("probabilities").append(
      text("li", label + ": " + (100 * p).toFixed(1) + " %"),
    );
  $("correction").value = current(i);
  $("correction").disabled = busy || !results[i];
  $("saveCorrection").disabled = busy || !results[i];
  $("resetCorrection").disabled = busy || !results[i];
  $("explain").disabled = busy || !ministral || !results[i];
  $("explanation").textContent = results[i]?.explanation || "";
  render();
}
$("import").onclick = async () => {
  const file = $("file").files[0];
  if (!file) {
    status("Bitte eine CSV-Datei wählen.");
    return;
  }
  if (file.size > 32 * 1024 * 1024) {
    status("Die Datei überschreitet 32 MiB.");
    return;
  }
  controller = new AbortController();
  setBusy(true);
  status("Datei wird eingelesen …");
  const body = new FormData();
  body.append("file", file);
  body.append("delimiter", separator());
  body.append("no_header", $("noHeader").checked);
  try {
    const imported = await api("import", body, controller.signal);
    data = {
      ...imported,
      delimiter: separator(),
      noHeader: $("noHeader").checked,
    };
    revision++;
    results = [];
    selected = -1;
    page = 0;
    $("detail").hidden = true;
    $("column").replaceChildren(new Option("Kompletter Datensatz", ""));
    data.headers.forEach((h, i) => $("column").add(new Option(h, String(i))));
    const index = data.headers.indexOf("text");
    if (index >= 0) $("column").value = String(index);
    $("progress").value = 0;
    render();
    status(file.name + " geladen. Teste zunächst fünf Zeilen.");
  } catch (e) {
    status(e.name === "AbortError" ? "Import abgebrochen." : e.message);
  } finally {
    controller = null;
    setBusy(false);
  }
};
for (const id of ["task", "type", "criteria", "column"])
  $(id).addEventListener("change", () => {
    if (id === "type") $("criteriaLabel").hidden = $("type").value === "noul";
    if (data) resetResults();
  });
$("resultColumn").addEventListener("input", render);
for (const id of ["filter", "threshold"])
  $(id).addEventListener("change", () => {
    page = 0;
    render();
  });
$("prev").onclick = () => {
  page--;
  render();
};
$("next").onclick = () => {
  page++;
  render();
};
async function run(limit) {
  if (!data) {
    status("Bitte zuerst eine Datei laden.");
    return;
  }
  let q;
  try {
    q = question();
  } catch (e) {
    status(e.message);
    return;
  }
  controller = new AbortController();
  setBusy(true);
  let count = 0;
  const indices = data.rows
    .map((_, i) => i)
    .slice(0, limit)
    .filter((i) => !results[i]);
  $("progress").max = indices.length || 1;
  $("progress").value = 0;
  try {
    for (const i of indices) {
      status(
        "Datensatz " +
          (i + 1) +
          " von " +
          Math.min(limit, data.rows.length) +
          " wird verarbeitet …",
      );
      const response = await api(
        "classify",
        {
          state: state(i),
          question: q,
          model_version: modelState.laya_version,
        },
        controller.signal,
      );
      const answer = response.answers.result;
      value(answer);
      results[i] = { answer };
      $("progress").value = ++count;
      render();
    }
    status(
      limit === 5
        ? "Vorschau fertig. Prüfe die Ergebnisse und starte anschließend die Gesamtverarbeitung."
        : "Verarbeitung abgeschlossen. Ergebnisse können geprüft und exportiert werden.",
    );
  } catch (e) {
    if (e.status === 409) await refreshModels().catch(() => {});
    status(
      e.name === "AbortError"
        ? "Abgebrochen. Bereits verarbeitete Zeilen bleiben erhalten."
        : "Verarbeitung gestoppt: " + e.message,
    );
  } finally {
    controller = null;
    setBusy(false);
    if (selected >= 0) showDetail(selected);
  }
}
$("preview").onclick = () => run(5);
$("run").onclick = () => run(data?.rows.length || 0);
$("cancel").onclick = () => controller?.abort();
$("saveCorrection").onclick = () => {
  if (!results[selected]) return;
  const v = $("correction").value.trim();
  if (!v) {
    status("Bitte ein Ergebnis eingeben.");
    return;
  }
  try {
    const q = question();
    if (q.type === "choice") {
      const labels = Array.isArray(q.criteria)
        ? q.criteria
        : Object.keys(q.criteria);
      if (!labels.includes(v))
        throw new Error("Bitte eine definierte Kategorie verwenden.");
    } else {
      const n = Number(v);
      const max = q.type === "noul" ? 1 : q.criteria.length - 1;
      if (!Number.isFinite(n) || n < 0 || n > max)
        throw new Error("Wert muss zwischen 0 und " + max + " liegen.");
    }
  } catch (e) {
    status(e.message);
    return;
  }
  if (v === value(results[selected].answer)) delete results[selected].override;
  else results[selected].override = v;
  render();
  status("Korrektur übernommen.");
};
$("resetCorrection").onclick = () => {
  if (results[selected]) {
    delete results[selected].override;
    showDetail(selected);
  }
};
$("suggest").onclick = async () => {
  controller = new AbortController();
  setBusy(true);
  status("Ministral erstellt einen Aufgabenvorschlag …");
  try {
    const response = await api(
      "assist",
      {
        mode: "suggest",
        task: $("task").value,
        model_version: modelState.ministral_version,
      },
      controller.signal,
    );
    $("task").value = response.instructions;
    $("criteria").value = JSON.stringify(response.criteria, null, 2);
    $("type").value = "choice";
    $("criteriaLabel").hidden = false;
    resetResults();
    status(
      "Vorschlag übernommen. Prüfe Frage und Kategorien vor der Verarbeitung.",
    );
  } catch (e) {
    status(e.name === "AbortError" ? "Abgebrochen." : e.message);
  } finally {
    controller = null;
    setBusy(false);
  }
};
$("explain").onclick = async () => {
  const i = selected,
    rev = revision;
  controller = new AbortController();
  setBusy(true);
  $("explanation").textContent = "Ministral formuliert eine Interpretation …";
  try {
    const response = await api(
      "assist",
      {
        mode: "explain",
        task: $("task").value,
        state: state(i),
        answer: results[i].answer,
        model_version: modelState.ministral_version,
        laya_version: modelState.laya_version,
      },
      controller.signal,
    );
    if (rev === revision) {
      results[i].explanation = response.text;
      if (selected === i) $("explanation").textContent = response.text;
    }
  } catch (e) {
    if (selected === i)
      $("explanation").textContent =
        e.name === "AbortError" ? "Abgebrochen." : e.message;
  } finally {
    controller = null;
    setBusy(false);
  }
};
$("export").onclick = async () => {
  const name = $("resultColumn").value.trim();
  const extra = [
    name,
    name + "_model",
    name + "_probability",
    name + "_corrected",
  ];
  if (!name || extra.some((h) => data.headers.includes(h))) {
    status("Bitte einen eindeutigen Namen für die Ergebnisspalte wählen.");
    return;
  }
  const headers = [...data.headers, ...extra],
    rows = data.rows.map((row, i) => [
      ...row,
      current(i),
      results[i] ? value(results[i].answer) : "",
      results[i] ? String(results[i].answer.answer_confidence) : "",
      results[i] ? String(results[i].override !== undefined) : "",
    ]);
  controller = new AbortController();
  setBusy(true);
  try {
    const blob = await api(
      "export",
      { headers, rows, delimiter: data.delimiter, no_header: data.noHeader },
      controller.signal,
    );
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = "classified.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status(
      "CSV exportiert. Noch offene Zeilen enthalten leere Ergebnisfelder.",
    );
  } catch (e) {
    status(e.name === "AbortError" ? "Abgebrochen." : e.message);
  } finally {
    controller = null;
    setBusy(false);
  }
};
function applyModels(snapshot) {
  const previous = modelState;
  modelState = snapshot;
  ministral = snapshot.ministral;
  if (previous && previous.laya_version !== snapshot.laya_version)
    resetResults(
      "Laya-Modell gewechselt. Ergebnisse wurden zurückgesetzt; die CSV bleibt geladen.",
    );
  if (previous && previous.ministral_version !== snapshot.ministral_version) {
    for (const result of results) if (result) delete result.explanation;
    $("explanation").textContent = "";
  }
  for (const [id, kind, active, empty] of [
    ["layaModel", "laya", snapshot.laya_id, "Kein Laya-Modell"],
    [
      "assistantModel",
      "ministral",
      snapshot.ministral_id,
      "Ohne Textassistent",
    ],
  ]) {
    const select = $(id);
    select.replaceChildren(new Option(empty, ""));
    for (const item of snapshot.options.filter(
      (option) => option.kind === kind,
    )) {
      const size = item.size_bytes
        ? " · " + (item.size_bytes / 1024 ** 3).toFixed(2) + " GiB"
        : "";
      select.add(new Option(item.name + size, item.id));
    }
    select.value = active || "";
  }
  const name = (id) =>
    snapshot.options.find((item) => item.id === id)?.name || "Nicht geladen";
  $("models").textContent =
    (snapshot.laya ? "Laya bereit" : "Laya nicht geladen") +
    " · " +
    (ministral ? "Textassistent bereit" : "Textassistent nicht geladen");
  $("modelStatus").textContent =
    "Laya: " +
    name(snapshot.laya_id) +
    "\nTextassistent: " +
    name(snapshot.ministral_id);
  setBusy(busy);
}
async function refreshModels() {
  const response = await fetch("/api/status");
  if (!response.ok) throw new Error("Modellstatus nicht erreichbar");
  applyModels(await response.json());
}
async function chooseModel(kind) {
  const id = $(kind === "laya" ? "layaModel" : "assistantModel").value;
  controller = new AbortController();
  setBusy(true);
  $("modelCancel").hidden = false;
  $("modelStatus").textContent = id
    ? "Modell wird geladen … Dies kann beim ersten Download einige Minuten dauern."
    : "Modell wird entladen …";
  let message;
  try {
    const snapshot = await api(
      "models/select",
      { kind, id },
      controller.signal,
    );
    applyModels(snapshot);
    message = id ? "Modell ist bereit." : "Modell wurde entladen.";
  } catch (e) {
    message =
      e.name === "AbortError"
        ? "Modellwechsel abgebrochen. Aktueller Zustand wird geprüft."
        : "Modell konnte nicht geladen werden: " + e.message;
  } finally {
    controller = null;
    $("modelCancel").hidden = true;
    try {
      await refreshModels();
    } catch (e) {
      message += " " + e.message;
    }
    setBusy(false);
    $("modelStatus").textContent =
      message + "\n" + $("modelStatus").textContent;
  }
}
$("modelCancel").onclick = () => controller?.abort();
$("loadLaya").onclick = () => chooseModel("laya");
$("loadAssistant").onclick = () => chooseModel("ministral");
$("refreshModels").onclick = async () => {
  setBusy(true);
  try {
    await refreshModels();
  } catch (e) {
    $("modelStatus").textContent = e.message;
  } finally {
    setBusy(false);
  }
};
setBusy(false);
refreshModels().catch((e) => {
  $("modelStatus").textContent = e.message;
  status(e.message);
});
