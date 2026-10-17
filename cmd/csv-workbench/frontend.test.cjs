// Model changes must never mix classifications or retain stale explanations.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
function setup(fetchImpl = () => new Promise(() => {})) {
  const nodes = new Map();
  function node() {
    return {
      value: "",
      hidden: false,
      disabled: false,
      textContent: "",
      children: [],
      append(...items) {
        this.children.push(...items);
      },
      replaceChildren(...items) {
        this.children = items;
      },
      add(item) {
        this.children.push(item);
      },
      addEventListener() {},
      setAttribute() {},
    };
  }
  const context = vm.createContext({
    document: {
      getElementById(id) {
        if (!nodes.has(id)) nodes.set(id, node());
        return nodes.get(id);
      },
      createElement: node,
    },
    Option: function (label, value) {
      return { label, value };
    },
    fetch: fetchImpl,
    FormData: class {},
    AbortController,
    URL,
    setTimeout,
  });
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, "web/app.js"), "utf8"),
    context,
  );
  return { context, nodes, run: (code) => vm.runInContext(code, context) };
}
function snapshot(layaVersion, ministralVersion, laya = true) {
  return JSON.stringify({
    laya,
    ministral: ministralVersion > 0,
    laya_id: laya ? "laya" : "",
    ministral_id: ministralVersion ? "assistant" : "",
    laya_version: layaVersion,
    ministral_version: ministralVersion,
    options: [
      { kind: "laya", id: "laya", name: "Classifier" },
      { kind: "ministral", id: "assistant", name: "Assistant" },
    ],
  });
}
test("assistant switch keeps corrected classifications but clears explanations", () => {
  const { run, nodes } = setup();
  run(
    `applyModels(${snapshot(1, 1)});results=[{answer:{choice:'Technik'},override:'Abrechnung',explanation:'old'}];`,
  );
  run(`applyModels(${snapshot(1, 2)});`);
  assert.equal(run("results.length"), 1);
  assert.equal(run("current(0)"), "Abrechnung");
  assert.equal(run("results[0].explanation"), undefined);
  assert.equal(nodes.get("explanation").textContent, "");
});
test("Laya switch clears results while preserving the imported dataset", () => {
  const { run, nodes } = setup();
  run(
    `applyModels(${snapshot(1, 1)});data={headers:['text'],rows:[['hello']]};results=[{answer:{choice:'Technik'},override:'Abrechnung'}];selected=0;`,
  );
  run(`applyModels(${snapshot(2, 1)});`);
  assert.equal(run("results.length"), 0);
  assert.equal(run("data.rows[0][0]"), "hello");
  assert.equal(run("selected"), -1);
  assert.equal(nodes.get("detail").hidden, true);
  assert.equal(nodes.get("export").disabled, true);
});
test("unloaded classifier and busy state disable inference and model switching", () => {
  const { run, nodes } = setup();
  run(`applyModels(${snapshot(0, 0, false)});`);
  assert.equal(nodes.get("preview").disabled, true);
  assert.equal(nodes.get("run").disabled, true);
  assert.equal(nodes.get("suggest").disabled, true);
  run(`applyModels(${snapshot(1, 1)});setBusy(true);`);
  for (const id of [
    "preview",
    "run",
    "loadLaya",
    "loadAssistant",
    "layaModel",
    "assistantModel",
    "suggest",
  ])
    assert.equal(nodes.get(id).disabled, true, id);
  run("setBusy(false)");
  assert.equal(nodes.get("preview").disabled, false);
  assert.equal(nodes.get("suggest").disabled, false);
});

test("rejected task suggestion preserves existing task and corrected results", async () => {
  const { run, nodes } = setup((path) =>
    path === "/api/assist"
      ? Promise.resolve({
          ok: false,
          status: 502,
          text: async () => "Ungültiger Aufgabenvorschlag",
        })
      : new Promise(() => {}),
  );
  run(
    `applyModels(${snapshot(1, 1)});data={headers:['text'],rows:[['hello']]};results=[{answer:{choice:'Technik'},override:'Abrechnung'}];selected=0;`,
  );
  nodes.get("task").value = "Existing task";
  nodes.get("criteria").value = '["Technik","Abrechnung"]';
  await nodes.get("suggest").onclick();
  assert.equal(nodes.get("task").value, "Existing task");
  assert.equal(nodes.get("criteria").value, '["Technik","Abrechnung"]');
  assert.equal(run("current(0)"), "Abrechnung");
  assert.equal(run("results.length"), 1);
  assert.equal(run("selected"), 0);
  assert.equal(nodes.get("status").textContent, "Ungültiger Aufgabenvorschlag");
  assert.equal(run("busy"), false);
});
