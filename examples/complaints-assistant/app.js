const orders = [
  { number: "NS-10482", product: "Northstar Echo Mini speaker", purchased_at: "12 Jul 2026", price: "€79.00", warranty_ends: "12 Jul 2028" },
  { number: "NS-09831", product: "Northstar Orbit USB-C charger", purchased_at: "03 Mar 2026", price: "€29.00", warranty_ends: "03 Mar 2028" },
  { number: "NS-09117", product: "Northstar Beam desk lamp", purchased_at: "18 Dec 2025", price: "€49.00", warranty_ends: "18 Dec 2027" }
];
let selectedOrder = null;
const $ = (id) => document.getElementById(id);
const ordersEl = $("orders"), form = $("claimForm"), submit = $("submit"), error = $("formError"), status = $("status"), result = $("result");

function renderOrders() {
  ordersEl.replaceChildren(...orders.map((order) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "order"; button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(selectedOrder === order.number));
    const dot = document.createElement("span"); dot.setAttribute("aria-hidden", "true"); dot.textContent = selectedOrder === order.number ? "●" : "○";
    const label = document.createElement("span");
    const name = document.createElement("b"); name.textContent = order.product;
    const meta = document.createElement("small"); meta.textContent = order.number + " · bought " + order.purchased_at;
    label.append(name, meta);
    const price = document.createElement("span"); price.className = "price"; price.textContent = order.price;
    button.replaceChildren(dot, label, price);
    button.addEventListener("click", () => { selectedOrder = order.number; renderOrders(); });
    return button;
  }));
}

function showResult(data) {
  $("classification").textContent = data.classification || "Needs review";
  const priority = $("priority"); priority.textContent = data.priority || "needs_human_review";
  priority.classList.toggle("safety", data.priority === "safety_review");
  $("actions").replaceChildren(...(Array.isArray(data.recommended_actions) ? data.recommended_actions : ["Review the customer description."]).map((action) => { const item = document.createElement("li"); item.textContent = action; return item; }));
  $("reply").textContent = data.customer_reply || "A human agent needs to prepare a reply.";
  const safety = $("safety"); safety.textContent = data.safety_notice || ""; safety.hidden = !data.safety_notice;
  result.hidden = false; status.hidden = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault(); error.textContent = "";
  const order = orders.find((item) => item.number === selectedOrder);
  const details = $("details").value.trim();
  if (!order || !details) { error.textContent = "Choose an order and describe the issue."; return; }
  submit.disabled = true; submit.textContent = "Preparing local draft…"; status.hidden = false; status.textContent = "Contacting the local GopherLLM server…"; result.hidden = true;
  try {
    const response = await fetch("/api/assist", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ order, issue_type: $("issueType").value, details }) });
    if (!response.ok) throw new Error((await response.text()) || "The local model did not respond.");
    showResult(await response.json());
  } catch (cause) {
    status.textContent = "No draft was generated."; error.textContent = cause.message || String(cause);
  } finally { submit.disabled = false; submit.textContent = "Ask local support assistant"; }
});
renderOrders();
