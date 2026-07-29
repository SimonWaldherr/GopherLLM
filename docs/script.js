(() => {
  "use strict";

  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const copyStatus = document.getElementById("copyStatus");
  const tabs = Array.from(document.querySelectorAll("[data-api-tab]"));
  const panels = Array.from(document.querySelectorAll("[data-api-panel]"));
  let toastTimer = 0;

  function preferredTheme() {
    const saved = localStorage.getItem("gopherllm-pages-theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }

  function setTheme(theme) {
    root.dataset.theme = theme;
    themeToggle.textContent = theme === "dark" ? "Light theme" : "Dark theme";
    themeToggle.setAttribute("aria-label", "Switch to " + (theme === "dark" ? "light" : "dark") + " theme");
  }

  function showCopyStatus() {
    window.clearTimeout(toastTimer);
    copyStatus.hidden = false;
    toastTimer = window.setTimeout(() => {
      copyStatus.hidden = true;
    }, 1800);
  }

  async function copyText(targetID, button) {
    const target = document.getElementById(targetID);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.textContent);
      const original = button.textContent;
      button.textContent = "Copied";
      showCopyStatus();
      window.setTimeout(() => {
        button.textContent = original;
      }, 1600);
    } catch (_) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  }

  function setAPITab(name, moveFocus) {
    const selected = tabs.find((tab) => tab.dataset.apiTab === name) || tabs[0];
    tabs.forEach((tab) => {
      const active = tab === selected;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.apiPanel !== selected.dataset.apiTab;
    });
    if (moveFocus) selected.focus();
  }

  setTheme(preferredTheme());
  document.getElementById("year").textContent = String(new Date().getFullYear());

  themeToggle.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("gopherllm-pages-theme", next);
    setTheme(next);
  });

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", () => copyText(button.dataset.copyTarget, button));
  });

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setAPITab(tab.dataset.apiTab, false));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const next = (index + direction + tabs.length) % tabs.length;
      setAPITab(tabs[next].dataset.apiTab, true);
    });
  });
})();
