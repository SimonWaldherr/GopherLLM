(() => {
  "use strict";

  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const themeToggleLabel = themeToggle.querySelector(".theme-toggle-label");
  const navToggle = document.getElementById("navToggle");
  const primaryNav = document.getElementById("primaryNav");
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
    themeToggleLabel.textContent = theme === "dark" ? "Light theme" : "Dark theme";
    themeToggle.setAttribute("aria-label", "Switch to " + (theme === "dark" ? "light" : "dark") + " theme");
  }

  function setNavigation(open, restoreFocus) {
    if (!navToggle || !primaryNav) return;
    primaryNav.classList.toggle("is-open", open);
    navToggle.setAttribute("aria-expanded", String(open));
    navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    navToggle.textContent = open ? "Close" : "Menu";
    if (open) {
      primaryNav.querySelector("a").focus();
    } else if (restoreFocus) {
      navToggle.focus();
    }
  }

  function showCopyStatus(message) {
    window.clearTimeout(toastTimer);
    copyStatus.textContent = message || "Copied to clipboard";
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
      showCopyStatus("Code selected — copy it");
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

  if (navToggle && primaryNav) {
    navToggle.addEventListener("click", () => {
      const opening = !primaryNav.classList.contains("is-open");
      setNavigation(opening, !opening);
    });

    primaryNav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => setNavigation(false));
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && primaryNav.classList.contains("is-open")) {
        setNavigation(false, true);
      }
    });

    document.addEventListener("click", (event) => {
      if (!primaryNav.classList.contains("is-open")) return;
      if (primaryNav.contains(event.target) || navToggle.contains(event.target)) return;
      setNavigation(false);
    });

    const desktopNavigation = window.matchMedia("(min-width: 981px)");
    desktopNavigation.addEventListener("change", (event) => {
      if (event.matches) setNavigation(false);
    });
  }

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", () => copyText(button.dataset.copyTarget, button));
  });

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setAPITab(tab.dataset.apiTab, false));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let next = index;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      setAPITab(tabs[next].dataset.apiTab, true);
    });
  });
})();
