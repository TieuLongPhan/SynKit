(function () {
  function initializeNavigation() {
    var body = document.getElementById("sk-globaltoc-tree");
    var toggle = document.querySelector(".sk-globaltoc__toggle");
    if (!body || !toggle || body.dataset.initialized) return;
    body.dataset.initialized = "true";

    toggle.addEventListener("click", function () {
      var expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      body.hidden = expanded;
      toggle.querySelector(".sk-globaltoc__chev").textContent = expanded ? "▸" : "▾";
    });

    body.querySelectorAll("li").forEach(function (item, index) {
      var children = Array.from(item.children);
      var link = children.find(function (child) { return child.tagName === "A"; });
      var submenu = children.find(function (child) { return child.tagName === "UL"; });
      if (!link || !submenu) return;

      var title = link.textContent.trim();
      var button = document.createElement("button");
      button.type = "button";
      button.className = "sk-globaltoc__branch-toggle";
      button.appendChild(document.createTextNode(title));
      var chevron = document.createElement("span");
      chevron.setAttribute("aria-hidden", "true");
      button.appendChild(chevron);
      submenu.id = "sk-nav-submenu-" + index;
      button.setAttribute("aria-controls", submenu.id);
      link.replaceWith(button);

      var overview = document.createElement("li");
      overview.className = "sk-globaltoc__overview";
      link.textContent = "Overview";
      link.setAttribute("aria-label", title + " overview");
      overview.appendChild(link);
      submenu.prepend(overview);

      function setExpanded(expanded) {
        submenu.hidden = !expanded;
        button.setAttribute("aria-expanded", String(expanded));
        button.setAttribute("aria-label", (expanded ? "Collapse " : "Expand ") + title);
        chevron.textContent = expanded ? "▾" : "▸";
      }
      setExpanded(false);
      button.addEventListener("click", function () {
        setExpanded(button.getAttribute("aria-expanded") !== "true");
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeNavigation);
  } else {
    initializeNavigation();
  }
})();
