/**
 * Interactive BPE tokenizer demo
 * Depends on bpe-tokenizer.js being loaded first.
 */
(async function () {
  const TOKEN_COLORS = [
    "#2e2a5b", "#3e4a3e", "#5b4a2e", "#633030",
    "#2e4a5b", "#4a2e5b", "#2e5b4a", "#5b2e3a",
  ];

  const root = document.getElementById("tokenizer-demo");
  if (!root) return;

  const statusEl = root.querySelector("#demo-status");
  const inputEl  = root.querySelector("#demo-input");
  const displayEl = root.querySelector("#demo-display");
  const idsEl    = root.querySelector("#demo-ids");
  const countEl  = root.querySelector("#demo-count");

  statusEl.textContent = "Loading tokenizer data…";
  inputEl.disabled = true;

  let tokenizer;
  try {
    tokenizer = await loadTokenizer("/data/tokenizer.json");
    statusEl.textContent = "";
    inputEl.disabled = false;
    update();
  } catch (err) {
    statusEl.textContent = `Could not load tokenizer: ${err.message}`;
    return;
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/ /g, "&nbsp;");
  }

  function update() {
    const text = inputEl.value;

    if (!text) {
      displayEl.innerHTML = "";
      idsEl.textContent   = "";
      countEl.textContent = "";
      return;
    }

    const tokens = tokenizer.tokenize(text);

    displayEl.innerHTML = tokens
      .map((t, i) => {
        const color = TOKEN_COLORS[i % TOKEN_COLORS.length];
        return `<span style="background-color:${color};color:#fff;padding:2px 5px;border-radius:3px;margin:1px 1px;display:inline-block;font-family:monospace">${escapeHtml(t.str)}</span>`;
      })
      .join("");

    idsEl.textContent   = "[" + tokens.map((t) => t.id).join(", ") + "]";
    countEl.textContent = `${tokens.length} token${tokens.length !== 1 ? "s" : ""}`;
  }

  inputEl.addEventListener("input", update);
})();
