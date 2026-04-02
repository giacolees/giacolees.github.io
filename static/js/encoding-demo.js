/**
 * Interactive encoding comparison demo.
 * Compares character-level (Unicode code points) vs. byte-level (UTF-8) encoding.
 */
(function () {
  const CHAR_COLORS = [
    "#2e2a5b", "#3e4a3e", "#5b4a2e", "#633030",
    "#2e4a5b", "#4a2e5b", "#2e5b4a", "#5b2e3a",
  ];

  const BYTE_COLORS = [
    "#1a3a5c", "#1e4d2b", "#5c3a1a", "#4d1e1e",
    "#1a4d5c", "#3a1a5c", "#1a5c3a", "#5c1a2e",
  ];

  const root = document.getElementById("encoding-demo");
  if (!root) return;

  const inputEl     = root.querySelector("#demo-input");
  const charDisplay = root.querySelector("#char-display");
  const charIds     = root.querySelector("#char-ids");
  const charCount   = root.querySelector("#char-count");
  const byteDisplay = root.querySelector("#byte-display");
  const byteIds     = root.querySelector("#byte-ids");
  const byteCount   = root.querySelector("#byte-count");

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/ /g, "&nbsp;");
  }

  function makeChip(label, color) {
    return `<span style="background-color:${color};color:#fff;padding:2px 5px;border-radius:3px;margin:1px 1px;display:inline-block;font-family:monospace">${escapeHtml(label)}</span>`;
  }

  function toUtf8Bytes(str) {
    return Array.from(new TextEncoder().encode(str));
  }

  function update() {
    const text = inputEl.value;

    if (!text) {
      charDisplay.innerHTML = "";
      charIds.textContent   = "";
      charCount.textContent = "";
      byteDisplay.innerHTML = "";
      byteIds.textContent   = "";
      byteCount.textContent = "";
      return;
    }

    // Character-level: one chip per Unicode code point
    const chars = Array.from(text);
    charDisplay.innerHTML = chars
      .map((ch, i) => makeChip(ch, CHAR_COLORS[i % CHAR_COLORS.length]))
      .join("");
    charIds.textContent   = "[" + chars.map((ch) => ch.codePointAt(0)).join(", ") + "]";
    charCount.textContent = `${chars.length} character${chars.length !== 1 ? "s" : ""}`;

    // Byte-level: one chip per UTF-8 byte
    const bytes = toUtf8Bytes(text);
    byteDisplay.innerHTML = bytes
      .map((b, i) => makeChip(String(b), BYTE_COLORS[i % BYTE_COLORS.length]))
      .join("");
    byteIds.textContent   = "[" + bytes.join(", ") + "]";
    byteCount.textContent = `${bytes.length} byte${bytes.length !== 1 ? "s" : ""}`;
  }

  inputEl.addEventListener("input", update);
  update();
})();
