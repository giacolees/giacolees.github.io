(function () {
  const el = document.getElementById("donut-ascii");
  if (!el) return;

  const W = 80, H = 24;
  let A = 1, B = 1;

  function frame() {
    const out   = new Array(W * H).fill(" ");
    const zbuf  = new Array(W * H).fill(0);
    const lum   = ".,-~:;=!*#$@";

    for (let j = 0; j < 6.28; j += 0.07) {
      for (let i = 0; i < 6.28; i += 0.02) {
        const sinI = Math.sin(i), cosI = Math.cos(i);
        const sinJ = Math.sin(j), cosJ = Math.cos(j);
        const sinA = Math.sin(A), cosA = Math.cos(A);
        const sinB = Math.sin(B), cosB = Math.cos(B);

        const h = cosJ + 2;
        const D = 1 / (sinI * h * sinA + sinJ * cosA + 5);
        const t = sinI * h * cosA - sinJ * sinA;

        const x = Math.floor(W / 2 + 30 * D * (cosI * h * cosB - t * sinB));
        const y = Math.floor(H / 2 + 15 * D * (cosI * h * sinB + t * cosB));
        const o = x + W * y;

        const N = Math.floor(
          8 * ((sinJ * sinA - sinI * cosJ * cosA) * cosB -
               sinI * cosJ * sinA - sinJ * cosA -
               cosI * cosJ * sinB)
        );

        if (y >= 0 && y < H && x >= 0 && x < W && D > zbuf[o]) {
          zbuf[o] = D;
          out[o] = lum[Math.max(0, N)];
        }
      }
    }

    let s = "";
    for (let k = 0; k < H; k++) s += out.slice(k * W, k * W + W).join("") + "\n";
    el.textContent = s;

    A += 0.07;
    B += 0.03;
  }

  setInterval(frame, 50);
  frame();
})();
