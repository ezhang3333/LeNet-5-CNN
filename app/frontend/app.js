const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const clearBtn = document.getElementById("clear");
const predictCanvasBtn = document.getElementById("predictCanvas");
const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const predictUploadBtn = document.getElementById("predictUpload");

const statusEl = document.getElementById("status");
const predEl = document.getElementById("pred");
const barsEl = document.getElementById("bars");

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setResult(predLabel, labels, probs) {
  predEl.textContent = predLabel;
  barsEl.innerHTML = "";
  for (let i = 0; i < labels.length; i++) {
    const row = document.createElement("div");
    row.className = "barRow";

    const left = document.createElement("div");
    const lbl = document.createElement("div");
    lbl.className = "barLabel";
    lbl.textContent = labels[i];
    const track = document.createElement("div");
    track.className = "barTrack";
    const fill = document.createElement("div");
    fill.className = "barFill";
    fill.style.width = `${(probs[i] * 100).toFixed(1)}%`;
    track.appendChild(fill);
    left.appendChild(lbl);
    left.appendChild(track);

    const pct = document.createElement("div");
    pct.className = "barPct";
    pct.textContent = `${(probs[i] * 100).toFixed(1)}%`;

    row.appendChild(left);
    row.appendChild(pct);
    barsEl.appendChild(row);
  }
}

async function predictFromDataURL(dataURL) {
  setStatus("Running inference...");
  predEl.textContent = "-";
  barsEl.innerHTML = "";

  const resp = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data_url: dataURL }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return await resp.json();
}

// --- Canvas drawing ---
function resetCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

resetCanvas();
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "black";
ctx.lineWidth = 18;

let drawing = false;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});

canvas.addEventListener("pointerup", () => {
  drawing = false;
});
canvas.addEventListener("pointercancel", () => {
  drawing = false;
});

clearBtn.addEventListener("click", () => {
  resetCanvas();
  setStatus("Cleared.");
});

predictCanvasBtn.addEventListener("click", async () => {
  try {
    const dataURL = canvas.toDataURL("image/png");
    const res = await predictFromDataURL(dataURL);
    setResult(res.pred_label, res.labels, res.probs);
    setStatus("Done.");
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  }
});

// --- Upload flow ---
fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  predictUploadBtn.disabled = false;
  setStatus("Upload ready.");
});

predictUploadBtn.addEventListener("click", async () => {
  const img = preview;
  if (!img || !img.src) return;

  const tmp = document.createElement("canvas");
  tmp.width = 280;
  tmp.height = 280;
  const tctx = tmp.getContext("2d");
  tctx.fillStyle = "white";
  tctx.fillRect(0, 0, tmp.width, tmp.height);
  tctx.drawImage(img, 0, 0, tmp.width, tmp.height);

  try {
    const dataURL = tmp.toDataURL("image/png");
    const res = await predictFromDataURL(dataURL);
    setResult(res.pred_label, res.labels, res.probs);
    setStatus("Done.");
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  }
});

