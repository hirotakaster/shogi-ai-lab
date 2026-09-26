const PIECES = {
  K: "玉",
  R: "飛",
  B: "角",
  G: "金",
  S: "銀",
  N: "桂",
  L: "香",
  P: "歩",
  "+R": "龍",
  "+B": "馬",
  "+S": "全",
  "+N": "圭",
  "+L": "杏",
  "+P": "と",
};

const state = {
  data: null,
  selected: null,
  selectedHand: null,
  selectedModel: null,
  legal: [],
  lastMove: null,
  gameStarted: false,
  thinking: false,
  timedOut: false,
  remainingMs: 0,
  turnStartedAt: 0,
  timerId: null,
  trainingMode: false,
  trainingRunning: false,
  trainingTimerId: null,
};

const boardEl = document.querySelector("#board");
const statusEl = document.querySelector("#status");
const turnEl = document.querySelector("#turn");
const moveNumberEl = document.querySelector("#moveNumber");
const checkEl = document.querySelector("#check");
const clockEl = document.querySelector("#clock");
const historyEl = document.querySelector("#history");
const sfenEl = document.querySelector("#sfen");
const searchEl = document.querySelector("#search");
const trainingPositionsEl = document.querySelector("#trainingPositions");
const trainingRecordsEl = document.querySelector("#trainingRecords");
const modelSelectEl = document.querySelector("#modelSelect");

modelSelectEl.addEventListener("change", () => {
  state.selectedModel = modelSelectEl.value || null;
});

document.querySelector("#startGame").addEventListener("click", startGame);
document.querySelector("#startTraining").addEventListener("click", startTrainingView);
document.querySelector("#stopTraining").addEventListener("click", stopTrainingView);

document.querySelector("#newGame").addEventListener("click", async () => {
  stopClock();
  stopTrainingLoop();
  state.trainingMode = false;
  state.gameStarted = false;
  state.timedOut = false;
  state.thinking = false;
  await post("/api/new", {});
  state.selected = null;
  state.selectedHand = null;
  searchEl.textContent = "-";
  clockEl.textContent = "-";
});

document.querySelector("#aiMove").addEventListener("click", () => aiMove({ manual: true }));

async function load() {
  const res = await fetch("/api/state");
  state.data = await res.json();
  render();
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = data.error || "エラー";
    return data;
  }
  state.data = data;
  render();
  return data;
}

async function startGame() {
  stopClock();
  stopTrainingLoop();
  state.trainingMode = false;
  state.gameStarted = true;
  state.timedOut = false;
  state.thinking = false;
  state.selected = null;
  state.selectedHand = null;
  searchEl.textContent = "-";
  await post("/api/new", {});
  resetTurnClock();
  autoPlayIfNeeded();
}

async function startTrainingView() {
  stopClock();
  stopTrainingLoop();
  state.trainingMode = true;
  state.trainingRunning = true;
  state.gameStarted = false;
  state.timedOut = false;
  state.thinking = false;
  state.selected = null;
  state.selectedHand = null;
  searchEl.textContent = "-";
  const data = await fetchJson("/api/training/new", {});
  state.data = data;
  state.lastMove = null;
  render();
  scheduleTrainingStep(0);
}

function stopTrainingView() {
  state.trainingRunning = false;
  stopTrainingLoop();
  updateControlState();
  statusEl.textContent = state.trainingMode ? "学習観戦停止中" : statusText();
}

function stopTrainingLoop() {
  if (state.trainingTimerId) {
    window.clearTimeout(state.trainingTimerId);
    state.trainingTimerId = null;
  }
  state.trainingRunning = false;
}

function scheduleTrainingStep(delay = Number(document.querySelector("#trainingInterval").value)) {
  stopTrainingLoop();
  state.trainingRunning = true;
  state.trainingTimerId = window.setTimeout(trainingStep, delay);
}

async function trainingStep() {
  if (!state.trainingMode || !state.trainingRunning || state.data?.result) {
    state.trainingRunning = false;
    updateControlState();
    return;
  }
  state.thinking = true;
  statusEl.textContent = "学習対局中...";
  updateControlState();
  try {
    const data = await fetchJson("/api/training/step", { ...engineRequest(), timeLimit: 0.8 });
    if (data.search) {
      state.lastMove = data.search.move;
      searchEl.textContent = describeSearch(data.search);
    }
    state.data = data;
    state.thinking = false;
    render();
    if (!state.data.result && state.trainingRunning) {
      scheduleTrainingStep();
    }
  } catch (err) {
    state.thinking = false;
    state.trainingRunning = false;
    statusEl.textContent = String(err);
    updateControlState();
  }
}

async function aiMove({ manual = false } = {}) {
  if (!state.gameStarted || state.thinking || state.timedOut || state.data?.result) return;
  if (!manual && currentPlayerType() !== "ai") return;
  state.thinking = true;
  startClock();
  updateControlState();
  statusEl.textContent = "AI思考中...";
  const strictTimeLimit = Math.max(0.1, selectedTimeMs() / 1000);
  const searchTimeLimit = Math.max(0.1, strictTimeLimit - 0.25);
  let data;
  try {
    data = await fetchJson("/api/ai", {
      ...engineRequest(),
      timeLimit: searchTimeLimit,
      strictTimeLimit,
    });
  } catch (err) {
    state.thinking = false;
    statusEl.textContent = String(err);
    updateControlState();
    return;
  }
  if (state.timedOut) return;
  if (data.timeout) {
    state.timedOut = true;
    state.thinking = false;
    stopClock();
    render();
    return;
  }
  if (data.search) {
    state.lastMove = data.search.move;
    searchEl.textContent = describeSearch(data.search);
  }
  state.data = data;
  render();
  state.thinking = false;
  if (!state.data.result && !state.timedOut) {
    resetTurnClock();
    autoPlayIfNeeded();
  }
  updateControlState();
}

function engineRequest() {
  return {
    depth: Number(document.querySelector("#depth").value),
    simulations: Number(document.querySelector("#simulations").value),
    engine: document.querySelector("#engineType").value,
    modelPath: state.selectedModel || modelSelectEl.value || null,
  };
}

function describeSearch(search) {
  if (!search) return "-";
  const parts = [search.move || "-", `評価 ${search.score}`];
  if (search.engine === "nn") {
    parts.push(`${search.simulations || search.nodes} sims`, `深さ ${search.depth}`);
  } else {
    parts.push(`${search.nodes} nodes`, `深さ ${search.depth}`);
  }
  parts.push(`${search.elapsed}s`);
  let text = parts.join(" / ");
  if (search.note) text += ` — ${search.note}`;
  return text;
}

async function fetchJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "API error");
  }
  return data;
}

function render() {
  const pos = state.data.position;
  state.legal = pos.legalMoves;
  boardEl.innerHTML = "";
  pos.board.forEach((row, r) => {
    row.forEach((piece, c) => {
      const sq = document.createElement("button");
      sq.className = "sq";
      sq.dataset.r = String(r);
      sq.dataset.c = String(c);
      const name = squareName(r, c);
      if (state.selected?.r === r && state.selected?.c === c) sq.classList.add("selected");
      if (isTarget(name)) sq.classList.add("target");
      if (state.lastMove?.slice(2, 4) === name) sq.classList.add("last");
      sq.addEventListener("click", () => clickSquare(r, c));
      if (piece) sq.appendChild(pieceNode(piece));
      boardEl.appendChild(sq);
    });
  });
  renderHand("black", document.querySelector("#blackHand"), pos.hands.black);
  renderHand("white", document.querySelector("#whiteHand"), pos.hands.white);
  turnEl.textContent = `${pos.turn === "black" ? "先手" : "後手"} (${currentPlayerLabel()})`;
  moveNumberEl.textContent = String(pos.moveNumber);
  checkEl.textContent = pos.inCheck ? "王手" : "なし";
  sfenEl.textContent = pos.sfen;
  statusEl.textContent = statusText();
  historyEl.innerHTML = "";
  state.data.history.forEach((m, i) => {
    const li = document.createElement("li");
    li.textContent = `${i + 1}. ${m}`;
    historyEl.appendChild(li);
  });
  updateClockDisplay();
  updateTrainingStats();
  const nnOption = document.querySelector("#nnOption");
  if (state.data.nn_available) {
    nnOption.disabled = false;
    nnOption.textContent = "学習モデル (Neural Net)";
    populateModelSelect(state.data.model_files || []);
  } else {
    nnOption.disabled = true;
    nnOption.textContent = state.data.torch_installed
      ? "学習モデル (data/ に .pt がありません)"
      : "学習モデル (PyTorch未導入)";
    document.querySelector("#engineType").value = "heuristic";
    populateModelSelect([]);
  }
  document.querySelector("#simulations").disabled =
    document.querySelector("#engineType").value !== "nn";
  updateControlState();
}

function populateModelSelect(files) {
  const current = state.selectedModel || modelSelectEl.value || "";
  modelSelectEl.innerHTML = '<option value="">未選択</option>';
  files.forEach((file) => {
    const option = document.createElement("option");
    option.value = file;
    option.textContent = file.replace(/^.*\//, "");
    option.selected = file === current;
    modelSelectEl.appendChild(option);
  });
  if (!files.includes(current)) {
    state.selectedModel = files[0] || null;
    modelSelectEl.value = state.selectedModel || "";
  }
}

function pieceNode(piece) {
  const div = document.createElement("div");
  div.className = `piece ${piece.color}`;
  if (piece.kind.startsWith("+")) div.classList.add("promoted");
  div.textContent = PIECES[piece.kind];
  return div;
}

function renderHand(color, el, hand) {
  el.innerHTML = "";
  Object.keys(hand).sort((a, b) => "RBGSLNP".indexOf(a) - "RBGSLNP".indexOf(b)).forEach((kind) => {
    const btn = document.createElement("button");
    btn.className = "hand-piece";
    if (state.selectedHand?.color === color && state.selectedHand?.kind === kind) btn.classList.add("selected");
    btn.textContent = PIECES[kind];
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = String(hand[kind]);
    btn.appendChild(count);
    btn.addEventListener("click", () => {
      if (!canHumanInteract() || state.data.position.turn !== color) return;
      state.selected = null;
      state.selectedHand = { color, kind };
      render();
    });
    el.appendChild(btn);
  });
}

async function clickSquare(r, c) {
  if (!canHumanInteract()) return;
  const pos = state.data.position;
  const piece = pos.board[r][c];
  const dst = squareName(r, c);
  if (state.selectedHand) {
    const move = `${state.selectedHand.kind}*${dst}`;
    if (state.legal.includes(move)) {
      state.selectedHand = null;
      stopClock();
      await post("/api/move", { move });
      if (!state.data.result) {
        resetTurnClock();
        autoPlayIfNeeded();
      }
      return;
    }
  }
  if (state.selected) {
    const src = squareName(state.selected.r, state.selected.c);
    const candidates = state.legal.filter((m) => m.startsWith(src + dst));
    if (candidates.length) {
      let move = candidates[0];
      if (candidates.length > 1 && candidates.some((m) => m.endsWith("+"))) {
        move = window.confirm("成りますか？") ? candidates.find((m) => m.endsWith("+")) : candidates.find((m) => !m.endsWith("+"));
      }
      state.selected = null;
      stopClock();
      await post("/api/move", { move });
      if (!state.data.result) {
        resetTurnClock();
        autoPlayIfNeeded();
      }
      return;
    }
  }
  state.selectedHand = null;
  if (piece && piece.color === pos.turn) {
    state.selected = { r, c };
  } else {
    state.selected = null;
  }
  render();
}

function isTarget(dst) {
  if (!canHumanInteract()) return false;
  if (state.selectedHand) return state.legal.includes(`${state.selectedHand.kind}*${dst}`);
  if (!state.selected) return false;
  const src = squareName(state.selected.r, state.selected.c);
  return state.legal.some((m) => m.startsWith(src + dst));
}

function squareName(r, c) {
  return "987654321"[c] + "abcdefghi"[r];
}

function currentPlayerType() {
  if (!state.data) return "human";
  const id = state.data.position.turn === "black" ? "#blackPlayer" : "#whitePlayer";
  return document.querySelector(id).value;
}

function currentPlayerLabel() {
  if (state.trainingMode) return "学習AI";
  return currentPlayerType() === "ai" ? "AI" : "あなた";
}

function canHumanInteract() {
  return !state.trainingMode && state.gameStarted && !state.thinking && !state.timedOut && !state.data?.result && currentPlayerType() === "human";
}

function statusText() {
  if (state.data.result) return `${state.data.result === "black" ? "先手" : "後手"}勝ち`;
  if (state.trainingMode) return state.trainingRunning ? "学習対局中..." : "学習観戦停止中";
  if (state.timedOut) return `${state.data.position.turn === "black" ? "先手" : "後手"}時間切れ`;
  if (!state.gameStarted) return "開始待ち";
  if (state.thinking) return "AI思考中...";
  return `${currentPlayerLabel()}の手番`;
}

function selectedTimeMs() {
  return Number(document.querySelector("#timeLimit").value) * 1000;
}

function resetTurnClock() {
  state.remainingMs = selectedTimeMs();
  state.turnStartedAt = Date.now();
  startClock();
}

function startClock() {
  stopClock();
  updateClockDisplay();
  state.timerId = window.setInterval(tickClock, 200);
}

function stopClock() {
  if (state.timerId) {
    window.clearInterval(state.timerId);
    state.timerId = null;
  }
}

function tickClock() {
  if (!state.gameStarted || state.data?.result || state.timedOut) {
    stopClock();
    updateClockDisplay();
    return;
  }
  state.remainingMs = Math.max(0, selectedTimeMs() - (Date.now() - state.turnStartedAt));
  if (currentPlayerType() === "ai") {
    // AI thinking consumes the same visible turn clock, but does not produce
    // an automatic loss when the server still has a legal fallback move.
    updateClockDisplay();
    statusEl.textContent = statusText();
    updateControlState();
    return;
  }
  if (state.remainingMs <= 0) {
    state.timedOut = true;
    state.thinking = false;
    state.selected = null;
    state.selectedHand = null;
    stopClock();
  }
  updateClockDisplay();
  statusEl.textContent = statusText();
  updateControlState();
}

function updateClockDisplay() {
  if (state.trainingMode || !state.gameStarted || !state.data || state.data.result) {
    clockEl.textContent = "-";
    clockEl.classList.remove("urgent");
    return;
  }
  const seconds = Math.ceil(state.remainingMs / 1000);
  clockEl.textContent = `${Math.max(0, seconds)}秒`;
  clockEl.classList.toggle("urgent", seconds <= 5 && currentPlayerType() === "human");
}

function updateTrainingStats() {
  const training = state.data?.training;
  trainingPositionsEl.textContent = training ? String(training.plies) : "-";
  trainingRecordsEl.textContent = training ? String(training.positions) : "-";
}

function autoPlayIfNeeded() {
  if (!state.gameStarted || state.thinking || state.timedOut || state.data?.result) return;
  if (currentPlayerType() === "ai") {
    window.setTimeout(() => aiMove(), 0);
  }
}

function updateControlState() {
  const started = state.gameStarted;
  const playing = started && !state.data?.result && !state.timedOut;
  document.querySelector("#startGame").disabled = state.trainingRunning || playing;
  document.querySelector("#aiMove").disabled = state.trainingMode || !started || state.thinking || state.timedOut || Boolean(state.data?.result);
  document.querySelector("#blackPlayer").disabled = state.trainingMode || playing;
  document.querySelector("#whitePlayer").disabled = state.trainingMode || playing;
  document.querySelector("#engineType").disabled = state.trainingMode || playing;
  document.querySelector("#modelSelect").disabled = state.trainingMode || playing;
  document.querySelector("#startTraining").disabled = state.trainingRunning || playing;
  document.querySelector("#stopTraining").disabled = !state.trainingRunning;
}

load().catch((err) => {
  statusEl.textContent = String(err);
});
