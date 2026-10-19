const PALETTE = ["#B8A5E8", "#F0B48B", "#92C9E8", "#A9D4A4", "#E8CD83", "#ECA6BB", "#9FB9D4"];

export default {
  render({ model, el }) {
    const root = document.createElement("section");
    root.className = "qh-game";
    root.innerHTML = `
      <div class="qh-top"><div><div class="qh-title">Queens · 7 × 7</div><div class="qh-meta"></div></div><div class="qh-timer">00:00</div></div>
      <div class="qh-instructions">Щелчок или проведение — закрыть / вернуть клетки.<br>Двойной щелчок — поставить / убрать королеву.</div>
      <div class="qh-board" role="group" aria-label="Поле Queens"></div>
      <div class="qh-bottom"><button class="qh-button qh-undo">↶ Назад</button><span class="qh-count"></span><button class="qh-button qh-next">Другая раскраска →</button></div>
      <div class="qh-feedback" role="status" aria-live="polite"></div>
      <div class="qh-rating" hidden><div class="qh-rating-title">Насколько сложной была раскраска?</div><div class="qh-rating-buttons" role="group" aria-label="Сложность от 0 до 10"></div><button class="qh-button qh-primary qh-save">Сохранить время и оценку</button></div>
      <label class="qh-player">Игрок <input maxlength="80" aria-label="Идентификатор игрока"><span class="qh-saved-count"></span></label>
    `;
    el.appendChild(root);
    const select = (value) => root.querySelector(value);
    const boardEl = select(".qh-board"), timerEl = select(".qh-timer");
    const undoButton = select(".qh-undo"), nextButton = select(".qh-next");
    const ratingEl = select(".qh-rating"), saveButton = select(".qh-save");
    const feedbackEl = select(".qh-feedback"), playerInput = select(".qh-player input");
    playerInput.value = model.get("player") || "local";
    let puzzle, n, queens, manual, history, cells, drag, pendingClick;
    let started, runningSince, elapsedBase, solved, confirmed, saved, saving;
    let actions, undos, rating, lastSingle, suppressClickUntil = 0;

    const send = (type, payload = {}) => model.send({ type, game_id: puzzle.game_id, ...payload });
    const snapshot = () => ({ queens: queens.slice(), manual: manual.slice() });
    const restore = (state) => { queens = state.queens.slice(); manual = state.manual.slice(); };
    const elapsed = () => elapsedBase + (runningSince === null ? 0 : (performance.now() - runningSince) / 1000);
    function startClock() {
      if (!started) {
        started = true; runningSince = performance.now();
        send("start", { player: playerInput.value.trim() || "local" });
      }
    }
    function renderClock() {
      const seconds = Math.floor(elapsed());
      timerEl.textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }
    function queenExcludes(q, i) {
      const r = Math.floor(q / n), c = q % n;
      const row = Math.floor(i / n), col = i % n;
      return row === r || col === c || puzzle.board[row][col] === puzzle.board[r][c]
        || (Math.abs(row - r) <= 1 && Math.abs(col - c) <= 1);
    }
    function blockedCells() {
      const blocked = new Array(n * n).fill(false);
      for (let q = 0; q < queens.length; q++) if (queens[q]) {
        for (let i = 0; i < blocked.length; i++) {
          if (queenExcludes(q, i)) blocked[i] = true;
        }
      }
      return blocked;
    }
    function visibleStates() {
      const blocked = blockedCells();
      return queens.map((q, i) => q ? 2 : (manual[i] === 1 || (manual[i] === -1 && blocked[i])) ? 1 : 0);
    }
    function isSolution() {
      const positions = queens.flatMap((q, i) => q ? [i] : []);
      if (positions.length !== n) return false;
      const rows = new Set(), columns = new Set(), colors = new Set();
      for (const i of positions) {
        const r = Math.floor(i / n), c = i % n;
        rows.add(r); columns.add(c); colors.add(puzzle.board[r][c]);
        for (const j of positions) if (i !== j && Math.abs(r - Math.floor(j / n)) <= 1 && Math.abs(c - j % n) <= 1) return false;
      }
      return rows.size === n && columns.size === n && colors.size === n;
    }
    function setFeedback(text, error = false) {
      feedbackEl.textContent = text;
      feedbackEl.classList.toggle("qh-error", error);
    }
    function paint() {
      const states = visibleStates();
      states.forEach((state, i) => {
        cells[i].classList.toggle("qh-gray", state === 1);
        cells[i].classList.toggle("qh-queen", state === 2);
        cells[i].textContent = state === 2 ? "♛" : "";
        cells[i].setAttribute("aria-label", `Строка ${Math.floor(i / n) + 1}, столбец ${i % n + 1}: ${["открыта", "закрыта", "королева"][state]}`);
      });
      undoButton.disabled = !history.length || saved || saving;
      nextButton.disabled = saving || (solved && !saved);
      nextButton.textContent = saved ? "Следующая раскраска →" : "Другая раскраска →";
      select(".qh-count").textContent = `${queens.filter(Boolean).length} / ${n} королев`;
      ratingEl.hidden = !solved;
      saveButton.disabled = rating === null || !confirmed || saved || saving;
      saveButton.textContent = saved ? "Сохранено ✓" : saving ? "Сохраняем…" : "Сохранить время и оценку";
      boardEl.classList.toggle("qh-finished", solved);
      playerInput.disabled = started;
    }
    function checkFinish() {
      if (!isSolution()) return;
      solved = true; confirmed = false;
      elapsedBase = elapsed(); runningSince = null;
      setFeedback("Решено. Проверяем расстановку…");
      send("solved", {
        queens: Array.from({ length: n }, (_, r) => queens.slice(r * n, (r + 1) * n).map(Number)),
        elapsed_seconds: elapsedBase, actions, undos,
      });
      renderClock();
    }
    function commit(before, kind, cell = null) {
      if (JSON.stringify(before) === JSON.stringify(snapshot())) return;
      history.push(before); actions++;
      lastSingle = kind === "single" ? { cell, at: performance.now(), before } : null;
      checkFinish(); paint();
    }
    function cancelPending() { if (pendingClick) clearTimeout(pendingClick.timer); pendingClick = null; }
    function flushPending() {
      if (!pendingClick) return;
      const i = pendingClick.cell; cancelPending(); toggleGray(i);
    }
    function toggleGray(i) {
      if (solved || saved || queens[i]) return;
      startClock();
      const before = snapshot();
      manual[i] = visibleStates()[i] === 1 ? 0 : 1;
      commit(before, "single", i);
    }
    function toggleQueen(i) {
      cancelPending();
      if (solved || saved) return;
      startClock();
      // A slow native double click may arrive after its first single-click timer.
      // Roll that one click back so the double click remains one undoable action.
      if (lastSingle && lastSingle.cell === i && performance.now() - lastSingle.at < 600) {
        restore(lastSingle.before); history.pop(); actions--; lastSingle = null;
      }
      const before = snapshot();
      queens[i] = !queens[i]; manual[i] = -1;
      if (queens[i]) manual.forEach((mark, j) => {
        if (queenExcludes(i, j) && mark === 0) manual[j] = -1;
      });
      commit(before, "queen");
    }
    function cellAt(event) {
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest(".qh-cell");
      return target && boardEl.contains(target) ? Number(target.dataset.index) : null;
    }
    function applyDragCell(i) {
      if (i === null || queens[i] || drag.visited.has(i)) return;
      drag.visited.add(i); manual[i] = drag.mark;
    }
    function onPointerDown(event) {
      if (event.button !== 0 || solved || saved) return;
      const i = cellAt(event); if (i === null) return;
      if (pendingClick && pendingClick.cell !== i) flushPending();
      startClock();
      drag = { first: i, before: snapshot(), visited: new Set(), active: false,
        mark: visibleStates()[i] === 1 ? 0 : 1, pointer: event.pointerId, capture: cells[i] };
      drag.capture.setPointerCapture(event.pointerId);
    }
    function onPointerMove(event) {
      if (!drag || event.pointerId !== drag.pointer) return;
      const i = cellAt(event);
      if (i !== null && i !== drag.first && !drag.active) {
        cancelPending(); drag.active = true; lastSingle = null; applyDragCell(drag.first);
      }
      if (drag.active) { applyDragCell(i); paint(); }
    }
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.pointer) return;
      if (drag.active) { commit(drag.before, "drag"); suppressClickUntil = performance.now() + 400; }
      const capture = drag.capture;
      drag = null;
      if (capture.hasPointerCapture(event.pointerId)) capture.releasePointerCapture(event.pointerId);
    }
    function onClick(event) {
      if (performance.now() < suppressClickUntil || event.detail > 1) return;
      const target = event.target.closest(".qh-cell"); if (!target) return;
      const i = Number(target.dataset.index);
      cancelPending();
      pendingClick = { cell: i, timer: setTimeout(() => { pendingClick = null; toggleGray(i); }, 280) };
    }
    function onDoubleClick(event) {
      event.preventDefault();
      const target = event.target.closest(".qh-cell");
      if (target && performance.now() >= suppressClickUntil) toggleQueen(Number(target.dataset.index));
    }
    function onKeyDown(event) {
      if (event.shiftKey && event.key === "Enter" && event.target.matches(".qh-cell")) {
        event.preventDefault(); toggleQueen(Number(event.target.dataset.index));
      }
    }
    function onUndo() {
      cancelPending();
      if (!history.length || saved || saving) return;
      if (solved) {
        solved = false; confirmed = false; rating = null; runningSince = performance.now();
        send("resume"); setFeedback("");
        root.querySelectorAll(".qh-score").forEach(button => button.setAttribute("aria-pressed", "false"));
      }
      restore(history.pop()); undos++; lastSingle = null; paint();
    }
    function onNext() {
      cancelPending();
      send("new_game", { player: playerInput.value.trim() || "local" });
    }
    function onSave() {
      if (saveButton.disabled) return;
      saving = true; paint(); send("save", { difficulty: rating });
    }
    function onFeedback() {
      const message = model.get("feedback");
      if (!message || message.game_id !== puzzle.game_id) return;
      if (message.kind === "solved" && !solved) return;
      if (message.kind === "solved") confirmed = true;
      if (message.kind === "saved") { saved = true; saving = false; }
      if (message.kind === "error") {
        saving = false;
        if (solved && !confirmed) { solved = false; runningSince = performance.now(); }
      }
      setFeedback(message.text || "", message.kind === "error"); paint();
      select(".qh-saved-count").textContent = `Сохранено: ${model.get("total_saved")}`;
    }
    function onPuzzle() {
      cancelPending();
      puzzle = model.get("puzzle"); n = puzzle.board.length;
      queens = new Array(n * n).fill(false); manual = new Array(n * n).fill(-1);
      history = []; cells = []; drag = null; lastSingle = null;
      started = false; runningSince = null; elapsedBase = 0;
      solved = false; confirmed = false; saved = false; saving = false;
      actions = 0; undos = 0; rating = null;
      boardEl.replaceChildren(); boardEl.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
      boardEl.style.gridTemplateRows = `repeat(${n}, 1fr)`;
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        const button = document.createElement("button");
        button.className = "qh-cell"; button.type = "button"; button.dataset.index = r * n + c;
        button.style.backgroundColor = PALETTE[puzzle.board[r][c] % PALETTE.length];
        if (c < n - 1) button.style.borderRight = puzzle.board[r][c] === puzzle.board[r][c + 1] ? "1px solid #ffffff88" : "2px solid #768296";
        if (r < n - 1) button.style.borderBottom = puzzle.board[r][c] === puzzle.board[r + 1][c] ? "1px solid #ffffff88" : "2px solid #768296";
        cells.push(button); boardEl.appendChild(button);
      }
      select(".qh-title").textContent = `Queens · ${n} × ${n}`;
      select(".qh-meta").textContent = `Раскраска #${puzzle.dataset_index} · таймер с первого действия`;
      root.querySelectorAll(".qh-score").forEach(button => button.setAttribute("aria-pressed", "false"));
      setFeedback(""); paint(); renderClock();
      select(".qh-saved-count").textContent = `Сохранено: ${model.get("total_saved")}`;
    }
    for (let score = 0; score <= 10; score++) {
      const button = document.createElement("button");
      button.className = "qh-button qh-score"; button.textContent = score;
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => {
        if (saved || saving) return;
        rating = score;
        root.querySelectorAll(".qh-score").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
        paint();
      });
      select(".qh-rating-buttons").appendChild(button);
    }
    boardEl.addEventListener("pointerdown", onPointerDown);
    boardEl.addEventListener("pointermove", onPointerMove);
    boardEl.addEventListener("pointerup", endDrag);
    boardEl.addEventListener("pointercancel", endDrag);
    boardEl.addEventListener("click", onClick);
    boardEl.addEventListener("dblclick", onDoubleClick);
    boardEl.addEventListener("keydown", onKeyDown);
    undoButton.addEventListener("click", onUndo);
    nextButton.addEventListener("click", onNext);
    saveButton.addEventListener("click", onSave);
    model.on("change:puzzle", onPuzzle);
    model.on("change:feedback", onFeedback);
    onPuzzle();
    const clock = setInterval(renderClock, 200);
    return () => {
      clearInterval(clock); cancelPending();
      model.off("change:puzzle", onPuzzle); model.off("change:feedback", onFeedback);
      root.remove();
    };
  },
};
