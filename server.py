from __future__ import annotations

import importlib.util
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from shogi_ai.core import Position, parse_usi
from shogi_ai.engine import AlphaBetaEngine
from shogi_ai.nn_model import get_player, list_model_files

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MODEL_DIR = ROOT / "data"

#: ``find_spec`` tells us whether the neural engine is usable without paying
#: for the torch import, which would slow down the dependency-free start-up.
TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


def available_models() -> list:
    """Checkpoint paths relative to the project root, re-scanned per request.

    Relative paths keep the server's absolute filesystem layout out of the
    browser; :func:`resolve_model` turns one back into a real path.
    """
    return [Path(path).name for path in list_model_files(str(MODEL_DIR))]


def resolve_model(name: str) -> str:
    """Map a name from the browser back onto a checkpoint inside ``data/``."""
    # Only a bare filename from our own listing is accepted, so a crafted
    # request cannot walk out of the model directory.
    if name in available_models():
        return str(MODEL_DIR / name)
    return ""


class GameState:
    def __init__(self):
        self.position = Position.start()
        self.history = []
        self.engine = AlphaBetaEngine(depth=3, time_limit=1.2)


STATE = GameState()
TRAINING_STATE = GameState()


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed = urlparse(path).path
        if parsed == "/":
            return str(WEB / "index.html")
        return str(WEB / parsed.lstrip("/"))

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self.send_json(payload(STATE))
            return
        if self.path.startswith("/api/training/state"):
            self.send_json(training_payload())
            return
        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        data = json.loads(body or "{}")
        try:
            if self.path.startswith("/api/new"):
                STATE.position = Position.start()
                STATE.history = []
                self.send_json(payload(STATE))
            elif self.path.startswith("/api/move"):
                move = parse_usi(data["move"], STATE.position)
                legal = {m.usi(): m for m in STATE.position.legal_moves()}
                if move.usi() not in legal:
                    self.send_json({"error": "illegal move", **payload(STATE)}, status=400)
                    return
                STATE.position = STATE.position.make_move(legal[move.usi()])
                STATE.history.append(move.usi())
                self.send_json(payload(STATE))
            elif self.path.startswith("/api/ai"):
                self.send_json(advance(STATE, data))
            elif self.path.startswith("/api/training/new"):
                TRAINING_STATE.position = Position.start()
                TRAINING_STATE.history = []
                self.send_json(training_payload())
            elif self.path.startswith("/api/training/step"):
                if TRAINING_STATE.position.result() is not None:
                    out = training_payload()
                    out["search"] = None
                    self.send_json(out)
                    return
                out = advance(TRAINING_STATE, data, strict=False)
                out.update(training_payload())
                self.send_json(out)
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc), **payload(STATE)}, status=500)

    def send_json(self, obj, status=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass


def advance(game: GameState, data: dict, strict: bool = True) -> dict:
    """Let the selected engine pick a move and play it."""
    time_limit = float(data.get("timeLimit", 1.2))
    strict_limit = float(data.get("strictTimeLimit", time_limit)) if strict else time_limit
    engine_type = data.get("engine", "heuristic")

    if engine_type == "nn":
        result, note = neural_move(game, data, time_limit)
        if result is None:
            # Fall back rather than stalling the game in the browser.
            result = heuristic_move(game, data, time_limit)
            engine_type = "heuristic"
    else:
        result = heuristic_move(game, data, time_limit)
        note = None

    timed_out = result.elapsed > max(0.1, min(strict_limit, 60.0))
    if result.move:
        game.position = game.position.make_move(result.move)
        game.history.append(result.move.usi())

    out = payload(game)
    out["search"] = {
        "move": result.move.usi() if result.move else None,
        "score": result.score,
        "nodes": result.nodes,
        "depth": result.depth,
        "elapsed": round(result.elapsed, 3),
        "engine": engine_type,
        "simulations": getattr(result, "simulations", 0),
        "timedOut": timed_out,
    }
    if note:
        out["search"]["note"] = note
    out["timeout"] = timed_out and result.move is None
    return out


def heuristic_move(game: GameState, data: dict, time_limit: float):
    game.engine.depth = max(1, min(int(data.get("depth", 3)), 5))
    game.engine.time_limit = max(0.1, min(time_limit, 60.0))
    return game.engine.search(game.position)


def neural_move(game: GameState, data: dict, time_limit: float):
    """Search with the selected checkpoint, or explain why we cannot."""
    if not TORCH_INSTALLED:
        return None, "PyTorch is not installed; played the heuristic engine instead"
    models = available_models()
    requested = data.get("modelPath") or (models[0] if models else "")
    if not requested:
        return None, "no .pt model in data/; played the heuristic engine instead"

    path = resolve_model(requested)
    if not path:
        return None, f"{requested} is not in data/; played the heuristic engine instead"

    player = get_player(path)
    if player is None:
        from shogi_ai.nn_model import load_error

        reason = load_error(path) or "could not be loaded"
        return None, f"{requested}: {reason}; played the heuristic engine instead"

    simulations = max(1, min(int(data.get("simulations", 120)), 4000))
    return player.search(game.position, simulations=simulations,
                         time_limit=max(0.1, min(time_limit, 60.0))), None


def payload(game):
    result = game.position.result()
    models = available_models()
    return {
        "position": game.position.json(),
        "history": list(game.history),
        "result": "black" if result == 1 else "white" if result == -1 else None,
        "nn_available": bool(models) and TORCH_INSTALLED,
        "torch_installed": TORCH_INSTALLED,
        "model_files": models,
    }


def training_payload():
    out = payload(TRAINING_STATE)
    out["training"] = {
        "plies": len(TRAINING_STATE.history),
        "positions": len(TRAINING_STATE.history),
    }
    return out


def main():
    addr = ("127.0.0.1", 8000)
    httpd = ThreadingHTTPServer(addr, Handler)
    models = available_models()
    print(f"Shogi AI Lab: http://{addr[0]}:{addr[1]}", flush=True)
    print(f"torch installed: {TORCH_INSTALLED} | "
          f"checkpoints in data/: {', '.join(models) if models else 'none'}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
