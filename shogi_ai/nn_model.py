"""Neural-network play for the Web server.

Holds one cached :class:`NeuralPlayer` per checkpoint path so that switching
models in the browser does not reload weights on every move, and exposes a
search result shaped like :class:`shogi_ai.engine.SearchResult` so the HTTP
layer can treat both engines the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_MODEL_DIR = "data"
#: Modest default: the Web server searches on the pure-Python backend, which is
#: far slower than cshogi, so a small budget keeps the UI responsive.
DEFAULT_SIMULATIONS = 120
DEFAULT_BATCH_SIZE = 8


def list_model_files(directory: str = DEFAULT_MODEL_DIR) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".pt")
    )


#: Kept for the existing callers and tests.
list_available_models = list_model_files


@dataclass
class NeuralSearchResult:
    move: Optional[object]
    score: int
    nodes: int
    depth: int
    elapsed: float
    simulations: int = 0
    value: float = 0.0


class NeuralPlayer:
    """MCTS over a policy/value checkpoint, driven from a ``core.Position``."""

    def __init__(self, path: str, device: str = "auto",
                 simulations: int = DEFAULT_SIMULATIONS,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        from .evaluator import TorchEvaluator
        from .network import describe, load_checkpoint
        from .evaluator import pick_device

        resolved = pick_device(device)
        model, payload = load_checkpoint(path, device=resolved)
        self.path = path
        self.device = resolved
        self.description = describe(model)
        self.steps = int(payload.get("steps", 0))
        self.evaluator = TorchEvaluator(model, device=resolved)
        self.simulations = simulations
        self.batch_size = batch_size

    def search(self, position, simulations: Optional[int] = None,
               time_limit: Optional[float] = None) -> NeuralSearchResult:
        import numpy as np

        from .backends import PurePythonBackend
        from .mcts import MCTS, select_move

        backend = PurePythonBackend(position)
        mcts = MCTS(self.evaluator, simulations=simulations or self.simulations,
                    batch_size=self.batch_size, add_noise=False,
                    rng=np.random.default_rng())
        root = mcts.run(backend, time_limit=time_limit)
        if not root.moves:
            return NeuralSearchResult(None, 0, 0, 0, mcts.elapsed)

        move, counts = select_move(root, 0.0, np.random.default_rng())
        value = root.mean_value()
        return NeuralSearchResult(
            move=move,
            # Report a centipawn-like number so the UI can show one scale.
            score=int(value * 1000),
            nodes=int(counts.sum()),
            depth=_principal_depth(root),
            elapsed=mcts.elapsed,
            simulations=mcts.simulations_done,
            value=value,
        )


def _principal_depth(root) -> int:
    """Length of the most-visited line, reported as the search depth."""
    import numpy as np

    depth = 0
    node = root
    while node is not None and node.child_visits is not None and node.child_visits.sum() > 0:
        index = int(np.argmax(node.child_visits))
        node = node.children[index]
        depth += 1
        if node is None or not node.expanded:
            break
    return depth


_PLAYERS: Dict[str, NeuralPlayer] = {}
_LOAD_ERRORS: Dict[str, str] = {}


def get_player(path: str, **kwargs) -> Optional[NeuralPlayer]:
    """Return a cached player for ``path``, or ``None`` if it cannot be loaded."""
    if not path:
        return None
    if path in _PLAYERS:
        return _PLAYERS[path]
    if path in _LOAD_ERRORS:
        return None
    if os.path.isdir(path):
        candidates = list_model_files(path)
        if not candidates:
            return None
        path = candidates[0]
        if path in _PLAYERS:
            return _PLAYERS[path]
    if not os.path.exists(path):
        _LOAD_ERRORS[path] = "file not found"
        return None
    try:
        player = NeuralPlayer(path, **kwargs)
    except SystemExit as exc:
        # A checkpoint from an incompatible version, or torch missing.
        _LOAD_ERRORS[path] = str(exc)
        print(f"could not load {path}: {exc}")
        return None
    _PLAYERS[path] = player
    print(f"loaded {path}: {player.description} on {player.device} "
          f"({player.steps} trained steps)")
    return player


def load_nn_model(path: str = "") -> bool:
    """Eagerly load a checkpoint; ``True`` when it is ready to play."""
    return get_player(path) is not None


def load_error(path: str) -> Optional[str]:
    return _LOAD_ERRORS.get(path)
