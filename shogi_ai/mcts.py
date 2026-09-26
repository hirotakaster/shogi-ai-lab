"""PUCT Monte-Carlo tree search with batched leaf evaluation.

The tree is edge-centric: a node keeps its children's statistics in NumPy
arrays so that selection is one vectorised expression rather than a Python loop
over child objects.  Leaves are gathered in batches using a virtual loss, which
is what keeps the GPU busy -- a single forward pass scores dozens of leaves.

Values are always stored from the point of view of the player to move at the
node that owns the edge, so a backup alternates sign on the way up.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import encode_planes, policy_index

#: AlphaZero's exploration schedule.
C_PUCT_INIT = 1.25
C_PUCT_BASE = 19652.0
#: Dirichlet noise at the root; 0.15 suits shogi's branching factor.
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPSILON = 0.25
#: Discourages two batched paths from picking the same edge.
VIRTUAL_LOSS = 1.0

Evaluator = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


class Node:
    __slots__ = ("moves", "raw_moves", "priors", "child_visits", "child_value",
                 "child_virtual", "children", "visits", "expanded", "terminal")

    def __init__(self):
        self.moves: List = []
        self.raw_moves: List = []
        self.priors: Optional[np.ndarray] = None
        self.child_visits: Optional[np.ndarray] = None
        self.child_value: Optional[np.ndarray] = None
        self.child_virtual: Optional[np.ndarray] = None
        self.children: List[Optional["Node"]] = []
        self.visits = 0
        self.expanded = False
        #: Set once a node is known to end the game, so it is never selected from.
        self.terminal: Optional[float] = None

    def expand(self, moves: Sequence, raw_moves: Sequence, priors: np.ndarray) -> None:
        count = len(moves)
        self.moves = list(moves)
        self.raw_moves = list(raw_moves)
        self.priors = priors
        self.child_visits = np.zeros(count, dtype=np.float32)
        self.child_value = np.zeros(count, dtype=np.float32)
        self.child_virtual = np.zeros(count, dtype=np.float32)
        self.children = [None] * count
        self.expanded = True

    def select(self) -> int:
        """Return the index of the edge with the highest PUCT score."""
        visits = self.child_visits + self.child_virtual
        total = float(visits.sum())
        # The virtual loss is subtracted from the running total as well as
        # counted as a visit, so a pending sibling looks like a loss.  That is
        # what steers the paths within one batch apart.
        denom = np.maximum(visits, 1e-8)
        q = np.where(visits > 0.0, (self.child_value - self.child_virtual) / denom, 0.0)
        c = math.log((1.0 + total + C_PUCT_BASE) / C_PUCT_BASE) + C_PUCT_INIT
        u = c * self.priors * math.sqrt(max(total, 1e-8)) / (1.0 + visits)
        return int(np.argmax(q + u))

    def visit_counts(self) -> np.ndarray:
        return self.child_visits.copy()

    def mean_value(self) -> float:
        total = float(self.child_visits.sum()) if self.child_visits is not None else 0.0
        if total == 0.0:
            return 0.0
        return float(self.child_value.sum() / total)


def softmax_over_legal(logits: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Softmax restricted to the legal policy indices, in the given order."""
    if not len(indices):
        return np.zeros(0, dtype=np.float32)
    uniform = np.full(len(indices), 1.0 / len(indices), dtype=np.float32)
    selected = logits[list(indices)].astype(np.float64)
    peak = selected.max()
    if not np.isfinite(peak):
        # An all -inf (or NaN) row carries no preference at all.
        return uniform
    exp = np.exp(selected - peak)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0.0:
        return uniform
    return (exp / total).astype(np.float32)


class MCTS:
    """Run PUCT search over a mutable backend board.

    ``evaluator`` maps a stacked ``(B, 85, 9, 9)`` float32 batch to
    ``(policy_logits (B, 2187), values (B,))``, each value expressed from the
    point of view of the side to move in that position.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int = 200,
        batch_size: int = 16,
        add_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.evaluator = evaluator
        self.simulations = simulations
        self.batch_size = max(1, batch_size)
        self.add_noise = add_noise
        self.rng = rng or np.random.default_rng()
        self.last_root: Optional[Node] = None
        self.root_value = 0.0
        self.evaluated = 0
        self.simulations_done = 0
        self.elapsed = 0.0

    # ------------------------------------------------------------------
    def run(self, backend, time_limit: Optional[float] = None) -> Node:
        """Search from ``backend``'s current position and return the root.

        ``time_limit`` caps the wall-clock budget in seconds.  It is checked
        between batches, so a search can overshoot by roughly the cost of one
        batch; callers that must not exceed a hard deadline should leave
        headroom.
        """
        self.evaluated = 0
        self.elapsed = 0.0
        started = time.time()
        root = Node()
        self.root_value = self._expand_root(root, backend)
        if not root.moves:
            self.simulations_done = 0
            self.elapsed = time.time() - started
            self.last_root = root
            return root
        if self.add_noise:
            self._apply_root_noise(root)

        done = 0
        while done < self.simulations:
            wanted = min(self.batch_size, self.simulations - done)
            paths, values = self._simulate_batch(root, backend, wanted)
            if not paths:
                break
            for path, value in zip(paths, values):
                self._backup(path, value)
            done += len(paths)
            if time_limit is not None and time.time() - started >= time_limit:
                break

        self.simulations_done = done
        self.elapsed = time.time() - started
        self.last_root = root
        return root

    # ------------------------------------------------------------------
    def _simulate_batch(self, root: Node, backend, wanted: int):
        """Descend ``wanted`` times, evaluate the new leaves in one batch."""
        paths: List[List[Tuple[Node, int]]] = []
        values: List[float] = []
        # Leaves needing a forward pass, de-duplicated by node identity so that
        # two paths landing on the same leaf cost a single evaluation.
        queue: List[Tuple[Node, np.ndarray, list, list, int]] = []
        slot_of_node: Dict[int, int] = {}
        waiters: List[Tuple[int, int]] = []

        for _ in range(wanted):
            node = root
            path: List[Tuple[Node, int]] = []
            pushed = 0
            value: Optional[float] = None
            while True:
                action = node.select()
                path.append((node, action))
                node.child_virtual[action] += VIRTUAL_LOSS
                backend.push(node.moves[action])
                pushed += 1

                child = node.children[action]
                if child is None:
                    child = Node()
                    node.children[action] = child

                if child.terminal is not None:
                    value = child.terminal
                    break
                if not child.expanded:
                    terminal = backend.terminal_value()
                    if terminal is not None:
                        child.terminal = float(terminal)
                        value = child.terminal
                    else:
                        key = id(child)
                        slot = slot_of_node.get(key)
                        if slot is None:
                            moves, raw_moves = backend.legal_moves()
                            slot = len(queue)
                            slot_of_node[key] = slot
                            queue.append((child, encode_planes(backend.snapshot()),
                                          moves, raw_moves, backend.turn))
                        waiters.append((len(paths), slot))
                    break
                node = child

            for _ in range(pushed):
                backend.pop()
            paths.append(path)
            values.append(0.0 if value is None else value)

        if queue:
            planes = np.stack([item[1] for item in queue])
            logits, leaf_values = self.evaluator(planes)
            self.evaluated += len(queue)
            for slot, (child, _planes, moves, raw_moves, turn) in enumerate(queue):
                indices = [policy_index(move, turn) for move in raw_moves]
                child.expand(moves, raw_moves, softmax_over_legal(logits[slot], indices))
            for path_index, slot in waiters:
                values[path_index] = float(leaf_values[slot])

        return paths, values

    def _expand_root(self, node: Node, backend) -> float:
        moves, raw_moves = backend.legal_moves()
        if not moves:
            node.expand([], [], np.zeros(0, dtype=np.float32))
            node.terminal = -1.0
            return -1.0
        logits, values = self.evaluator(encode_planes(backend.snapshot())[None])
        self.evaluated += 1
        indices = [policy_index(move, backend.turn) for move in raw_moves]
        node.expand(moves, raw_moves, softmax_over_legal(logits[0], indices))
        return float(values[0])

    def _apply_root_noise(self, root: Node) -> None:
        noise = self.rng.dirichlet([DIRICHLET_ALPHA] * len(root.moves)).astype(np.float32)
        root.priors = (1.0 - DIRICHLET_EPSILON) * root.priors + DIRICHLET_EPSILON * noise

    @staticmethod
    def _backup(path: Sequence[Tuple[Node, int]], value: float) -> None:
        # ``value`` is from the leaf mover's point of view; flipping it before
        # each edge turns it into that edge owner's point of view.
        for node, action in reversed(path):
            value = -value
            node.child_virtual[action] -= VIRTUAL_LOSS
            node.child_visits[action] += 1.0
            node.child_value[action] += value
            node.visits += 1


def select_move(root: Node, temperature: float, rng: np.random.Generator):
    """Pick a move from the root's visit counts at the given temperature."""
    counts = root.visit_counts() if root.child_visits is not None else np.zeros(0)
    if counts.size == 0 or counts.sum() == 0:
        return (root.moves[0] if root.moves else None), counts
    if temperature <= 1e-6:
        return root.moves[int(np.argmax(counts))], counts
    scaled = counts.astype(np.float64) ** (1.0 / temperature)
    total = scaled.sum()
    probs = scaled / total if total > 0 else np.full(counts.size, 1.0 / counts.size)
    return root.moves[int(rng.choice(counts.size, p=probs))], counts


def policy_targets(root: Node, turn: int) -> List[Tuple[int, float]]:
    """Normalised visit counts as ``(policy_index, probability)`` pairs.

    This is the policy target stored in the training records: a sparse
    distribution over the moves the search actually considered.
    """
    counts = root.visit_counts() if root.child_visits is not None else np.zeros(0)
    total = float(counts.sum())
    if total <= 0.0:
        return []
    pairs = []
    for move, count in zip(root.raw_moves, counts):
        if count > 0.0:
            pairs.append((policy_index(move, turn), float(count) / total))
    return pairs
