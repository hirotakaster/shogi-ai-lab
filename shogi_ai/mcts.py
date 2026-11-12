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
#: Gumbel AlphaZero (Danihelka et al., 2022): the completed-Q transform
#: ``sigma(q) = (C_VISIT + max visits) * C_SCALE * q``.  C_SCALE is tuned
#: for values in [-1, 1]; with raw (unnormalised) Q this keeps the search
#: scores on the same scale as the log-priors they are added to.
GUMBEL_C_VISIT = 50.0
GUMBEL_C_SCALE = 0.1
#: Root actions considered by a Gumbel search.  Shogi's branching factor
#: is large, but at a few hundred simulations only a handful of moves can
#: be searched meaningfully; 16 is the paper's AlphaZero setting.
GUMBEL_CANDIDATES = 16

Evaluator = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


class Node:
    __slots__ = ("moves", "raw_moves", "priors", "raw_priors", "child_visits",
                 "child_value", "child_virtual", "children", "visits",
                 "expanded", "terminal", "improved_policy")

    def __init__(self):
        self.moves: List = []
        self.raw_moves: List = []
        self.priors: Optional[np.ndarray] = None
        #: The network's priors before root noise.  Noise is always mixed
        #: from these, never from an already-noised distribution, so a root
        #: retained across moves keeps its exploration anchored to what the
        #: network actually said.
        self.raw_priors: Optional[np.ndarray] = None
        self.child_visits: Optional[np.ndarray] = None
        self.child_value: Optional[np.ndarray] = None
        self.child_virtual: Optional[np.ndarray] = None
        self.children: List[Optional["Node"]] = []
        self.visits = 0
        self.expanded = False
        #: Set once a node is known to end the game, so it is never selected from.
        self.terminal: Optional[float] = None
        #: Gumbel search only: the improved policy ``softmax(log pi +
        #: sigma(completed Q))`` over the considered root actions.  When
        #: present it, not the visit counts, is the policy target and the
        #: distribution moves are selected from.
        self.improved_policy: Optional[np.ndarray] = None

    def expand(self, moves: Sequence, raw_moves: Sequence, priors: np.ndarray) -> None:
        count = len(moves)
        self.moves = list(moves)
        self.raw_moves = list(raw_moves)
        self.priors = priors
        self.raw_priors = priors.copy()
        self.child_visits = np.zeros(count, dtype=np.float32)
        self.child_value = np.zeros(count, dtype=np.float32)
        self.child_virtual = np.zeros(count, dtype=np.float32)
        self.children = [None] * count
        self.expanded = True
        self.improved_policy = None

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


def completed_qvalues(root: Node, fallback: float) -> np.ndarray:
    """Each root action's Q, with unvisited actions filled by ``fallback``.

    Gumbel AlphaZero scores actions by a *completed* Q: an action the
    search never tried still needs a value, and the root's own value
    estimate is the principled stand-in (it is what the network believed
    before any action was searched).
    """
    if root.child_visits is None:
        return np.zeros(0, dtype=np.float64)
    visits = root.child_visits.astype(np.float64)
    totals = root.child_value.astype(np.float64)
    return np.where(visits > 0.0, totals / np.maximum(visits, 1e-8), fallback)


def _sample_gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    uniform = np.clip(rng.random(size), 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(uniform))


def gumbel_scores(root: Node, fallback: float, gumbel: np.ndarray,
                  c_visit: float = GUMBEL_C_VISIT,
                  c_scale: float = GUMBEL_C_SCALE) -> np.ndarray:
    """``gumbel + log pi + sigma(completed Q)`` for every root action."""
    priors = root.raw_priors if root.raw_priors is not None else root.priors
    logits = np.log(np.clip(priors.astype(np.float64), 1e-12, None))
    max_visits = float(root.child_visits.max()) if root.child_visits is not None else 0.0
    sigma = (c_visit + max_visits) * c_scale
    return gumbel + logits + sigma * completed_qvalues(root, fallback)


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
        search_mode: str = "puct",
        gumbel_candidates: int = GUMBEL_CANDIDATES,
    ):
        if search_mode not in ("puct", "gumbel"):
            raise ValueError(f"unknown search_mode {search_mode!r}")
        self.evaluator = evaluator
        self.simulations = simulations
        self.batch_size = max(1, batch_size)
        self.add_noise = add_noise
        self.rng = rng or np.random.default_rng()
        self.search_mode = search_mode
        self.gumbel_candidates = max(1, gumbel_candidates)
        self.last_root: Optional[Node] = None
        #: Subtree retained by :meth:`advance`, consumed by the next run.
        self._pending_root: Optional[Node] = None
        self.root_value = 0.0
        self.evaluated = 0
        self.simulations_done = 0
        self.elapsed = 0.0

    # ------------------------------------------------------------------
    def run(self, backend, time_limit: Optional[float] = None,
            simulations: Optional[int] = None) -> Node:
        """Search from ``backend``'s current position and return the root.

        ``time_limit`` caps the wall-clock budget in seconds.  It is checked
        between batches, so a search can overshoot by roughly the cost of one
        batch; callers that must not exceed a hard deadline should leave
        headroom.  ``simulations`` overrides the configured budget for this
        call only -- playout-cap randomisation in self-play uses it to give
        most plies a small budget and a few plies the full one.
        """
        budget = self.simulations if simulations is None else max(0, simulations)
        self.evaluated = 0
        self.elapsed = 0.0
        started = time.time()
        # A subtree retained by advance() becomes the root when the caller
        # has played on from exactly the position it describes; anything
        # else starts a fresh tree.
        root = self._pending_root if self._pending_root is not None else Node()
        self._pending_root = None
        root.improved_policy = None
        if root.terminal is not None:
            self.root_value = root.terminal
            self.simulations_done = 0
            self.elapsed = time.time() - started
            self.last_root = root
            return root
        if root.expanded:
            # Retained subtree: its gathered statistics are the root value
            # estimate until the new simulations refine it.
            self.root_value = root.mean_value()
        else:
            self.root_value = self._expand_root(root, backend)
            if not root.moves:
                self.simulations_done = 0
                self.elapsed = time.time() - started
                self.last_root = root
                return root
        if self.search_mode == "gumbel":
            done = self._run_gumbel(root, backend, budget, started, time_limit)
            self.simulations_done = done
            self.elapsed = time.time() - started
            self.last_root = root
            return root
        if self.add_noise:
            self._apply_root_noise(root)

        done = 0
        while done < budget:
            wanted = min(self.batch_size, budget - done)
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
    def _run_gumbel(self, root: Node, backend, budget: int,
                    started: float, time_limit: Optional[float]) -> int:
        """Gumbel AlphaZero at the root: Sequential Halving over top moves.

        Only the ``gumbel_candidates`` actions with the highest
        ``gumbel + log pi`` are considered.  The budget is spent in
        halving phases -- every survivor gets an equal share, then the
        worse half by ``gumbel + log pi + sigma(completed Q)`` is dropped
        -- and any remainder goes to the final winner.  Exploration comes
        from the Gumbel sample itself, so Dirichlet noise is *not* also
        applied; with ``add_noise`` off the sample is zero and the search
        is deterministic, which is what evaluation play wants.

        Below the root the search is ordinary PUCT: the guarantee the
        paper needs is about how the root budget is allocated and how the
        improved policy is read off, both of which live here.
        """
        count = len(root.moves)
        fallback = float(self.root_value)
        if count == 0 or budget <= 0:
            root.improved_policy = (root.raw_priors.copy()
                                    if root.raw_priors is not None else None)
            return 0
        priors = root.raw_priors if root.raw_priors is not None else root.priors
        logits = np.log(np.clip(priors.astype(np.float64), 1e-12, None))
        gumbel = (_sample_gumbel(self.rng, count) if self.add_noise
                  else np.zeros(count, dtype=np.float64))
        considered = int(min(self.gumbel_candidates, count))
        candidates = [int(i) for i in
                      np.argsort(-(gumbel + logits))[:considered]]

        done = 0
        remaining = budget
        while len(candidates) > 1 and remaining > 0:
            phases_left = int(math.ceil(math.log2(len(candidates))))
            per_candidate = max(1, remaining // (len(candidates) * phases_left))
            for action in list(candidates):
                if remaining <= 0:
                    break
                spent = self._run_forced(root, backend, action,
                                         min(per_candidate, remaining))
                done += spent
                remaining -= spent
                if time_limit is not None and time.time() - started >= time_limit:
                    remaining = 0
                    break
            if remaining <= 0:
                break
            scores = gumbel_scores(root, fallback, gumbel)
            ranked = sorted(candidates, key=lambda a: scores[a], reverse=True)
            candidates = ranked[: int(math.ceil(len(ranked) / 2))]
        if remaining > 0 and candidates:
            scores = gumbel_scores(root, fallback, gumbel)
            winner = max(candidates, key=lambda a: scores[a])
            spent = self._run_forced(root, backend, winner, remaining)
            done += spent
            remaining -= spent

        # The improved policy: softmax of log pi + sigma(completed Q),
        # restricted to the considered actions (others get exactly zero,
        # which is the point -- at low budgets the target should not
        # pretend unsearched moves were evaluated).
        scores = gumbel_scores(root, fallback, np.zeros(count, dtype=np.float64))
        improved = np.full(count, -np.inf, dtype=np.float64)
        initial = np.argsort(-(gumbel + logits))[:considered]
        improved[initial] = scores[initial]
        peak = float(np.max(improved[np.isfinite(improved)]))
        exp = np.where(np.isfinite(improved), np.exp(improved - peak), 0.0)
        total = float(exp.sum())
        root.improved_policy = (exp / total).astype(np.float32) if total > 0 else priors.copy()
        return done

    def _run_forced(self, root: Node, backend, action: int, budget: int) -> int:
        """Run ``budget`` simulations whose first edge is ``action``."""
        done = 0
        while done < budget:
            wanted = min(self.batch_size, budget - done)
            paths, values = self._simulate_batch(root, backend, wanted,
                                                 forced_root_action=action)
            if not paths:
                break
            for path, value in zip(paths, values):
                self._backup(path, value)
            done += len(paths)
        return done

    # ------------------------------------------------------------------
    def _simulate_batch(self, root: Node, backend, wanted: int,
                        forced_root_action: Optional[int] = None):
        """Descend ``wanted`` times, evaluate the new leaves in one batch.

        ``forced_root_action`` pins the first edge of every path; below
        the root (and everywhere, when it is None) selection is PUCT.
        """
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
                if node is root and forced_root_action is not None:
                    action = forced_root_action
                else:
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
        base = root.raw_priors if root.raw_priors is not None else root.priors
        noise = self.rng.dirichlet([DIRICHLET_ALPHA] * len(root.moves)).astype(np.float32)
        root.priors = (1.0 - DIRICHLET_EPSILON) * base + DIRICHLET_EPSILON * noise

    def advance(self, move) -> bool:
        """Retain the subtree under ``move`` as the next search's root.

        Callers that play a game move by move -- self-play above all --
        would otherwise throw away every visit the search just gathered
        and rebuild the tree from nothing on the next ply.  The retained
        child's statistics describe exactly the position the caller
        reaches by playing ``move``, so the next run starts from them
        instead of from zero.  Returns False when there is nothing to
        retain (no search yet, the move was not in the tree, or its child
        was never created).
        """
        self._pending_root = None
        root = self.last_root
        if root is None:
            return False
        for index, candidate in enumerate(root.moves):
            if candidate == move:
                child = root.children[index]
                if child is None:
                    return False
                self._pending_root = child
                return True
        return False

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
    """Pick a move at the given temperature.

    PUCT roots are sampled from their visit counts.  A Gumbel root
    carries an improved policy instead: greedy play takes its argmax
    and tempered play samples from it sharpened / flattened by the
    temperature, which is the distribution self-play should explore.
    """
    counts = root.visit_counts() if root.child_visits is not None else np.zeros(0)
    improved = getattr(root, "improved_policy", None)
    if improved is not None and len(root.moves) and float(improved.sum()) > 0.0:
        if temperature <= 1e-6:
            return root.moves[int(np.argmax(improved))], counts
        scaled = improved.astype(np.float64) ** (1.0 / temperature)
        total = scaled.sum()
        probs = scaled / total if total > 0 else np.full(len(root.moves), 1.0 / len(root.moves))
        return root.moves[int(rng.choice(len(root.moves), p=probs))], counts
    if counts.size == 0 or counts.sum() == 0:
        return (root.moves[0] if root.moves else None), counts
    if temperature <= 1e-6:
        return root.moves[int(np.argmax(counts))], counts
    scaled = counts.astype(np.float64) ** (1.0 / temperature)
    total = scaled.sum()
    probs = scaled / total if total > 0 else np.full(counts.size, 1.0 / counts.size)
    return root.moves[int(rng.choice(counts.size, p=probs))], counts


def policy_targets(root: Node, turn: int) -> List[Tuple[int, float]]:
    """The policy target as ``(policy_index, probability)`` pairs.

    For a PUCT root this is the normalised visit distribution; for a
    Gumbel root it is the improved policy, which is the distribution
    that search actually guarantees improvement over the network's.
    Either way it is sparse: only moves the search considered appear.
    """
    improved = getattr(root, "improved_policy", None)
    if improved is not None:
        pairs = []
        for move, prob in zip(root.raw_moves, improved):
            if prob > 0.0:
                pairs.append((policy_index(move, turn), float(prob)))
        return pairs
    counts = root.visit_counts() if root.child_visits is not None else np.zeros(0)
    total = float(counts.sum())
    if total <= 0.0:
        return []
    pairs = []
    for move, count in zip(root.raw_moves, counts):
        if count > 0.0:
            pairs.append((policy_index(move, turn), float(count) / total))
    return pairs
