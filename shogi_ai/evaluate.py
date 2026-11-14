"""Play matches between two players to decide whether a generation improved.

Self-play training only helps if each new checkpoint is actually stronger, and
a falling loss does not prove that.  This module plays a match and reports the
challenger's score, so a training loop can keep or discard a generation.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .backends import PurePythonBackend, make_backend
from .core import BLACK
from .mcts import MCTS, select_move


class MctsPlayer:
    """Plays with MCTS over a policy/value network."""

    def __init__(self, evaluator, simulations: int = 200, batch_size: int = 16,
                 name: str = "mcts", search_mode: str = "puct",
                 gumbel_candidates: int = 16, add_noise: bool = False):
        self.evaluator = evaluator
        self.simulations = simulations
        self.batch_size = batch_size
        self.name = name
        self.search_mode = search_mode
        self.gumbel_candidates = gumbel_candidates
        self.add_noise = add_noise

    def choose(self, backend, ply: int, temperature: float, rng):
        mcts = MCTS(self.evaluator, simulations=self.simulations,
                    batch_size=self.batch_size, add_noise=self.add_noise, rng=rng,
                    search_mode=self.search_mode,
                    gumbel_candidates=self.gumbel_candidates)
        root = mcts.run(backend)
        if not root.moves:
            return None
        move, _counts = select_move(root, temperature, rng)
        return move


class HeuristicPlayer:
    """Plays with the dependency-free alpha-beta engine, as a fixed baseline."""

    name = "heuristic"

    def __init__(self, depth: int = 3, time_limit: float = 1.0):
        from .engine import AlphaBetaEngine

        self.engine = AlphaBetaEngine(depth=depth, time_limit=time_limit)

    def choose(self, backend, ply: int, temperature: float, rng):
        if not isinstance(backend, PurePythonBackend):
            raise SystemExit("the heuristic player needs --backend python")
        result = self.engine.search(backend.position)
        return result.move


@dataclass
class MatchResult:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    plies: int = 0
    seconds: float = 0.0

    @property
    def score(self) -> float:
        """Challenger score with draws counted as a half point."""
        if not self.games:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def elo_diff(self) -> Optional[float]:
        s = self.score
        if s <= 0.0 or s >= 1.0:
            return None
        return -400.0 * math.log10(1.0 / s - 1.0)

    def summary(self, challenger: str = "challenger", champion: str = "champion") -> str:
        elo = self.elo_diff
        elo_text = f"{elo:+.0f} Elo" if elo is not None else "Elo n/a (no losses or no wins)"
        return (f"{challenger} vs {champion}: {self.wins}W-{self.losses}L-{self.draws}D "
                f"in {self.games} games, score {self.score:.3f} ({elo_text}), "
                f"{self.plies / max(self.games, 1):.0f} plies/game, {self.seconds:.1f}s")


def _play_game(challenger, champion, game_index: int, backend_kind: str,
               max_moves: int, opening_moves: int, seed: int):
    """Play one match game; return (outcome, plies).

    ``outcome`` is from Black's point of view: +1 Black won, 0 draw,
    -1 White won.  The challenger takes Black in even-indexed games.
    The per-game RNG depends only on (seed, game_index), so a game plays
    out identically no matter which process runs it -- that is what makes
    the parallel match below a pure scheduling change.
    """
    rng = np.random.default_rng(seed + game_index * 7919)
    backend = make_backend(backend_kind)
    challenger_is_black = game_index % 2 == 0
    outcome: Optional[int] = None
    ply = 0

    while ply < max_moves:
        terminal = backend.terminal_value()
        if terminal is not None:
            outcome = int(round(terminal)) * (1 if backend.turn == BLACK else -1)
            break
        mover_is_challenger = (backend.turn == BLACK) == challenger_is_black
        player = challenger if mover_is_challenger else champion
        # A little opening randomness keeps the games from being identical.
        temperature = 1.0 if ply < opening_moves else 0.0
        move = player.choose(backend, ply, temperature, rng)
        if move is None:
            outcome = -1 if backend.turn == BLACK else 1
            break
        backend.push(move)
        ply += 1

    if outcome is None:
        outcome = 0
    return outcome, ply


def _record(result: MatchResult, game_index: int, outcome: int, plies: int) -> None:
    """Tally one finished game (challenger's perspective) into ``result``."""
    result.games += 1
    result.plies += plies
    if outcome == 0:
        result.draws += 1
    elif (outcome > 0) == (game_index % 2 == 0):
        result.wins += 1
    else:
        result.losses += 1


def _progress(result: MatchResult, games: int) -> None:
    print(f"game {result.games}/{games}: "
          f"{result.wins}W-{result.losses}L-{result.draws}D "
          f"(score {result.score:.3f})", flush=True)


def play_match(challenger, champion, games: int = 20, backend_kind: str = "auto",
               max_moves: int = 320, opening_moves: int = 8,
               seed: int = 20260926, verbose: bool = True) -> MatchResult:
    """Play ``games`` games, alternating who takes Black."""
    result = MatchResult()
    started = time.time()

    for game in range(games):
        outcome, plies = _play_game(challenger, champion, game,
                                    backend_kind, max_moves, opening_moves, seed)
        _record(result, game, outcome, plies)
        if verbose:
            _progress(result, games)

    result.seconds = time.time() - started
    return result


def _build_player(spec: str, args, name: str):
    if spec == "heuristic":
        return HeuristicPlayer(depth=args.heuristic_depth, time_limit=args.heuristic_time)
    if spec == "random":
        from .evaluator import UniformEvaluator

        return MctsPlayer(UniformEvaluator(), simulations=args.simulations,
                          batch_size=args.batch_size, name="random-net",
                          search_mode=args.search_mode,
                          gumbel_candidates=args.gumbel_candidates,
                          add_noise=getattr(args, "add_noise", False))
    from .evaluator import load_evaluator

    evaluator = load_evaluator(spec, preset=args.preset, device=args.device)
    return MctsPlayer(evaluator, simulations=args.simulations,
                      batch_size=args.batch_size, name=name,
                      search_mode=args.search_mode,
                      gumbel_candidates=args.gumbel_candidates,
                      add_noise=getattr(args, "add_noise", False))


#: Per-process state for the parallel match, set by :func:`_init_match_worker`.
_WORKER_PLAYERS = None
_WORKER_CONFIG = None


def _init_match_worker(challenger_spec: str, champion_spec: str, args_dict: dict) -> None:
    """Build this worker's two players once, from their spec strings."""
    global _WORKER_PLAYERS, _WORKER_CONFIG
    args = SimpleNamespace(**args_dict)
    _WORKER_PLAYERS = (_build_player(challenger_spec, args, "challenger"),
                       _build_player(champion_spec, args, "champion"))
    _WORKER_CONFIG = args


def _play_index(game_index: int):
    """Play one match game with the worker's already-built players.

    One game per task, as in self-play: the pool hands out work dynamically,
    so a worker that draws a run of long games does not become the critical
    path while the others sit idle.
    """
    challenger, champion = _WORKER_PLAYERS
    args = _WORKER_CONFIG
    outcome, plies = _play_game(challenger, champion, game_index, args.backend,
                                args.max_moves, args.opening_moves, args.seed)
    return game_index, outcome, plies


def play_match_parallel(challenger_spec: str, champion_spec: str, args,
                        games: int, processes: int, verbose: bool = True) -> MatchResult:
    """Play a match across ``processes`` workers.

    Same games, colours, per-game seeds and tallying as :func:`play_match`;
    players are given as spec strings because each worker must build its
    own (a loaded network cannot be handed to another process).  Results
    are tallied in the order games finish.
    """
    result = MatchResult()
    started = time.time()
    # spawn keeps each worker's CUDA context independent, as in self-play.
    context = mp.get_context("spawn")
    with context.Pool(processes, initializer=_init_match_worker,
                      initargs=(challenger_spec, champion_spec, vars(args))) as pool:
        for game_index, outcome, plies in pool.imap_unordered(
                _play_index, range(games), chunksize=1):
            _record(result, game_index, outcome, plies)
            if verbose:
                _progress(result, games)
    result.seconds = time.time() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a gating match between two players.")
    parser.add_argument("--challenger", required=True,
                        help="checkpoint path, or 'heuristic' / 'random'")
    parser.add_argument("--champion", required=True,
                        help="checkpoint path, or 'heuristic' / 'random'")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--processes", type=int, default=1,
                        help="workers playing match games in parallel (1 = sequential)")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--search-mode", choices=("puct", "gumbel"), default="puct",
                        help="root search mode for both MCTS players")
    parser.add_argument("--gumbel-candidates", type=int, default=16)
    parser.add_argument("--max-moves", type=int, default=320)
    parser.add_argument("--opening-moves", type=int, default=8)
    parser.add_argument("--add-noise", action="store_true",
                        help="per-game root noise in the gating match "
                             "(Gumbel sample / Dirichlet noise, seeded by "
                             "seed + game index, so a match stays "
                             "reproducible and parallel-invariant)")
    parser.add_argument("--backend", choices=("auto", "cshogi", "python"), default="auto")
    parser.add_argument("--preset", default="base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--heuristic-depth", type=int, default=3)
    parser.add_argument("--heuristic-time", type=float, default=1.0)
    parser.add_argument("--gate", type=float, default=0.0,
                        help="if set, exit non-zero when the score falls below this")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    if args.challenger == "heuristic" or args.champion == "heuristic":
        if args.backend == "cshogi":
            raise SystemExit("the heuristic player needs --backend python or auto")
        args.backend = "python"

    if args.processes < 1:
        raise SystemExit("--processes must be at least 1")
    if args.processes > 1:
        result = play_match_parallel(args.challenger, args.champion, args,
                                     games=args.games, processes=args.processes)
    else:
        challenger = _build_player(args.challenger, args, "challenger")
        champion = _build_player(args.champion, args, "champion")
        result = play_match(challenger, champion, games=args.games,
                            backend_kind=args.backend, max_moves=args.max_moves,
                            opening_moves=args.opening_moves, seed=args.seed)
    print(result.summary(args.challenger, args.champion))

    if args.gate > 0.0:
        if result.score < args.gate:
            raise SystemExit(f"gate failed: {result.score:.3f} < {args.gate:.3f}")
        print(f"gate passed: {result.score:.3f} >= {args.gate:.3f}")


if __name__ == "__main__":
    main()
