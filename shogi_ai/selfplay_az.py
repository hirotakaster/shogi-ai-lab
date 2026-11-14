"""AlphaZero-style self-play: MCTS games in, training records out.

Each record holds the search's own visit distribution as the policy target and
the eventual game result as the value target, both from the point of view of the
side to move.  The check flag is stored alongside the SFEN because the feature
encoder needs it and it cannot be recovered from an SFEN without a rules engine.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .backends import make_backend
from .core import BLACK
from .encoding import policy_index
from .mcts import MCTS, policy_targets, select_move


@dataclass
class SelfPlayConfig:
    simulations: int = 200
    batch_size: int = 16
    max_moves: int = 320
    #: Plies played at temperature 1.0 before switching to greedy play.
    temperature_moves: int = 30
    temperature: float = 1.0
    #: Resign when the root value stays below this for ``resign_patience``
    #: consecutive moves.  Set to -1.0 to never resign.
    resign_threshold: float = -0.95
    resign_patience: int = 4
    backend: str = "auto"
    model: Optional[str] = None
    preset: str = "base"
    device: str = "auto"
    fp16: Optional[bool] = None
    #: "puct" (default) or "gumbel" (Gumbel AlphaZero at the root).
    search_mode: str = "puct"
    gumbel_candidates: int = 16
    #: Playout-cap randomisation (KataGo): when > 0, only this fraction
    #: of plies is searched with the full ``simulations`` budget; the
    #: rest get ``playout_cap_simulations`` and are *played but not
    #: recorded* -- a capped search's policy target is too noisy to
    #: train on, so only full plies become training positions.  The
    #: saving is spent on more games instead of more simulations.
    playout_cap_fraction: float = 0.0
    #: Budget for capped plies; 0 means a quarter of ``simulations``.
    playout_cap_simulations: int = 0
    #: Games played concurrently in one process (threads) sharing a
    #: :class:`shogi_ai.evaluator.BatchingEvaluator`, so leaf evaluations
    #: from different games coalesce into larger network batches.  Only
    #: the single-process path uses it; multiprocess workers each own a
    #: GPU context already and play one game at a time per task.
    parallel_games: int = 1
    #: Largest batch the coalescing evaluator may form across games.
    inference_batch_size: int = 256
    #: How long the coalescing evaluator waits for more planes (ms).
    inference_wait_ms: float = 5.0
    #: First-play urgency reduction for PUCT selection below the root;
    #: None keeps the legacy behaviour (unvisited edges score 0.0).
    fpu_reduction: Optional[float] = None

    def capped_simulations(self) -> int:
        if self.playout_cap_simulations > 0:
            return self.playout_cap_simulations
        return max(8, self.simulations // 4)


@dataclass
class GameSummary:
    records: List[dict] = field(default_factory=list)
    outcome: int = 0
    plies: int = 0
    resigned: bool = False
    reason: str = ""
    #: Plies played with a capped search (not recorded) under playout-cap
    #: randomisation.
    low_cap_plies: int = 0


def play_one_game(evaluator, config: SelfPlayConfig, seed: int) -> GameSummary:
    """Play a single self-play game and return its training records."""
    rng = np.random.default_rng(seed)
    backend = make_backend(config.backend)
    if not 0.0 <= config.playout_cap_fraction <= 1.0:
        raise ValueError("playout_cap_fraction must lie in [0, 1]")
    cap_enabled = config.playout_cap_fraction > 0.0
    mcts = MCTS(evaluator, simulations=config.simulations,
                batch_size=config.batch_size, add_noise=True, rng=rng,
                search_mode=config.search_mode,
                gumbel_candidates=config.gumbel_candidates,
                fpu_reduction=config.fpu_reduction)

    pending: List[dict] = []
    outcome: Optional[int] = None
    reason = ""
    resigned = False
    low_value_streak = 0
    low_cap_plies = 0
    ply = 0

    while ply < config.max_moves:
        terminal = backend.terminal_value()
        if terminal is not None:
            # Expressed for the side to move; translate to Black's view.
            outcome = int(round(terminal)) * (1 if backend.turn == BLACK else -1)
            reason = "draw" if terminal == 0.0 else "mate"
            break

        # Playout-cap randomisation decides this ply's budget before the
        # search: full plies are recorded, capped plies only move the
        # game forward.
        is_full = (not cap_enabled) or float(rng.random()) < config.playout_cap_fraction
        ply_simulations = (config.simulations if is_full
                           else config.capped_simulations())
        root = mcts.run(backend, simulations=ply_simulations)
        if not root.moves:
            outcome = -1 if backend.turn == BLACK else 1
            reason = "no legal move"
            break

        root_value = root.mean_value()
        temperature = config.temperature if ply < config.temperature_moves else 0.0
        move, _counts = select_move(root, temperature, rng)

        if is_full:
            pending.append({
                "sfen": backend.sfen(),
                "turn": "black" if backend.turn == BLACK else "white",
                "check": bool(backend.snapshot().in_check),
                "policy": [[int(i), round(float(p), 6)] for i, p in policy_targets(root, backend.turn)],
                "q": round(root_value, 4),
                "ply": ply,
            })
        else:
            low_cap_plies += 1

        # A capped search's value is too noisy to resign on; only full
        # plies count toward the streak.
        if is_full and config.resign_threshold > -1.0:
            low_value_streak = low_value_streak + 1 if root_value < config.resign_threshold else 0
            if low_value_streak >= config.resign_patience:
                outcome = -1 if backend.turn == BLACK else 1
                reason = "resign"
                resigned = True
                break

        backend.push(move)
        # Keep the played move's subtree: the next ply's search starts
        # from its statistics instead of rebuilding the tree from zero.
        mcts.advance(move)
        ply += 1

    if outcome is None:
        outcome = 0
        reason = "max moves"

    for record in pending:
        sign = 1 if record["turn"] == "black" else -1
        record["value"] = float(outcome * sign)
        if reason == "max moves":
            # Not a played-out result: training marks these so the value
            # target can come from the search instead of a forced draw.
            record["capped"] = True
    return GameSummary(records=pending, outcome=outcome, plies=ply,
                       resigned=resigned, reason=reason,
                       low_cap_plies=low_cap_plies)


#: Per-process state, set once by :func:`_init_worker`.
_WORKER_EVALUATOR = None
_WORKER_CONFIG: Optional[SelfPlayConfig] = None


def _init_worker(config: SelfPlayConfig, seed: int) -> None:
    """Load the network once per worker process."""
    global _WORKER_EVALUATOR, _WORKER_CONFIG
    from .evaluator import load_evaluator

    _WORKER_CONFIG = config
    _WORKER_EVALUATOR = load_evaluator(config.model, preset=config.preset,
                                       device=config.device, fp16=config.fp16,
                                       seed=seed)


def _play_seed(seed: int) -> GameSummary:
    """Play one game with the worker's already-loaded network.

    One game per task rather than a fixed slice each: the pool then hands out
    work dynamically, so a worker that draws a run of long games no longer
    becomes the critical path while the others sit idle, and each finished game
    is reported as it lands instead of in a burst at the end of a slice.
    """
    return play_one_game(_WORKER_EVALUATOR, _WORKER_CONFIG, seed)


def _concurrent_summaries(evaluator, config: SelfPlayConfig, seeds):
    """Yield game summaries from games played concurrently in threads.

    Each game owns its backend, search and RNG; the only shared state is
    the evaluator, wrapped so leaf batches from different games coalesce
    into single network calls.  Summaries are yielded as games finish.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .evaluator import BatchingEvaluator

    batching = BatchingEvaluator(
        evaluator,
        max_batch=config.inference_batch_size,
        max_wait=config.inference_wait_ms / 1000.0,
    )
    try:
        with ThreadPoolExecutor(max_workers=config.parallel_games) as pool:
            futures = [pool.submit(play_one_game, batching, config, seed)
                       for seed in seeds]
            for future in as_completed(futures):
                yield future.result()
    finally:
        batching.close()


def generate(config: SelfPlayConfig, games: int, out_path: str,
             processes: int = 1, seed: int = 20260926) -> dict:
    """Run ``games`` self-play games and append them to ``out_path``."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    seeds = [seed + i * 7919 for i in range(games)]
    started = time.time()
    stats = {"games": 0, "positions": 0, "black": 0, "white": 0, "draw": 0,
             "resigned": 0, "plies": 0, "low_cap_plies": 0}

    # Write to a scratch file and rename only on success.  An interrupted run
    # must not leave a truncated -- or worse, empty -- file behind, because a
    # resuming learn_loop would mistake that for a finished round.
    partial_path = out_path + ".partial"
    try:
        with open(partial_path, "w", encoding="utf-8") as sink:
            if processes > 1:
                # spawn keeps each worker's CUDA context independent.
                context = mp.get_context("spawn")
                with context.Pool(processes, initializer=_init_worker,
                                  initargs=(config, seed)) as pool:
                    for summary in pool.imap_unordered(_play_seed, seeds, chunksize=1):
                        _write(sink, summary, stats)
                        _report(stats, games, started)
            else:
                from .evaluator import load_evaluator

                evaluator = load_evaluator(config.model, preset=config.preset,
                                           device=config.device, fp16=config.fp16, seed=seed)
                if config.parallel_games > 1:
                    summaries = _concurrent_summaries(evaluator, config, seeds)
                else:
                    summaries = (play_one_game(evaluator, config, game_seed)
                                 for game_seed in seeds)
                for summary in summaries:
                    _write(sink, summary, stats)
                    _report(stats, games, started)
            sink.flush()
            os.fsync(sink.fileno())
    except BaseException:
        # Includes KeyboardInterrupt, which is the common way a Colab run dies.
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise

    if stats["positions"] == 0:
        os.remove(partial_path)
        raise SystemExit(
            f"self-play produced no positions from {games} games; "
            f"not writing {out_path}"
        )
    os.replace(partial_path, out_path)
    stats["seconds"] = round(time.time() - started, 1)
    return stats


def _write(sink, summary: GameSummary, stats: dict) -> None:
    game_id = stats["games"]
    for record in summary.records:
        record["game"] = game_id
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["games"] += 1
    stats["positions"] += len(summary.records)
    stats["plies"] += summary.plies
    stats["low_cap_plies"] += summary.low_cap_plies
    stats["resigned"] += int(summary.resigned)
    stats["black" if summary.outcome > 0 else "white" if summary.outcome < 0 else "draw"] += 1


def _report(stats: dict, total: int, started: float) -> None:
    done = stats["games"]
    elapsed = time.time() - started
    rate = done / elapsed if elapsed else 0.0
    print(f"game {done}/{total}: {stats['positions']} positions, "
          f"B/W/D {stats['black']}/{stats['white']}/{stats['draw']}, "
          f"{rate:.2f} games/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-style self-play.")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="leaves evaluated per network call")
    parser.add_argument("--max-moves", type=int, default=320)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--resign-threshold", type=float, default=-0.95,
                        help="-1.0 disables resignation")
    parser.add_argument("--backend", choices=("auto", "cshogi", "python"), default="auto")
    parser.add_argument("--search-mode", choices=("puct", "gumbel"), default="puct",
                        help="root search: PUCT, or Gumbel AlphaZero for low budgets")
    parser.add_argument("--gumbel-candidates", type=int, default=16)
    parser.add_argument("--fpu-reduction", type=float, default=None,
                        help="first-play urgency: unvisited edges inherit the "
                             "node's mean value minus this; unset keeps 0.0")
    parser.add_argument("--playout-cap-fraction", type=float, default=0.0,
                        help="fraction of plies searched at full budget; "
                             "0 disables playout-cap randomisation")
    parser.add_argument("--playout-cap-simulations", type=int, default=0,
                        help="budget for capped plies; 0 means simulations // 4")
    parser.add_argument("--parallel-games", type=int, default=1,
                        help="games played concurrently in-process, sharing "
                             "batched inference (single-process path only)")
    parser.add_argument("--inference-batch-size", type=int, default=256,
                        help="largest cross-game inference batch")
    parser.add_argument("--inference-wait-ms", type=float, default=5.0,
                        help="how long inference waits to coalesce a batch (ms)")
    parser.add_argument("--model", default="", help="checkpoint; omitted means a random network")
    parser.add_argument("--preset", default="base", help="architecture when --model is omitted")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", default="data/selfplay_az.jsonl")
    args = parser.parse_args()

    config = SelfPlayConfig(
        simulations=args.simulations,
        batch_size=args.batch_size,
        max_moves=args.max_moves,
        temperature_moves=args.temperature_moves,
        resign_threshold=args.resign_threshold,
        backend=args.backend,
        model=args.model or None,
        preset=args.preset,
        device=args.device,
        search_mode=args.search_mode,
        gumbel_candidates=args.gumbel_candidates,
        fpu_reduction=args.fpu_reduction,
        playout_cap_fraction=args.playout_cap_fraction,
        playout_cap_simulations=args.playout_cap_simulations,
        parallel_games=args.parallel_games,
        inference_batch_size=args.inference_batch_size,
        inference_wait_ms=args.inference_wait_ms,
    )
    stats = generate(config, args.games, args.out, processes=args.processes, seed=args.seed)
    print(f"wrote {stats['positions']} positions from {stats['games']} games "
          f"to {args.out} in {stats['seconds']}s "
          f"(B/W/D {stats['black']}/{stats['white']}/{stats['draw']}, "
          f"resigned {stats['resigned']}, low-cap plies {stats['low_cap_plies']})")


if __name__ == "__main__":
    main()
