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


@dataclass
class GameSummary:
    records: List[dict] = field(default_factory=list)
    outcome: int = 0
    plies: int = 0
    resigned: bool = False
    reason: str = ""


def play_one_game(evaluator, config: SelfPlayConfig, seed: int) -> GameSummary:
    """Play a single self-play game and return its training records."""
    rng = np.random.default_rng(seed)
    backend = make_backend(config.backend)
    mcts = MCTS(evaluator, simulations=config.simulations,
                batch_size=config.batch_size, add_noise=True, rng=rng)

    pending: List[dict] = []
    outcome: Optional[int] = None
    reason = ""
    resigned = False
    low_value_streak = 0
    ply = 0

    while ply < config.max_moves:
        terminal = backend.terminal_value()
        if terminal is not None:
            # Expressed for the side to move; translate to Black's view.
            outcome = int(round(terminal)) * (1 if backend.turn == BLACK else -1)
            reason = "draw" if terminal == 0.0 else "mate"
            break

        root = mcts.run(backend)
        if not root.moves:
            outcome = -1 if backend.turn == BLACK else 1
            reason = "no legal move"
            break

        root_value = root.mean_value()
        temperature = config.temperature if ply < config.temperature_moves else 0.0
        move, _counts = select_move(root, temperature, rng)

        pending.append({
            "sfen": backend.sfen(),
            "turn": "black" if backend.turn == BLACK else "white",
            "check": bool(backend.snapshot().in_check),
            "policy": [[int(i), round(float(p), 6)] for i, p in policy_targets(root, backend.turn)],
            "q": round(root_value, 4),
            "ply": ply,
        })

        if config.resign_threshold > -1.0:
            low_value_streak = low_value_streak + 1 if root_value < config.resign_threshold else 0
            if low_value_streak >= config.resign_patience:
                outcome = -1 if backend.turn == BLACK else 1
                reason = "resign"
                resigned = True
                break

        backend.push(move)
        ply += 1

    if outcome is None:
        outcome = 0
        reason = "max moves"

    for record in pending:
        sign = 1 if record["turn"] == "black" else -1
        record["value"] = float(outcome * sign)
    return GameSummary(records=pending, outcome=outcome, plies=ply,
                       resigned=resigned, reason=reason)


def _worker(payload):
    """Play a slice of the games in a separate process."""
    config, seeds = payload
    from .evaluator import load_evaluator

    evaluator = load_evaluator(config.model, preset=config.preset,
                               device=config.device, fp16=config.fp16,
                               seed=seeds[0])
    return [play_one_game(evaluator, config, seed) for seed in seeds]


def generate(config: SelfPlayConfig, games: int, out_path: str,
             processes: int = 1, seed: int = 20260926) -> dict:
    """Run ``games`` self-play games and append them to ``out_path``."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    seeds = [seed + i * 7919 for i in range(games)]
    started = time.time()
    stats = {"games": 0, "positions": 0, "black": 0, "white": 0, "draw": 0,
             "resigned": 0, "plies": 0}

    # Write to a scratch file and rename only on success.  An interrupted run
    # must not leave a truncated -- or worse, empty -- file behind, because a
    # resuming learn_loop would mistake that for a finished round.
    partial_path = out_path + ".partial"
    try:
        with open(partial_path, "w", encoding="utf-8") as sink:
            if processes > 1:
                chunks = [seeds[i::processes] for i in range(processes)]
                context = mp.get_context("spawn")
                with context.Pool(processes) as pool:
                    for summaries in pool.imap_unordered(_worker, [(config, c) for c in chunks if c]):
                        for summary in summaries:
                            _write(sink, summary, stats)
                            _report(stats, games, started)
            else:
                from .evaluator import load_evaluator

                evaluator = load_evaluator(config.model, preset=config.preset,
                                           device=config.device, fp16=config.fp16, seed=seed)
                for game_seed in seeds:
                    _write(sink, play_one_game(evaluator, config, game_seed), stats)
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
    )
    stats = generate(config, args.games, args.out, processes=args.processes, seed=args.seed)
    print(f"wrote {stats['positions']} positions from {stats['games']} games "
          f"to {args.out} in {stats['seconds']}s "
          f"(B/W/D {stats['black']}/{stats['white']}/{stats['draw']}, "
          f"resigned {stats['resigned']})")


if __name__ == "__main__":
    main()
