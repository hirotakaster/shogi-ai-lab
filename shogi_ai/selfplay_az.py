"""AlphaZero-style self-play: MCTS games in, training records out.

Each record holds the search's own visit distribution as the policy target and
the eventual game result as the value target, both from the point of view of the
side to move.  The check flag is stored alongside the SFEN because the feature
encoder needs it and it cannot be recovered from an SFEN without a rules engine.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .backends import make_backend, raw_move_from_usi
from .core import BLACK
from .encoding import policy_index
from .mcts import MCTS, policy_targets, sample_prior_move, select_move


@dataclass
class SelfPlayConfig:
    simulations: int = 200
    batch_size: int = 16
    max_moves: int = 320
    #: Plies played at temperature 1.0 before switching to greedy play.
    temperature_moves: int = 30
    temperature: float = 1.0
    #: Opening stage: the first ``opening_moves`` plies are sampled at
    #: ``opening_temperature`` instead of ``temperature``.  Campaign 3's
    #: self-play funnelled through a single opening line (at ply 2 only
    #: 2 distinct positions across ~2,000 recorded games), so the opening
    #: stage runs hotter to widen the funnel.  0 disables the stage and
    #: keeps the legacy single-temperature schedule.
    opening_moves: int = 0
    opening_temperature: float = 1.5
    #: Temperature curve: ``(ply, temperature)`` control points in
    #: ascending order.  When non-empty it replaces the opening/regular
    #: schedule above: the temperature is flat at the first point's
    #: value up to its ply, geometrically (log-linearly) interpolated
    #: between neighbouring points, and flat at the last point's value
    #: from its ply on; a point at temperature 0 is a step to greedy
    #: play, not an interpolation target.  Campaign 3 block 4 runs
    #: ``0:4.0,4:4.0,39:1.0,50:1.0,51:0.0`` -- flat at 4.0 for the
    #: first four plies, a geometric decay to 1.0 by ply 39 (passing
    #: 2.0 near ply 21), held to ply 50, greedy after -- because the
    #: net is nowhere near discovering joseki yet: its crystallised
    #: "main line" is early self-play noise that locked in, and
    #: breadth is what the value net is starving for (block 3's milder
    #: diversity fix bought +105 Elo on the external yardstick while
    #: internal gate scores stayed flat).
    temperature_points: Tuple[Tuple[int, float], ...] = ()
    #: Moves banned at the root on specific plies, as ``(ply, USI)``
    #: pairs.  A banned move is removed from the root's candidate list
    #: before the search: never searched, never played, and absent
    #: from the recorded policy target, so the net learns the position
    #: without it.  Campaign 3's records showed why: sente's first
    #: move was the central rook swing 2h5h in 82.2% of recorded games
    #: and the pair 2h5h + 8b5b in 65.5% (rounds 21-30 and 36-45
    #: files; only 13 distinct first-move positions) -- a habit locked
    #: in by early self-play noise, not a discovered joseki.  Empty
    #: (the default) bans nothing.  The ban binds self-play only;
    #: gating and real play keep every move, and the training targets
    #: already teach the banned move ~zero.
    banned_moves: Tuple[Tuple[int, str], ...] = ()
    #: Prior-sampled opening stage: for the first ``prior_sample_moves``
    #: plies the played move is sampled from the network's raw priors
    #: at ``prior_sample_temperature`` instead of from the search's
    #: improved policy.  The recorded policy target is unchanged (the
    #: search's distribution); only the move actually played explores.
    #: This exists because temperature on the improved policy has a
    #: ceiling: the search concentrates on the move its Q values favour,
    #: unfavoured first moves are then never played, so their values
    #: are never corrected by real outcomes -- the self-confirming loop
    #: behind the 82.2 percent first-move concentration above.
    #: Playing from the prior breaks the loop without banning anything:
    #: every legal move keeps the probability the network gives it.
    #: 0 (the default) disables the stage.
    prior_sample_moves: int = 0
    prior_sample_temperature: float = 1.0
    #: Gumbel candidate width during the prior-sample stage; 0 keeps
    #: ``gumbel_candidates``.  A wider root lets the search *consider*
    #: more first moves, so the recorded targets for opening positions
    #: spread over more than the production handful.  Ignored in puct
    #: mode and outside the stage.
    opening_candidates: int = 0
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

    def temperature_at(self, ply: int) -> float:
        """Sampling temperature for ``ply``.

        An explicit ``temperature_points`` curve wins when present;
        the opening stage (when enabled) comes next, then the regular
        temperature window, then greedy play.
        """
        points = self.temperature_points
        if points:
            if ply <= points[0][0]:
                return points[0][1]
            for (p0, t0), (p1, t1) in zip(points, points[1:]):
                if ply < p1:
                    if t0 <= 0.0 or t1 <= 0.0:
                        # A zero endpoint is a step to greedy play,
                        # not something to interpolate towards.
                        return t0
                    frac = (ply - p0) / (p1 - p0)
                    return math.exp(math.log(t0)
                                    + (math.log(t1) - math.log(t0)) * frac)
            return points[-1][1]
        if ply < self.opening_moves:
            return self.opening_temperature
        if ply < self.temperature_moves:
            return self.temperature
        return 0.0


def parse_temperature_points(text: str) -> Tuple[Tuple[int, float], ...]:
    """Parse ``"0:4.0,4:4.0,39:1.0,50:1.0,51:0.0"`` into point tuples.

    Each point is ``ply:temperature``; plies must be non-negative and
    strictly increasing, temperatures non-negative.  An empty string
    yields no points (the legacy schedule applies).
    """
    if not text:
        return ()
    points: List[Tuple[int, float]] = []
    for part in text.split(","):
        ply_text, sep, temp_text = part.partition(":")
        if not sep:
            raise ValueError(
                f"temperature point {part!r} is not ply:temperature")
        try:
            point_ply = int(ply_text)
            temperature = float(temp_text)
        except ValueError:
            raise ValueError(
                f"temperature point {part!r} is not ply:temperature") from None
        if point_ply < 0 or temperature < 0.0:
            raise ValueError(
                f"temperature point {part!r} needs a ply >= 0 "
                "and a temperature >= 0")
        if points and point_ply <= points[-1][0]:
            raise ValueError(
                "temperature point plies must be strictly increasing")
        points.append((point_ply, temperature))
    return tuple(points)


#: A USI board move (``2h5h``, optionally ``+``-suffixed) or drop (``R*5h``).
_USI_MOVE = re.compile(r"^(?:[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i])$")


def parse_banned_moves(text: str) -> Tuple[Tuple[int, str], ...]:
    """Parse ``"0:2h5h,1:8b5b"`` into ``(ply, USI)`` pairs.

    Each entry is ``ply:usi``; plies must be non-negative (repeats are
    allowed -- several moves can be banned on the same ply) and the
    USI must be a board move or a drop.  An empty string yields no
    bans.
    """
    if not text:
        return ()
    entries: List[Tuple[int, str]] = []
    for part in text.split(","):
        ply_text, sep, usi = part.partition(":")
        if not sep or not _USI_MOVE.match(usi):
            raise ValueError(
                f"banned move {part!r} is not ply:usi (e.g. 0:2h5h)")
        try:
            ban_ply = int(ply_text)
        except ValueError:
            raise ValueError(
                f"banned move {part!r} is not ply:usi (e.g. 0:2h5h)") from None
        if ban_ply < 0:
            raise ValueError(f"banned move {part!r} needs a ply >= 0")
        entries.append((ban_ply, usi))
    return tuple(entries)


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
    if config.opening_moves < 0 or config.opening_temperature < 0.0:
        raise ValueError("opening_moves and opening_temperature must be >= 0")
    _plies = [point_ply for point_ply, _ in config.temperature_points]
    if (any(point_ply < 0 for point_ply in _plies)
            or _plies != sorted(set(_plies))
            or any(temp < 0.0 for _, temp in config.temperature_points)):
        raise ValueError("temperature_points must have non-negative, "
                         "strictly increasing plies and temperatures >= 0")
    banned_by_ply: dict = {}
    for ban_ply, usi in config.banned_moves:
        if ban_ply < 0 or not _USI_MOVE.match(usi):
            raise ValueError("banned_moves entries need a ply >= 0 "
                             "and a USI move")
        banned_by_ply.setdefault(ban_ply, []).append(usi)
    if config.prior_sample_moves < 0 or config.prior_sample_temperature < 0.0:
        raise ValueError("prior_sample_moves and prior_sample_temperature "
                         "must be >= 0")
    if config.opening_candidates < 0:
        raise ValueError("opening_candidates must be >= 0")
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
        banned_indices = None
        if ply in banned_by_ply:
            banned_indices = {
                policy_index(raw_move_from_usi(usi), backend.turn)
                for usi in banned_by_ply[ply]}
        prior_stage = ply < config.prior_sample_moves
        root = mcts.run(
            backend, simulations=ply_simulations,
            banned_indices=banned_indices,
            gumbel_candidates=(config.opening_candidates
                               if prior_stage and config.opening_candidates > 0
                               else None))
        if not root.moves:
            outcome = -1 if backend.turn == BLACK else 1
            reason = "no legal move"
            break

        root_value = root.mean_value()
        move = None
        if prior_stage:
            # Opening exploration stage: the played move comes from the
            # network's raw priors, not the search result, so moves the
            # search's Q values have written off still reach real
            # positions and real outcomes.  The record below still
            # stores the search's own policy target.
            move = sample_prior_move(
                root, config.prior_sample_temperature, rng)
        if move is None:
            move, _counts = select_move(root, config.temperature_at(ply), rng)

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
    parser.add_argument("--opening-moves", type=int, default=0,
                        help="plies sampled at --opening-temperature before "
                             "the regular temperature window; 0 disables")
    parser.add_argument("--opening-temperature", type=float, default=1.5,
                        help="sampling temperature for the opening stage")
    parser.add_argument("--temperature-points", type=parse_temperature_points,
                        default=(),
                        help="comma-separated ply:temperature control points, "
                             "e.g. 0:4.0,4:4.0,39:1.0,50:1.0,51:0.0; the "
                             "schedule interpolates geometrically between "
                             "points and replaces the opening/regular "
                             "schedule when given")
    parser.add_argument("--banned-moves", type=parse_banned_moves,
                        default=(),
                        help="comma-separated ply:usi moves banned at the "
                             "self-play root, e.g. 0:2h5h,1:8b5b; a banned "
                             "move is never searched, played or recorded "
                             "on that ply")
    parser.add_argument("--prior-sample-moves", type=int, default=0,
                        help="opening plies whose played move is sampled "
                             "from the network's raw priors instead of the "
                             "search result; the recorded target stays the "
                             "search's; 0 disables")
    parser.add_argument("--prior-sample-temperature", type=float, default=1.0,
                        help="sampling temperature for --prior-sample-moves")
    parser.add_argument("--opening-candidates", type=int, default=0,
                        help="Gumbel candidate width during the "
                             "prior-sample stage; 0 keeps --gumbel-candidates")
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
        opening_moves=args.opening_moves,
        opening_temperature=args.opening_temperature,
        temperature_points=args.temperature_points,
        banned_moves=args.banned_moves,
        prior_sample_moves=args.prior_sample_moves,
        prior_sample_temperature=args.prior_sample_temperature,
        opening_candidates=args.opening_candidates,
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
