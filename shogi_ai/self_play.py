"""Heuristic self-play, kept as a cheap way to bootstrap training data.

The records it writes carry the played move rather than a search distribution,
which ``shogi_ai.train_az`` turns into a one-hot policy target.  For real
strength use ``shogi_ai.selfplay_az`` instead.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from typing import List, Optional

from .core import BLACK, Position
from .engine import AlphaBetaEngine


def play_game(
    depth: int,
    max_moves: int,
    time_limit: float = 0.4,
    temperature: float = 0.0,
    top_k: int = 1,
    seed: Optional[int] = None,
) -> List[dict]:
    pos = Position.start()
    engine = AlphaBetaEngine(depth=depth, time_limit=time_limit, seed=seed)
    rng = random.Random(seed)
    records = []
    for _ in range(max_moves):
        result = pos.result()
        if result is not None:
            value = 1 if result == BLACK else -1
            for rec in records:
                rec["result"] = value
            return records
        search = engine.search(pos, depth=depth)
        if search.move is None:
            break
        selected_move = search.move
        if temperature > 0 and top_k > 1 and search.candidates:
            candidates = list(search.candidates[:top_k])
            best = candidates[0][1]
            weights = [pow(2.718281828, (score - best) / (300.0 * temperature)) for _, score in candidates]
            selected_move = rng.choices([move for move, _ in candidates], weights=weights, k=1)[0]
        records.append({
            "sfen": pos.to_sfen(),
            "turn": "black" if pos.turn == BLACK else "white",
            "move": selected_move.usi(),
            "score": search.score,
            "result": 0,
        })
        pos = pos.make_move(selected_move)
    return records


def _seed(game: int) -> int:
    return 20260921 + game * 1009


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-moves", type=int, default=180)
    parser.add_argument("--time-limit", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature; 0 keeps the best move")
    parser.add_argument("--top-k", type=int, default=1, help="sample among the top K searched moves")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--out", default="data/selfplay.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as f:
        if args.processes > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.processes) as executor:
                futures = []
                for game in range(args.games):
                    futures.append(executor.submit(play_game, args.depth, args.max_moves, args.time_limit, args.temperature, args.top_k, _seed(game)))
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    records = future.result()
                    for rec in records:
                        rec["game"] = i
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += len(records)
                    print(f"game {i + 1}/{args.games}: {len(records)} positions")
        else:
            for game in range(args.games):
                records = play_game(args.depth, args.max_moves, args.time_limit, args.temperature, args.top_k, _seed(game))
                for rec in records:
                    rec["game"] = game
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += len(records)
                print(f"game {game + 1}/{args.games}: {len(records)} positions")
    print(f"wrote {total} positions to {args.out}")


if __name__ == "__main__":
    main()

