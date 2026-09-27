"""Drive the AlphaZero loop: self-play, train, then gate on a match.

Each round runs as a subprocess so that a CUDA context is released before the
next stage starts -- on a shared GPU that matters more than the small startup
cost.  A round only replaces the current best model when the freshly trained
network actually beats it over a match, which is what stops a bad generation
from poisoning every later round.

Rounds are resumable: existing per-round artefacts are reused, so re-running
after a disconnect picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def count_records(path: Path) -> int:
    """Non-blank lines in a JSONL file, or ``-1`` when it does not exist."""
    if not path.exists():
        return -1
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total += 1
    return total


def usable_data(path: Path) -> bool:
    """True when a round's data file exists and actually holds records.

    Existence alone is not enough: an interrupted run used to leave an empty
    file, which a resuming loop then reused as if the round were finished.
    """
    records = count_records(path)
    if records > 0:
        return True
    if records == 0:
        print(f"{path} exists but is empty (an interrupted run); regenerating it")
        path.unlink()
    return False


def _load_history(path: Path) -> List[dict]:
    """Read the round history, tolerating a truncated file."""
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"could not read {path} ({exc}); starting a fresh history")
        return []
    return loaded if isinstance(loaded, list) else []


def _print_plan(args, data_dir: Path, best_model: Path, initial_data,
                finished_rounds, current) -> None:
    """Say what will be reused before doing any work.

    Pointing --data-dir somewhere new silently restarts the whole run, which is
    easy to do by accident when moving output onto Drive, so the state is
    spelled out up front rather than discovered ten minutes in.
    """
    print("\nplan")
    print(f"  data dir    : {data_dir}")
    print(f"  champion    : {best_model} "
          f"({'found' if current is not None else 'MISSING -> round 1 starts from a random network'})")
    if initial_data is not None:
        print(f"  initial data: {initial_data}")

    reusable = []
    for round_number in range(1, args.rounds + 1):
        data_path = data_dir / f"selfplay_{round_number:03d}.jsonl"
        if round_number in finished_rounds:
            reusable.append(f"{round_number}(done)")
        elif count_records(data_path) > 0:
            reusable.append(f"{round_number}(data)")
    if reusable:
        print(f"  reusing     : rounds {', '.join(reusable)}")
    else:
        print(f"  reusing     : nothing -- all {args.rounds} rounds will run from scratch")
    print()


def run(cmd: List[str], cwd: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play, train, and gate in rounds.")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--games", type=int, default=50, help="self-play games per round")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16, help="MCTS leaf batch")
    parser.add_argument("--max-moves", type=int, default=320)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--resign-threshold", type=float, default=-0.95)
    parser.add_argument("--processes", type=int, default=1, help="self-play workers")
    parser.add_argument("--backend", choices=("auto", "cshogi", "python"), default="auto")

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max-positions", type=int, default=400_000,
                        help="replay buffer cap across rounds")
    parser.add_argument("--preset", default="base")
    parser.add_argument("--workers", type=int, default=2, help="data loader workers")

    parser.add_argument("--eval-games", type=int, default=20,
                        help="gating match length; 0 promotes every round unchecked")
    parser.add_argument("--gate", type=float, default=0.55,
                        help="score the challenger must reach to be promoted")
    parser.add_argument("--eval-simulations", type=int, default=0,
                        help="defaults to --simulations")

    parser.add_argument("--initial-data", default="",
                        help="existing JSONL used for round 1 instead of self-play")
    parser.add_argument("--data-dir", default="data/learn")
    parser.add_argument("--out-model", default="data/policy_value.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")

    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    best_model = Path(args.out_model)
    best_model.parent.mkdir(parents=True, exist_ok=True)

    initial_data = Path(args.initial_data) if args.initial_data else None
    if initial_data is not None and not initial_data.exists():
        raise SystemExit(f"initial data does not exist: {initial_data}")

    current: Optional[Path] = best_model if best_model.exists() else None
    if current:
        print(f"starting from the existing checkpoint {current}")
    else:
        print("no checkpoint yet: round 1 self-play will use a random network")

    log_path = data_dir / "history.json"
    history = _load_history(log_path)
    # A round recorded in the history finished its gating match too, so there is
    # nothing left to do for it.  Without this a re-run would replay every
    # gating match while reusing the data and the checkpoint.
    finished_rounds = {entry["round"] for entry in history
                       if isinstance(entry, dict) and isinstance(entry.get("round"), int)}
    eval_simulations = args.eval_simulations or args.simulations

    _print_plan(args, data_dir, best_model, initial_data, finished_rounds, current)

    for round_number in range(1, args.rounds + 1):
        if round_number in finished_rounds:
            print(f"round {round_number}/{args.rounds}: already finished in an "
                  f"earlier run; skipping", flush=True)
            continue

        started = time.time()
        print(f"\n{'=' * 60}\nround {round_number}/{args.rounds}\n{'=' * 60}", flush=True)

        # ---- 1. data -------------------------------------------------
        if round_number == 1 and initial_data is not None:
            data_path = initial_data
            print(f"round 1 trains from the supplied data {data_path}")
        else:
            data_path = data_dir / f"selfplay_{round_number:03d}.jsonl"
            if usable_data(data_path):
                print(f"reusing existing {data_path} "
                      f"({count_records(data_path)} positions)")
            else:
                cmd = [
                    sys.executable, "-m", "shogi_ai.selfplay_az",
                    "--games", str(args.games),
                    "--simulations", str(args.simulations),
                    "--batch-size", str(args.batch_size),
                    "--max-moves", str(args.max_moves),
                    "--temperature-moves", str(args.temperature_moves),
                    "--resign-threshold", str(args.resign_threshold),
                    "--processes", str(args.processes),
                    "--backend", args.backend,
                    "--preset", args.preset,
                    "--device", args.device,
                    "--seed", str(args.seed + round_number * 104729),
                    "--out", str(data_path),
                ]
                if current is not None:
                    cmd += ["--model", str(current)]
                run(cmd, root)

        # ---- 2. train ------------------------------------------------
        candidate = data_dir / f"policy_value_{round_number:03d}.pt"
        # Checkpoints are saved atomically, so a file that is here is complete.
        if candidate.exists() and candidate.stat().st_size > 0:
            print(f"reusing existing {candidate}")
        else:
            # Train on everything generated so far, capped to the buffer size.
            buffer: List[str] = []
            if initial_data is not None:
                buffer.append(str(initial_data))
            buffer += [str(p) for p in sorted(data_dir.glob("selfplay_*.jsonl"))
                       if count_records(p) > 0]
            if not buffer:
                raise SystemExit("no training data available for this round")
            cmd = [
                sys.executable, "-m", "shogi_ai.train_az",
                "--data", *buffer,
                "--max-positions", str(args.max_positions),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.train_batch_size),
                "--lr", str(args.lr),
                "--preset", args.preset,
                "--workers", str(args.workers),
                "--device", args.device,
                "--seed", str(args.seed + round_number),
                "--out", str(candidate),
            ]
            if current is not None:
                cmd += ["--init", str(current)]
            run(cmd, root)

        # ---- 3. gate -------------------------------------------------
        promoted = True
        score = None
        if current is None or args.eval_games <= 0:
            reason = "no champion to beat" if current is None else "gating disabled"
            print(f"promoting without a match ({reason})")
        else:
            cmd = [
                sys.executable, "-m", "shogi_ai.evaluate",
                "--challenger", str(candidate),
                "--champion", str(current),
                "--games", str(args.eval_games),
                "--simulations", str(eval_simulations),
                "--batch-size", str(args.batch_size),
                "--max-moves", str(args.max_moves),
                "--backend", args.backend,
                "--preset", args.preset,
                "--device", args.device,
                "--seed", str(args.seed + round_number * 31),
            ]
            match = subprocess.run(cmd, cwd=root, check=True,
                                   capture_output=True, text=True)
            print(match.stdout.strip())
            score = _parse_score(match.stdout)
            if score is None:
                print("could not read the match score; keeping the champion")
                promoted = False
            else:
                promoted = score >= args.gate
                verdict = "promoted" if promoted else "rejected"
                print(f"round {round_number}: score {score:.3f} vs gate {args.gate:.3f} -> {verdict}")

        if promoted:
            shutil.copyfile(candidate, best_model)
            current = best_model
            print(f"round {round_number}: {best_model} now holds the best model")
        else:
            print(f"round {round_number}: keeping the previous champion")

        history.append({
            "round": round_number,
            "data": str(data_path),
            "candidate": str(candidate),
            "score": score,
            "promoted": promoted,
            "seconds": round(time.time() - started, 1),
        })
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\ndone. {sum(1 for h in history if h['promoted'])}/{len(history)} "
          f"rounds promoted. history: {log_path}")


def _parse_score(output: str) -> Optional[float]:
    """Pull ``score 0.612`` out of the match summary line."""
    for line in reversed(output.splitlines()):
        if "score" in line:
            for token in line.replace(",", " ").split():
                try:
                    value = float(token)
                except ValueError:
                    continue
                if 0.0 <= value <= 1.0 and "." in token:
                    return value
    return None


if __name__ == "__main__":
    main()
