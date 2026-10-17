"""Train the policy/value network on self-play records.

Reads one or more JSONL files written by :mod:`shogi_ai.selfplay_az` and, as a
bootstrap, also understands the older ``{"sfen", "move", "result"}`` records: a
played move becomes a one-hot policy target, which is ordinary supervised
learning on whatever games are already on disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .backends import raw_move_from_usi, snapshot_from_sfen
from .encoding import (NUM_FEATURE_PLANES, POLICY_SIZE, encode_planes,
                       mirror_policy_index, policy_index)
from .network import DEFAULT_PRESET, PRESETS, build_network, describe, load_checkpoint, save_checkpoint


def load_records(patterns: Sequence[str], max_positions: Optional[int] = None) -> List[dict]:
    """Load records from files or globs, newest last, trimmed to a buffer size."""
    paths: List[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            if not os.path.exists(pattern):
                raise SystemExit(f"no training data matched {pattern!r}")
            matched = [pattern]
        paths.extend(matched)

    rows: List[dict] = []
    legacy = 0
    empty: List[str] = []
    for path in paths:
        before = len(rows)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "policy" not in row:
                    legacy += 1
                rows.append(row)
        if len(rows) == before:
            empty.append(path)
        print(f"  {path}: {len(rows)} positions so far")

    if empty and not rows:
        raise SystemExit(
            "every training file is empty: " + ", ".join(empty) + "\n"
            "A self-play run was probably interrupted. Delete the empty files "
            "and run self-play again."
        )

    if legacy:
        print(f"  {legacy} legacy records will use a one-hot policy target")
    if max_positions and len(rows) > max_positions:
        # Keep the most recent slice: later files are later generations.
        rows = rows[-max_positions:]
        print(f"  replay buffer trimmed to the newest {len(rows)} positions")
    return rows


def _prepare(row: dict) -> dict:
    """Normalise a record into ``sfen``/``check``/``value``/``policy`` form."""
    sfen_fields = row.get("sfen", "").split()
    if len(sfen_fields) < 2 or sfen_fields[1] not in ("b", "w"):
        raise ValueError(f"record has an invalid SFEN side-to-move: {row.get('sfen')!r}")
    sfen_turn = "black" if sfen_fields[1] == "b" else "white"
    raw_turn = row.get("turn", sfen_turn)
    turn_aliases = {"b": "black", "black": "black", "w": "white", "white": "white"}
    turn = turn_aliases.get(str(raw_turn).lower())
    if turn is None:
        raise ValueError(f"record has an invalid turn {raw_turn!r}: {row.get('sfen')!r}")
    if turn != sfen_turn:
        raise ValueError(
            f"record turn {turn!r} disagrees with SFEN side-to-move "
            f"{sfen_turn!r}: {row.get('sfen')!r}"
        )
    sign = 1 if turn == "black" else -1

    if "value" in row:
        value = float(row["value"])
    else:
        result = float(row.get("result", 0))
        if result not in (-1.0, 0.0, 1.0):
            raise ValueError(
                f"legacy result must be Black outcome -1/0/1, got {result!r}"
            )
        value = result * sign

    if "policy" in row:
        policy = [(int(i), float(p)) for i, p in row["policy"]]
    elif "move" in row:
        index = policy_index(raw_move_from_usi(row["move"]), 1 if turn == "black" else -1)
        policy = [(index, 1.0)]
    else:
        raise SystemExit(f"record has neither a policy nor a move: {row}")

    if "check" in row:
        check = bool(row["check"])
    else:
        # Legacy records predate the check plane; recover it from the rules.
        from .core import BLACK, WHITE, position_from_sfen

        position = position_from_sfen(row["sfen"])
        check = position.is_in_check(BLACK if turn == "black" else WHITE)

    return {"sfen": row["sfen"], "check": check, "value": value, "policy": policy}


class SelfPlayDataset:
    """Decodes SFEN records into planes and dense policy targets on demand.

    Deliberately a plain class rather than a ``torch.utils.data.Dataset``
    subclass, and it returns NumPy arrays: that keeps it free of torch and
    picklable, which is what lets ``DataLoader`` spawn worker processes.
    ``default_collate`` turns the arrays into tensors.
    """

    def __init__(self, rows: Sequence[dict]):
        self.rows = [_prepare(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        planes = encode_planes(snapshot_from_sfen(row["sfen"], row["check"]))
        target = np.zeros(POLICY_SIZE, dtype=np.float32)
        for policy_idx, probability in row["policy"]:
            target[policy_idx] = probability
        total = target.sum()
        if total > 0:
            target /= total
        return planes, target, np.array([row["value"]], dtype=np.float32)


_MIRROR_POLICY_INDICES = np.asarray(
    [mirror_policy_index(index) for index in range(POLICY_SIZE)], dtype=np.int64
)


class RandomMirrorDataset:
    """Apply a random horizontal mirror without augmenting validation data."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        planes, policy_target, value_target = self.dataset[index]
        if np.random.random() < 0.5:
            planes = np.flip(planes, axis=-1).copy()
            policy_target = policy_target[_MIRROR_POLICY_INDICES].copy()
        return planes, policy_target, value_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the policy/value network.")
    parser.add_argument("--data", nargs="+", default=["data/selfplay_az.jsonl"],
                        help="JSONL files or globs, oldest first")
    parser.add_argument("--max-positions", type=int, default=0,
                        help="replay buffer cap; 0 keeps everything")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    parser.add_argument("--init", default="", help="checkpoint to continue from")
    parser.add_argument("--out", default="data/policy_value.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--no-mirror-augmentation", action="store_true",
                        help="disable random left-right mirror augmentation")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--metrics-out", default="", help="append per-epoch metrics as JSONL")
    parser.add_argument("--round", type=int, default=0, help="round number for metrics logging")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training: pip install torch") from exc

    from .evaluator import pick_device

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"training on {device}")

    print("loading data")
    rows = load_records(args.data, args.max_positions or None)
    if not rows:
        raise SystemExit("no training records found")
    dataset = SelfPlayDataset(rows)
    print(f"{len(dataset)} positions, {NUM_FEATURE_PLANES} planes, {POLICY_SIZE} policy outputs")

    val_size = int(len(dataset) * args.val_fraction)
    if val_size > 0:
        train_set, val_set = random_split(
            dataset, [len(dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set, val_set = dataset, None
    if not args.no_mirror_augmentation:
        train_set = RandomMirrorDataset(train_set)

    loader_kwargs = {"num_workers": args.workers, "pin_memory": device == "cuda"}
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **loader_kwargs)
    val_loader = (DataLoader(val_set, batch_size=args.batch_size, **loader_kwargs)
                  if val_set else None)

    steps = 0
    if args.init:
        model, payload = load_checkpoint(args.init, device=device)
        steps = int(payload.get("steps", 0))
        print(f"continuing from {args.init}: {describe(model)} at {steps} steps")
    else:
        model = build_network(args.preset).to(device)
        print(f"new network: {describe(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    # OneCycleLR needs both of its phases to contain at least one step, which a
    # very small dataset cannot supply; a constant rate is fine at that size.
    if total_steps >= 8:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.25,
        )
    else:
        print(f"only {total_steps} optimiser steps: using a constant learning rate")
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0,
                                                        total_iters=total_steps)
    use_amp = device == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("mixed precision enabled")

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        totals = {"policy": 0.0, "value": 0.0, "n": 0}
        for planes, policy_target, value_target in train_loader:
            planes = planes.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            value_target = value_target.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_logits, value = model(planes)
                # Soft-target cross entropy: the search's visit distribution.
                policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(value, value_target)
                loss = policy_loss + args.value_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            steps += 1

            batch = planes.size(0)
            totals["policy"] += float(policy_loss.item()) * batch
            totals["value"] += float(value_loss.item()) * batch
            totals["n"] += batch

        seen = max(1, totals["n"])
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - started
        train_policy = totals["policy"] / seen
        train_value = totals["value"] / seen
        line = (f"epoch {epoch}/{args.epochs}: policy={train_policy:.4f} "
                f"value={train_value:.4f} "
                f"lr={current_lr:.2e} {elapsed:.1f}s")
        record: Dict = {
            "round": args.round, "epoch": epoch,
            "policy": round(train_policy, 4), "value": round(train_value, 4),
            "lr": round(current_lr, 8), "steps": steps,
        }
        if val_loader:
            val_str, val_dict = _validate(model, val_loader, device, args.value_weight, use_amp)
            line += " | " + val_str
            record.update(val_dict)
        print(line, flush=True)
        if args.metrics_out:
            with open(args.metrics_out, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(record) + "\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_checkpoint(args.out, model, steps=steps,
                    extra={"positions": len(dataset), "data": list(args.data)})
    print(f"saved {args.out} ({describe(model)}, {steps} steps)")


def _validate(model, loader, device: str, value_weight: float, use_amp: bool):
    """Return ``(display_string, metrics_dict)``."""
    import torch
    import torch.nn.functional as F

    model.eval()
    policy_total = value_total = agree = count = 0.0
    with torch.no_grad():
        for planes, policy_target, value_target in loader:
            planes = planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            with torch.amp.autocast("cuda", enabled=False):
                policy_logits, value = model(planes)
            policy_total += float(-(policy_target * F.log_softmax(policy_logits.float(), dim=1)
                                    ).sum(dim=1).sum().item())
            value_total += float(F.mse_loss(value.float(), value_target, reduction="sum").item())
            agree += float((policy_logits.argmax(dim=1) == policy_target.argmax(dim=1)).sum().item())
            count += planes.size(0)
    model.train()
    count = max(count, 1.0)
    vp = round(policy_total / count, 4)
    vv = round(value_total / count, 4)
    t1 = round(agree / count, 3)
    return (f"val policy={vp:.4f} value={vv:.4f} top1={t1:.3f}",
            {"val_policy": vp, "val_value": vv, "top1": t1})


if __name__ == "__main__":
    main()
