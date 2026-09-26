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
from .encoding import NUM_FEATURE_PLANES, POLICY_SIZE, encode_planes, policy_index
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
    turn = row.get("turn", "black")
    sign = 1 if turn == "black" else -1

    if "value" in row:
        value = float(row["value"])
    else:
        value = float(row.get("result", 0)) * sign

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
    parser.add_argument("--seed", type=int, default=20260926)
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
        line = (f"epoch {epoch}/{args.epochs}: policy={totals['policy'] / seen:.4f} "
                f"value={totals['value'] / seen:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} {time.time() - started:.1f}s")
        if val_loader:
            line += " | " + _validate(model, val_loader, device, args.value_weight, use_amp)
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_checkpoint(args.out, model, steps=steps,
                    extra={"positions": len(dataset), "data": list(args.data)})
    print(f"saved {args.out} ({describe(model)}, {steps} steps)")


def _validate(model, loader, device: str, value_weight: float, use_amp: bool) -> str:
    import torch
    import torch.nn.functional as F

    model.eval()
    policy_total = value_total = agree = count = 0.0
    with torch.no_grad():
        for planes, policy_target, value_target in loader:
            planes = planes.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                policy_logits, value = model(planes)
            policy_total += float(-(policy_target * F.log_softmax(policy_logits.float(), dim=1)
                                    ).sum(dim=1).sum().item())
            value_total += float(F.mse_loss(value.float(), value_target, reduction="sum").item())
            agree += float((policy_logits.argmax(dim=1) == policy_target.argmax(dim=1)).sum().item())
            count += planes.size(0)
    model.train()
    count = max(count, 1.0)
    return (f"val policy={policy_total / count:.4f} value={value_total / count:.4f} "
            f"top1={agree / count:.3f}")


if __name__ == "__main__":
    main()
