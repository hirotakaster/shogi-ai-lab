"""Batched network evaluation for search.

An evaluator turns a stack of feature planes into policy logits and values.
Keeping this behind a tiny callable means :class:`shogi_ai.mcts.MCTS` never
imports torch, so the search can be tested -- and the Web server can run --
without it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .encoding import POLICY_SIZE


def pick_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to the best device this machine actually has."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon: useful for local smoke tests, not for training runs.
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchEvaluator:
    """Runs a :class:`shogi_ai.network.PolicyValueNet` on batches of planes."""

    def __init__(self, model, device: str = "auto", fp16: Optional[bool] = None):
        import torch

        self.torch = torch
        self.device = pick_device(device)
        self.model = model.to(self.device)
        self.model.eval()
        # Half precision pays off on CUDA only; it is a slowdown on CPU and is
        # not reliably supported on MPS.  This uses autocast rather than
        # ``model.half()`` on purpose: autocast keeps batch norm in float32,
        # where casting the whole module would run its running statistics in
        # half and can under/overflow them.
        self.fp16 = (self.device == "cuda") if fp16 is None else fp16
        self.positions = 0
        self.batches = 0

    def __call__(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        self.positions += planes.shape[0]
        self.batches += 1
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(planes)).to(self.device).float()
            with torch.amp.autocast(self.device, enabled=self.fp16):
                logits, value = self.model(x)
            return (
                logits.float().cpu().numpy(),
                value.float().cpu().numpy().reshape(-1),
            )

    def stats(self) -> str:
        if not self.batches:
            return "no evaluations"
        return (f"{self.positions} positions in {self.batches} batches "
                f"(mean batch {self.positions / self.batches:.1f})")


class UniformEvaluator:
    """Flat policy, neutral value.  Used as a baseline opponent and in tests."""

    def __call__(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = planes.shape[0]
        return (np.zeros((batch, POLICY_SIZE), dtype=np.float32),
                np.zeros(batch, dtype=np.float32))


def load_evaluator(model_path: Optional[str], preset: str = "base",
                   device: str = "auto", fp16: Optional[bool] = None,
                   seed: Optional[int] = None) -> TorchEvaluator:
    """Load a checkpoint, or build a fresh random network when none is given.

    A randomly initialised network is exactly how an AlphaZero run starts, so
    ``--model`` is optional for the first generation of self-play.
    """
    from .network import build_network, describe, load_checkpoint

    if model_path:
        model, payload = load_checkpoint(model_path, device=pick_device(device))
        print(f"loaded {model_path}: {describe(model)} (trained steps: {payload.get('steps', 0)})")
    else:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        model = build_network(preset)
        print(f"fresh random network: {describe(model)}")
    return TorchEvaluator(model, device=device, fp16=fp16)
