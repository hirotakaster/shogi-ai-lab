"""Batched network evaluation for search.

An evaluator turns a stack of feature planes into policy logits and values.
Keeping this behind a tiny callable means :class:`shogi_ai.mcts.MCTS` never
imports torch, so the search can be tested -- and the Web server can run --
without it.
"""

from __future__ import annotations

import queue
import threading
import time
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


class _BatchSlot:
    __slots__ = ("event", "logits", "value", "error")

    def __init__(self):
        self.event = threading.Event()
        self.logits = None
        self.value = 0.0
        self.error: Optional[BaseException] = None


_STOP = object()


class BatchingEvaluator:
    """Coalesce concurrent single-game evaluations into larger batches.

    One MCTS search only ever offers ``batch_size`` leaves at a time, so
    on a GPU the network is fed in small slices even when several games
    are being played at once.  This wrapper sits between concurrent
    games and the real evaluator: every plane is queued individually, a
    background thread gathers whatever has arrived (up to ``max_batch``,
    waiting at most ``max_wait`` seconds for more), runs one forward
    pass, and hands each caller its rows back.  Callers keep the plain
    synchronous evaluator interface and block until their batch lands.

    The underlying evaluator is only ever called from the worker thread,
    which also keeps a CUDA context single-threaded.
    """

    def __init__(self, underlying, max_batch: int = 256, max_wait: float = 0.005):
        self.underlying = underlying
        self.max_batch = max(1, int(max_batch))
        self.max_wait = max(0.0, float(max_wait))
        self.positions = 0
        self.batches = 0
        self._queue: "queue.Queue" = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="batching-evaluator")
        self._thread.start()

    def __call__(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("BatchingEvaluator is closed")
        planes = np.ascontiguousarray(planes)
        count = planes.shape[0]
        if count == 0:
            return (np.zeros((0, POLICY_SIZE), dtype=np.float32),
                    np.zeros(0, dtype=np.float32))
        slots = []
        for i in range(count):
            slot = _BatchSlot()
            slots.append(slot)
            self._queue.put((planes[i], slot))
        logits = np.empty((count, POLICY_SIZE), dtype=np.float32)
        values = np.empty(count, dtype=np.float32)
        for i, slot in enumerate(slots):
            slot.event.wait()
            if slot.error is not None:
                raise slot.error
            logits[i] = slot.logits
            values[i] = slot.value
        return logits, values

    def _serve(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            items = [first]
            stop_after = False
            deadline = time.monotonic() + self.max_wait
            while len(items) < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0.0:
                    break
                try:
                    nxt = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    stop_after = True
                    break
                items.append(nxt)
            planes = np.stack([plane for plane, _slot in items])
            try:
                logits, values = self.underlying(planes)
                for j, (_plane, slot) in enumerate(items):
                    slot.logits = logits[j]
                    slot.value = float(values[j])
                    slot.event.set()
                self.positions += len(items)
                self.batches += 1
            except BaseException as exc:  # propagate to every waiter
                for _plane, slot in items:
                    slot.error = exc
                    slot.event.set()
            if stop_after:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def stats(self) -> str:
        if not self.batches:
            return "no evaluations"
        return (f"{self.positions} positions in {self.batches} batches "
                f"(mean batch {self.positions / self.batches:.1f})")


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
