"""The single definition of the policy/value network.

Checkpoints are self-describing: :func:`save_checkpoint` stores the
architecture next to the weights and :func:`load_checkpoint` rebuilds it, so a
model file never has to be matched against a hard-coded class by hand.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from .encoding import NUM_FEATURE_PLANES, POLICY_SIZE

#: Named architectures.  ``base`` is the default: it fits comfortably on a
#: 24 GB card and is the sweet spot for a few hours of self-play.
PRESETS: Dict[str, Dict[str, int]] = {
    "small": {"blocks": 6, "channels": 96},
    "base": {"blocks": 10, "channels": 192},
    "large": {"blocks": 15, "channels": 256},
}
DEFAULT_PRESET = "base"

CHECKPOINT_VERSION = 2


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise SystemExit(
            "PyTorch is required for the neural network. Install it with "
            "`pip install torch` (CUDA build on Colab)."
        ) from exc


def build_network(preset: str = DEFAULT_PRESET, blocks: Optional[int] = None,
                  channels: Optional[int] = None):
    """Instantiate a :class:`PolicyValueNet` from a preset name or explicit size."""
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    shape = PRESETS[preset]
    return PolicyValueNet(
        blocks=blocks if blocks is not None else shape["blocks"],
        channels=channels if channels is not None else shape["channels"],
    )


def _define():
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            h = F.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return F.relu(x + h)

    class PolicyValueNet(nn.Module):
        """ResNet trunk with a 2187-wide policy head and a scalar value head."""

        def __init__(self, blocks: int = 10, channels: int = 192):
            super().__init__()
            self.blocks = blocks
            self.channels = channels
            self.stem = nn.Sequential(
                nn.Conv2d(NUM_FEATURE_PLANES, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.trunk = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            # 27 move planes x 81 squares, produced directly by a 1x1 conv.
            self.policy_head = nn.Conv2d(channels, POLICY_SIZE // 81, 1)
            self.value_head = nn.Sequential(
                nn.Conv2d(channels, 32, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * 81, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            """Return ``(policy_logits, value)`` with shapes ``(N, 2187)``, ``(N, 1)``."""
            h = self.trunk(self.stem(x))
            policy = self.policy_head(h).flatten(1)
            return policy, self.value_head(h)

        def config(self) -> Dict[str, int]:
            return {"blocks": self.blocks, "channels": self.channels}

    return PolicyValueNet


_NET_CLASS = None


def _net_class():
    global _NET_CLASS
    if _NET_CLASS is None:
        _NET_CLASS = _define()
    return _NET_CLASS


class _LazyNet:
    """Defer the torch import until a network is actually constructed."""

    def __call__(self, *args, **kwargs):
        return _net_class()(*args, **kwargs)

    def __instancecheck__(self, instance):
        return isinstance(instance, _net_class())


PolicyValueNet = _LazyNet()


def save_checkpoint(path: str, model, *, steps: int = 0, extra: Optional[dict] = None) -> None:
    torch = _torch()
    payload = {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "config": model.config(),
        "feature_planes": NUM_FEATURE_PLANES,
        "policy_size": POLICY_SIZE,
        "steps": steps,
    }
    if extra:
        payload.update(extra)
    # Same reasoning as the self-play output: a half-written .pt that still
    # exists on disk would be picked up as a finished round.
    partial = f"{path}.partial"
    try:
        torch.save(payload, partial)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    os.replace(partial, path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[object, dict]:
    """Rebuild the network described by a checkpoint and load its weights."""
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=True)

    if payload.get("version") != CHECKPOINT_VERSION:
        planes = payload.get("feature_planes")
        raise SystemExit(
            f"{path} was written by an incompatible version "
            f"(feature planes: {planes}, expected {NUM_FEATURE_PLANES}). "
            "Retrain it with shogi_ai.train_az."
        )
    if payload.get("feature_planes") != NUM_FEATURE_PLANES:
        raise SystemExit(
            f"{path} expects {payload.get('feature_planes')} feature planes but this "
            f"build produces {NUM_FEATURE_PLANES}."
        )

    config = payload.get("config", PRESETS[DEFAULT_PRESET])
    model = _net_class()(blocks=config["blocks"], channels=config["channels"])
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, payload


def describe(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{model.blocks} blocks x {model.channels}ch, {total / 1e6:.2f}M params"
