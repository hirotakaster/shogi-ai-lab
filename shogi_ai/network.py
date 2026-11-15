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
#: ``attn`` is the experimental hybrid: the base trunk with a self-attention
#: block after every ``attn_every`` residual blocks (and after the last
#: one), so long-range relations (sliding pieces, king and castle) can be
#: wired directly instead of through many convolutions.  Attention keys
#: default to 0 (absent) in the plain presets, and old checkpoints whose
#: config lacks them still load as pure ResNets.
PRESETS: Dict[str, Dict[str, int]] = {
    "small": {"blocks": 6, "channels": 96},
    "base": {"blocks": 10, "channels": 192},
    "large": {"blocks": 15, "channels": 256},
    "attn": {"blocks": 10, "channels": 192, "attn_every": 5, "attn_heads": 8},
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
        attn_every=shape.get("attn_every", 0),
        attn_heads=shape.get("attn_heads", 8),
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

    class AttentionBlock(nn.Module):
        """One pre-norm transformer layer over the 81 squares.

        The feature map is read as 81 tokens (one per square) with a
        learned positional embedding, so a piece can attend to any other
        square in one step; the result is folded back into a feature map
        of the same shape, which keeps the trunk interface unchanged.
        """

        def __init__(self, channels: int, heads: int):
            super().__init__()
            self.pos = nn.Parameter(torch.zeros(1, 81, channels))
            nn.init.normal_(self.pos, std=0.02)
            self.norm1 = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(
                channels, heads, dropout=0.0, batch_first=True
            )
            self.norm2 = nn.LayerNorm(channels)
            self.mlp = nn.Sequential(
                nn.Linear(channels, 2 * channels),
                nn.GELU(),
                nn.Linear(2 * channels, channels),
            )

        def forward(self, x):
            n, c, h, w = x.shape
            t = x.flatten(2).transpose(1, 2) + self.pos
            a = self.norm1(t)
            t = t + self.attn(a, a, a, need_weights=False)[0]
            t = t + self.mlp(self.norm2(t))
            return t.transpose(1, 2).reshape(n, c, h, w)

    class PolicyValueNet(nn.Module):
        """ResNet trunk with a 2187-wide policy head and a scalar value head.

        With ``attn_every`` > 0 the trunk interleaves an
        :class:`AttentionBlock` after every ``attn_every`` residual
        blocks and after the final block.
        """

        def __init__(self, blocks: int = 10, channels: int = 192,
                     attn_every: int = 0, attn_heads: int = 8):
            super().__init__()
            self.blocks = blocks
            self.channels = channels
            self.attn_every = attn_every
            self.attn_heads = attn_heads
            self.stem = nn.Sequential(
                nn.Conv2d(NUM_FEATURE_PLANES, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            layers = []
            for i in range(1, blocks + 1):
                layers.append(ResidualBlock(channels))
                if attn_every > 0 and (i % attn_every == 0 or i == blocks):
                    layers.append(AttentionBlock(channels, attn_heads))
            self.trunk = nn.Sequential(*layers)
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
            return {
                "blocks": self.blocks,
                "channels": self.channels,
                "attn_every": self.attn_every,
                "attn_heads": self.attn_heads,
            }

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
    model = _net_class()(
        blocks=config["blocks"],
        channels=config["channels"],
        # Checkpoints written before the attention hybrid have no attn
        # keys; they describe pure ResNets and must keep loading.
        attn_every=config.get("attn_every", 0),
        attn_heads=config.get("attn_heads", 8),
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, payload


def describe(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{model.blocks} blocks x {model.channels}ch, {total / 1e6:.2f}M params"
