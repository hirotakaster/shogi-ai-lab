"""The policy/value network: presets, the attention hybrid, checkpoints."""

import os
import tempfile
import unittest

try:
    import torch
except ImportError:
    # PyTorch is an optional training dependency. Without it these tests
    # skip, so the suite still runs green on the dependency-free setup
    # the project promises for local play.
    torch = None

from shogi_ai.encoding import NUM_FEATURE_PLANES, POLICY_SIZE
from shogi_ai.network import build_network, load_checkpoint, save_checkpoint


def _attention_blocks(model):
    return [m for m in model.trunk if type(m).__name__ == "AttentionBlock"]


def _params(model):
    return sum(p.numel() for p in model.parameters())


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AttentionPresetTests(unittest.TestCase):
    def test_attn_preset_interleaves_two_attention_blocks(self):
        model = build_network("attn")
        self.assertEqual(len(_attention_blocks(model)), 2)
        self.assertEqual(model.config()["attn_every"], 5)
        self.assertEqual(model.config()["attn_heads"], 8)

    def test_plain_presets_have_no_attention(self):
        for preset in ("small", "base", "large"):
            self.assertEqual(_attention_blocks(build_network(preset)), [])

    def test_attention_lands_on_the_final_block_when_every_misses(self):
        # blocks=4 with attn_every=5: no multiple of 5, but the trunk
        # still ends with an attention block.
        model = build_network("attn", blocks=4, channels=32)
        self.assertEqual(len(_attention_blocks(model)), 1)

    def test_attn_forward_shapes_and_value_range(self):
        model = build_network("attn", blocks=2, channels=32)
        model.eval()
        x = torch.zeros(2, NUM_FEATURE_PLANES, 9, 9)
        with torch.no_grad():
            policy, value = model(x)
        self.assertEqual(tuple(policy.shape), (2, POLICY_SIZE))
        self.assertEqual(tuple(value.shape), (2, 1))
        self.assertTrue(bool((value.abs() <= 1).all()))

    def test_attn_has_more_params_than_base_but_stays_close(self):
        base = _params(build_network("base"))
        attn = _params(build_network("attn"))
        self.assertGreater(attn, base)
        self.assertLess(attn, base + 1_000_000)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AttentionCheckpointTests(unittest.TestCase):
    def test_attn_checkpoint_round_trip(self):
        model = build_network("attn", blocks=2, channels=32)
        model.eval()
        x = torch.randn(1, NUM_FEATURE_PLANES, 9, 9)
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "attn.pt")
            save_checkpoint(path, model, steps=7)
            loaded, payload = load_checkpoint(path)
        loaded.eval()
        self.assertEqual(loaded.config(), model.config())
        self.assertEqual(payload["steps"], 7)
        with torch.no_grad():
            actual = loaded(x)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_legacy_config_without_attn_keys_still_loads(self):
        # Checkpoints written before the hybrid carry only blocks and
        # channels in their config; they must load as pure ResNets.
        model = build_network("base")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pt")
            save_checkpoint(path, model, steps=3)
            payload = torch.load(path, weights_only=True)
            payload["config"] = {
                "blocks": payload["config"]["blocks"],
                "channels": payload["config"]["channels"],
            }
            torch.save(payload, path)
            loaded, _ = load_checkpoint(path)
        self.assertEqual(loaded.config()["attn_every"], 0)
        self.assertEqual(_attention_blocks(loaded), [])


if __name__ == "__main__":
    unittest.main()
