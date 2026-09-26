"""Checkpoint discovery, loading, and the Web server's neural player."""

import os
import tempfile
import unittest

import torch

from shogi_ai import nn_model
from shogi_ai.core import Position, position_from_sfen
from shogi_ai.network import (
    CHECKPOINT_VERSION,
    build_network,
    describe,
    load_checkpoint,
    save_checkpoint,
)


def _write_checkpoint(path: str, preset: str = "small", steps: int = 5):
    model = build_network(preset)
    save_checkpoint(path, model, steps=steps)
    return model


def _write_legacy_checkpoint(path: str):
    """A checkpoint in the pre-policy-head format, which must be refused."""
    torch.save({"model": {"unused": torch.zeros(1)}, "channels": 28}, path)


class CheckpointTests(unittest.TestCase):
    def test_list_model_files_returns_only_pt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["b.pt", "a.pt", "notes.txt"]:
                open(os.path.join(tmpdir, name), "wb").close()
            files = nn_model.list_model_files(tmpdir)
            self.assertEqual(
                files,
                [os.path.join(tmpdir, "a.pt"), os.path.join(tmpdir, "b.pt")],
            )

    def test_missing_directory_is_empty_not_an_error(self):
        self.assertEqual(nn_model.list_model_files("/nonexistent/path"), [])

    def test_checkpoint_round_trip_restores_the_architecture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.pt")
            original = _write_checkpoint(path, preset="small", steps=11)
            restored, payload = load_checkpoint(path)
            self.assertEqual(describe(restored), describe(original))
            self.assertEqual(payload["steps"], 11)
            self.assertEqual(payload["version"], CHECKPOINT_VERSION)
            self.assertEqual(payload["feature_planes"], 85)

    def test_presets_round_trip_independently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for preset in ("small", "base"):
                path = os.path.join(tmpdir, f"{preset}.pt")
                _write_checkpoint(path, preset=preset)
                restored, _payload = load_checkpoint(path)
                self.assertEqual(restored.config(), build_network(preset).config())

    def test_legacy_checkpoint_is_refused_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old.pt")
            _write_legacy_checkpoint(path)
            with self.assertRaises(SystemExit) as caught:
                load_checkpoint(path)
            self.assertIn("train_az", str(caught.exception))


class NeuralPlayerTests(unittest.TestCase):
    def setUp(self):
        nn_model._PLAYERS.clear()
        nn_model._LOAD_ERRORS.clear()

    def test_load_nn_model_accepts_a_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.pt")
            _write_checkpoint(path)
            self.assertTrue(nn_model.load_nn_model(path))

    def test_load_nn_model_reports_failure_for_a_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "old.pt")
            _write_legacy_checkpoint(path)
            self.assertFalse(nn_model.load_nn_model(path))
            self.assertIn("train_az", nn_model.load_error(path))

    def test_missing_file_is_reported_not_raised(self):
        self.assertFalse(nn_model.load_nn_model("/nonexistent/model.pt"))
        self.assertEqual(nn_model.load_error("/nonexistent/model.pt"), "file not found")

    def test_empty_path_loads_nothing(self):
        self.assertFalse(nn_model.load_nn_model(""))
        self.assertIsNone(nn_model.get_player(""))

    def test_a_directory_resolves_to_its_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_checkpoint(os.path.join(tmpdir, "a.pt"))
            player = nn_model.get_player(tmpdir)
            self.assertIsNotNone(player)
            self.assertTrue(player.path.endswith("a.pt"))

    def test_player_is_cached_across_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.pt")
            _write_checkpoint(path)
            self.assertIs(nn_model.get_player(path), nn_model.get_player(path))

    def test_search_returns_a_legal_move_from_the_start_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.pt")
            _write_checkpoint(path)
            player = nn_model.get_player(path, simulations=24, batch_size=4)
            position = Position.start()
            result = player.search(position, simulations=24)
            self.assertIsNotNone(result.move)
            self.assertIn(result.move.usi(), [m.usi() for m in position.legal_moves()])
            self.assertGreater(result.nodes, 0)
            self.assertGreaterEqual(result.depth, 1)
            self.assertLessEqual(abs(result.value), 1.0)

    def test_search_respects_a_time_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.pt")
            _write_checkpoint(path)
            player = nn_model.get_player(path, simulations=100000, batch_size=4)
            result = player.search(Position.start(), time_limit=0.4)
            # Checked between batches, so allow generous slack for one batch.
            self.assertLess(result.elapsed, 3.0)
            self.assertLess(result.simulations, 100000)

    def test_search_on_a_finished_game_returns_no_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.pt")
            _write_checkpoint(path)
            player = nn_model.get_player(path, simulations=16, batch_size=4)
            # White is mated: the rook on 1b is guarded by the king on 2c.
            mated = position_from_sfen("7pk/8R/7K1/9/9/9/9/9/9 w - 2")
            self.assertEqual(mated.legal_moves(), [])
            self.assertIsNone(player.search(mated).move)


if __name__ == "__main__":
    unittest.main()
