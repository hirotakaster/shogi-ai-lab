import unittest
from unittest.mock import patch

import numpy as np

from shogi_ai.encoding import POLICY_SIZE, mirror_policy_index
from shogi_ai.train_az import RandomMirrorDataset, _prepare


INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


class MirrorPolicyTests(unittest.TestCase):
    def test_mirroring_twice_restores_every_policy_index(self):
        for index in range(POLICY_SIZE):
            self.assertEqual(mirror_policy_index(mirror_policy_index(index)), index)

    def test_mirrors_destination_and_keeps_drop_plane(self):
        drop_plane = 20
        square = 2 * 9 + 3
        index = drop_plane * 81 + square
        mirrored = mirror_policy_index(index)
        self.assertEqual(mirrored // 81, drop_plane)
        self.assertEqual(mirrored % 81, 2 * 9 + 5)

    def test_rejects_invalid_policy_index(self):
        with self.assertRaises(ValueError):
            mirror_policy_index(POLICY_SIZE)


class LegacyRecordTests(unittest.TestCase):
    def test_infers_missing_turn_from_sfen(self):
        row = {"sfen": INITIAL_SFEN, "move": "7g7f", "result": 1}
        prepared = _prepare(row)
        self.assertEqual(prepared["value"], 1.0)

    def test_accepts_short_turn_alias_when_it_matches_sfen(self):
        row = {"sfen": INITIAL_SFEN, "turn": "b", "move": "7g7f", "result": 1}
        self.assertEqual(_prepare(row)["value"], 1.0)

    def test_rejects_turn_that_disagrees_with_sfen(self):
        row = {"sfen": INITIAL_SFEN, "turn": "white", "move": "7g7f", "result": 1}
        with self.assertRaisesRegex(ValueError, "disagrees with SFEN"):
            _prepare(row)

    def test_rejects_result_that_is_not_a_black_outcome(self):
        row = {"sfen": INITIAL_SFEN, "move": "7g7f", "result": 100}
        with self.assertRaisesRegex(ValueError, "Black outcome -1/0/1"):
            _prepare(row)


class RandomMirrorDatasetTests(unittest.TestCase):
    def test_mirrors_planes_and_policy_target_together(self):
        policy_index = 1 * 81 + 2 * 9 + 3
        planes = np.zeros((85, 9, 9), dtype=np.float32)
        planes[0, 1, 2] = 1.0
        target = np.zeros(POLICY_SIZE, dtype=np.float32)
        target[policy_index] = 1.0

        class OneSample:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return planes.copy(), target.copy(), np.array([1.0], dtype=np.float32)

        augmented = RandomMirrorDataset(OneSample())
        with patch("shogi_ai.train_az.np.random.random", return_value=0.0):
            mirrored_planes, mirrored_target, _value = augmented[0]

        self.assertEqual(mirrored_planes[0, 1, 6], 1.0)
        self.assertEqual(int(np.argmax(mirrored_target)), mirror_policy_index(policy_index))
        self.assertEqual(float(mirrored_target.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
