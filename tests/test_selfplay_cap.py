"""Playout-cap randomisation and Gumbel self-play (Batch 3).

KataGo's trick is a data-quality trade: most plies are searched with a
small budget and are *not* recorded, a few get the full budget and
become the training positions.  These tests pin that accounting with a
stand-in network, so they run without torch in seconds.
"""

import unittest

from shogi_ai.evaluator import UniformEvaluator
from shogi_ai.selfplay_az import SelfPlayConfig, play_one_game


def play(**overrides):
    kwargs = dict(simulations=16, batch_size=4, max_moves=40,
                  backend="python", resign_threshold=-1.0)
    kwargs.update(overrides)
    config = SelfPlayConfig(**kwargs)
    return play_one_game(UniformEvaluator(), config, seed=7)


class PlayoutCapTests(unittest.TestCase):
    def test_disabled_cap_records_every_ply(self):
        summary = play()
        self.assertEqual(summary.low_cap_plies, 0)
        self.assertEqual(len(summary.records), summary.plies)

    def test_full_fraction_records_every_ply(self):
        summary = play(playout_cap_fraction=1.0)
        self.assertEqual(summary.low_cap_plies, 0)
        self.assertEqual(len(summary.records), summary.plies)

    def test_capped_plies_are_played_but_not_recorded(self):
        summary = play(playout_cap_fraction=0.5)
        self.assertGreater(summary.low_cap_plies, 0)
        self.assertLess(len(summary.records), summary.plies)
        # Every ply is accounted for: recorded, or played at the low cap.
        self.assertEqual(len(summary.records) + summary.low_cap_plies,
                         summary.plies)
        for record in summary.records:
            self.assertAlmostEqual(
                sum(p for _i, p in record["policy"]), 1.0, places=4)

    def test_capped_budget_defaults_to_a_quarter(self):
        config = SelfPlayConfig(simulations=200)
        self.assertEqual(config.capped_simulations(), 50)
        config = SelfPlayConfig(simulations=200, playout_cap_simulations=30)
        self.assertEqual(config.capped_simulations(), 30)

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            play(playout_cap_fraction=1.5)


class GumbelSelfPlayTests(unittest.TestCase):
    def test_gumbel_games_produce_valid_records(self):
        summary = play(search_mode="gumbel", simulations=24)
        self.assertEqual(len(summary.records), summary.plies)
        for record in summary.records:
            self.assertTrue(record["policy"])
            self.assertAlmostEqual(
                sum(p for _i, p in record["policy"]), 1.0, places=4)

    def test_gumbel_and_cap_compose(self):
        summary = play(search_mode="gumbel", simulations=24,
                       playout_cap_fraction=0.5)
        self.assertEqual(len(summary.records) + summary.low_cap_plies,
                         summary.plies)


if __name__ == "__main__":
    unittest.main()
