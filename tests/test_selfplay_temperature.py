"""Two-stage self-play temperature schedule (Batch 16).

Campaign 3's self-play funnelled through a single opening line (at
ply 2 only 2 distinct positions across ~2,000 recorded games), so the
first plies can now be sampled hotter than the rest of the temperature
window.  These tests pin the schedule and its legacy default.
"""

import unittest

from shogi_ai.evaluator import UniformEvaluator
from shogi_ai.selfplay_az import SelfPlayConfig, play_one_game


class TemperatureScheduleTests(unittest.TestCase):
    def test_default_keeps_the_legacy_schedule(self):
        config = SelfPlayConfig()
        self.assertEqual(config.temperature_at(0), 1.0)
        self.assertEqual(config.temperature_at(29), 1.0)
        self.assertEqual(config.temperature_at(30), 0.0)
        self.assertEqual(config.temperature_at(200), 0.0)

    def test_opening_stage_runs_hotter_then_falls_back(self):
        config = SelfPlayConfig(opening_moves=10, opening_temperature=1.5)
        self.assertEqual(config.temperature_at(0), 1.5)
        self.assertEqual(config.temperature_at(9), 1.5)
        self.assertEqual(config.temperature_at(10), 1.0)
        self.assertEqual(config.temperature_at(29), 1.0)
        self.assertEqual(config.temperature_at(30), 0.0)

    def test_opening_stage_can_outlast_the_regular_window(self):
        config = SelfPlayConfig(opening_moves=40, opening_temperature=1.5,
                                temperature_moves=30)
        self.assertEqual(config.temperature_at(35), 1.5)
        self.assertEqual(config.temperature_at(40), 0.0)

    def test_custom_temperature_is_respected(self):
        config = SelfPlayConfig(temperature=0.7, temperature_moves=5)
        self.assertEqual(config.temperature_at(4), 0.7)
        self.assertEqual(config.temperature_at(5), 0.0)


class OpeningValidationTests(unittest.TestCase):
    def play(self, **overrides):
        kwargs = dict(simulations=16, batch_size=4, max_moves=40,
                      backend="python", resign_threshold=-1.0)
        kwargs.update(overrides)
        return play_one_game(UniformEvaluator(), SelfPlayConfig(**kwargs),
                             seed=7)

    def test_negative_opening_moves_rejected(self):
        with self.assertRaises(ValueError):
            self.play(opening_moves=-1)

    def test_negative_opening_temperature_rejected(self):
        with self.assertRaises(ValueError):
            self.play(opening_moves=10, opening_temperature=-0.5)

    def test_opening_stage_game_completes(self):
        summary = self.play(opening_moves=10, opening_temperature=1.5)
        self.assertGreater(summary.plies, 0)
        self.assertEqual(len(summary.records), summary.plies)


if __name__ == "__main__":
    unittest.main()
