"""Two-stage self-play temperature schedule (Batch 16), generalised
to a control-point curve with geometric interpolation in Batch 17.

Campaign 3's self-play funnelled through a single opening line (at
ply 2 only 2 distinct positions across ~2,000 recorded games), so the
first plies can now be sampled hotter than the rest of the temperature
window.  These tests pin the schedule and its legacy default.
"""

import unittest

from shogi_ai.evaluator import UniformEvaluator
from shogi_ai.selfplay_az import (SelfPlayConfig, parse_temperature_points,
                                  play_one_game)


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


class TemperaturePointTests(unittest.TestCase):
    BLOCK4 = "0:4.0,4:4.0,39:1.0,50:1.0,51:0.0"

    def test_parse_the_block4_curve(self):
        self.assertEqual(parse_temperature_points(self.BLOCK4),
                         ((0, 4.0), (4, 4.0), (39, 1.0), (50, 1.0),
                          (51, 0.0)))

    def test_parse_empty_means_no_points(self):
        self.assertEqual(parse_temperature_points(""), ())

    def test_parse_rejects_malformed_points(self):
        for text in ("4", "4:", ":2.0", "x:2.0", "4:x", "-4:2.0",
                     "4:-1.0", "16:2.0,4:3.0", "4:2.0,4:1.0"):
            with self.assertRaises(ValueError, msg=text):
                parse_temperature_points(text)

    def test_curve_endpoints_and_plateaus(self):
        config = SelfPlayConfig(
            temperature_points=parse_temperature_points(self.BLOCK4))
        self.assertEqual(config.temperature_at(0), 4.0)
        self.assertEqual(config.temperature_at(3), 4.0)
        self.assertEqual(config.temperature_at(4), 4.0)
        self.assertEqual(config.temperature_at(39), 1.0)
        self.assertEqual(config.temperature_at(49), 1.0)
        self.assertEqual(config.temperature_at(50), 1.0)
        self.assertEqual(config.temperature_at(51), 0.0)
        self.assertEqual(config.temperature_at(200), 0.0)

    def test_curve_decays_geometrically(self):
        config = SelfPlayConfig(
            temperature_points=parse_temperature_points(self.BLOCK4))
        # The 4.0 -> 1.0 segment spans plies 4..39; at ply 21 the
        # log-linear interpolation gives 4 ** (1 - 17/35) ~= 2.04.
        self.assertAlmostEqual(config.temperature_at(21),
                               4 ** (1 - 17 / 35), places=6)
        samples = [config.temperature_at(ply) for ply in (5, 15, 25, 35, 38)]
        self.assertEqual(samples, sorted(samples, reverse=True))
        self.assertGreater(samples[-1], 1.0)

    def test_points_take_precedence_over_the_opening_stage(self):
        config = SelfPlayConfig(opening_moves=10, opening_temperature=9.9,
                                temperature_points=((0, 3.0), (4, 3.0)))
        self.assertEqual(config.temperature_at(0), 3.0)
        self.assertEqual(config.temperature_at(9), 3.0)


class PointValidationTests(unittest.TestCase):
    def play(self, **overrides):
        kwargs = dict(simulations=16, batch_size=4, max_moves=40,
                      backend="python", resign_threshold=-1.0)
        kwargs.update(overrides)
        return play_one_game(UniformEvaluator(), SelfPlayConfig(**kwargs),
                             seed=7)

    def test_non_increasing_points_rejected(self):
        with self.assertRaises(ValueError):
            self.play(temperature_points=((16, 2.0), (4, 3.0)))

    def test_negative_point_temperature_rejected(self):
        with self.assertRaises(ValueError):
            self.play(temperature_points=((4, -1.0),))

    def test_negative_point_ply_rejected(self):
        with self.assertRaises(ValueError):
            self.play(temperature_points=((-1, 2.0),))

    def test_curve_game_completes(self):
        summary = self.play(temperature_points=parse_temperature_points(
            TemperaturePointTests.BLOCK4))
        self.assertGreater(summary.plies, 0)
        self.assertEqual(len(summary.records), summary.plies)


if __name__ == "__main__":
    unittest.main()
