"""Match play for the learning loop's gate: per-game extraction and the
parallel match.

A gating game is seeded by (seed, game index) alone, so which process plays
it cannot change how it plays out.  These tests pin that contract: the
sequential tally, the worker path used by the pool, and a real two-process
match must all agree game by game.
"""

import unittest
from types import SimpleNamespace

from shogi_ai.evaluate import (MatchResult, _build_player, _init_match_worker,
                               _play_game, _play_index, _record,
                               play_match_parallel)
from shogi_ai.learn_loop import evaluation_processes


def _args(**overrides):
    values = dict(
        challenger="random", champion="random", games=4, processes=1,
        simulations=8, batch_size=4, search_mode="puct", gumbel_candidates=16,
        max_moves=80, opening_moves=8, backend="python", preset="base",
        device="cpu", heuristic_depth=3, heuristic_time=1.0, gate=0.0,
        seed=12345,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sequential_outcomes(args, games):
    challenger = _build_player("random", args, "challenger")
    champion = _build_player("random", args, "champion")
    return [_play_game(challenger, champion, index, args.backend,
                       args.max_moves, args.opening_moves, args.seed)
            for index in range(games)]


class RecordTests(unittest.TestCase):
    def test_outcomes_are_tallied_from_the_challengers_seat(self):
        result = MatchResult()
        _record(result, 0, +1, 100)   # challenger is Black and Black won
        _record(result, 1, +1, 90)    # challenger is White and Black won
        _record(result, 2, -1, 80)    # challenger is Black and White won
        _record(result, 3, -1, 70)    # challenger is White and White won
        _record(result, 4, 0, 60)     # draw
        self.assertEqual((result.games, result.wins, result.losses, result.draws),
                         (5, 2, 2, 1))
        self.assertEqual(result.plies, 400)
        self.assertAlmostEqual(result.score, 0.5)


class PerGameTests(unittest.TestCase):
    def test_a_game_depends_only_on_seed_and_index(self):
        args = _args()
        first = _sequential_outcomes(args, 2)
        second = _sequential_outcomes(args, 2)
        self.assertEqual(first, second)
        for outcome, plies in first:
            self.assertIn(outcome, (-1, 0, 1))
            self.assertGreater(plies, 0)

    def test_worker_path_matches_the_sequential_game(self):
        args = _args()
        expected = _sequential_outcomes(args, 2)
        _init_match_worker("random", "random", vars(args))
        for index in range(2):
            played_index, outcome, plies = _play_index(index)
            self.assertEqual(played_index, index)
            self.assertEqual((outcome, plies), expected[index])


class ParallelMatchTests(unittest.TestCase):
    def test_parallel_match_tallies_the_same_games(self):
        args = _args()
        outcomes = _sequential_outcomes(args, args.games)
        expected = MatchResult()
        for index, (outcome, plies) in enumerate(outcomes):
            _record(expected, index, outcome, plies)

        result = play_match_parallel("random", "random", args,
                                     games=args.games, processes=2, verbose=False)
        self.assertEqual(result.games, expected.games)
        self.assertEqual(result.wins, expected.wins)
        self.assertEqual(result.losses, expected.losses)
        self.assertEqual(result.draws, expected.draws)
        self.assertEqual(result.plies, expected.plies)


class NoisyMatchTests(unittest.TestCase):
    """Root noise in the gate: a noisy match must still be reproducible
    from (seed, game index) alone, and parallel play must agree with
    sequential play -- otherwise the noise would make gating flaky
    instead of merely discriminating.

    Background: the 2026-11-13 production round 1 gate scored exactly
    0.500 with the result alternating every game -- with add_noise off,
    play after the 8 opening plies is deterministic, so two near-identical
    nets replay the same line and the colour, not the strength, decides.
    """

    def test_noisy_game_depends_only_on_seed_and_index(self):
        args = _args(add_noise=True)
        first = _sequential_outcomes(args, 2)
        second = _sequential_outcomes(args, 2)
        self.assertEqual(first, second)

    def test_parallel_noisy_match_tallies_the_same_games(self):
        args = _args(add_noise=True)
        outcomes = _sequential_outcomes(args, args.games)
        expected = MatchResult()
        for index, (outcome, plies) in enumerate(outcomes):
            _record(expected, index, outcome, plies)

        result = play_match_parallel("random", "random", args,
                                     games=args.games, processes=2, verbose=False)
        self.assertEqual(result.games, expected.games)
        self.assertEqual(result.wins, expected.wins)
        self.assertEqual(result.losses, expected.losses)
        self.assertEqual(result.draws, expected.draws)
        self.assertEqual(result.plies, expected.plies)

    def test_noise_flag_reaches_the_player(self):
        args = _args(add_noise=True)
        self.assertTrue(_build_player("random", args, "challenger").add_noise)
        self.assertFalse(_build_player("random", _args(), "challenger").add_noise)


class EvaluationProcessesTests(unittest.TestCase):
    def test_explicit_value_wins(self):
        args = SimpleNamespace(eval_processes=3, processes=12)
        self.assertEqual(evaluation_processes(args), 3)

    def test_zero_falls_back_to_selfplay_processes(self):
        args = SimpleNamespace(eval_processes=0, processes=12)
        self.assertEqual(evaluation_processes(args), 12)


if __name__ == "__main__":
    unittest.main()
