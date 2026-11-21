"""Prior-sampled opening moves in self-play (Batch 20).

Campaign 3's first move stayed locked on the central rook swing
(82.2 percent of recorded games) through two temperature fixes,
because temperature acts on the search's *improved* policy: the
search concentrates on the move its Q values favour, unfavoured
moves are never played, and their values are never corrected -- a
self-confirming loop.  The prior-sample stage instead plays the
first plies from the network's raw priors (``sample_prior_move``),
so every legal move keeps the probability the network gives it and
written-off moves still reach real positions and real outcomes.
The recorded policy target remains the search's own distribution.
``--opening-candidates`` widens the Gumbel root during the stage so
the search considers more first moves than the production width.
These tests pin the sampler, the candidate override and the
end-to-end self-play behaviour.  All seeds are fixed, so every
statistical assertion is deterministic.
"""

import unittest

import numpy as np

from shogi_ai.backends import PurePythonBackend, raw_move_from_usi
from shogi_ai.core import BLACK
from shogi_ai.encoding import POLICY_SIZE, policy_index
from shogi_ai.mcts import MCTS, Node, policy_targets, sample_prior_move
from shogi_ai.selfplay_az import SelfPlayConfig, play_one_game

PAWN_PUSH = "7g7f"   # the human-standard first move
ROOK_SWING = "2h5h"  # the locked-in central rook swing
IDX_PAWN = policy_index(raw_move_from_usi(PAWN_PUSH), BLACK)
IDX_ROOK = policy_index(raw_move_from_usi(ROOK_SWING), BLACK)


def fixed_logits_evaluator(peaks):
    """A stand-in network with hand-set policy logits; value 0."""
    def evaluator(planes):
        batch = planes.shape[0]
        logits = np.zeros((batch, POLICY_SIZE), dtype=np.float32)
        for index, value in peaks.items():
            logits[:, index] = value
        return logits, np.zeros(batch, dtype=np.float32)
    return evaluator


#: Prior: rook swing 0.61, pawn push 0.08, the other 28 first
#: moves ~0.011 each.
SKEWED_EVALUATOR = fixed_logits_evaluator({IDX_ROOK: 4.0, IDX_PAWN: 2.0})


def raw_of(root, move):
    """The RawMove for a backend move taken from ``root.moves``."""
    for candidate, raw in zip(root.moves, root.raw_moves):
        if candidate is move or candidate == move:
            return raw
    raise AssertionError("move not in root")


def build_root(evaluator, seed=7, **mcts_kwargs):
    backend = PurePythonBackend()
    mcts = MCTS(evaluator, simulations=8, batch_size=16,
                rng=np.random.default_rng(seed), **mcts_kwargs)
    return backend, mcts.run(backend)


class SamplePriorMoveTests(unittest.TestCase):
    def test_zero_temperature_takes_the_prior_argmax(self):
        evaluator = fixed_logits_evaluator({IDX_PAWN: 2.0, IDX_ROOK: 1.9})
        _, root = build_root(evaluator)
        move = sample_prior_move(root, 0.0, np.random.default_rng(1))
        self.assertEqual(raw_of(root, move), raw_move_from_usi(PAWN_PUSH))

    def test_frequencies_follow_the_priors(self):
        # Priors: pawn e^2 / (e^2 + e^1.9 + 28) = 0.176, rook 0.159.
        evaluator = fixed_logits_evaluator({IDX_PAWN: 2.0, IDX_ROOK: 1.9})
        _, root = build_root(evaluator)
        rng = np.random.default_rng(12345)
        counts = {PAWN_PUSH: 0, ROOK_SWING: 0}
        draws = 3000
        for _ in range(draws):
            raw = raw_of(root, sample_prior_move(root, 1.0, rng))
            if raw == raw_move_from_usi(PAWN_PUSH):
                counts[PAWN_PUSH] += 1
            elif raw == raw_move_from_usi(ROOK_SWING):
                counts[ROOK_SWING] += 1
        self.assertAlmostEqual(counts[PAWN_PUSH] / draws, 0.176, delta=0.03)
        self.assertAlmostEqual(counts[ROOK_SWING] / draws, 0.159, delta=0.03)

    def test_root_without_priors_returns_none(self):
        self.assertIsNone(
            sample_prior_move(Node(), 1.0, np.random.default_rng(1)))

    def test_banned_move_is_never_sampled(self):
        evaluator = fixed_logits_evaluator({IDX_ROOK: 4.0, IDX_PAWN: 2.0})
        backend = PurePythonBackend()
        mcts = MCTS(evaluator, simulations=8, batch_size=16,
                    rng=np.random.default_rng(7))
        root = mcts.run(backend, banned_indices={IDX_ROOK})
        rng = np.random.default_rng(99)
        for _ in range(500):
            raw = raw_of(root, sample_prior_move(root, 1.0, rng))
            self.assertNotEqual(raw, raw_move_from_usi(ROOK_SWING))


class OpeningCandidatesTests(unittest.TestCase):
    def run_gumbel(self, **run_kwargs):
        backend = PurePythonBackend()
        mcts = MCTS(fixed_logits_evaluator({}), simulations=64,
                    batch_size=16, rng=np.random.default_rng(3),
                    search_mode="gumbel", gumbel_candidates=4)
        return mcts.run(backend, **run_kwargs)

    def test_configured_width_limits_the_improved_policy(self):
        root = self.run_gumbel()
        support = sum(1 for prob in root.improved_policy if prob > 0.0)
        self.assertEqual(support, 4)

    def test_per_call_override_widens_the_root(self):
        root = self.run_gumbel(gumbel_candidates=9)
        support = sum(1 for prob in root.improved_policy if prob > 0.0)
        self.assertEqual(support, 9)
        targets = policy_targets(root, BLACK)
        self.assertEqual(len(targets), 9)


class SelfPlayPriorStageTests(unittest.TestCase):
    # Production mode (Gumbel root, 8 candidates): the improved
    # policy is the prior restricted to the considered set, so plain
    # temperature-1.0 play lands on the swing ~0.8 of the time;
    # sampling the raw prior at temperature 2.0 flattens it to
    # ~0.19.  Fixed seeds make both shares exact.
    GAMES = 36

    def swing_share(self, **config_kwargs):
        config = SelfPlayConfig(simulations=16, batch_size=16, max_moves=2,
                                backend="python", resign_threshold=-1.0,
                                search_mode="gumbel", gumbel_candidates=8,
                                **config_kwargs)
        swings = 0
        target_argmax_rook = 0
        for game in range(self.GAMES):
            summary = play_one_game(SKEWED_EVALUATOR, config,
                                    seed=20261122 + game)
            by_ply = {record["ply"]: record for record in summary.records}
            rows = by_ply[1]["sfen"].split(" ")[0].split("/")
            if rows[7] == "1B2R4":
                swings += 1
            top = max(by_ply[0]["policy"], key=lambda pair: pair[1])[0]
            if top == IDX_ROOK:
                target_argmax_rook += 1
        return swings / self.GAMES, target_argmax_rook / self.GAMES

    def test_prior_stage_spreads_the_first_move(self):
        control_share, _ = self.swing_share()
        prior_share, target_share = self.swing_share(
            prior_sample_moves=1, prior_sample_temperature=2.0)
        self.assertGreaterEqual(control_share, 0.55)
        self.assertLessEqual(prior_share, 0.35)
        self.assertLess(prior_share, control_share)
        # The recorded target is still the search's distribution:
        # its argmax stays the rook swing the search believes in.
        self.assertGreaterEqual(target_share, 0.8)

    def test_invalid_config_entries_raise(self):
        config = SelfPlayConfig(backend="python", prior_sample_moves=-1)
        with self.assertRaises(ValueError):
            play_one_game(SKEWED_EVALUATOR, config, seed=1)
        config = SelfPlayConfig(backend="python", opening_candidates=-1)
        with self.assertRaises(ValueError):
            play_one_game(SKEWED_EVALUATOR, config, seed=1)


if __name__ == "__main__":
    unittest.main()
