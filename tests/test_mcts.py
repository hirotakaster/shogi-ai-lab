import unittest

import numpy as np

from shogi_ai.backends import PurePythonBackend
from shogi_ai.core import position_from_sfen as _position_from_sfen
from shogi_ai.encoding import POLICY_SIZE, policy_index
from shogi_ai.mcts import MCTS, Node, policy_targets, select_move, softmax_over_legal


#: The white king on 1a is walled in by its own pawn on 2a, and both 1b and 2b
#: are covered by the black king on 2c.  Several rook moves finish the game at
#: once -- 9b1b is mate, while 9b9a leaves White with no legal move at all
#: (the 2a pawn is pinned against the rook), which shogi also scores as a loss.
MATE_IN_ONE = "7pk/R8/7K1/9/9/9/9/9/9 b - 1"
MATE_MOVE = "9b1b"


def uniform_evaluator(planes):
    """A stand-in network: flat policy, neutral value. Needs no torch."""
    batch = planes.shape[0]
    return np.zeros((batch, POLICY_SIZE), dtype=np.float32), np.zeros(batch, dtype=np.float32)


class BiasedEvaluator:
    """Scores one policy index highly so the search must concentrate on it."""

    def __init__(self, favoured, value=0.0):
        self.favoured = favoured
        self.value = value
        self.calls = 0
        self.batch_sizes = []

    def __call__(self, planes):
        self.calls += 1
        self.batch_sizes.append(planes.shape[0])
        batch = planes.shape[0]
        logits = np.zeros((batch, POLICY_SIZE), dtype=np.float32)
        logits[:, self.favoured] = 8.0
        return logits, np.full(batch, self.value, dtype=np.float32)


class SoftmaxTests(unittest.TestCase):
    def test_masks_to_legal_indices_and_sums_to_one(self):
        logits = np.arange(POLICY_SIZE, dtype=np.float32)
        probs = softmax_over_legal(logits, [5, 7, 9])
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        # Higher logits must get higher probability.
        self.assertLess(probs[0], probs[2])

    def test_degenerate_logits_fall_back_to_uniform(self):
        logits = np.full(POLICY_SIZE, -np.inf, dtype=np.float32)
        probs = softmax_over_legal(logits, [1, 2])
        np.testing.assert_allclose(probs, [0.5, 0.5])

    def test_empty_legal_set(self):
        self.assertEqual(softmax_over_legal(np.zeros(POLICY_SIZE), []).shape, (0,))


class SearchTests(unittest.TestCase):
    def test_visits_match_the_simulation_budget(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=64, batch_size=8)
        root = mcts.run(backend)
        self.assertEqual(int(root.visit_counts().sum()), 64)

    def test_search_leaves_the_board_untouched(self):
        backend = PurePythonBackend()
        before = backend.sfen()
        MCTS(uniform_evaluator, simulations=48, batch_size=8).run(backend)
        self.assertEqual(backend.sfen(), before)
        self.assertFalse(backend.is_repetition_draw())

    def test_virtual_loss_is_fully_unwound(self):
        backend = PurePythonBackend()
        root = MCTS(uniform_evaluator, simulations=40, batch_size=8).run(backend)
        self.assertEqual(float(np.abs(root.child_virtual).sum()), 0.0)

    def test_batching_actually_groups_leaves(self):
        backend = PurePythonBackend()
        evaluator = BiasedEvaluator(favoured=0)
        MCTS(evaluator, simulations=64, batch_size=16).run(backend)
        # One single-position root call, then batched calls.
        self.assertGreater(max(evaluator.batch_sizes), 1)
        self.assertLess(evaluator.calls, 64)

    def test_priors_steer_the_visits(self):
        backend = PurePythonBackend()
        _native, raw = backend.legal_moves()
        favoured_move = raw[3]
        favoured = policy_index(favoured_move, backend.turn)
        mcts = MCTS(BiasedEvaluator(favoured), simulations=128, batch_size=8)
        root = mcts.run(backend)
        counts = root.visit_counts()
        self.assertEqual(int(np.argmax(counts)), 3)

    def test_root_noise_changes_priors_but_keeps_a_distribution(self):
        backend = PurePythonBackend()
        plain = MCTS(uniform_evaluator, simulations=1, batch_size=1, add_noise=False)
        noisy = MCTS(uniform_evaluator, simulations=1, batch_size=1, add_noise=True,
                     rng=np.random.default_rng(0))
        flat = plain.run(backend).priors
        perturbed = noisy.run(backend).priors
        self.assertAlmostEqual(float(perturbed.sum()), 1.0, places=5)
        self.assertFalse(np.allclose(flat, perturbed))

    def test_forced_win_is_found_without_network_help(self):
        """The search must settle on a move that ends the game immediately."""
        backend = PurePythonBackend(_position_from_sfen(MATE_IN_ONE))
        # A flat, neutral network: every evaluated position scores 0, so only a
        # real terminal score can produce the +1 that wins the search.
        root = MCTS(uniform_evaluator, simulations=256, batch_size=8).run(backend)
        best_index = int(np.argmax(root.visit_counts()))
        winner = root.children[best_index]
        self.assertEqual(winner.terminal, -1.0, "chosen move does not end the game")
        self.assertIn(root.moves[best_index].usi(), self._immediate_wins(backend))

    def test_mate_move_really_is_mate(self):
        """Pin the test fixture itself, independently of the search."""
        backend = PurePythonBackend(_position_from_sfen(MATE_IN_ONE))
        backend.push(backend.parse_usi(MATE_MOVE))
        self.assertTrue(backend.position.is_in_check(backend.turn))
        self.assertEqual(backend.position.legal_moves(), [])

    @staticmethod
    def _immediate_wins(backend):
        """USI moves after which the opponent has no legal reply."""
        wins = set()
        native, _raw = backend.legal_moves()
        for move in native:
            backend.push(move)
            if backend.terminal_value() == -1.0:
                wins.add(backend.move_usi(move))
            backend.pop()
        return wins

    def test_search_from_a_mated_position_returns_empty_root(self):
        backend = PurePythonBackend(_position_from_sfen(MATE_IN_ONE))
        backend.push(backend.parse_usi(MATE_MOVE))
        root = MCTS(uniform_evaluator, simulations=16, batch_size=4).run(backend)
        self.assertEqual(root.moves, [])
        self.assertEqual(root.terminal, -1.0)


class TreeReuseTests(unittest.TestCase):
    def test_advance_retains_the_played_childs_statistics(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=64, batch_size=8)
        root = mcts.run(backend)
        move, _counts = select_move(root, 0.0, np.random.default_rng(0))
        child = root.children[root.moves.index(move)]
        self.assertIsNotNone(child)
        retained_visits = child.visits
        self.assertGreater(retained_visits, 0)

        backend.push(move)
        self.assertTrue(mcts.advance(move))
        reused = mcts.run(backend)
        self.assertIs(reused, child)
        # The second search adds its simulations on top of the retained
        # ones instead of starting the count from zero again.
        self.assertEqual(reused.visits, retained_visits + 64)

    def test_advance_with_an_unknown_move_retains_nothing(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=16, batch_size=4)
        mcts.run(backend)
        self.assertFalse(mcts.advance("not a move"))
        fresh = mcts.run(backend)
        self.assertEqual(fresh.visits, 16)

    def test_advance_before_any_search_retains_nothing(self):
        mcts = MCTS(uniform_evaluator, simulations=16, batch_size=4)
        self.assertFalse(mcts.advance("7g7f"))

    def test_root_noise_on_a_reused_root_stays_anchored_to_network_priors(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=32, batch_size=8,
                    add_noise=True, rng=np.random.default_rng(0))
        root = mcts.run(backend)
        move, _counts = select_move(root, 0.0, np.random.default_rng(0))
        backend.push(move)
        self.assertTrue(mcts.advance(move))
        reused = mcts.run(backend)
        # Still a distribution, and still mixed from the raw priors: the
        # network's own ranking must survive repeated noise applications.
        self.assertAlmostEqual(float(reused.priors.sum()), 1.0, places=5)
        base = reused.raw_priors
        self.assertTrue(np.all(reused.priors <= 0.75 * base + 0.25 + 1e-6))


class MoveSelectionTests(unittest.TestCase):
    def test_zero_temperature_picks_the_most_visited(self):
        root = Node()
        root.expand(["a", "b", "c"], [None] * 3, np.array([0.3, 0.4, 0.3], dtype=np.float32))
        root.child_visits[:] = [3.0, 11.0, 5.0]
        move, counts = select_move(root, 0.0, np.random.default_rng(0))
        self.assertEqual(move, "b")
        np.testing.assert_allclose(counts, [3.0, 11.0, 5.0])

    def test_high_temperature_can_pick_other_moves(self):
        root = Node()
        root.expand(["a", "b"], [None] * 2, np.array([0.5, 0.5], dtype=np.float32))
        root.child_visits[:] = [40.0, 60.0]
        rng = np.random.default_rng(1)
        picked = {select_move(root, 1.0, rng)[0] for _ in range(60)}
        self.assertEqual(picked, {"a", "b"})

    def test_unvisited_root_still_returns_a_move(self):
        root = Node()
        root.expand(["a"], [None], np.array([1.0], dtype=np.float32))
        move, _counts = select_move(root, 1.0, np.random.default_rng(0))
        self.assertEqual(move, "a")


class PolicyTargetTests(unittest.TestCase):
    def test_targets_are_a_normalised_distribution_over_legal_indices(self):
        backend = PurePythonBackend()
        root = MCTS(uniform_evaluator, simulations=64, batch_size=8).run(backend)
        targets = policy_targets(root, backend.turn)
        self.assertTrue(targets)
        self.assertAlmostEqual(sum(p for _i, p in targets), 1.0, places=5)
        legal = {policy_index(m, backend.turn) for m in root.raw_moves}
        self.assertTrue({i for i, _p in targets} <= legal)
        for index, _p in targets:
            self.assertLess(index, POLICY_SIZE)

    def test_empty_root_has_no_targets(self):
        self.assertEqual(policy_targets(Node(), 1), [])


class GumbelSearchTests(unittest.TestCase):
    def test_visits_match_the_simulation_budget(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=64, batch_size=8,
                    search_mode="gumbel", add_noise=True,
                    rng=np.random.default_rng(0))
        root = mcts.run(backend)
        self.assertEqual(int(root.visit_counts().sum()), 64)
        self.assertEqual(mcts.simulations_done, 64)

    def test_budget_smaller_than_the_candidate_count_still_works(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=8, batch_size=4,
                    search_mode="gumbel", gumbel_candidates=16,
                    add_noise=True, rng=np.random.default_rng(1))
        root = mcts.run(backend)
        self.assertEqual(int(root.visit_counts().sum()), 8)
        self.assertAlmostEqual(float(root.improved_policy.sum()), 1.0, places=5)

    def test_improved_policy_is_restricted_to_the_candidates(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=64, batch_size=8,
                    search_mode="gumbel", gumbel_candidates=8,
                    add_noise=True, rng=np.random.default_rng(0))
        root = mcts.run(backend)
        self.assertIsNotNone(root.improved_policy)
        self.assertAlmostEqual(float(root.improved_policy.sum()), 1.0, places=5)
        self.assertLessEqual(int((root.improved_policy > 0).sum()), 8)
        # Moves outside the candidate set were never forced at the root,
        # so they carry no visits either.
        outside = root.improved_policy == 0
        self.assertTrue(np.all(root.visit_counts()[outside] == 0))

    def test_policy_targets_use_the_improved_policy(self):
        backend = PurePythonBackend()
        root = MCTS(uniform_evaluator, simulations=64, batch_size=8,
                    search_mode="gumbel", add_noise=True,
                    rng=np.random.default_rng(0)).run(backend)
        targets = policy_targets(root, backend.turn)
        self.assertAlmostEqual(sum(p for _i, p in targets), 1.0, places=5)
        self.assertEqual(len(targets), int((root.improved_policy > 0).sum()))

    def test_greedy_selection_takes_the_improved_argmax(self):
        backend = PurePythonBackend()
        root = MCTS(uniform_evaluator, simulations=64, batch_size=8,
                    search_mode="gumbel", add_noise=True,
                    rng=np.random.default_rng(0)).run(backend)
        move, _counts = select_move(root, 0.0, np.random.default_rng(0))
        self.assertEqual(move, root.moves[int(np.argmax(root.improved_policy))])

    def test_deterministic_without_noise(self):
        def run_once():
            backend = PurePythonBackend()
            return MCTS(uniform_evaluator, simulations=48, batch_size=8,
                        search_mode="gumbel", add_noise=False,
                        rng=np.random.default_rng(3)).run(backend)

        first, second = run_once(), run_once()
        np.testing.assert_allclose(first.improved_policy, second.improved_policy)
        np.testing.assert_allclose(first.visit_counts(), second.visit_counts())

    def test_priors_steer_the_improved_policy(self):
        backend = PurePythonBackend()
        _native, raw = backend.legal_moves()
        favoured = policy_index(raw[3], backend.turn)
        root = MCTS(BiasedEvaluator(favoured), simulations=128, batch_size=8,
                    search_mode="gumbel", add_noise=False).run(backend)
        self.assertEqual(int(np.argmax(root.improved_policy)), 3)

    def test_forced_win_is_found_without_network_help(self):
        backend = PurePythonBackend(_position_from_sfen(MATE_IN_ONE))
        root = MCTS(uniform_evaluator, simulations=64, batch_size=8,
                    search_mode="gumbel", add_noise=False).run(backend)
        move, _counts = select_move(root, 0.0, np.random.default_rng(0))
        self.assertIn(move.usi(), SearchTests._immediate_wins(backend))

    def test_virtual_loss_is_fully_unwound(self):
        backend = PurePythonBackend()
        root = MCTS(uniform_evaluator, simulations=40, batch_size=8,
                    search_mode="gumbel", add_noise=True,
                    rng=np.random.default_rng(0)).run(backend)
        self.assertEqual(float(np.abs(root.child_virtual).sum()), 0.0)

    def test_search_leaves_the_board_untouched(self):
        backend = PurePythonBackend()
        before = backend.sfen()
        MCTS(uniform_evaluator, simulations=48, batch_size=8,
             search_mode="gumbel", add_noise=True,
             rng=np.random.default_rng(0)).run(backend)
        self.assertEqual(backend.sfen(), before)

    def test_unknown_search_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MCTS(uniform_evaluator, search_mode="banana")


class SimulationOverrideTests(unittest.TestCase):
    def test_run_simulations_override_spends_only_the_override(self):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=200, batch_size=8)
        root = mcts.run(backend, simulations=24)
        self.assertEqual(int(root.visit_counts().sum()), 24)
        # The configured budget is untouched for the next call.
        self.assertEqual(mcts.simulations, 200)


if __name__ == "__main__":
    unittest.main()
