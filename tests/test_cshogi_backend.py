"""Pin the cshogi backend against the pure-Python one.

cshogi cannot be built on Apple Silicon (its setup.py forces the x86 SSE/AVX
flags), so these tests skip locally and run on Colab, where they are the check
that the fast backend really produces the same features, the same legal moves
and the same policy indices as the reference implementation.  If cshogi ever
renames part of its API, this is what fails.
"""

import unittest

import numpy as np

from shogi_ai.backends import PurePythonBackend
from shogi_ai.encoding import encode_planes, policy_index

from shogi_ai.backends import CshogiBackend, import_cshogi

_CSHOGI, _CSHOGI_ERROR = import_cshogi()
HAVE_CSHOGI = _CSHOGI is not None
SKIP_REASON = _CSHOGI_ERROR or ""

#: Quiet development moves: no captures, empty hands throughout.
OPENING_LINE = ["7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e", "8h7g", "3d3e"]
#: A bishop trade, so both players end up holding a piece.
CAPTURE_LINE = ["7g7f", "3c3d", "8h2b+", "3a2b"]
#: Continues the trade until both sides have dropped their bishop back on.
DROP_LINE = CAPTURE_LINE + ["B*4e", "2b3c", "6i7h", "B*5e"]
ALL_LINES = (OPENING_LINE, CAPTURE_LINE, DROP_LINE)


class FixtureTests(unittest.TestCase):
    """Check the move lists above are legal.

    Deliberately outside the cshogi-only class: these fixtures are only
    exercised where cshogi is installed, so a bad line used to sit unnoticed
    until someone ran the suite on a GPU machine.  Validating them against the
    dependency-free engine catches that everywhere.
    """

    def test_every_line_is_legal_from_the_opening(self):
        for name, line in (("OPENING_LINE", OPENING_LINE),
                           ("CAPTURE_LINE", CAPTURE_LINE),
                           ("DROP_LINE", DROP_LINE)):
            backend = PurePythonBackend()
            for ply, usi in enumerate(line):
                legal = {m.usi() for m in backend.position.legal_moves()}
                self.assertIn(usi, legal, f"{name}[{ply}] = {usi} is not legal")
                backend.push(backend.parse_usi(usi))

    def test_capture_line_fills_both_hands(self):
        """Otherwise the hand planes would never be exercised."""
        backend = PurePythonBackend()
        for usi in CAPTURE_LINE:
            backend.push(backend.parse_usi(usi))
        black, white = backend.snapshot().hands
        self.assertGreater(sum(black), 0, "Black holds nothing")
        self.assertGreater(sum(white), 0, "White holds nothing")

    def test_drop_line_contains_a_drop_for_each_side(self):
        drops = [usi for usi in DROP_LINE if "*" in usi]
        self.assertGreaterEqual(len(drops), 2, drops)


@unittest.skipUnless(HAVE_CSHOGI, SKIP_REASON)
class CshogiAgreementTests(unittest.TestCase):
    def _pair(self):
        return PurePythonBackend(), CshogiBackend()

    def test_turn_mapping(self):
        python, fast = self._pair()
        self.assertEqual(python.turn, fast.turn)
        python.push(python.parse_usi("7g7f"))
        fast.push(fast.parse_usi("7g7f"))
        self.assertEqual(python.turn, fast.turn)

    def test_sfen_agrees_along_a_line(self):
        for line in ALL_LINES:
            python, fast = self._pair()
            for usi in line:
                python.push(python.parse_usi(usi))
                fast.push(fast.parse_usi(usi))
                self.assertEqual(python.sfen(), fast.sfen(), f"after {usi}")

    def test_snapshots_and_planes_agree(self):
        for line in ALL_LINES:
            python, fast = self._pair()
            for usi in [None] + line:
                if usi:
                    python.push(python.parse_usi(usi))
                    fast.push(fast.parse_usi(usi))
                a, b = python.snapshot(), fast.snapshot()
                self.assertEqual(list(a.squares), list(b.squares), f"squares after {usi}")
                self.assertEqual([list(h) for h in a.hands],
                                 [list(h) for h in b.hands], f"hands after {usi}")
                self.assertEqual(a.turn, b.turn, f"turn after {usi}")
                self.assertEqual(a.in_check, b.in_check, f"check after {usi}")
                np.testing.assert_array_equal(encode_planes(a), encode_planes(b))

    def test_hands_are_populated_after_captures(self):
        python, fast = self._pair()
        for usi in CAPTURE_LINE:
            python.push(python.parse_usi(usi))
            fast.push(fast.parse_usi(usi))
        snapshot = fast.snapshot()
        # Both sides hold a bishop here, so a hand mix-up cannot hide.
        self.assertGreater(sum(snapshot.hands[0]), 0)
        self.assertGreater(sum(snapshot.hands[1]), 0)
        self.assertEqual([list(h) for h in snapshot.hands],
                         [list(h) for h in python.snapshot().hands])

    def test_drops_agree_with_the_reference(self):
        python, fast = self._pair()
        for usi in DROP_LINE:
            python.push(python.parse_usi(usi))
            fast.push(fast.parse_usi(usi))
            self.assertEqual(python.sfen(), fast.sfen(), f"after {usi}")

    def test_legal_move_sets_agree(self):
        for line in ALL_LINES:
            python, fast = self._pair()
            for usi in [None] + line:
                if usi:
                    python.push(python.parse_usi(usi))
                    fast.push(fast.parse_usi(usi))
                py_native, _py_raw = python.legal_moves()
                fast_native, _fast_raw = fast.legal_moves()
                py_usi = sorted(python.move_usi(m) for m in py_native)
                fast_usi = sorted(fast.move_usi(m) for m in fast_native)
                self.assertEqual(py_usi, fast_usi, f"legal moves after {usi}")

    def test_policy_indices_agree(self):
        for line in ALL_LINES:
            python, fast = self._pair()
            for usi in [None] + line:
                if usi:
                    python.push(python.parse_usi(usi))
                    fast.push(fast.parse_usi(usi))
                py_native, py_raw = python.legal_moves()
                fast_native, fast_raw = fast.legal_moves()
                py_map = {python.move_usi(m): policy_index(r, python.turn)
                          for m, r in zip(py_native, py_raw)}
                fast_map = {fast.move_usi(m): policy_index(r, fast.turn)
                            for m, r in zip(fast_native, fast_raw)}
                self.assertEqual(py_map, fast_map, f"policy indices after {usi}")

    def test_pop_restores_the_position(self):
        python, fast = self._pair()
        before = fast.sfen()
        for usi in OPENING_LINE:
            fast.push(fast.parse_usi(usi))
        for _ in OPENING_LINE:
            fast.pop()
        self.assertEqual(fast.sfen(), before)

    def test_terminal_value_is_none_while_the_game_is_live(self):
        _python, fast = self._pair()
        self.assertIsNone(fast.terminal_value())

    def test_mate_is_detected(self):
        fast = CshogiBackend("7pk/R8/7K1/9/9/9/9/9/9 b - 1")
        fast.push(fast.parse_usi("9b1b"))
        self.assertEqual(fast.terminal_value(), -1.0)

    def test_search_runs_on_the_fast_backend(self):
        from shogi_ai.evaluator import UniformEvaluator
        from shogi_ai.mcts import MCTS

        fast = CshogiBackend()
        before = fast.sfen()
        root = MCTS(UniformEvaluator(), simulations=64, batch_size=8).run(fast)
        self.assertEqual(int(root.visit_counts().sum()), 64)
        self.assertEqual(fast.sfen(), before, "search must leave the board untouched")

    def test_self_play_produces_records(self):
        from shogi_ai.evaluator import UniformEvaluator
        from shogi_ai.selfplay_az import SelfPlayConfig, play_one_game

        config = SelfPlayConfig(simulations=16, batch_size=4, max_moves=10,
                                backend="cshogi", resign_threshold=-1.0)
        summary = play_one_game(UniformEvaluator(), config, seed=1)
        self.assertTrue(summary.records)
        for record in summary.records:
            self.assertIn("policy", record)
            self.assertAlmostEqual(sum(p for _i, p in record["policy"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
