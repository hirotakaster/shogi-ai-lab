"""Banned opening moves in self-play (Batch 19).

Campaign 3's self-play records funnel through one first move: sente
plays the central rook swing 2h5h in 82.2% of recorded games and the
pair 2h5h + 8b5b in 65.5%.  ``--banned-moves 0:2h5h,1:8b5b`` removes
those moves from the root's candidate list on the named plies, so
they are never searched, played or recorded.  These tests pin the
parser, the MCTS root restriction (both root modes, plus the
retained-subtree path self-play actually exercises) and the
end-to-end self-play behaviour.
"""

import unittest

import numpy as np

from shogi_ai.backends import PurePythonBackend, raw_move_from_usi
from shogi_ai.core import BLACK
from shogi_ai.encoding import POLICY_SIZE, policy_index
from shogi_ai.mcts import MCTS, policy_targets
from shogi_ai.selfplay_az import (SelfPlayConfig, parse_banned_moves,
                                  play_one_game)

ROOK_SWING_BLACK = "2h5h"  # sente's first move: rook to the centre file
ROOK_SWING_WHITE = "8b5b"  # gote's matching reply


def uniform_evaluator(planes):
    """A stand-in network: flat policy, neutral value. Needs no torch."""
    batch = planes.shape[0]
    return (np.zeros((batch, POLICY_SIZE), dtype=np.float32),
            np.zeros(batch, dtype=np.float32))


def banned_index(usi, turn):
    return policy_index(raw_move_from_usi(usi), turn)


class ParseBannedMovesTests(unittest.TestCase):
    def test_empty_string_bans_nothing(self):
        self.assertEqual(parse_banned_moves(""), ())

    def test_parses_ply_usi_pairs(self):
        self.assertEqual(parse_banned_moves("0:2h5h,1:8b5b"),
                         ((0, "2h5h"), (1, "8b5b")))

    def test_repeated_ply_and_drops_are_allowed(self):
        self.assertEqual(parse_banned_moves("0:2h5h,0:7g7f,3:R*5h"),
                         ((0, "2h5h"), (0, "7g7f"), (3, "R*5h")))

    def test_malformed_entries_raise(self):
        for text in ("2h5h", "x:2h5h", "-1:2h5h", "0:2h", "0:2h5x",
                     "0:", "0:2h5h,", "0:2H5H"):
            with self.assertRaises(ValueError, msg=text):
                parse_banned_moves(text)


class RootRestrictionTests(unittest.TestCase):
    def run_root(self, banned, **mcts_kwargs):
        backend = PurePythonBackend()
        mcts = MCTS(uniform_evaluator, simulations=32, batch_size=16,
                    rng=np.random.default_rng(7), **mcts_kwargs)
        return backend, mcts.run(backend, banned_indices=banned)

    def test_banned_move_absent_from_root_and_targets(self):
        banned = {banned_index(ROOK_SWING_BLACK, BLACK)}
        backend, root = self.run_root(banned)
        _, legal_raw = PurePythonBackend().legal_moves()
        self.assertEqual(len(root.moves), len(legal_raw) - 1)
        self.assertNotIn(raw_move_from_usi(ROOK_SWING_BLACK), root.raw_moves)
        target_indices = {index for index, _ in policy_targets(root, backend.turn)}
        self.assertNotIn(banned_index(ROOK_SWING_BLACK, BLACK), target_indices)
        self.assertAlmostEqual(
            sum(prob for _, prob in policy_targets(root, backend.turn)),
            1.0, places=5)

    def test_without_ban_the_move_is_present(self):
        _, root = self.run_root(None)
        self.assertIn(raw_move_from_usi(ROOK_SWING_BLACK), root.raw_moves)

    def test_gumbel_root_is_restricted_too(self):
        banned = {banned_index(ROOK_SWING_BLACK, BLACK)}
        backend, root = self.run_root(
            banned, search_mode="gumbel", gumbel_candidates=8)
        self.assertNotIn(raw_move_from_usi(ROOK_SWING_BLACK), root.raw_moves)
        self.assertIsNotNone(root.improved_policy)
        self.assertEqual(len(root.improved_policy), len(root.moves))
        target_indices = {index for index, _ in policy_targets(root, backend.turn)}
        self.assertNotIn(banned_index(ROOK_SWING_BLACK, BLACK), target_indices)

    def test_ban_covering_every_move_is_ignored(self):
        backend = PurePythonBackend()
        _, legal_raw = backend.legal_moves()
        banned = {policy_index(move, backend.turn) for move in legal_raw}
        _, root = self.run_root(banned)
        self.assertEqual(len(root.moves), len(legal_raw))


class SelfPlayBanTests(unittest.TestCase):
    def test_banned_first_moves_are_never_played_or_recorded(self):
        config = SelfPlayConfig(
            simulations=24, batch_size=16, max_moves=6, backend="python",
            resign_threshold=-1.0,
            banned_moves=((0, ROOK_SWING_BLACK), (1, ROOK_SWING_WHITE)))
        summary = play_one_game(uniform_evaluator, config, seed=20261121)
        by_ply = {record["ply"]: record for record in summary.records}
        self.assertIn(0, by_ply)
        self.assertIn(1, by_ply)
        self.assertIn(2, by_ply)

        # Ply 0: the banned swing is absent from the recorded target.
        sente_targets = {index for index, _ in by_ply[0]["policy"]}
        self.assertNotIn(banned_index(ROOK_SWING_BLACK, BLACK), sente_targets)
        # Ply 1's position is the one after sente's first move: the
        # rook must not stand on the centre file (SFEN rank-8 row).
        rows = by_ply[1]["sfen"].split(" ")[0].split("/")
        self.assertNotEqual(rows[7], "1B2R4")
        # Ply 1: gote's banned swing is absent from the target too.
        gote_targets = {index for index, _ in by_ply[1]["policy"]}
        self.assertNotIn(banned_index(ROOK_SWING_WHITE, -BLACK), gote_targets)
        # Ply 2's position: gote's rook must not stand on the centre
        # file either (SFEN rank-2 row).
        rows = by_ply[2]["sfen"].split(" ")[0].split("/")
        self.assertNotEqual(rows[1], "4r2b1")

    def test_invalid_config_entries_raise(self):
        config = SelfPlayConfig(backend="python",
                                banned_moves=((-1, ROOK_SWING_BLACK),))
        with self.assertRaises(ValueError):
            play_one_game(uniform_evaluator, config, seed=1)
        config = SelfPlayConfig(backend="python",
                                banned_moves=((0, "not-a-move"),))
        with self.assertRaises(ValueError):
            play_one_game(uniform_evaluator, config, seed=1)


if __name__ == "__main__":
    unittest.main()
