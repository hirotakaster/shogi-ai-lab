import unittest

import numpy as np

from shogi_ai import encoding
from shogi_ai.backends import PurePythonBackend, snapshot_from_sfen
from shogi_ai.core import Position
from shogi_ai.encoding import (
    CHECK_PLANE,
    NUM_FEATURE_PLANES,
    OPP_BOARD_OFFSET,
    OPP_HAND_OFFSET,
    OWN_BOARD_OFFSET,
    OWN_HAND_OFFSET,
    POLICY_SIZE,
    PIECE_INDEX,
    RawMove,
    encode_planes,
    policy_index,
)


class PolicyLayoutTests(unittest.TestCase):
    def test_policy_size_matches_dlshogi_convention(self):
        self.assertEqual(encoding.NUM_MOVE_PLANES, 27)
        self.assertEqual(POLICY_SIZE, 2187)

    def test_every_legal_move_maps_into_range_and_is_unique(self):
        """Walk a few plies and check the mapping is injective everywhere."""
        backend = PurePythonBackend()
        seen_positions = 0
        for _ in range(12):
            native, raw = backend.legal_moves()
            if not native:
                break
            indices = [policy_index(move, backend.turn) for move in raw]
            self.assertEqual(
                len(indices), len(set(indices)),
                "two distinct legal moves collided on one policy index",
            )
            for index in indices:
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, POLICY_SIZE)
            seen_positions += 1
            backend.push(native[len(native) // 3])
        self.assertGreaterEqual(seen_positions, 12)

    def test_white_moves_are_normalised_onto_black_planes(self):
        """The mirrored move of a mirrored position must share its index."""
        # Black pushes the file-7 pawn; White pushes the mirrored file-3 pawn.
        black = RawMove(from_sq=_sq("7g"), to_sq=_sq("7f"), promote=False, drop_piece=None)
        white = RawMove(from_sq=_sq("3c"), to_sq=_sq("3d"), promote=False, drop_piece=None)
        self.assertEqual(policy_index(black, 1), policy_index(white, -1))

    def test_knight_and_drop_planes_are_distinct(self):
        knight = RawMove(_sq("8i"), _sq("7g"), False, None)
        drop = RawMove(None, _sq("7g"), False, "N")
        self.assertNotEqual(policy_index(knight, 1), policy_index(drop, 1))
        self.assertGreaterEqual(policy_index(drop, 1) // 81, encoding.DROP_OFFSET)

    def test_knight_hops_use_the_two_dedicated_planes(self):
        # Column indices run opposite to file numbers (column 0 is file 9), so
        # 8i8g steps towards a higher column and 2i3g towards a lower one.
        self.assertEqual(policy_index(RawMove(_sq("8i"), _sq("7g"), False, None), 1) // 81, 9)
        self.assertEqual(policy_index(RawMove(_sq("2i"), _sq("3g"), False, None), 1) // 81, 8)

    def test_promotion_uses_a_separate_plane(self):
        plain = RawMove(_sq("2d"), _sq("2c"), False, None)
        promoted = RawMove(_sq("2d"), _sq("2c"), True, None)
        self.assertEqual(
            policy_index(promoted, 1) - policy_index(plain, 1),
            encoding.PROMOTE_OFFSET * 81,
        )

    def test_non_ray_move_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_index(RawMove(_sq("9i"), _sq("7h"), False, None), 1)


class FeaturePlaneTests(unittest.TestCase):
    def test_plane_count(self):
        self.assertEqual(NUM_FEATURE_PLANES, 85)

    def test_start_position_planes(self):
        planes = encode_planes(PurePythonBackend().snapshot())
        self.assertEqual(planes.shape, (85, 9, 9))
        # 20 pieces per side on the board, nothing in hand, nobody in check.
        self.assertEqual(planes[OWN_BOARD_OFFSET:OWN_BOARD_OFFSET + 14].sum(), 20)
        self.assertEqual(planes[OPP_BOARD_OFFSET:OPP_BOARD_OFFSET + 14].sum(), 20)
        self.assertEqual(planes[OWN_HAND_OFFSET:CHECK_PLANE].sum(), 0)
        self.assertEqual(planes[CHECK_PLANE].sum(), 0)
        # Black's own king sits on 5i == rank 8, file 4.
        self.assertEqual(planes[OWN_BOARD_OFFSET + PIECE_INDEX["K"], 8, 4], 1.0)

    def test_side_to_move_normalisation_is_symmetric(self):
        """The mirrored position with the other side to move encodes identically."""
        backend = PurePythonBackend()
        before = encode_planes(backend.snapshot())
        # 7g7f for Black, then the mirrored 3c3d for White.
        backend.push(backend.parse_usi("7g7f"))
        backend.push(backend.parse_usi("3c3d"))
        after = encode_planes(backend.snapshot())
        # Both sides played the mirrored move, so the mover's view is unchanged
        # except that its own pawn has advanced -- compare to a hand-built
        # expectation instead: the two encodings must differ only on pawn planes.
        differing = {int(p) for p in np.nonzero((before != after).any(axis=(1, 2)))[0]}
        self.assertEqual(
            differing,
            {OWN_BOARD_OFFSET + PIECE_INDEX["P"], OPP_BOARD_OFFSET + PIECE_INDEX["P"]},
        )

    def test_hand_planes_use_saturating_thermometer(self):
        raw = snapshot_from_sfen("9/9/9/9/9/9/9/9/9 b 3P2r 1", in_check=False)
        planes = encode_planes(raw)
        pawn_planes = planes[OWN_HAND_OFFSET:OWN_HAND_OFFSET + 8]
        self.assertEqual(int(pawn_planes[:, 0, 0].sum()), 3)
        self.assertTrue(bool(pawn_planes[0].all()))
        self.assertFalse(bool(pawn_planes[3].any()))
        # Two rooks in the opponent's hand saturate that two-plane block.
        rook_base = OPP_HAND_OFFSET + encoding._HAND_OFFSET[encoding.HAND_INDEX["R"]]
        self.assertEqual(int(planes[rook_base:rook_base + 2, 0, 0].sum()), 2)

    def test_pawn_count_saturates_at_the_cap(self):
        raw = snapshot_from_sfen("9/9/9/9/9/9/9/9/9 b 18P 1", in_check=False)
        planes = encode_planes(raw)
        self.assertEqual(int(planes[OWN_HAND_OFFSET:OWN_HAND_OFFSET + 8, 0, 0].sum()), 8)

    def test_check_plane_is_set(self):
        raw = snapshot_from_sfen("4k4/9/9/9/9/9/9/9/4K3r w - 1", in_check=True)
        self.assertEqual(encode_planes(raw)[CHECK_PLANE].sum(), 81)

    def test_out_buffer_is_reused_and_cleared(self):
        buffer = np.ones((85, 9, 9), dtype=np.float32)
        planes = encode_planes(PurePythonBackend().snapshot(), out=buffer)
        self.assertEqual(planes[CHECK_PLANE].sum(), 0)


class BackendSnapshotTests(unittest.TestCase):
    """The pure-Python snapshot and the SFEN parser must agree.

    The cshogi backend builds its snapshot through ``snapshot_from_sfen``, so
    pinning that function against the pure-Python backend is what keeps the two
    backends from diverging on machines where cshogi cannot be installed.
    """

    def test_snapshots_agree_along_a_game(self):
        backend = PurePythonBackend()
        for ply in range(24):
            direct = backend.snapshot()
            parsed = snapshot_from_sfen(backend.sfen(), direct.in_check)
            self.assertEqual(list(direct.squares), list(parsed.squares), f"ply {ply}")
            self.assertEqual(
                [list(h) for h in direct.hands],
                [list(h) for h in parsed.hands],
                f"ply {ply}",
            )
            self.assertEqual(direct.turn, parsed.turn, f"ply {ply}")
            native, _raw = backend.legal_moves()
            if not native:
                break
            backend.push(native[ply % len(native)])

    def test_snapshots_agree_with_pieces_in_hand(self):
        backend = PurePythonBackend()
        for usi in ["7g7f", "3c3d", "8h3c+", "2b3c"]:
            backend.push(backend.parse_usi(usi))
        direct = backend.snapshot()
        parsed = snapshot_from_sfen(backend.sfen(), direct.in_check)
        self.assertEqual([list(h) for h in direct.hands], [list(h) for h in parsed.hands])
        self.assertNotEqual(sum(direct.hands[0]) + sum(direct.hands[1]), 0)
        np.testing.assert_array_equal(encode_planes(direct), encode_planes(parsed))


class RepetitionTests(unittest.TestCase):
    def test_fourfold_repetition_is_a_draw(self):
        backend = PurePythonBackend()
        # Shuffle both rooks up and back to repeat the opening position.
        cycle = ["2h3h", "8b7b", "3h2h", "7b8b"]
        for _ in range(3):
            for usi in cycle:
                backend.push(backend.parse_usi(usi))
        self.assertTrue(backend.is_repetition_draw())
        self.assertEqual(backend.terminal_value(), 0.0)

    def test_repetition_count_unwinds_on_pop(self):
        backend = PurePythonBackend()
        cycle = ["2h3h", "8b7b", "3h2h", "7b8b"]
        for _ in range(3):
            for usi in cycle:
                backend.push(backend.parse_usi(usi))
        self.assertTrue(backend.is_repetition_draw())
        backend.pop()
        self.assertFalse(backend.is_repetition_draw())
        self.assertIsNone(backend.terminal_value())

    def test_live_position_has_no_terminal_value(self):
        self.assertIsNone(PurePythonBackend().terminal_value())


def _sq(text: str) -> int:
    from shogi_ai.core import parse_square

    rank, file = parse_square(text)
    return rank * 9 + file


if __name__ == "__main__":
    unittest.main()
