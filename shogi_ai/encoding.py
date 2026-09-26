"""Feature planes and policy indices shared by every backend.

The network only ever sees a "Black to move" board: when White is to move the
whole position is rotated 180 degrees and the two sides are swapped.  Both the
pure-Python rules engine and the cshogi backend hand this module a *raw*
snapshot in absolute coordinates, and the normalisation happens here exactly
once, so the two backends cannot drift apart.

Square indices are ``sq = rank * 9 + file`` using the same orientation as
``core.Position.board`` (rank 0 is White's back rank).  Rotating the board 180
degrees is therefore just ``80 - sq``.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Piece and hand layout
# ---------------------------------------------------------------------------

#: Board plane order.  Index + 1 is the signed code used in a raw snapshot.
PIECE_ORDER: Tuple[str, ...] = (
    "P", "L", "N", "S", "G", "B", "R", "K",
    "+P", "+L", "+N", "+S", "+B", "+R",
)
PIECE_INDEX: Dict[str, int] = {kind: i for i, kind in enumerate(PIECE_ORDER)}
NUM_PIECE_PLANES = len(PIECE_ORDER)  # 14

#: Hand plane order with the cap used for the thermometer encoding.  A player
#: can hold 18 pawns in theory, but counts above the cap carry almost no extra
#: information and cost planes, so they saturate.
HAND_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("P", 8), ("L", 4), ("N", 4), ("S", 4), ("G", 4), ("B", 2), ("R", 2),
)
HAND_ORDER: Tuple[str, ...] = tuple(kind for kind, _ in HAND_LAYOUT)
HAND_INDEX: Dict[str, int] = {kind: i for i, kind in enumerate(HAND_ORDER)}
NUM_HAND_PLANES = sum(cap for _, cap in HAND_LAYOUT)  # 28

#: Offset of each hand piece inside one side's hand plane block.
_HAND_OFFSET: Tuple[int, ...] = tuple(
    sum(cap for _, cap in HAND_LAYOUT[:i]) for i in range(len(HAND_LAYOUT))
)

OWN_BOARD_OFFSET = 0
OPP_BOARD_OFFSET = NUM_PIECE_PLANES
OWN_HAND_OFFSET = 2 * NUM_PIECE_PLANES
OPP_HAND_OFFSET = OWN_HAND_OFFSET + NUM_HAND_PLANES
CHECK_PLANE = OPP_HAND_OFFSET + NUM_HAND_PLANES

#: 14 own + 14 opponent board planes, 28 + 28 hand planes, 1 check plane.
NUM_FEATURE_PLANES = CHECK_PLANE + 1  # 85

# ---------------------------------------------------------------------------
# Policy layout: 27 planes of 81 destination squares (dlshogi convention)
# ---------------------------------------------------------------------------

#: Unit directions from the mover's point of view.  "Up" is towards the
#: opponent, which after normalisation always means a decreasing rank.
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),    # 0 up
    (-1, -1),   # 1 up-left
    (-1, 1),    # 2 up-right
    (0, -1),    # 3 left
    (0, 1),     # 4 right
    (1, 0),     # 5 down
    (1, -1),    # 6 down-left
    (1, 1),     # 7 down-right
    (-2, -1),   # 8 knight up-left
    (-2, 1),    # 9 knight up-right
)
_DIRECTION_INDEX: Dict[Tuple[int, int], int] = {d: i for i, d in enumerate(DIRECTIONS)}
NUM_DIRECTIONS = len(DIRECTIONS)  # 10

PROMOTE_OFFSET = NUM_DIRECTIONS          # 10
DROP_OFFSET = 2 * NUM_DIRECTIONS         # 20
NUM_MOVE_PLANES = DROP_OFFSET + len(HAND_ORDER)  # 27
NUM_SQUARES = 81
POLICY_SIZE = NUM_MOVE_PLANES * NUM_SQUARES  # 2187


class RawSnapshot(NamedTuple):
    """A position in absolute coordinates, as produced by a backend.

    ``squares`` holds 81 signed piece codes indexed by ``rank * 9 + file``:
    ``0`` is empty, ``+k`` is a Black ``PIECE_ORDER[k - 1]`` and ``-k`` the
    White equivalent.  ``hands`` is ``(black_counts, white_counts)`` with seven
    counts each in :data:`HAND_ORDER`.  ``turn`` is ``+1`` for Black.
    """

    squares: Sequence[int]
    hands: Tuple[Sequence[int], Sequence[int]]
    turn: int
    in_check: bool


class RawMove(NamedTuple):
    """A move in absolute coordinates, as produced by a backend."""

    from_sq: Optional[int]
    to_sq: int
    promote: bool
    drop_piece: Optional[str]


def encode_planes(raw: RawSnapshot, out=None):
    """Return an ``(85, 9, 9)`` float32 array from the mover's point of view."""
    import numpy as np

    if out is None:
        out = np.zeros((NUM_FEATURE_PLANES, NUM_SQUARES), dtype=np.float32)
    else:
        out = out.reshape(NUM_FEATURE_PLANES, NUM_SQUARES)
        out.fill(0.0)

    black_to_move = raw.turn > 0
    squares = raw.squares
    for sq in range(NUM_SQUARES):
        code = squares[sq]
        if not code:
            continue
        # A Black piece belongs to the mover only when Black is to move.
        own = (code > 0) == black_to_move
        plane = (OWN_BOARD_OFFSET if own else OPP_BOARD_OFFSET) + abs(code) - 1
        out[plane, sq if black_to_move else 80 - sq] = 1.0

    own_hand, opp_hand = (raw.hands if black_to_move else (raw.hands[1], raw.hands[0]))
    for base, counts in ((OWN_HAND_OFFSET, own_hand), (OPP_HAND_OFFSET, opp_hand)):
        for i, (_kind, cap) in enumerate(HAND_LAYOUT):
            filled = min(int(counts[i]), cap)
            if filled:
                start = base + _HAND_OFFSET[i]
                out[start:start + filled, :] = 1.0

    if raw.in_check:
        out[CHECK_PLANE, :] = 1.0
    return out.reshape(NUM_FEATURE_PLANES, 9, 9)


def policy_index(move: RawMove, turn: int) -> int:
    """Map an absolute move to its index in the ``2187``-wide policy head."""
    black_to_move = turn > 0
    to_sq = move.to_sq if black_to_move else 80 - move.to_sq

    if move.drop_piece is not None:
        try:
            plane = DROP_OFFSET + HAND_INDEX[move.drop_piece]
        except KeyError:
            raise ValueError(f"not a droppable piece: {move.drop_piece!r}") from None
        return plane * NUM_SQUARES + to_sq

    if move.from_sq is None:
        raise ValueError("a board move needs from_sq")
    from_sq = move.from_sq if black_to_move else 80 - move.from_sq

    dr = to_sq // 9 - from_sq // 9
    dc = to_sq % 9 - from_sq % 9
    if (dr, dc) in ((-2, -1), (-2, 1)):
        direction = _DIRECTION_INDEX[(dr, dc)]
    else:
        unit = (_sign(dr), _sign(dc))
        if unit not in _DIRECTION_INDEX or (dr and dc and abs(dr) != abs(dc)):
            raise ValueError(f"move is not along a shogi ray: {move!r}")
        direction = _DIRECTION_INDEX[unit]

    plane = (PROMOTE_OFFSET + direction) if move.promote else direction
    return plane * NUM_SQUARES + to_sq


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def legal_policy_indices(moves: Sequence[RawMove], turn: int) -> List[int]:
    """Policy indices for a list of legal moves, in the same order."""
    return [policy_index(move, turn) for move in moves]
