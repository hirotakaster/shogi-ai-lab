from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

BLACK = 1
WHITE = -1

FILES = "987654321"
RANKS = "abcdefghi"
HAND_ORDER = ["R", "B", "G", "S", "N", "L", "P"]

PROMOTE = {
    "P": "+P",
    "L": "+L",
    "N": "+N",
    "S": "+S",
    "B": "+B",
    "R": "+R",
}
DEMOTE = {v: k for k, v in PROMOTE.items()}
PIECE_VALUES = {
    "K": 0,
    "R": 950,
    "B": 800,
    "G": 620,
    "S": 540,
    "N": 360,
    "L": 320,
    "P": 100,
    "+R": 1150,
    "+B": 1000,
    "+S": 610,
    "+N": 560,
    "+L": 540,
    "+P": 520,
}


@dataclass(frozen=True)
class Piece:
    kind: str
    color: int

    def base(self) -> str:
        return DEMOTE.get(self.kind, self.kind)

    def can_promote(self) -> bool:
        return self.kind in PROMOTE

    def promoted(self) -> "Piece":
        return Piece(PROMOTE[self.kind], self.color)

    def demoted(self) -> "Piece":
        return Piece(self.base(), self.color)

    def sfen(self) -> str:
        text = self.kind
        if text.startswith("+"):
            text = "+" + text[1]
        return text if self.color == BLACK else text.lower()


@dataclass(frozen=True)
class Move:
    from_sq: Optional[Tuple[int, int]]
    to_sq: Tuple[int, int]
    piece: str
    promote: bool = False
    drop: bool = False

    def usi(self) -> str:
        dst = square_name(self.to_sq)
        if self.drop:
            return f"{self.piece}*{dst}"
        assert self.from_sq is not None
        return f"{square_name(self.from_sq)}{dst}{'+' if self.promote else ''}"


@dataclass
class Position:
    board: List[List[Optional[Piece]]] = field(default_factory=lambda: [[None for _ in range(9)] for _ in range(9)])
    hands: Dict[int, Dict[str, int]] = field(default_factory=lambda: {BLACK: {}, WHITE: {}})
    turn: int = BLACK
    move_number: int = 1

    @staticmethod
    def start() -> "Position":
        p = Position()
        rows = [
            ["l", "n", "s", "g", "k", "g", "s", "n", "l"],
            [None, "r", None, None, None, None, None, "b", None],
            ["p", "p", "p", "p", "p", "p", "p", "p", "p"],
            [None] * 9,
            [None] * 9,
            [None] * 9,
            ["P", "P", "P", "P", "P", "P", "P", "P", "P"],
            [None, "B", None, None, None, None, None, "R", None],
            ["L", "N", "S", "G", "K", "G", "S", "N", "L"],
        ]
        for r, row in enumerate(rows):
            for c, token in enumerate(row):
                if token:
                    color = BLACK if token.isupper() else WHITE
                    p.board[r][c] = Piece(token.upper(), color)
        return p

    def clone(self) -> "Position":
        return Position(
            board=[[cell for cell in row] for row in self.board],
            hands={BLACK: dict(self.hands[BLACK]), WHITE: dict(self.hands[WHITE])},
            turn=self.turn,
            move_number=self.move_number,
        )

    def piece_at(self, sq: Tuple[int, int]) -> Optional[Piece]:
        r, c = sq
        return self.board[r][c]

    def set_piece(self, sq: Tuple[int, int], piece: Optional[Piece]) -> None:
        r, c = sq
        self.board[r][c] = piece

    def make_move(self, move: Move) -> "Position":
        nxt = self.clone()
        mover = nxt.turn
        if move.drop:
            count = nxt.hands[mover].get(move.piece, 0)
            if count <= 0:
                raise ValueError("piece is not in hand")
            nxt.hands[mover][move.piece] = count - 1
            if nxt.hands[mover][move.piece] == 0:
                del nxt.hands[mover][move.piece]
            nxt.set_piece(move.to_sq, Piece(move.piece, mover))
        else:
            if move.from_sq is None:
                raise ValueError("normal move needs from_sq")
            piece = nxt.piece_at(move.from_sq)
            if piece is None or piece.color != mover:
                raise ValueError("no mover piece on source square")
            captured = nxt.piece_at(move.to_sq)
            nxt.set_piece(move.from_sq, None)
            moved = piece.promoted() if move.promote else piece
            nxt.set_piece(move.to_sq, moved)
            if captured:
                base = captured.base()
                nxt.hands[mover][base] = nxt.hands[mover].get(base, 0) + 1
        nxt.turn = -mover
        nxt.move_number += 1
        return nxt

    def legal_moves(self) -> List[Move]:
        pseudo = list(self.pseudo_legal_moves())
        legal: List[Move] = []
        for move in pseudo:
            try:
                nxt = self.make_move(move)
            except ValueError:
                continue
            if not nxt.is_in_check(-nxt.turn):
                legal.append(move)
        return legal

    def pseudo_legal_moves(self) -> Iterable[Move]:
        for r in range(9):
            for c in range(9):
                piece = self.board[r][c]
                if piece and piece.color == self.turn:
                    yield from self._piece_moves((r, c), piece)
        yield from self._drop_moves()

    def _piece_moves(self, sq: Tuple[int, int], piece: Piece) -> Iterable[Move]:
        for nr, nc in attacks_from(self.board, sq, piece, include_friendly=False):
            for promote in promotion_options(piece, sq, (nr, nc)):
                if must_promote(piece, (nr, nc)):
                    if promote:
                        yield Move(sq, (nr, nc), piece.kind, promote=True)
                else:
                    yield Move(sq, (nr, nc), piece.kind, promote=promote)

    def _drop_moves(self) -> Iterable[Move]:
        hand = self.hands[self.turn]
        for kind, count in list(hand.items()):
            if count <= 0:
                continue
            for r in range(9):
                for c in range(9):
                    if self.board[r][c] is not None:
                        continue
                    if not can_drop_on(kind, self.turn, (r, c)):
                        continue
                    if kind == "P" and self._file_has_unpromoted_pawn(c, self.turn):
                        continue
                    yield Move(None, (r, c), kind, drop=True)

    def _file_has_unpromoted_pawn(self, c: int, color: int) -> bool:
        for r in range(9):
            p = self.board[r][c]
            if p and p.color == color and p.kind == "P":
                return True
        return False

    def king_square(self, color: int) -> Optional[Tuple[int, int]]:
        for r in range(9):
            for c in range(9):
                p = self.board[r][c]
                if p and p.color == color and p.kind == "K":
                    return (r, c)
        return None

    def is_in_check(self, color: int) -> bool:
        king = self.king_square(color)
        if king is None:
            return True
        attacker = -color
        for r in range(9):
            for c in range(9):
                p = self.board[r][c]
                if p and p.color == attacker:
                    if king in attacks_from(self.board, (r, c), p, include_friendly=True):
                        return True
        return False

    def result(self) -> Optional[int]:
        if self.legal_moves():
            return None
        return -self.turn

    def to_sfen(self) -> str:
        rows = []
        for r in range(9):
            empty = 0
            row = ""
            for c in range(9):
                p = self.board[r][c]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += p.sfen()
            if empty:
                row += str(empty)
            rows.append(row)
        hand = ""
        for color in (BLACK, WHITE):
            for kind in HAND_ORDER:
                n = self.hands[color].get(kind, 0)
                if n:
                    token = kind if color == BLACK else kind.lower()
                    hand += (str(n) if n > 1 else "") + token
        return f"{'/'.join(rows)} {'b' if self.turn == BLACK else 'w'} {hand or '-'} {self.move_number}"

    def json(self) -> dict:
        return {
            "board": [[piece_to_json(p) for p in row] for row in self.board],
            "hands": {
                "black": dict(self.hands[BLACK]),
                "white": dict(self.hands[WHITE]),
            },
            "turn": "black" if self.turn == BLACK else "white",
            "moveNumber": self.move_number,
            "sfen": self.to_sfen(),
            "inCheck": self.is_in_check(self.turn),
            "legalMoves": [m.usi() for m in self.legal_moves()],
        }


def position_from_sfen(sfen: str) -> Position:
    """Build a :class:`Position` from an SFEN string.

    The inverse of :meth:`Position.to_sfen`, used to resume a game from a
    position and to recover the check flag of a stored training record.
    """
    fields = sfen.split()
    if len(fields) < 3:
        raise ValueError(f"malformed SFEN: {sfen!r}")
    board_text, turn_text, hand_text = fields[0], fields[1], fields[2]

    position = Position()
    rows = board_text.split("/")
    if len(rows) != 9:
        raise ValueError(f"SFEN needs 9 ranks, got {len(rows)}: {sfen!r}")
    for r, row in enumerate(rows):
        c = 0
        i = 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                c += int(ch)
                i += 1
                continue
            promoted = ch == "+"
            if promoted:
                i += 1
                ch = row[i]
            kind = ("+" if promoted else "") + ch.upper()
            if kind not in PIECE_VALUES:
                raise ValueError(f"unknown piece {kind!r} in {sfen!r}")
            if c > 8:
                raise ValueError(f"rank {r} overflows in {sfen!r}")
            position.board[r][c] = Piece(kind, BLACK if ch.isupper() else WHITE)
            c += 1
            i += 1

    position.turn = BLACK if turn_text == "b" else WHITE
    if hand_text != "-":
        count = 0
        for ch in hand_text:
            if ch.isdigit():
                count = count * 10 + int(ch)
                continue
            color = BLACK if ch.isupper() else WHITE
            kind = ch.upper()
            position.hands[color][kind] = position.hands[color].get(kind, 0) + (count or 1)
            count = 0
    if len(fields) > 3 and fields[3].isdigit():
        position.move_number = int(fields[3])
    return position


def piece_to_json(piece: Optional[Piece]) -> Optional[dict]:
    if not piece:
        return None
    return {"kind": piece.kind, "color": "black" if piece.color == BLACK else "white"}


def square_name(sq: Tuple[int, int]) -> str:
    r, c = sq
    return FILES[c] + RANKS[r]


def parse_square(text: str) -> Tuple[int, int]:
    return (RANKS.index(text[1]), FILES.index(text[0]))


def parse_usi(text: str, pos: Position) -> Move:
    if "*" in text:
        piece = text[0].upper()
        return Move(None, parse_square(text[2:4]), piece, drop=True)
    from_sq = parse_square(text[0:2])
    to_sq = parse_square(text[2:4])
    piece = pos.piece_at(from_sq)
    if piece is None:
        raise ValueError("source square is empty")
    return Move(from_sq, to_sq, piece.kind, promote=text.endswith("+"))


def inside(r: int, c: int) -> bool:
    return 0 <= r < 9 and 0 <= c < 9


def forward(color: int) -> int:
    return -1 if color == BLACK else 1


def promotion_zone(color: int, r: int) -> bool:
    return r <= 2 if color == BLACK else r >= 6


def promotion_options(piece: Piece, src: Tuple[int, int], dst: Tuple[int, int]) -> List[bool]:
    if not piece.can_promote():
        return [False]
    if promotion_zone(piece.color, src[0]) or promotion_zone(piece.color, dst[0]):
        return [False, True]
    return [False]


def must_promote(piece: Piece, dst: Tuple[int, int]) -> bool:
    r = dst[0]
    if piece.kind in ("P", "L"):
        return r == 0 if piece.color == BLACK else r == 8
    if piece.kind == "N":
        return r <= 1 if piece.color == BLACK else r >= 7
    return False


def can_drop_on(kind: str, color: int, sq: Tuple[int, int]) -> bool:
    r = sq[0]
    if kind in ("P", "L"):
        return r != (0 if color == BLACK else 8)
    if kind == "N":
        return r not in ((0, 1) if color == BLACK else (7, 8))
    return True


def attacks_from(
    board: List[List[Optional[Piece]]],
    sq: Tuple[int, int],
    piece: Piece,
    include_friendly: bool,
) -> List[Tuple[int, int]]:
    r, c = sq
    f = forward(piece.color)
    kind = piece.kind
    gold_steps = [(f, -1), (f, 0), (f, 1), (0, -1), (0, 1), (-f, 0)]
    king_steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    moves: List[Tuple[int, int]] = []

    if kind in ("+P", "+L", "+N", "+S", "G"):
        steps = gold_steps
    elif kind == "S":
        steps = [(f, -1), (f, 0), (f, 1), (-f, -1), (-f, 1)]
    elif kind == "N":
        steps = [(2 * f, -1), (2 * f, 1)]
    elif kind == "K":
        steps = king_steps
    else:
        steps = []
    for dr, dc in steps:
        add_step(board, moves, r + dr, c + dc, piece.color, include_friendly)

    if kind in ("L",):
        slide(board, moves, r, c, f, 0, piece.color, include_friendly)
    if kind in ("B", "+B"):
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            slide(board, moves, r, c, dr, dc, piece.color, include_friendly)
    if kind in ("R", "+R"):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            slide(board, moves, r, c, dr, dc, piece.color, include_friendly)
    if kind == "+B":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            add_step(board, moves, r + dr, c + dc, piece.color, include_friendly)
    if kind == "+R":
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            add_step(board, moves, r + dr, c + dc, piece.color, include_friendly)
    if kind == "P":
        add_step(board, moves, r + f, c, piece.color, include_friendly)
    return moves


def add_step(board, moves, r, c, color, include_friendly) -> None:
    if not inside(r, c):
        return
    target = board[r][c]
    if target is None or target.color != color or include_friendly:
        moves.append((r, c))


def slide(board, moves, r, c, dr, dc, color, include_friendly) -> None:
    nr, nc = r + dr, c + dc
    while inside(nr, nc):
        target = board[nr][nc]
        if target is None:
            moves.append((nr, nc))
        else:
            if target.color != color or include_friendly:
                moves.append((nr, nc))
            break
        nr += dr
        nc += dc

