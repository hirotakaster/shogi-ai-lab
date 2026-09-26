"""Position backends for search and self-play.

Two implementations share one interface:

``PurePythonBackend``
    Wraps :class:`shogi_ai.core.Position`.  No third-party dependency, so the
    local Web server can always play.  Move generation is slow.

``CshogiBackend``
    Wraps ``cshogi.Board``.  Orders of magnitude faster, used for self-play on
    Colab.  It deliberately exchanges data with the shared encoder through SFEN
    and USI strings: that costs a little speed but means the features and
    policy indices are produced by exactly the same code as the pure-Python
    backend, so the two cannot disagree about what a position looks like.

Both report terminal values from the **side to move**'s point of view.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .core import BLACK, Move, Position, parse_square, position_from_sfen
from .encoding import HAND_ORDER, PIECE_INDEX, RawMove, RawSnapshot

#: Fourfold repetition is adjudicated as a draw (sennichite).
REPETITION_LIMIT = 4


def snapshot_from_sfen(sfen: str, in_check: bool) -> RawSnapshot:
    """Parse an SFEN board/turn/hand field into a :class:`RawSnapshot`.

    Shared by the cshogi backend and by the tests that pin the pure-Python
    backend against it.
    """
    parts = sfen.split()
    board_text, turn_text, hand_text = parts[0], parts[1], parts[2]

    squares = [0] * 81
    for rank, row in enumerate(board_text.split("/")):
        file = 0
        i = 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                file += int(ch)
                i += 1
                continue
            promoted = ch == "+"
            if promoted:
                i += 1
                ch = row[i]
            code = PIECE_INDEX[("+" if promoted else "") + ch.upper()] + 1
            squares[rank * 9 + file] = code if ch.isupper() else -code
            file += 1
            i += 1

    black_hand = [0] * len(HAND_ORDER)
    white_hand = [0] * len(HAND_ORDER)
    if hand_text != "-":
        count = 0
        for ch in hand_text:
            if ch.isdigit():
                count = count * 10 + int(ch)
                continue
            hand = black_hand if ch.isupper() else white_hand
            hand[HAND_ORDER.index(ch.upper())] += count or 1
            count = 0

    return RawSnapshot(
        squares=squares,
        hands=(black_hand, white_hand),
        turn=BLACK if turn_text == "b" else -BLACK,
        in_check=in_check,
    )


def raw_move_from_usi(usi: str) -> RawMove:
    """Parse a USI move into absolute-coordinate form."""
    if "*" in usi:
        return RawMove(None, _sq(usi[2:4]), False, usi[0].upper())
    return RawMove(_sq(usi[0:2]), _sq(usi[2:4]), usi.endswith("+"), None)


def _sq(text: str) -> int:
    rank, file = parse_square(text)
    return rank * 9 + file


class PurePythonBackend:
    """Backend over the dependency-free rules engine."""

    name = "python"

    def __init__(self, position: Optional[Position] = None):
        self.position = position or Position.start()
        self._stack: List[Position] = []
        self._repetition: Dict[str, int] = {}
        self._count_repetition(self.position)

    # -- position queries ------------------------------------------------
    @property
    def turn(self) -> int:
        return self.position.turn

    def snapshot(self) -> RawSnapshot:
        pos = self.position
        squares = [0] * 81
        for rank in range(9):
            row = pos.board[rank]
            for file in range(9):
                piece = row[file]
                if piece is not None:
                    code = PIECE_INDEX[piece.kind] + 1
                    squares[rank * 9 + file] = code if piece.color == BLACK else -code
        hands = tuple(
            [pos.hands[color].get(kind, 0) for kind in HAND_ORDER]
            for color in (BLACK, -BLACK)
        )
        return RawSnapshot(
            squares=squares,
            hands=(hands[0], hands[1]),
            turn=pos.turn,
            in_check=pos.is_in_check(pos.turn),
        )

    def legal_moves(self) -> Tuple[List[Move], List[RawMove]]:
        native = self.position.legal_moves()
        raw = [
            RawMove(
                None if move.drop else move.from_sq[0] * 9 + move.from_sq[1],
                move.to_sq[0] * 9 + move.to_sq[1],
                move.promote,
                move.piece if move.drop else None,
            )
            for move in native
        ]
        return native, raw

    def sfen(self) -> str:
        return self.position.to_sfen()

    def move_usi(self, move: Move) -> str:
        return move.usi()

    def parse_usi(self, usi: str) -> Move:
        from .core import parse_usi

        return parse_usi(usi, self.position)

    # -- tree traversal --------------------------------------------------
    def push(self, move: Move) -> None:
        self._stack.append(self.position)
        self.position = self.position.make_move(move)
        self._count_repetition(self.position)

    def pop(self) -> None:
        key = self._repetition_key(self.position)
        remaining = self._repetition.get(key, 0) - 1
        if remaining > 0:
            self._repetition[key] = remaining
        else:
            self._repetition.pop(key, None)
        self.position = self._stack.pop()

    def terminal_value(self) -> Optional[float]:
        """``None`` while the game is live, else the value for the mover."""
        if self.is_repetition_draw():
            return 0.0
        if not self.position.legal_moves():
            # No legal reply: the side to move has been mated.
            return -1.0
        return None

    def is_repetition_draw(self) -> bool:
        return self._repetition.get(self._repetition_key(self.position), 0) >= REPETITION_LIMIT

    def _count_repetition(self, position: Position) -> None:
        key = self._repetition_key(position)
        self._repetition[key] = self._repetition.get(key, 0) + 1

    @staticmethod
    def _repetition_key(position: Position) -> str:
        # The move number must not take part in the comparison.
        return " ".join(position.to_sfen().split()[:3])


def import_cshogi():
    """Import cshogi, returning ``(module, error_text)``.

    A failing ``import cshogi`` is not always a missing package: on a fresh
    Colab runtime the usual cause is a NumPy ABI mismatch, which also raises
    ImportError.  Reporting "not installed" for that sends people off to
    reinstall a package they already have, so the real message is kept.
    """
    try:
        import cshogi

        return cshogi, None
    except ImportError as exc:
        missing = isinstance(exc, ModuleNotFoundError) and exc.name == "cshogi"
        if missing:
            return None, "cshogi is not installed"
        return None, f"cshogi is installed but will not import: {exc}"


def _resolve_move_to_usi(cshogi, board):
    """Find cshogi's move-to-USI conversion.

    It lives on the module in current releases and has been a ``Board`` method
    in others, so both are accepted rather than pinning one spelling.
    """
    if hasattr(cshogi, "move_to_usi"):
        return cshogi.move_to_usi
    if hasattr(board, "move_to_usi"):
        return board.move_to_usi
    raise ImportError(
        "this cshogi build exposes neither cshogi.move_to_usi(move) nor "
        "Board.move_to_usi(move); shogi_ai cannot convert its moves"
    )


class CshogiBackend:
    """Backend over ``cshogi.Board``, used for fast self-play."""

    name = "cshogi"

    def __init__(self, sfen: Optional[str] = None):
        cshogi, error = import_cshogi()
        if cshogi is None:
            raise ImportError(error)

        self._cshogi = cshogi
        self.board = cshogi.Board(sfen) if sfen else cshogi.Board()
        self._move_to_usi = _resolve_move_to_usi(cshogi, self.board)
        try:
            self._black = cshogi.BLACK
        except AttributeError as exc:
            raise ImportError("cshogi does not expose BLACK") from exc

    @property
    def turn(self) -> int:
        return BLACK if self.board.turn == self._black else -BLACK

    def snapshot(self) -> RawSnapshot:
        return snapshot_from_sfen(self.board.sfen(), self.board.is_check())

    def legal_moves(self) -> Tuple[List[int], List[RawMove]]:
        native = list(self.board.legal_moves)
        to_usi = self._move_to_usi
        raw = [raw_move_from_usi(to_usi(move)) for move in native]
        return native, raw

    def sfen(self) -> str:
        return self.board.sfen()

    def move_usi(self, move: int) -> str:
        return self._move_to_usi(move)

    def parse_usi(self, usi: str) -> int:
        return self.board.move_from_usi(usi)

    def push(self, move: int) -> None:
        self.board.push(move)

    def pop(self) -> None:
        self.board.pop()

    def terminal_value(self) -> Optional[float]:
        board = self.board
        # is_draw() covers repetition; a repetition with perpetual check is a
        # loss for the checking side, which cshogi reports separately.
        draw = board.is_draw()
        if draw == self._cshogi.REPETITION_DRAW:
            return 0.0
        if draw == self._cshogi.REPETITION_WIN:
            return 1.0
        if draw == self._cshogi.REPETITION_LOSE:
            return -1.0
        if board.is_game_over():
            return -1.0
        return None


#: Set once so an automatic fallback is reported a single time, not per game.
_WARNED_ABOUT_FALLBACK = False


def make_backend(kind: str = "auto", sfen: Optional[str] = None):
    """Build a backend; ``auto`` prefers cshogi and falls back to Python."""
    global _WARNED_ABOUT_FALLBACK

    if kind in ("auto", "cshogi"):
        cshogi, error = import_cshogi()
        if cshogi is not None:
            return CshogiBackend(sfen)
        if kind == "cshogi":
            raise SystemExit(
                f"{error}\n"
                "Fix it with `pip install cshogi` (and restart the runtime if a\n"
                "NumPy version changed), or pass --backend python to use the\n"
                "dependency-free engine -- which is roughly 1000x slower at move\n"
                "generation, so self-play will crawl."
            )
        if not _WARNED_ABOUT_FALLBACK:
            # Falling back silently would hand the caller a 1000x slowdown
            # while it still looks like the fast path was taken.
            _WARNED_ABOUT_FALLBACK = True
            print(f"WARNING: {error}\n"
                  "         falling back to the pure-Python backend, which is far "
                  "slower.\n"
                  "         Pass --backend cshogi to make this an error instead.",
                  flush=True)
    return PurePythonBackend(position_from_sfen(sfen) if sfen else Position.start())


#: The opening position, used to pin cshogi's conventions against ours.
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
START_LEGAL_MOVES = 30


def verify_cshogi_api() -> List[str]:
    """Exercise every cshogi call this module relies on.

    Returns the list of problems found, empty when the backend is sound.  Each
    step is checked separately so a mismatched cshogi release reports all of
    its differences at once instead of one per run, and the conventions that
    would silently corrupt training data -- colour mapping above all -- are
    compared against the pure-Python engine rather than assumed.
    """
    problems: List[str] = []

    cshogi, error = import_cshogi()
    if cshogi is None:
        return [error]

    try:
        backend = CshogiBackend()
    except ImportError as exc:
        return [str(exc)]
    except Exception as exc:  # pragma: no cover - depends on the cshogi build
        return [f"cshogi.Board() failed: {type(exc).__name__}: {exc}"]

    reference = PurePythonBackend()

    def check(label, action):
        try:
            return action(), None
        except Exception as exc:
            problems.append(f"{label}: {type(exc).__name__}: {exc}")
            return None, exc

    sfen, failed = check("Board.sfen()", backend.sfen)
    if not failed and sfen != START_SFEN:
        problems.append(f"Board.sfen() gave {sfen!r}, expected the opening position")

    turn, failed = check("Board.turn / cshogi.BLACK", lambda: backend.turn)
    if not failed and turn != BLACK:
        problems.append(
            f"colour mapping is wrong: the opening position reported turn={turn}, "
            "expected Black. Training data built from this would be sign-flipped."
        )

    check("Board.is_check()", backend.board.is_check)
    check("Board.is_game_over()", backend.board.is_game_over)
    check("Board.is_draw() / REPETITION_*", backend.terminal_value)

    moves, failed = check("Board.legal_moves + move_to_usi", backend.legal_moves)
    if not failed:
        native, _raw = moves
        if len(native) != START_LEGAL_MOVES:
            problems.append(
                f"the opening position has {len(native)} legal moves, "
                f"expected {START_LEGAL_MOVES}"
            )
        reference_usi = sorted(m.usi() for m in reference.legal_moves()[0])
        fast_usi = sorted(backend.move_usi(m) for m in native)
        if fast_usi != reference_usi:
            problems.append("legal move sets disagree with the pure-Python engine")

        if native:
            _value, failed = check(
                "Board.move_from_usi()",
                lambda: backend.parse_usi(backend.move_usi(native[0])),
            )

    snapshot, failed = check("snapshot()", backend.snapshot)
    if not failed:
        expected = reference.snapshot()
        if list(snapshot.squares) != list(expected.squares):
            problems.append("board squares disagree with the pure-Python engine")
        if [list(h) for h in snapshot.hands] != [list(h) for h in expected.hands]:
            problems.append("hand counts disagree with the pure-Python engine")

    def push_pop():
        before = backend.sfen()
        move = backend.parse_usi("7g7f")
        backend.push(move)
        after = backend.sfen()
        backend.pop()
        if backend.sfen() != before:
            raise AssertionError("pop() did not restore the position")
        if after == before:
            raise AssertionError("push() did not change the position")
        return True

    check("Board.push() / Board.pop()", push_pop)
    return problems


def main() -> None:
    problems = verify_cshogi_api()
    if not problems:
        cshogi, _error = import_cshogi()
        version = getattr(cshogi, "__version__", "(version unknown)")
        print(f"cshogi backend OK (cshogi {version})")
        return
    print(f"cshogi backend has {len(problems)} problem(s):")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
