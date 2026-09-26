"""Pin the shape of the cshogi API that ``CshogiBackend`` calls.

cshogi cannot be installed on every development machine, and a name that has
moved -- ``move_to_usi`` is a module function, not a ``Board`` method -- only
shows up at run time on the machine that does have it.  These tests stand a
fake module in cshogi's place whose behaviour is delegated to the pure-Python
engine, so the adapter's *call structure* is checked everywhere: which names it
looks up, on the module or on the board, and whether it honours the colour
constant rather than assuming one.

They cannot validate real cshogi behaviour. ``tests/test_cshogi_backend.py``
does that, and only runs where cshogi is installed.
"""

import sys
import types
import unittest
from contextlib import contextmanager

from shogi_ai.core import BLACK, Position, position_from_sfen
from shogi_ai.backends import PurePythonBackend


def _build_fake(move_to_usi_on_board: bool = False, black_value: int = 0):
    """A stand-in cshogi whose moves are USI strings."""
    module = types.ModuleType("cshogi")
    module.__version__ = "fake"
    module.BLACK = black_value
    module.WHITE = 1 - black_value
    module.REPETITION_NONE = 0
    module.REPETITION_DRAW = 1
    module.REPETITION_WIN = 2
    module.REPETITION_LOSE = 3

    class FakeBoard:
        def __init__(self, sfen=None):
            start = position_from_sfen(sfen) if sfen else Position.start()
            self._inner = PurePythonBackend(start)

        @property
        def turn(self):
            return black_value if self._inner.turn == BLACK else 1 - black_value

        @property
        def legal_moves(self):
            return [m.usi() for m in self._inner.position.legal_moves()]

        def sfen(self):
            return self._inner.sfen()

        def is_check(self):
            return self._inner.position.is_in_check(self._inner.turn)

        def is_game_over(self):
            return not self._inner.position.legal_moves()

        def is_draw(self):
            return module.REPETITION_DRAW if self._inner.is_repetition_draw() else module.REPETITION_NONE

        def push(self, usi):
            self._inner.push(self._inner.parse_usi(usi))

        def pop(self):
            self._inner.pop()

        def move_from_usi(self, usi):
            return usi

    if move_to_usi_on_board:
        FakeBoard.move_to_usi = staticmethod(lambda move: move)
    else:
        module.move_to_usi = lambda move: move

    module.Board = FakeBoard
    return module


@contextmanager
def fake_cshogi(**kwargs):
    saved = sys.modules.get("cshogi")
    sys.modules["cshogi"] = _build_fake(**kwargs)
    try:
        yield
    finally:
        if saved is None:
            del sys.modules["cshogi"]
        else:
            sys.modules["cshogi"] = saved


class ApiShapeTests(unittest.TestCase):
    def test_module_level_move_to_usi_is_used(self):
        """The real cshogi exposes move_to_usi on the module, not the board."""
        from shogi_ai.backends import verify_cshogi_api

        with fake_cshogi(move_to_usi_on_board=False):
            self.assertEqual(verify_cshogi_api(), [])

    def test_board_method_move_to_usi_is_also_accepted(self):
        from shogi_ai.backends import verify_cshogi_api

        with fake_cshogi(move_to_usi_on_board=True):
            self.assertEqual(verify_cshogi_api(), [])

    def test_missing_move_to_usi_is_reported_clearly(self):
        from shogi_ai.backends import verify_cshogi_api

        fake = _build_fake()
        del fake.move_to_usi
        saved = sys.modules.get("cshogi")
        sys.modules["cshogi"] = fake
        try:
            problems = verify_cshogi_api()
        finally:
            if saved is None:
                del sys.modules["cshogi"]
            else:
                sys.modules["cshogi"] = saved
        self.assertEqual(len(problems), 1)
        self.assertIn("move_to_usi", problems[0])

    def test_missing_black_constant_is_reported(self):
        from shogi_ai.backends import verify_cshogi_api

        fake = _build_fake()
        del fake.BLACK
        saved = sys.modules.get("cshogi")
        sys.modules["cshogi"] = fake
        try:
            problems = verify_cshogi_api()
        finally:
            if saved is None:
                del sys.modules["cshogi"]
            else:
                sys.modules["cshogi"] = saved
        self.assertEqual(problems, ["cshogi does not expose BLACK"])

    def test_a_flipped_colour_constant_is_caught(self):
        """The check that matters most: a sign-flipped colour ruins the data."""
        from shogi_ai.backends import verify_cshogi_api

        # BLACK == 1 while Board.turn keeps reporting 0 for the opening.
        fake = _build_fake(black_value=0)
        fake.BLACK = 1
        saved = sys.modules.get("cshogi")
        sys.modules["cshogi"] = fake
        try:
            problems = verify_cshogi_api()
        finally:
            if saved is None:
                del sys.modules["cshogi"]
            else:
                sys.modules["cshogi"] = saved
        self.assertTrue(any("colour mapping is wrong" in p for p in problems), problems)


class BackendBehaviourTests(unittest.TestCase):
    def test_backend_agrees_with_the_reference_on_the_opening(self):
        from shogi_ai.backends import CshogiBackend

        with fake_cshogi():
            fast = CshogiBackend()
            reference = PurePythonBackend()
            self.assertEqual(fast.sfen(), reference.sfen())
            self.assertEqual(fast.turn, reference.turn)
            self.assertEqual(
                sorted(fast.move_usi(m) for m in fast.legal_moves()[0]),
                sorted(m.usi() for m in reference.legal_moves()[0]),
            )
            self.assertEqual(list(fast.snapshot().squares),
                             list(reference.snapshot().squares))

    def test_policy_indices_agree_with_the_reference(self):
        from shogi_ai.backends import CshogiBackend
        from shogi_ai.encoding import policy_index

        with fake_cshogi():
            fast = CshogiBackend()
            reference = PurePythonBackend()
            fast_native, fast_raw = fast.legal_moves()
            ref_native, ref_raw = reference.legal_moves()
            self.assertEqual(
                {fast.move_usi(m): policy_index(r, fast.turn)
                 for m, r in zip(fast_native, fast_raw)},
                {reference.move_usi(m): policy_index(r, reference.turn)
                 for m, r in zip(ref_native, ref_raw)},
            )

    def test_push_and_pop_round_trip(self):
        from shogi_ai.backends import CshogiBackend

        with fake_cshogi():
            fast = CshogiBackend()
            before = fast.sfen()
            fast.push(fast.parse_usi("7g7f"))
            self.assertNotEqual(fast.sfen(), before)
            fast.pop()
            self.assertEqual(fast.sfen(), before)

    def test_make_backend_prefers_cshogi_when_it_imports(self):
        from shogi_ai.backends import make_backend

        with fake_cshogi():
            self.assertEqual(make_backend("auto").name, "cshogi")
            self.assertEqual(make_backend("cshogi").name, "cshogi")
            self.assertEqual(make_backend("python").name, "python")


if __name__ == "__main__":
    unittest.main()
