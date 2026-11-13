"""Round-artefact validation for the learning loop.

These cover the resume path: a round is only skipped when its artefacts are
really usable.  An interrupted self-play run used to leave an empty JSONL
behind, which the loop then reused as if the round had finished, and training
died with "no training records found".
"""

import json
import tempfile
import unittest
from pathlib import Path

from shogi_ai.learn_loop import _parse_score, count_records, usable_data


class CountRecordsTests(unittest.TestCase):
    def test_missing_file_is_minus_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(count_records(Path(tmp) / "nope.jsonl"), -1)

    def test_empty_file_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.touch()
            self.assertEqual(count_records(path), 0)

    def test_blank_lines_do_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text("\n  \n\n", encoding="utf-8")
            self.assertEqual(count_records(path), 0)

    def test_records_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(7)),
                            encoding="utf-8")
            self.assertEqual(count_records(path), 7)


class UsableDataTests(unittest.TestCase):
    def test_populated_file_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text(json.dumps({"sfen": "x"}) + "\n", encoding="utf-8")
            self.assertTrue(usable_data(path))
            self.assertTrue(path.exists(), "a good file must be left alone")

    def test_missing_file_is_not_usable_and_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            self.assertFalse(usable_data(path))
            self.assertFalse(path.exists())

    def test_empty_file_is_removed_so_the_round_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.touch()
            self.assertFalse(usable_data(path))
            self.assertFalse(path.exists(), "the stale empty file must be deleted")


class ParseScoreTests(unittest.TestCase):
    def test_reads_the_match_score(self):
        line = ("a.pt vs b.pt: 11W-7L-2D in 20 games, score 0.600 "
                "(+70 Elo), 180 plies/game, 30.0s")
        self.assertEqual(_parse_score(line), 0.6)

    def test_returns_none_without_a_score(self):
        self.assertIsNone(_parse_score("nothing to see here"))

    def test_ignores_integers_and_out_of_range_values(self):
        self.assertIsNone(_parse_score("score 20 games"))


class _Args:
    lr = 2e-3
    epochs = 4
    round_lr = 0.0
    round_epochs = 0


class TrainingRegimeTests(unittest.TestCase):
    """The finetuning regime exists because the first Colab run showed
    the from-scratch rate wrecking a trained champion on tiny data."""

    def test_from_scratch_uses_base_values(self):
        from shogi_ai.learn_loop import training_regime

        args = _Args()
        args.round_lr, args.round_epochs = 5e-4, 2
        self.assertEqual(training_regime(args, has_champion=False), (2e-3, 4))

    def test_champion_uses_round_values_when_set(self):
        from shogi_ai.learn_loop import training_regime

        args = _Args()
        args.round_lr, args.round_epochs = 5e-4, 2
        self.assertEqual(training_regime(args, has_champion=True), (5e-4, 2))

    def test_champion_falls_back_when_round_values_unset(self):
        from shogi_ai.learn_loop import training_regime

        self.assertEqual(training_regime(_Args(), has_champion=True), (2e-3, 4))


class BuildBufferTests(unittest.TestCase):
    def _seed(self, tmp, names):
        data_dir = Path(tmp)
        for name in names:
            (data_dir / name).write_text(json.dumps({"sfen": "x"}) + "\n",
                                         encoding="utf-8")
        return data_dir

    def test_collects_initial_then_selfplay_in_order(self):
        from shogi_ai.learn_loop import build_buffer

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._seed(tmp, ["selfplay_002.jsonl", "selfplay_001.jsonl"])
            initial = data_dir / "initial.jsonl"
            initial.write_text(json.dumps({"sfen": "y"}) + "\n", encoding="utf-8")
            buffer = build_buffer(initial, data_dir, data_dir / "selfplay_002.jsonl", 1)
            self.assertEqual([Path(p).name for p in buffer],
                             ["initial.jsonl", "selfplay_001.jsonl", "selfplay_002.jsonl"])

    def test_latest_repeat_oversamples_the_newest_round(self):
        from shogi_ai.learn_loop import build_buffer

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._seed(tmp, ["selfplay_001.jsonl", "selfplay_002.jsonl"])
            buffer = build_buffer(None, data_dir, data_dir / "selfplay_002.jsonl", 3)
            self.assertEqual([Path(p).name for p in buffer],
                             ["selfplay_001.jsonl"] + ["selfplay_002.jsonl"] * 3)

    def test_repeat_is_skipped_when_latest_is_the_only_data(self):
        from shogi_ai.learn_loop import build_buffer

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._seed(tmp, ["selfplay_001.jsonl"])
            buffer = build_buffer(None, data_dir, data_dir / "selfplay_001.jsonl", 3)
            self.assertEqual(len(buffer), 1)

    def test_repeat_is_skipped_for_the_initial_file(self):
        from shogi_ai.learn_loop import build_buffer

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._seed(tmp, ["selfplay_001.jsonl"])
            initial = data_dir / "initial.jsonl"
            initial.write_text(json.dumps({"sfen": "y"}) + "\n", encoding="utf-8")
            buffer = build_buffer(initial, data_dir, initial, 3)
            self.assertEqual([Path(p).name for p in buffer],
                             ["initial.jsonl", "selfplay_001.jsonl"])

    def test_empty_selfplay_files_are_left_out(self):
        from shogi_ai.learn_loop import build_buffer

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._seed(tmp, ["selfplay_001.jsonl"])
            (data_dir / "selfplay_002.jsonl").touch()
            buffer = build_buffer(None, data_dir, data_dir / "selfplay_001.jsonl", 1)
            self.assertEqual([Path(p).name for p in buffer], ["selfplay_001.jsonl"])


if __name__ == "__main__":
    unittest.main()
