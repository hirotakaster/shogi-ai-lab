"""Cross-game batched inference (Batch 4).

A BatchingEvaluator must be transparent -- every caller gets exactly the
rows its own planes produce -- while actually coalescing concurrent
calls into fewer, larger underlying batches.  The recording evaluator
below makes each row identifiable by the constant the caller filled its
planes with, so a mix-up between callers cannot hide.
"""

import threading
import time
import unittest

import numpy as np

from shogi_ai.encoding import POLICY_SIZE
from shogi_ai.evaluator import BatchingEvaluator, UniformEvaluator


class RecordingEvaluator:
    """Values and logits identify the input plane's constant fill."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.batch_sizes = []
        self.lock = threading.Lock()

    def __call__(self, planes):
        with self.lock:
            self.batch_sizes.append(planes.shape[0])
        if self.delay:
            time.sleep(self.delay)
        fills = planes[:, 0, 0, 0]
        logits = np.repeat(fills[:, None], POLICY_SIZE, axis=1).astype(np.float32)
        return logits, fills.astype(np.float32)


def constant_planes(fill, count=1):
    return np.full((count, 85, 9, 9), fill, dtype=np.float32)


class BatchingEvaluatorTests(unittest.TestCase):
    def test_single_call_is_transparent(self):
        underlying = RecordingEvaluator(delay=0.0)
        with BatchingEvaluator(underlying, max_batch=8, max_wait=0.01) as ev:
            logits, values = ev(constant_planes(3.0, count=2))
        self.assertEqual(logits.shape, (2, POLICY_SIZE))
        np.testing.assert_allclose(values, [3.0, 3.0])
        np.testing.assert_allclose(logits[:, 0], [3.0, 3.0])

    def test_concurrent_calls_coalesce_and_stay_separate(self):
        underlying = RecordingEvaluator(delay=0.05)
        ev = BatchingEvaluator(underlying, max_batch=64, max_wait=0.2)
        try:
            results = {}

            def call(name, fill):
                results[name] = ev(constant_planes(fill, count=3))

            threads = [threading.Thread(target=call, args=(f"g{i}", float(i + 1)))
                       for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            ev.close()
        # Each caller got its own rows back.
        for i in range(4):
            _logits, values = results[f"g{i}"]
            np.testing.assert_allclose(values, [float(i + 1)] * 3)
        # And the twelve planes were not evaluated one call at a time:
        # some underlying batch must hold more than one game's planes.
        self.assertEqual(sum(underlying.batch_sizes), 12)
        self.assertGreater(max(underlying.batch_sizes), 3)

    def test_errors_reach_the_caller(self):
        class Broken:
            def __call__(self, planes):
                raise RuntimeError("boom")

        ev = BatchingEvaluator(Broken(), max_wait=0.01)
        try:
            with self.assertRaises(RuntimeError):
                ev(constant_planes(1.0))
        finally:
            ev.close()

    def test_call_after_close_is_rejected(self):
        ev = BatchingEvaluator(UniformEvaluator(), max_wait=0.01)
        ev.close()
        ev.close()  # idempotent
        with self.assertRaises(RuntimeError):
            ev(constant_planes(1.0))

    def test_empty_batch_short_circuits(self):
        with BatchingEvaluator(UniformEvaluator(), max_wait=0.01) as ev:
            logits, values = ev(np.zeros((0, 85, 9, 9), dtype=np.float32))
        self.assertEqual(logits.shape, (0, POLICY_SIZE))
        self.assertEqual(values.shape, (0,))


class ConcurrentSelfPlayTests(unittest.TestCase):
    def test_concurrent_games_produce_valid_summaries(self):
        from shogi_ai.selfplay_az import SelfPlayConfig, _concurrent_summaries

        config = SelfPlayConfig(simulations=16, batch_size=4, max_moves=20,
                                backend="python", resign_threshold=-1.0,
                                parallel_games=2)
        summaries = list(_concurrent_summaries(
            UniformEvaluator(), config, [11, 22]))
        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            self.assertTrue(summary.records)
            for record in summary.records:
                self.assertAlmostEqual(
                    sum(p for _i, p in record["policy"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
