"""Phase 6 tests for genuine RNN training and careful experimental claims."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from phineubio.competition import CompetitionConfig, TrainableRNN, MODELS, main, run_competition
from phineubio.temporal import make_sequences


class TrainableRNNTests(unittest.TestCase):
    def test_state_shapes_and_parameter_count(self):
        model = TrainableRNN(4, seed=17)
        states, probabilities = model.forward([0, 1, 0, 1])
        self.assertEqual(len(states), 5)
        self.assertEqual(len(states[0]), 4)
        self.assertEqual(len(probabilities), 4)
        self.assertEqual(model.parameter_count, 4 * 4 + 3 * 4 + 1)

    def test_analytical_recurrent_gradient_matches_finite_difference(self):
        model = TrainableRNN(3, seed=21)
        observed = [1, 0, 1, 1, 0, 1]
        clean = [1, 0, 1, 1, 0, 1]
        _, gradients = model.loss_and_gradients(observed, clean, delay=2, start=3)
        before = model.wh[0][1]
        epsilon = 1e-5
        model.wh[0][1] = before + epsilon
        high = model.loss_and_gradients(observed, clean, delay=2, start=3)[0]
        model.wh[0][1] = before - epsilon
        low = model.loss_and_gradients(observed, clean, delay=2, start=3)[0]
        model.wh[0][1] = before
        numerical = (high - low) / (2 * epsilon)
        self.assertAlmostEqual(gradients["wh"][0][1], numerical, places=5)

    def test_recurrent_weights_actually_train(self):
        model = TrainableRNN(4, seed=5)
        sequence = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        original = [row[:] for row in model.wh]
        model.train_sequence(sequence, sequence, delay=1, start=3, learning_rate=0.1, clip=5.0)
        self.assertNotEqual(original, model.wh)

    def test_train_is_reproducible(self):
        sequences = make_sequences(8, 16, 108, 0.9, 0.1)
        a, b = TrainableRNN(5, 11), TrainableRNN(5, 11)
        curve_a = a.train(sequences, 1, 4, epochs=5, learning_rate=0.05, clip=5.0, seed=31)
        curve_b = b.train(sequences, 1, 4, epochs=5, learning_rate=0.05, clip=5.0, seed=31)
        self.assertEqual(curve_a, curve_b)
        self.assertEqual(a.wh, b.wh)
        self.assertTrue(all(math.isfinite(v) for v in curve_a))

    def test_invalid_windows_and_input(self):
        model = TrainableRNN(3, 1)
        with self.assertRaises(ValueError):
            model.forward([0, 2, 1])
        with self.assertRaises(ValueError):
            model.loss_and_gradients([0, 1], [0], delay=1, start=1)
        with self.assertRaises(ValueError):
            model.loss_and_gradients([0, 1], [0, 1], delay=2, start=1)


class CompetitionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CompetitionConfig(
            seeds=(17,), delays=(1, 2), noise_levels=(0.0,),
            stay_probability=0.85, train_sequences=4, test_sequences=3,
            sequence_length=14, burn_in=3, neurons=4, epochs=3
        )

    def test_paired_results_and_repeatable_accuracy(self):
        a = run_competition(self.cfg)
        b = run_competition(self.cfg)
        self.assertEqual(a["schema_version"], "6.0")
        self.assertEqual(len(a["runs"]), 2)
        self.assertEqual(len(a["summary"]), 2)
        for left, right in zip(a["runs"], b["runs"]):
            self.assertEqual(left["metrics"], right["metrics"])
            self.assertEqual(left["trained_rnn"], right["trained_rnn"])
            self.assertEqual(left["samples"], right["samples"])
            self.assertEqual(left["samples"]["train"], 4 * 11)
            self.assertEqual(left["samples"]["test"], 3 * 11)
            self.assertEqual(set(left["metrics"]), set(MODELS))
            self.assertTrue(all(0 <= x <= 1 for x in left["metrics"].values()))
            self.assertEqual(left["trained_rnn"]["trainable_parameters"], 4 * 4 + 3 * 4 + 1)
            for profile in left["profiling"].values():
                self.assertGreater(profile["total_ns"], 0)
                self.assertGreaterEqual(profile["peak_traced_python_bytes"], 0)

    def test_rejects_bad_config(self):
        for kwargs in [{"neurons": 0}, {"epochs": 0}, {"delays": (14,)},
                       {"noise_levels": (0.5, 0.5)}, {"seeds": (1, 1)},
                       {"learning_rate": 0}, {"gradient_clip": -2}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CompetitionConfig(**{**self.cfg.__dict__, **kwargs})

    def test_cli_writes_versioned_json_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            jp = Path(folder) / "competition.json"
            cp = Path(folder) / "competition.csv"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "--seed-count", "1", "--delays", "1", "--noise-levels", "0",
                    "--train-sequences", "3", "--test-sequences", "2",
                    "--sequence-length", "10", "--burn-in", "3", "--neurons", "3",
                    "--epochs", "2", "--json-output", str(jp), "--csv-output", str(cp)
                ]), 0)
            report = json.loads(jp.read_text(encoding="utf-8"))
            with cp.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(report["runs"]), 1)
            self.assertEqual(len(rows), 1)
            self.assertIn("trained_rnn_accuracy", rows[0])


if __name__ == "__main__":
    unittest.main()
