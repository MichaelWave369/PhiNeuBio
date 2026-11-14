"""Phase 5 regression suite: deterministic science and nondeterministic profiling."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.temporal import (
    MODELS, TemporalConfig, TanhReservoir, make_sequences,
    extract_features, run_temporal, main,
)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = TemporalConfig(
            seeds=(17, 18), delays=(1, 3), noise_levels=(0.0, 0.2),
            stay_probability=0.8, train_sequences=5, test_sequences=3,
            sequence_length=14, burn_in=4, neurons=6, epochs=5,
        )

    def test_signal_generation_is_reproducible_and_noise_masks_nested(self):
        clean = make_sequences(5, 30, 94, 0.9, 0.0)
        noisy = make_sequences(5, 30, 94, 0.9, 0.2)
        more_noisy = make_sequences(5, 30, 94, 0.9, 0.4)
        self.assertEqual([a for a, _ in clean], [a for a, _ in noisy])
        self.assertEqual(clean, make_sequences(5, 30, 94, 0.9, 0.0))
        for (truth, observed), (_, middle), (_, high) in zip(clean, noisy, more_noisy):
            self.assertEqual(truth, observed)
            for a, b, c in zip(truth, middle, high):
                self.assertFalse(b != a and c == a)

    def test_tanh_is_reproducible_and_resets(self):
        model = TanhReservoir(neurons=6, seed=99)
        inputs = [1, 1, 0, 1, 0]
        a = [model.step(bit) for bit in inputs]
        model.reset()
        self.assertEqual(a, [model.step(bit) for bit in inputs])
        fresh = TanhReservoir(6, 99)\n        self.assertEqual(a, [fresh.step(bit) for bit in inputs])
        self.assertEqual(len(a[0]), 6)

    def test_all_models_use_same_targets_and_sample_window(self):
        sequences = make_sequences(3, 14, 101, 0.8, 0.2)
        targets = []
        for name in MODELS:
            features, labels = extract_features(name, sequences, delay=3, start=5, history_length=4, neurons=6, seed=17)
            targets.append(labels)
            self.assertEqual(len(features), 3 * 9)
        self.assertTrue(all(y == targets[0] for y in targets[1:]))

    def test_metrics_deterministic_but_time_is_not_an_accuracy_metric(self):
        a = run_temporal(self.cfg)
        b = run_temporal(self.cfg)
        self.assertEqual(a["schema_version"], "5.0")
        self.assertEqual(len(a["runs"]), 8)
        self.assertEqual(len(a["summary"]), 4)
        self.assertEqual(
            [(x["seed"], x["noise"], x["delay"], x["metrics"]) for x in a["runs"]],
            [(x["seed"], x["noise"], x["delay"], x["metrics"]) for x in b["runs"]]
        )
        for row in a["runs"]:
            self.assertEqual(row["samples"]["train"], 5 * 10)
            self.assertEqual(row["samples"]["test"], 3 * 10)
            self.assertEqual(set(row["metrics"]), set(MODELS))
            for accuracy in row["metrics"].values():
                self.assertTrue(0 <= accuracy <= 1)
            for profile in row["profiling"].values():
                self.assertGreater(profile["total_ns"], 0)
                self.assertGreaterEqual(profile["peak_traced_python_bytes"], 0)
                self.assertEqual(
                    profile["total_ns"],
                    profile["extraction_ns"] + profile["fit_ns"] + profile["evaluation_ns"]
                )

    def test_invalid_configs(self):
        invalid = [
            {"seeds": ()}, {"seeds": (1, 1)}, {"delays": ()},
            {"delays": (0,)}, {"noise_levels": ()},
            {"noise_levels": (-0.1,)}, {"noise_levels": (0.6,)},
            {"stay_probability": 0.49}, {"stay_probability": 1.1},
            {"burn_in": 14}, {"window_size": 2}, {"neurons": 0},
            {"epochs": 0}, {"train_sequences": 0}, {"delays": (20,)},
        ]
        for replacement in invalid:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                TemporalConfig(**{**self.cfg.__dict__, **replacement})

    def test_cli_json_csv_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "temporal.json"
            csv_path = Path(folder) / "temporal.csv"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "--seed-count", "1", "--delays", "1", "2",
                    "--noise-levels", "0", "0.1",
                    "--train-sequences", "3", "--test-sequences", "2",
                    "--sequence-length", "12", "--burn-in", "3",
                    "--neurons", "5", "--epochs", "3",
                    "--json-output", str(json_path), "--csv-output", str(csv_path),
                ]), 0)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(report["runs"]), 4)
            self.assertEqual(rows[0]["seed"], "369")


if __name__ == "__main__":
    unittest.main()
