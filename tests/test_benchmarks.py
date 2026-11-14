"""No-leakage controls, determinism, and output-contract regression tests."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.benchmarks import (
    BenchmarkConfig, _make_sequences, main, run_benchmark,
)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = BenchmarkConfig(
            seeds=(17, 21), delays=(1, 4), train_sequences=8,
            test_sequences=4, sequence_length=20, burn_in=5,
            neurons=10, epochs=10,
        )

    def test_repeatable_paired_seed_delay_runs(self):
        first = run_benchmark(self.config)
        self.assertEqual(first, run_benchmark(self.config))
        self.assertEqual(first["schema_version"], "3.0")
        self.assertEqual(len(first["runs"]), 4)
        self.assertEqual(len(first["summary"]), 2)
        for row in first["runs"]:
            self.assertEqual(row["samples"]["train"], 8 * 15)
            self.assertEqual(row["samples"]["test"], 4 * 15)
            self.assertEqual(set(row["metrics"]), set(first["models"]))
            for accuracy in row["metrics"].values():
                self.assertTrue(0 <= accuracy <= 1)

    def test_direct_delayed_tap_actually_has_memory(self):
        report = run_benchmark(self.config)
        for row in report["runs"]:
            self.assertGreaterEqual(row["metrics"]["delayed_input_tap"], 0.9)

    def test_distinct_data_splits(self):
        train = _make_sequences(5, 20, 121, 0.2)
        test = _make_sequences(5, 20, 222, 0.2)
        self.assertNotEqual(train, test)
        self.assertEqual(train, _make_sequences(5, 20, 121, 0.2))

    def test_flip_one_half_degrades_perfect_delay_tap(self):
        noisy = run_benchmark(BenchmarkConfig(
            seeds=(17,), delays=(4,), train_sequences=16,
            test_sequences=12, sequence_length=20, burn_in=5,
            neurons=10, epochs=10, flip_probability=0.5,
        ))
        self.assertLess(noisy["runs"][0]["metrics"]["delayed_input_tap"], 0.8)

    def test_config_rejects_invalid_experiments(self):
        failures = [
            {"seeds": ()}, {"seeds": (4, 4)}, {"delays": ()},
            {"delays": (1, 1)}, {"delays": (0,)},
            {"delays": (30,), "sequence_length": 20},
            {"train_sequences": 0}, {"burn_in": 20, "sequence_length": 20},
            {"flip_probability": -0.1}, {"flip_probability": 0.7},
            {"neurons": 0}, {"epochs": 0},
        ]
        for override in failures:
            with self.subTest(override=override), self.assertRaises(ValueError):
                BenchmarkConfig(**override)

    def test_cli_json_and_csv_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "report.json"
            csv_path = Path(folder) / "report.csv"
            args = [
                "--seed-start", "4", "--seed-count", "2",
                "--delays", "1", "3", "--neurons", "8",
                "--train-sequences", "5", "--test-sequences", "3",
                "--sequence-length", "12", "--burn-in", "4", "--epochs", "4",
                "--json-output", str(json_path), "--csv-output", str(csv_path),
            ]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(report["runs"]))
            self.assertEqual(len(report["summary"]), 2)
            self.assertEqual(rows[0]["seed"], "4")


if __name__ == "__main__":
    unittest.main()
