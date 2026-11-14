import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.benchmarks import _make_sequences
from phineubio.robustness import RobustnessConfig, run_robustness, main


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.config = RobustnessConfig(
            seeds=(11, 14), delays=(1, 3), noise_levels=(0.0, 0.25),
            train_sequences=6, test_sequences=3, sequence_length=16,
            burn_in=4, neurons=8, epochs=5,
        )

    def test_report_deterministic_and_paired(self):
        a = run_robustness(self.config)
        self.assertEqual(a, run_robustness(self.config))
        self.assertEqual(a["schema_version"], "4.0")
        self.assertEqual(len(a["runs"]), 8)
        self.assertEqual(len(a["summary"]), 4)
        self.assertEqual(a["configuration"]["window_size"], 4)
        for row in a["runs"]:
            self.assertEqual(row["samples"]["train"], 6 * 12)
            self.assertEqual(row["samples"]["test"], 3 * 12)
            self.assertEqual(len(row["metrics"]), 6)
            self.assertTrue(all(0 <= x <= 1 for x in row["metrics"].values()))
            self.assertEqual(row["work_proxies"]["history_window"]["feature_dimensions"], 4)
            self.assertEqual(row["work_proxies"]["spiking_recurrent"]["dense_recurrent_pair_scans"], 9 * 16 * 8 * 8)

    def test_noise_mask_is_nested_and_clean_sequence_fixed(self):
        no_noise = _make_sequences(6, 32, 201, 0.0)
        medium = _make_sequences(6, 32, 201, 0.1)
        high = _make_sequences(6, 32, 201, 0.25)
        self.assertEqual([a for a, _ in no_noise], [a for a, _ in high])
        for (clean, noiseless), (_, mid), (_, noisy) in zip(no_noise, medium, high):
            self.assertEqual(clean, noiseless)
            self.assertTrue(all(not(m != c) or (n != c) for c, m, n in zip(clean, mid, noisy)))

    def test_noiseless_window_can_decode_delayed_bit(self):
        result = run_robustness(RobustnessConfig(
            seeds=(23,), delays=(1, 3), noise_levels=(0.0,),
            train_sequences=16, test_sequences=8, sequence_length=24,
            burn_in=4, neurons=8, epochs=12,
        ))
        self.assertTrue(all(row["metrics"]["delayed_input_tap"] > 0.95 for row in result["runs"]))
        self.assertTrue(all(row["metrics"]["history_window"] > 0.8 for row in result["runs"]))

    def test_unequal_history_window_is_rejected(self):
        for override in [
            {"window_size": 2}, {"noise_levels": ()}, {"noise_levels": (0.1, 0.1)},
            {"noise_levels": (-0.1,)}, {"noise_levels": (0.7,)},
            {"seeds": (3, 3)}, {"delays": (1, 1)},
            {"epochs": 0}, {"neurons": 0},
            {"burn_in": 16}, {"train_sequences": 0},
            {"delays": (20,)},
        ]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                RobustnessConfig(**{**self.config.__dict__, **override})

    def test_cli_exports_csv_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            csv_path = Path(folder) / "out.csv"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "--seed-count", "1", "--delays", "1", "2",
                    "--noise-levels", "0", "0.25", "--train-sequences", "4",
                    "--test-sequences", "2", "--sequence-length", "12",
                    "--burn-in", "3", "--neurons", "6", "--epochs", "4",
                    "--json-output", str(json_path), "--csv-output", str(csv_path),
                ]), 0)
            result = json.loads(json_path.read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(result["runs"]))
            self.assertEqual(len(result["runs"]), 4)
            self.assertEqual(rows[0]["seed"], "369")


if __name__ == "__main__":
    unittest.main()
