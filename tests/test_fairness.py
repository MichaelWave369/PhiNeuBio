"""Phase 8 reproducibility, signal controls and validation isolation."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.fairness import (
    FairnessConfig, MODELS, generate_sequences, run_fairness,
    select_readout, select_trainable, main,
)
from phineubio.validation import widths_under_budget


class FairnessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = FairnessConfig(
            seeds=(7,), delays=(1, 2), noise_levels=(0.0,),
            signal_families=("markov", "periodic"),
            train_sequences=3, validation_sequences=2,
            test_sequences=2, probe_sequences=1,
            sequence_length=12, burn_in=3, epochs=2,
            learning_rates=(0.03, 0.1),
            parameter_budget=55, inference_repeats=2,
        )

    def test_signal_streams_reproducible_and_nested_noise(self):
        for family in ("markov", "periodic"):
            clean = generate_sequences(family, 4, 24, 123, 0.85, 6, 0.0)
            medium = generate_sequences(family, 4, 24, 123, 0.85, 6, 0.1)
            noisy = generate_sequences(family, 4, 24, 123, 0.85, 6, 0.3)
            self.assertEqual(clean, generate_sequences(family, 4, 24, 123, 0.85, 6, 0.0))
            self.assertEqual([a for a, _ in clean], [a for a, _ in noisy])
            for (truth, observed), (_, middle), (_, high) in zip(clean, medium, noisy):
                self.assertEqual(truth, observed)
                self.assertFalse(any(mid != actual and strong == actual
                                     for actual, mid, strong in zip(truth, middle, high)))

    def test_validation_selection_is_independent_of_test_data(self):
        train = generate_sequences("periodic", 4, 12, 11, 0.85, 6, 0.1)
        val = generate_sequences("periodic", 2, 12, 22, 0.85, 6, 0.1)
        alternate_test = generate_sequences("periodic", 2, 12, 333, 0.85, 6, 0.1)
        # Neither selection function accepts a test parameter at all.
        for name in ("trainable_gru", "trainable_elman"):
            _, report = select_trainable(name, self.cfg, 7, 1, train, val, 3,
                                         widths_under_budget(55))
            self.assertIn(report["selected_learning_rate"], self.cfg.learning_rates)
            self.assertEqual(len(report["candidate_validation_scores"]), 2)
        for name in ("spiking_lif", "tanh_reservoir", "history_window", "delayed_tap"):
            _, report = select_readout(name, self.cfg, 7, 1, train, val, 3, 6)
            self.assertIn(report["selected_learning_rate"], self.cfg.learning_rates)
            self.assertEqual(len(report["candidate_validation_scores"]), 2)
        self.assertNotEqual(train, alternate_test)

    def test_deterministic_accuracy_but_observed_times(self):
        first = run_fairness(self.cfg)
        second = run_fairness(self.cfg)
        self.assertEqual(first["schema_version"], "8.0")
        self.assertEqual(len(first["runs"]), 4)
        self.assertEqual(len(first["summary"]), 4)
        for a, b in zip(first["runs"], second["runs"]):
            self.assertEqual(a["metrics"], b["metrics"])
            # Avoid asserting wall-clock times are deterministic.
            a_select = {name: (value["selected_learning_rate"], value["validation_cross_entropy"])
                        for name, value in a["selection"].items()}
            b_select = {name: (value["selected_learning_rate"], value["validation_cross_entropy"])
                        for name, value in b["selection"].items()}
            self.assertEqual(a_select, b_select)
            self.assertEqual(a["samples"]["test"], 18)
            self.assertEqual(set(a["metrics"]), set(MODELS))
            for model in MODELS:
                self.assertTrue(0 <= a["metrics"][model] <= 1)
                profile = a["inference_profiling"][model]
                self.assertEqual(len(profile["elapsed_ns"]), 2)
                self.assertGreater(profile["median_elapsed_ns"], 0)
                self.assertEqual(profile["retained_samples_per_repeat"], 9)

    def test_invalid_configuration(self):
        invalid = [
            {"seeds": ()}, {"delays": (0,)},
            {"signal_families": ("other",)}, {"signal_families": ("markov", "markov")},
            {"noise_levels": (0.7,)}, {"validation_sequences": 0},
            {"inference_repeats": 0}, {"learning_rates": ()},
            {"parameter_budget": 8}, {"period": 99},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                FairnessConfig(**{**self.cfg.__dict__, **change})

    def test_cli_report_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "run.json"
            csv_file = Path(folder) / "run.csv"
            args = [
                "--seed-count", "1", "--delays", "1", "--noise-levels", "0",
                "--signal-families", "periodic", "--train-sequences", "3",
                "--validation-sequences", "2", "--test-sequences", "2",
                "--probe-sequences", "1", "--sequence-length", "12",
                "--burn-in", "3", "--epochs", "2", "--parameter-budget", "55",
                "--inference-repeats", "2", "--json-output", str(dest),
                "--csv-output", str(csv_file),
            ]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(dest.read_text(encoding="utf-8"))
            with csv_file.open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 1)
            self.assertEqual(report["task"], "validation_tuned_multisignal_comparison")
            self.assertEqual(records[0]["signal_family"], "periodic")


if __name__ == "__main__":
    unittest.main()
