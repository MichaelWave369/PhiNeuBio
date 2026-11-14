import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.validation import (
    ValidationConfig, run_validation, select_and_test, widths_under_budget
)
from phineubio.temporal import make_sequences


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ValidationConfig(
            seeds=(17,), delays=(1, 2), noise_levels=(0.0,),
            train_sequences=4, validation_sequences=3, test_sequences=2,
            sequence_length=12, burn_in=3, epochs=2,
            learning_rates=(0.03, 0.1), parameter_budget=55,
        )

    def test_parameter_budget_caps_both_trainable_models(self):
        sizes = widths_under_budget(95)
        self.assertEqual(sizes["trainable_gru"]["trainable_parameters"], 77)
        self.assertEqual(sizes["trainable_elman"]["trainable_parameters"], 89)
        self.assertTrue(all(d["trainable_parameters"] <= 95 for d in sizes.values()))

    def test_train_validation_test_are_distinct(self):
        seed = 17 * 1009
        train = make_sequences(4, 12, seed + 101, 0.85, 0.1)
        validation = make_sequences(4, 12, seed + 303, 0.85, 0.1)
        test = make_sequences(4, 12, seed + 202, 0.85, 0.1)
        self.assertNotEqual(train, validation)
        self.assertNotEqual(validation, test)
        self.assertNotEqual(train, test)

    def test_selection_does_not_use_test_targets(self):
        seed = 17 * 1009
        train = make_sequences(4, 12, seed + 101, 0.85, 0.1)
        val = make_sequences(3, 12, seed + 303, 0.85, 0.1)
        test = make_sequences(2, 12, seed + 202, 0.85, 0.1)
        altered = [([1 - bit for bit in clean], observed[:]) for clean, observed in test]
        for name in ("trainable_gru", "trainable_elman"):
            with self.subTest(name=name):
                a = select_and_test(name, self.cfg, 17, 1, train, val, test)
                b = select_and_test(name, self.cfg, 17, 1, train, val, altered)
                self.assertEqual(a["selected_learning_rate"], b["selected_learning_rate"])
                self.assertEqual(a["candidate_validation_scores"], b["candidate_validation_scores"])
                self.assertEqual(a["training_loss_by_epoch"], b["training_loss_by_epoch"])
                self.assertEqual(a["trainable_parameters"], b["trainable_parameters"])

    def test_report_accuracy_repeatable_but_not_timing(self):
        a = run_validation(self.cfg)
        b = run_validation(self.cfg)
        self.assertEqual(a["schema_version"], "7.0")
        self.assertEqual(len(a["runs"]), 2)
        self.assertEqual(len(a["summary"]), 2)
        for first, second in zip(a["runs"], b["runs"]):
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(first["samples"], second["samples"])
            self.assertEqual(first["samples"], {"train": 36, "validation": 27, "test": 18})
            for model in ("trainable_gru", "trainable_elman"):
                x, y = first["selection"][model], second["selection"][model]
                self.assertEqual(x["selected_learning_rate"], y["selected_learning_rate"])
                self.assertEqual(x["validation_cross_entropy"], y["validation_cross_entropy"])
                self.assertGreater(x["selection_elapsed_ns"], 0)
                self.assertGreater(x["one_test_evaluation_ns"], 0)

    def test_invalid_validation_config(self):
        for changes in [
            {"seeds": ()}, {"seeds": (1, 1)}, {"delays": (12,)},
            {"noise_levels": (0.1, 0.1)}, {"noise_levels": (0.6,)},
            {"learning_rates": ()}, {"learning_rates": (-0.1,)},
            {"parameter_budget": 10}, {"validation_sequences": 0},
            {"sequence_length": 2}, {"epochs": 0},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ValidationConfig(**{**self.cfg.__dict__, **changes})

    def test_cli_json_csv(self):
        # Use the main function directly to test its actual argument plumbing.
        from phineubio.validation import main
        with tempfile.TemporaryDirectory() as folder:
            j = Path(folder) / "out.json"
            c = Path(folder) / "out.csv"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main([
                    "--seed-count", "1", "--delays", "1",
                    "--noise-levels", "0",
                    "--train-sequences", "3", "--validation-sequences", "2",
                    "--test-sequences", "2", "--sequence-length", "10",
                    "--burn-in", "3", "--epochs", "2",
                    "--parameter-budget", "55", "--json-output", str(j),
                    "--csv-output", str(c),
                ]), 0)
            report = json.loads(j.read_text(encoding="utf-8"))
            with c.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(report["schema_version"], "7.0")
            self.assertEqual(len(rows), 1)
            self.assertIn("trainable_gru_test_accuracy", rows[0])


if __name__ == "__main__":
    unittest.main()
