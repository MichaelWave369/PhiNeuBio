"""Phase 9: paired uncertainty never treats time steps as independent seeds."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.audit import MIN_INTERVAL_SEEDS, paired_interval, run_audit, main
from phineubio.fairness import MODELS


def fixture(seeds=(1, 2, 3, 4, 5)):
    # Minimal deliberately controlled, valid Phase 8 source grid.
    return {
        "schema_version": "8.0", "substrate": "software_simulation",
        "task": "validation_tuned_multisignal_comparison",
        "configuration": {
            "seeds": list(seeds), "signal_families": ["iid"],
            "noise_levels": [0.0], "delays": [1],
        },
        "runs": [{
            "signal_family": "iid", "noise": 0.0, "delay": 1, "seed": seed,
            "samples": {"train": 16, "validation": 8, "test": 8, "profiling_probe": 8},
            "metrics": {model: (0.75 if model == "spiking_lif" else 0.5) for model in MODELS},
        } for seed in seeds],
    }


class AuditTests(unittest.TestCase):
    def test_positive_difference_and_bootstrap_reproducible(self):
        report = run_audit(fixture(), resamples=400)
        self.assertEqual(report, run_audit(fixture(), resamples=400))
        self.assertEqual(report["schema_version"], "9.0")
        self.assertEqual(len(report["conditions"]), 1)
        effect = report["conditions"][0]["comparisons"][0]["effect"]
        self.assertEqual(effect["paired_seed_count"], 5)
        self.assertEqual(effect["mean_difference"], 0.25)
        self.assertEqual(effect["bootstrap_95_interval"], [0.25, 0.25])
        self.assertEqual(effect["positive_seed_count"], 5)
        self.assertEqual(len(report["source"]["canonical_json_sha256"]), 64)

    def test_two_seed_sweep_never_pretends_to_be_confident(self):
        report = run_audit(fixture((1, 2)), resamples=400)
        effect = report["conditions"][0]["comparisons"][0]["effect"]
        self.assertIsNone(effect["bootstrap_95_interval"])
        self.assertEqual(effect["interpretation"], "insufficient_seeds")
        self.assertEqual(MIN_INTERVAL_SEEDS, 5)

    def test_wide_interval_spans_zero_for_mixed_wins(self):
        result = paired_interval([0.4, -0.4, 0.4, -0.4, 0], resamples=800)
        self.assertLess(result["bootstrap_95_interval"][0], 0)
        self.assertGreater(result["bootstrap_95_interval"][1], 0)

    def test_rejects_duplicate_missing_nonfinite_and_unpaired_data(self):
        broken = fixture()
        broken["runs"].append(broken["runs"][0])
        with self.assertRaises(ValueError):
            run_audit(broken, resamples=400)
        missing = fixture()
        missing["runs"].pop()
        with self.assertRaises(ValueError):
            run_audit(missing, resamples=400)
        bad = fixture()
        bad["runs"][0]["metrics"]["spiking_lif"] = float("nan")
        with self.assertRaises(ValueError):
            run_audit(bad, resamples=400)
        uneven = fixture()
        uneven["runs"][0]["samples"]["test"] = 9
        with self.assertRaises(ValueError):
            run_audit(uneven, resamples=400)

    def test_cli_json_csv_and_no_rebenchmarking(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "phase8.json"
            output = Path(folder) / "phase9.json"
            csv_output = Path(folder) / "effects.csv"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--input", str(source), "--output", str(output),
                             "--csv-output", str(csv_output), "--resamples", "400"])
            self.assertEqual(code, 0)
            audit = json.loads(output.read_text(encoding="utf-8"))
            with csv_output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(audit["source"]["source_run_count"], 5)
            self.assertEqual(rows[0]["bootstrap_95_low"], "0.25")


if __name__ == "__main__":
    unittest.main()
