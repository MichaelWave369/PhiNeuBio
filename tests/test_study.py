"""Phase 10 predeclared analysis tests; isolated from nondeterministic timing."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.audit import COMPARISONS
from phineubio.fairness import MODELS
from phineubio.study import (StudyConfig, exact_two_sided_sign_p, holm_adjust,
                             load_plan, run_study, summarize_study, main)


class Phase10Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=StudyConfig(
            seeds=(11,12),train_sizes=(2,3),signal_families=("iid",),
            delays=(1,),noise_levels=(0.0,),validation_sequences=2,
            test_sequences=2,probe_sequences=1,sequence_length=12,burn_in=3,
            epochs=2,learning_rates=(0.03,0.1),parameter_budget=55,
            bootstrap_resamples=200,
        )

    def test_sign_test_exact_ties_and_insufficient_seeds(self):
        self.assertIsNone(exact_two_sided_sign_p(2,1))
        self.assertAlmostEqual(exact_two_sided_sign_p(10,0),2/1024)
        self.assertEqual(exact_two_sided_sign_p(5,5,2),1.0)
        self.assertIsNone(exact_two_sided_sign_p(2,2,4))
        with self.assertRaises(ValueError):
            exact_two_sided_sign_p(-1,5)

    def test_holm_adjustment_is_monotonic_and_conservative(self):
        self.assertEqual(holm_adjust([0.01,0.02,0.04,1]),[0.04,0.06,0.08,1.0])
        with self.assertRaises(ValueError):
            holm_adjust([float("nan"),0.2])

    def test_plan_roundtrip_and_tamper_rejection(self):
        plan={
            "schema_version":"10.0","research_question":"Does training size affect paired accuracy?",
            "configuration":__import__("dataclasses").asdict(self.cfg),
            "comparisons":[list(pair) for pair in COMPARISONS],
        }
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"plan.json"
            path.write_text(json.dumps(plan),encoding="utf-8")
            parsed,raw=load_plan(path)
            self.assertEqual(parsed,self.cfg)
            self.assertEqual(raw,json.loads(json.dumps(plan)))
            result=run_study(parsed,raw)
            self.assertEqual(result["plan"],raw)
            plan["comparisons"]=[["spiking_lif","majority"]]
            path.write_text(json.dumps(plan),encoding="utf-8")
            with self.assertRaises(ValueError):
                load_plan(path)

    def test_paired_train_sizes_and_deterministic_metrics(self):
        first=run_study(self.cfg)
        second=run_study(self.cfg)
        self.assertEqual(first,second)
        self.assertEqual(first["schema_version"],"10.0")
        self.assertEqual(len(first["records"]),4)
        self.assertEqual(len(first["summary"]),2)
        self.assertEqual(len(first["plan_sha256"]),64)
        for record in first["records"]:
            self.assertEqual(set(record["metrics"]),set(MODELS))
            self.assertEqual(record["samples"]["test"],18)
            self.assertEqual(record["samples"]["validation"],18)
        for cell in first["summary"]:
            self.assertEqual(len(cell["contrasts"]),4)
            for effect in cell["contrasts"]:
                self.assertIsNone(effect["exact_sign_p"])
                self.assertIsNone(effect["holm_adjusted_sign_p"])
                self.assertIsNone(effect["effect"]["bootstrap_95_interval"])

    def test_incomplete_and_duplicate_results_rejected(self):
        report=run_study(self.cfg)
        records=report["records"]
        with self.assertRaises(ValueError):
            summarize_study(self.cfg,records[:-1])
        with self.assertRaises(ValueError):
            summarize_study(self.cfg,records[:-1]+[records[0]])

    def test_cli_json_and_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"study.json"
            csv_path=Path(folder)/"study.csv"
            args=[
                "--seed-count","1","--training-sizes","2","3",
                "--signal-families","iid","--delays","1",
                "--noise-levels","0","--train-epochs","2",
                "--sequence-length","12","--burn-in","3",
                "--validation-sequences","2","--test-sequences","2",
                "--probe-sequences","1","--parameter-budget","55",
                "--resamples","200","--output",str(output),
                "--csv-output",str(csv_path),
            ]
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),0)
            data=json.loads(output.read_text(encoding="utf-8"))
            with csv_path.open(newline="",encoding="utf-8") as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(len(data["records"]),2)
            self.assertEqual(len(rows),8)
            self.assertEqual(data["plan"]["configuration"]["train_sizes"],[2,3])

    def test_config_cost_guard(self):
        with self.assertRaises(ValueError):
            StudyConfig(seeds=tuple(range(12)),train_sizes=tuple(range(1,9)),
                        signal_families=("iid","markov","periodic"),
                        delays=(1,3),noise_levels=(0,.15))
        with self.assertRaises(ValueError):
            StudyConfig(train_sizes=(8,4))


if __name__=="__main__":
    unittest.main()
