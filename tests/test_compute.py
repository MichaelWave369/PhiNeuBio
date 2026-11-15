"""Phase 11: matched structural budgets and independent local CPU probes."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.compute import (
    ComputeConfig, architecture_costs, benchmark_probes,
    match_recurrent_widths, main, run_compute,
)


class CostLabTests(unittest.TestCase):
    def setUp(self):
        self.cfg=ComputeConfig(
            seeds=(17,),signal_families=("iid",),
            noise_levels=(0.0,),delays=(1,),
            pair_visit_budget=75,
            train_sequences=3,validation_sequences=2,test_sequences=2,
            probe_sequences=2,sequence_length=12,burn_in=3,
            epochs=2,inference_repeats=2,
        )

    def test_widths_satisfy_budget_and_maximize_capacity(self):
        widths=match_recurrent_widths(108)
        self.assertEqual(widths["trainable_gru"]["hidden_width"],6)
        self.assertEqual(widths["trainable_gru"]["recurrent_pair_visits_per_step"],108)
        for name in ("trainable_elman","spiking_lif","tanh_reservoir"):
            self.assertEqual(widths[name]["hidden_width"],10)
            self.assertEqual(widths[name]["recurrent_pair_visits_per_step"],100)
        for name,d in widths.items():
            self.assertLessEqual(d["recurrent_pair_visits_per_step"],108)
            if d["hidden_width"]<32:
                self.assertGreater(d["recurrent_matrices_visited"]*(d["hidden_width"]+1)**2,108)

    def test_accounting_explicitly_separates_fixed_and_trainable_slots(self):
        costs=architecture_costs(match_recurrent_widths(108),3)
        self.assertEqual(costs["trainable_gru"]["trainable_parameter_slots"],3*6*6+7*6+1)
        self.assertEqual(costs["trainable_elman"]["trainable_parameter_slots"],10*10+3*10+1)
        self.assertEqual(costs["spiking_lif"]["trainable_parameter_slots"],2*10+1)
        self.assertEqual(costs["spiking_lif"]["fixed_parameter_slots"],10*10+10)
        self.assertEqual(costs["tanh_reservoir"]["fixed_parameter_slots"],10*10+20)
        self.assertFalse(costs["history_window"]["matched_on_recurrent_pair_visits_only"])
        self.assertEqual(costs["history_window"]["readout_feature_width"],3)

    def test_timing_rounds_randomized_and_checksum_stable(self):
        from phineubio.fairness import MODELS
        runners={name:(lambda n=name:len(n)) for name in MODELS}
        first,orders=benchmark_probes(runners,4,seed=17,samples_per_repeat=20)
        second,other_orders=benchmark_probes(runners,4,seed=17,samples_per_repeat=20)
        self.assertEqual(orders,other_orders)
        self.assertEqual(len(orders),4)
        self.assertGreater(len({tuple(order) for order in orders}),1)
        for name in MODELS:
            self.assertEqual(len(first[name]["elapsed_ns"]),4)
            self.assertGreater(first[name]["median_elapsed_ns"],0)
            self.assertEqual(first[name]["prediction_checksum"],len(name))
            self.assertEqual(first[name]["retained_samples_per_repeat"],20)
            self.assertGreater(second[name]["median_elapsed_ns"],0)

    def test_seeded_accuracy_repeats_but_elapsed_time_not_asserted(self):
        a=run_compute(self.cfg)
        b=run_compute(self.cfg)
        self.assertEqual(a["schema_version"],"11.0")
        self.assertEqual(len(a["runs"]),1)
        self.assertEqual(a["runs"][0]["accuracy"],b["runs"][0]["accuracy"])
        self.assertEqual(a["runs"][0]["selected_validation"],b["runs"][0]["selected_validation"])
        self.assertEqual(a["runs"][0]["round_order"],b["runs"][0]["round_order"])
        self.assertEqual(a["runs"][0]["samples"],{"train":27,"validation":18,"test":18,"probe":18})
        self.assertEqual(set(a["runs"][0]["accuracy"]),set(a["structural_cost_profiles"]))
        for name,value in a["runs"][0]["accuracy"].items():
            self.assertTrue(0<=value<=1,name)

    def test_bad_budget_or_sample_protocol(self):
        for changes in [
            {"pair_visit_budget":2},{"pair_visit_budget":108.0},
            {"inference_repeats":0},{"signal_families":("bogus",)},
            {"delays":(20,)},{"seeds":(17,17)},{"train_sequences":0},
        ]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                ComputeConfig(**{**self.cfg.__dict__,**changes})
        with self.assertRaises(ValueError):
            match_recurrent_widths(1)

    def test_cli_exports_json_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            jp=Path(tmp)/"report.json"
            cp=Path(tmp)/"report.csv"
            args=[
                "--seed-count","1","--signal-families","iid",
                "--noise-levels","0","--delays","1",
                "--pair-visit-budget","75","--train-sequences","3",
                "--validation-sequences","2","--test-sequences","2",
                "--probe-sequences","2","--sequence-length","12",
                "--burn-in","3","--epochs","2","--inference-repeats","2",
                "--json-output",str(jp),"--csv-output",str(cp),
            ]
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),0)
            d=json.loads(jp.read_text(encoding="utf-8"))
            with cp.open(newline="",encoding="utf-8") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),7)
            self.assertEqual(len(d["runs"]),1)
            self.assertEqual(rows[0]["model"],"trainable_gru")
            self.assertTrue(d["runtime_environment"]["python_version"])


if __name__=="__main__":
    unittest.main()
