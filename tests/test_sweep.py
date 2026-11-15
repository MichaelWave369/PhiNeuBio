"""Phase 12 work event ledger, controlled budget sweep and exports."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from phineubio.compute import architecture_costs, match_recurrent_widths
from phineubio.fairness import generate_sequences
from phineubio.reservoir import Reservoir, ReservoirConfig
from phineubio.sweep import SweepConfig, lif_activity, operation_ledger, pareto, run_sweep, main


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.cfg=SweepConfig(
            budgets=(12,27),seeds=(5,),signal_families=("iid",),
            noise_levels=(0.0,),delays=(1,),train_sequences=3,
            validation_sequences=2,test_sequences=2,probe_sequences=1,
            sequence_length=12,burn_in=3,epochs=2,inference_repeats=2,
        )

    def test_lif_event_counter_matches_direct_enumeration(self):
        sequences=generate_sequences("iid",3,14,123,0.85,6,0)
        result=lif_activity(8,4,sequences)
        net=Reservoir(ReservoirConfig(seed=8,neurons=4))
        adds=nonzero=0
        for _,observed in sequences:
            net.reset()
            for bit in observed:
                for row in net.weights:
                    for j,w in enumerate(row):
                        if net.spikes[j]:
                            adds+=1
                            nonzero+=int(w!=0)
                net.step(float(bit))
        self.assertEqual(result["steps"],42)
        self.assertEqual(result["active_weight_addends"],adds)
        self.assertEqual(result["nonzero_active_edges"],nonzero)

    def test_operation_categories_are_not_fake_flops(self):
        costs=architecture_costs(match_recurrent_widths(27),3)
        activity={"steps":24,"active_weight_addends":15,"nonzero_active_edges":4,"emitted_spikes":8}
        events=operation_ledger(costs,24,18,activity)
        self.assertEqual(events["trainable_gru"]["recurrent_dense_scalar_products"],
                         costs["trainable_gru"]["recurrent_pair_visits_per_step"]*24)
        self.assertEqual(events["spiking_lif"]["recurrent_dense_scalar_products"],0)
        self.assertEqual(events["spiking_lif"]["lif_active_weight_addends"],15)
        self.assertEqual(events["delayed_tap"]["readout_scalar_products"],18)
        with self.assertRaises(ValueError):
            operation_ledger(costs,24,25,activity)

    def test_frontier_checks_cost_and_accuracy(self):
        points=[
            {"model":"spiking_lif","visits":9,"accuracy":0.6},
            {"model":"trainable_gru","visits":12,"accuracy":0.7},
            {"model":"trainable_elman","visits":12,"accuracy":0.4},
            {"model":"tanh_reservoir","visits":20,"accuracy":0.7},
        ]
        self.assertEqual(pareto(points),points[:2])
        with self.assertRaises(ValueError):
            pareto([{"model":"majority","visits":0,"accuracy":1.0}])

    def test_accuracy_and_operations_are_repeatable_not_walltime(self):
        a,b=run_sweep(self.cfg),run_sweep(self.cfg)
        self.assertEqual(len(a["budgets"]),2)
        self.assertEqual(a["schema_version"],"12.0")
        self.assertEqual(len(a["frontiers"]),1)
        for x,y in zip(a["budgets"],b["budgets"]):
            self.assertEqual(x["runs"][0]["accuracy"],y["runs"][0]["accuracy"])
            self.assertEqual(x["probe_operations"],y["probe_operations"])
            self.assertEqual(x["runs"][0]["round_order"],y["runs"][0]["round_order"])

    def test_invalid_budget_sequences(self):
        for kw in [{"budgets":()},{"budgets":(27,12)},
                   {"budgets":(12,12)},{"budgets":(2,)},
                   {"seeds":(5,5)},{"inference_repeats":0}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                SweepConfig(**{**self.cfg.__dict__,**kw})

    def test_cli_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Path(folder)/"sweep.json"
            c=Path(folder)/"sweep.csv"
            argv=["--budgets","12","27","--seed-count","1",
                  "--signal-families","iid","--noise-levels","0",
                  "--delays","1","--train-sequences","3",
                  "--validation-sequences","2","--test-sequences","2",
                  "--probe-sequences","1","--sequence-length","12",
                  "--burn-in","3","--epochs","2","--inference-repeats","2",
                  "--json-output",str(j),"--csv-output",str(c)]
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv),0)
            result=json.loads(j.read_text(encoding="utf-8"))
            with c.open(newline="",encoding="utf-8") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(result["budgets"]),2)
            self.assertEqual(len(rows),14)
            self.assertEqual(rows[0]["model"],"trainable_gru")


if __name__=="__main__":
    unittest.main()
