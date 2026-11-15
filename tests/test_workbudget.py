"""Phase 14 cap accounting, burst noise, paired splits and reproducibility."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.compute import architecture_costs, match_recurrent_widths
from phineubio.fairness import MODELS
from phineubio.workbudget import (
    WorkBudgetConfig, generate_work_sequences, work_profile, schedule_epochs,
    run_work_budget, main,
)


class WorkBudgetTests(unittest.TestCase):
    def setUp(self):
        self.cfg=WorkBudgetConfig(
            work_caps=(10000,25000),seeds=(7,),
            signal_families=("burst",),noise_levels=(0,),
            delays=(1,),pair_visit_budget=27,
            train_sequences=3,validation_sequences=2,test_sequences=2,
            sequence_length=12,burn_in=3,max_epochs=3,burst_block=3,
        )

    def test_burst_stream_deterministic_and_noise_nested_by_block(self):
        a=generate_work_sequences("burst",4,48,17,.85,6,0,5,4)
        b=generate_work_sequences("burst",4,48,17,.85,6,.2,5,4)
        c=generate_work_sequences("burst",4,48,17,.85,6,.45,5,4)
        self.assertEqual(a,generate_work_sequences("burst",4,48,17,.85,6,0,5,4))
        for (clean,observed),(_,low),(_,high) in zip(a,b,c):
            self.assertEqual(clean,observed)
            self.assertTrue(all(not(v!=lo and v==hi) for v,lo,hi in zip(clean,low,high)))
            for start in range(0,48,4):
                self.assertEqual(len({int(x!=y) for x,y in zip(clean[start:start+4],low[start:start+4])}),1)

    def test_symbolic_components_match_declared_training_loop_formula(self):
        widths=match_recurrent_widths(27)
        costs=architecture_costs(widths,2)
        gru=work_profile("trainable_gru",costs,3,2,12,3,2)
        n=widths["trainable_gru"]["hidden_width"]
        e=gru["per_candidate_epoch_components"]
        self.assertEqual(e["recurrent_forward_pair_terms"],3*n*n*12*3)
        self.assertEqual(e["recurrent_backward_pair_terms"],6*n*n*12*3)
        self.assertEqual(gru["per_candidate_epoch_units"],sum(e.values()))
        lif=work_profile("spiking_lif",costs,3,2,12,3,2)
        lif_width=widths["spiking_lif"]["hidden_width"]
        self.assertEqual(lif["shared_feature_events"],lif_width*lif_width*12*5)

    def test_allocations_never_exceed_caps_and_do_not_shrink(self):
        costs=architecture_costs(match_recurrent_widths(27),2)
        for name in MODELS:
            p=work_profile(name,costs,3,2,12,3,2)
            low=schedule_epochs(name,p,10000,3)
            high=schedule_epochs(name,p,25000,3)
            self.assertGreaterEqual(high["epochs"],low["epochs"])
            if name!="majority":
                for b,cap in ((low,10000),(high,25000)):
                    self.assertGreaterEqual(b["epochs"],1)
                    self.assertLessEqual(b["modeled_selection_units"],cap)
                    self.assertEqual(b["unused_cap"],cap-b["modeled_selection_units"])
                    self.assertEqual(sum(b["selection_proxy_components"].values()),
                                     b["modeled_selection_units"])
            else:
                self.assertFalse(low["budget_applies"])
        with self.assertRaises(ValueError):
            schedule_epochs("trainable_gru",work_profile("trainable_gru",costs,3,2,12,3,2),10,3)

    def test_seeded_accuracy_and_budget_selection_reproducible(self):
        a=run_work_budget(self.cfg)
        b=run_work_budget(self.cfg)
        self.assertEqual(a["schema_version"],"14.0")
        self.assertEqual(len(a["runs"]),2)
        self.assertEqual(a["preselected_allocations"],b["preselected_allocations"])
        for one,two in zip(a["runs"],b["runs"]):
            self.assertEqual(one["accuracy"],two["accuracy"])
            self.assertEqual(one["selection"],two["selection"])
            self.assertEqual(one["samples"]["test"],18)
            self.assertEqual(set(one["accuracy"]),set(MODELS))
            self.assertTrue(all(x["selection_elapsed_ns"]>0 for x in one["training_effort"].values()))

    def test_invalid_configuration(self):
        for kw in [{"work_caps":()},{"work_caps":(25000,10000)},
                   {"work_caps":(10000,10000)},{"work_caps":(0,)},
                   {"seeds":(7,7)},{"signal_families":("unknown",)},
                   {"burst_block":100},{"max_epochs":0},
                   {"pair_visit_budget":2}]:
            with self.subTest(change=kw),self.assertRaises(ValueError):
                WorkBudgetConfig(**{**self.cfg.__dict__,**kw})

    def test_cli_json_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Path(folder)/"budget.json"
            c=Path(folder)/"budget.csv"
            args=["--work-caps","10000","25000","--seed-count","1",
                  "--signal-families","burst","--noise-levels","0",
                  "--delays","1","--pair-visit-budget","27",
                  "--train-sequences","3","--validation-sequences","2",
                  "--test-sequences","2","--sequence-length","12",
                  "--burn-in","3","--max-epochs","3","--burst-block","3",
                  "--json-output",str(j),"--csv-output",str(c)]
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),0)
            data=json.loads(j.read_text(encoding="utf-8"))
            with c.open(newline="",encoding="utf-8") as h:
                rows=list(csv.DictReader(h))
            self.assertEqual(len(data["runs"]),2)
            self.assertEqual(len(rows),14)
            self.assertEqual(rows[0]["model"],"trainable_gru")


if __name__=="__main__":
    unittest.main()
