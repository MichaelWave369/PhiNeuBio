"""Phase 13 learning exposure, data pairing and export tests."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.fairness import MODELS
from phineubio.training import (
    TrainingConfig, generate_training_sequences, effort, run_training, main,
)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=TrainingConfig(
            epochs_budgets=(1,2),seeds=(5,),
            signal_families=("iid","switching"),noise_levels=(0,),
            delays=(1,),pair_visit_budget=27,
            train_sequences=3,validation_sequences=2,test_sequences=2,
            sequence_length=12,burn_in=3,switching_block=3,
        )

    def test_switching_signal_has_nested_noise_and_repeatable_clean(self):
        a=generate_training_sequences("switching",4,60,123,.85,6,0.,5)
        b=generate_training_sequences("switching",4,60,123,.85,6,.15,5)
        c=generate_training_sequences("switching",4,60,123,.85,6,.35,5)
        self.assertEqual(a,generate_training_sequences("switching",4,60,123,.85,6,0.,5))
        for (clean,observed),(_,low),(_,high) in zip(a,b,c):
            self.assertEqual(clean,observed)
            self.assertTrue(all(not(v!=lo and v==hi) for v,lo,hi in zip(clean,low,high)))

    def test_same_labels_but_unequal_optimizer_steps(self):
        gru=effort("trainable_gru",2,2,3,12,3)
        lif=effort("spiking_lif",2,2,3,12,3)
        self.assertEqual(gru["target_presentations"],108)
        self.assertEqual(lif["target_presentations"],108)
        self.assertEqual(gru["optimizer_updates"],12)
        self.assertEqual(lif["optimizer_updates"],108)
        self.assertEqual(effort("majority",2,2,3,12,3)["optimizer_updates"],0)

    def test_same_seed_yields_same_accuracy_not_same_elapsed_time(self):
        x,y=run_training(self.cfg),run_training(self.cfg)
        self.assertEqual(x["schema_version"],"13.0")
        self.assertEqual(len(x["runs"]),4)
        self.assertEqual(len(x["summary"]),4)
        for a,b in zip(x["runs"],y["runs"]):
            self.assertEqual(a["metrics"],b["metrics"])
            self.assertEqual(a["selection"],b["selection"])
            self.assertEqual(set(a["metrics"]),set(MODELS))
            self.assertEqual(a["samples"]["test"],18)
            self.assertTrue(all(v["selection_elapsed_ns"]>0 for v in a["training_effort"].values()))

    def test_epoch_budgets_increase_exposure(self):
        report=run_training(self.cfg)
        for family in self.cfg.signal_families:
            group=sorted([r for r in report["runs"] if r["signal_family"]==family],
                         key=lambda r:r["epochs_budget"])
            for name in MODELS[:-1]:
                self.assertEqual(2*group[0]["training_effort"][name]["target_presentations"],
                                 group[1]["training_effort"][name]["target_presentations"])

    def test_bad_parameters_rejected(self):
        for changes in [{"epochs_budgets":()},{"epochs_budgets":(2,1)},
                        {"epochs_budgets":(1,1)},{"seeds":(5,5)},
                        {"signal_families":("no",)},{"switching_block":200},
                        {"pair_visit_budget":2},{"train_sequences":0}]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                TrainingConfig(**{**self.cfg.__dict__,**changes})

    def test_cli_json_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Path(folder)/"training.json"
            c=Path(folder)/"training.csv"
            args=["--epochs-budgets","1","2","--seed-count","1",
                  "--signal-families","switching","--noise-levels","0",
                  "--delays","1","--pair-visit-budget","27","--train-sequences","3",
                  "--validation-sequences","2","--test-sequences","2",
                  "--sequence-length","12","--burn-in","3","--switching-block","3",
                  "--json-output",str(j),"--csv-output",str(c)]
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),0)
            data=json.loads(j.read_text(encoding="utf-8"))
            with c.open(newline="",encoding="utf-8") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(data["runs"]),2)
            self.assertEqual(len(rows),14)
            self.assertEqual(rows[0]["model"],"trainable_gru")


if __name__=="__main__":
    unittest.main()
