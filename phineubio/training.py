"""Phase 13: matched target exposures and measured selection costs.

All learning rates are tuned on validation only; final test labels are unseen
until candidate selection completes. Runtime is Python elapsed time, not energy.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns
from types import SimpleNamespace

from .compute import ComputeConfig, architecture_costs, match_recurrent_widths
from .fairness import FAMILIES, MODELS, READOUTS, TRAINED, generate_sequences, select_readout, select_trainable
from .temporal import extract_features


@dataclass(frozen=True)
class TrainingConfig:
    epochs_budgets: tuple[int, ...] = (2, 5)
    seeds: tuple[int, ...] = (369, 370)
    signal_families: tuple[str, ...] = ("markov", "switching")
    noise_levels: tuple[float, ...] = (.15,)
    delays: tuple[int, ...] = (2,)
    pair_visit_budget: int = 75
    train_sequences: int = 6
    validation_sequences: int = 3
    test_sequences: int = 3
    sequence_length: int = 24
    burn_in: int = 6
    learning_rates: tuple[float, ...] = (.03, .1)
    gradient_clip: float = 5.
    stay_probability: float = .85
    period: int = 6
    switching_block: int = 6

    def __post_init__(self):
        if (not self.epochs_budgets or len(self.epochs_budgets)>6 or
            any(type(x) is not int or not 1<=x<=100 for x in self.epochs_budgets) or
            self.epochs_budgets != tuple(sorted(set(self.epochs_budgets)))):
            raise ValueError("epoch budgets must be 1-6 distinct sorted integers in [1,100]")
        if (not self.seeds or len(self.seeds)>12 or len(set(self.seeds))!=len(self.seeds)
            or any(type(x) is not int for x in self.seeds)):
            raise ValueError("invalid seed list")
        if (not self.signal_families or len(set(self.signal_families))!=len(self.signal_families)
            or any(x not in FAMILIES+("switching",) for x in self.signal_families)):
            raise ValueError("invalid signal family")
        if type(self.switching_block) is not int or not 2<=self.switching_block<=self.sequence_length:
            raise ValueError("switching_block must fit sequence_length")
        if len(self.epochs_budgets)*len(self.seeds)*len(self.signal_families)*len(self.noise_levels)*len(self.delays)>120:
            raise ValueError("at most 120 study cells")
        ComputeConfig(
            seeds=self.seeds, signal_families=("markov",), delays=self.delays,
            noise_levels=self.noise_levels, pair_visit_budget=self.pair_visit_budget,
            train_sequences=self.train_sequences, validation_sequences=self.validation_sequences,
            test_sequences=self.test_sequences, probe_sequences=1,
            sequence_length=self.sequence_length,burn_in=self.burn_in,
            epochs=self.epochs_budgets[0], learning_rates=self.learning_rates,
            stay_probability=self.stay_probability, period=self.period,
        )
        if not math.isfinite(self.gradient_clip) or self.gradient_clip<=0:
            raise ValueError("gradient clip must be finite positive")


def generate_training_sequences(family, count, length, seed, stay, period, noise, block):
    """Alternate .95/.55 persistence blocks; noise draws follow clean generation."""
    if family!="switching":
        return generate_sequences(family,count,length,seed,stay,period,noise)
    rng=random.Random(seed)
    sequences=[]
    for _ in range(count):
        bit=rng.randrange(2)
        clean=[bit]
        for t in range(1,length):
            chance=.95 if ((t-1)//block)%2==0 else .55
            if rng.random()>=chance: bit ^= 1
            clean.append(bit)
        observed=[v ^ int(rng.random()<noise) for v in clean]
        sequences.append((clean,observed))
    return sequences


def effort(name,epochs,candidates,sequences,length,start):
    if name not in MODELS or min(epochs,candidates,sequences,length-start)<1:
        raise ValueError("invalid effort parameters")
    retained=sequences*(length-start)
    if name=="majority":
        return {"candidate_models":0,"target_presentations":retained,
                "optimizer_updates":0,"bptt_steps":0}
    presentations=candidates*epochs*retained
    return {
        "candidate_models":candidates,
        "target_presentations":presentations,
        "optimizer_updates":candidates*epochs*(sequences if name in TRAINED else retained),
        "bptt_steps":candidates*epochs*sequences*length if name in TRAINED else 0,
    }


def fit_and_test(name,opts,seed,delay,train,validation,test,start,widths,history):
    """Time candidate selection without test; evaluate test only after selection."""
    selection_start=perf_counter_ns()
    if name in TRAINED:
        model,info=select_trainable(name,opts,seed,delay,train,validation,start,widths)
    elif name in READOUTS:
        n=widths[name]["hidden_width"] if name in widths else history
        model,info=select_readout(name,opts,seed,delay,train,validation,start,n)
    else:
        labels=[c[t-delay] for c,_ in train for t in range(start,len(c))]
        model=int(2*sum(labels)>=len(labels))
        info={"selected_learning_rate":None,"validation_cross_entropy":None,
              "candidate_validation_scores":[],"trainable_parameters":0}
    select_ns=max(1,perf_counter_ns()-selection_start)
    test_start=perf_counter_ns()
    if name in TRAINED:
        accuracy=model.accuracy(test,delay,start)
    elif name in READOUTS:
        n=widths[name]["hidden_width"] if name in widths else history
        features,labels=extract_features(name,test,delay,start,n,n,seed)
        accuracy=model.score(features,labels)
    else:
        targets=[c[t-delay] for c,_ in test for t in range(start,len(c))]
        accuracy=sum(model==y for y in targets)/len(targets)
    return accuracy,info,select_ns,max(1,perf_counter_ns()-test_start)


def run_training(config=None):
    cfg=config or TrainingConfig()
    widths=match_recurrent_widths(cfg.pair_visit_budget)
    costs=architecture_costs(widths,max(cfg.delays)+1)
    start=max(cfg.burn_in,max(cfg.delays))
    runs=[]
    for family in cfg.signal_families:
        for seed in cfg.seeds:
            for noise in cfg.noise_levels:
                streams={
                    key:generate_training_sequences(
                        family,count,cfg.sequence_length,seed*1009+offset,
                        cfg.stay_probability,cfg.period,noise,cfg.switching_block)
                    for key,count,offset in (
                        ("train",cfg.train_sequences,101),
                        ("validation",cfg.validation_sequences,303),
                        ("test",cfg.test_sequences,202),
                    )
                }
                for delay in cfg.delays:
                    for epochs in cfg.epochs_budgets:
                        opts=SimpleNamespace(epochs=epochs,learning_rates=cfg.learning_rates,
                                             gradient_clip=cfg.gradient_clip)
                        metrics,selection,accounting={},{},{}
                        for name in MODELS:
                            acc,chosen,ns,eval_ns=fit_and_test(
                                name,opts,seed,delay,streams["train"],streams["validation"],
                                streams["test"],start,widths,max(cfg.delays)+1)
                            metrics[name]=acc
                            selection[name]=chosen
                            accounting[name]={
                                **effort(name,epochs,len(cfg.learning_rates),cfg.train_sequences,
                                         cfg.sequence_length,start),
                                "selection_elapsed_ns":ns,
                                "test_evaluation_elapsed_ns":eval_ns,
                            }
                        runs.append({
                            "signal_family":family,"seed":seed,"noise":noise,"delay":delay,
                            "epochs_budget":epochs,
                            "samples":{
                                "train":cfg.train_sequences*(cfg.sequence_length-start),
                                "validation":cfg.validation_sequences*(cfg.sequence_length-start),
                                "test":cfg.test_sequences*(cfg.sequence_length-start),
                            },
                            "metrics":metrics,"selection":selection,"training_effort":accounting,
                        })
    summary=[]
    for family in cfg.signal_families:
        for noise in cfg.noise_levels:
            for delay in cfg.delays:
                for epochs in cfg.epochs_budgets:
                    group=[x for x in runs if
                           (x["signal_family"],x["noise"],x["delay"],x["epochs_budget"])==
                           (family,noise,delay,epochs)]
                    summary.append({
                        "signal_family":family,"noise":noise,"delay":delay,
                        "epochs_budget":epochs,"seeds":len(group),
                        "accuracy":{
                            name:{
                                "mean":statistics.mean(x["metrics"][name] for x in group),
                                "population_stdev":statistics.pstdev(x["metrics"][name] for x in group),
                            } for name in MODELS
                        },
                        "median_selection_elapsed_ns":{
                            name:statistics.median(
                                x["training_effort"][name]["selection_elapsed_ns"] for x in group)
                            for name in MODELS
                        },
                    })
    return {
        "schema_version":"13.0","project":"PhiNeuBio","substrate":"software_simulation",
        "task":"training_exposure_and_selection_cost",
        "configuration":asdict(cfg),
        "architecture_costs":costs,
        "protocol":{
            "paired_data":"Same seeded train, validation and held-out test streams at each epoch budget",
            "selection":"Learning rates selected by validation BCE only; test labels first read after selection",
            "timing":"Selection elapsed includes model initialization, readout feature extraction, optimization and validation scoring, not held-out test evaluation or data generation",
            "budget":"Equal retained target presentations per trained model; optimizer update counts and arithmetic workloads are not equal",
            "switching":"Alternating .95/.55 Markov persistence blocks with independent observation corruption",
            "hardware":"No CPU instruction counts, power meters or joule measurements",
        },
        "runs":runs,"summary":summary,
        "limitations":[
            "Equal target presentations do not mean equal optimization compute, FLOPs or training time.",
            "Sequence-wise BPTT makes fewer optimizer updates than pointwise readout SGD.",
            "Selection wall time includes feature extraction and validation, not just backward passes.",
            "Repeated epoch budgets share the same seed data; they are paired rather than independent experiments.",
            "Toy digital signals are not living biological neurons or evidence of consciousness.",
        ],
    }


def write_csv(report,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=("signal_family","seed","noise","delay","epochs_budget","model","test_accuracy",
            "target_presentations","optimizer_updates","selection_elapsed_ns")
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in report["runs"]:
            for name in MODELS:
                e=row["training_effort"][name]
                writer.writerow({
                    "signal_family":row["signal_family"],"seed":row["seed"],
                    "noise":row["noise"],"delay":row["delay"],"epochs_budget":row["epochs_budget"],
                    "model":name,"test_accuracy":row["metrics"][name],
                    "target_presentations":e["target_presentations"],
                    "optimizer_updates":e["optimizer_updates"],
                    "selection_elapsed_ns":e["selection_elapsed_ns"],
                })


def main(argv=None):
    p=argparse.ArgumentParser(description="PhiNeuBio Phase 13 learning exposure study")
    p.add_argument("--epochs-budgets",type=int,nargs="+",default=[2,5])
    p.add_argument("--seed-start",type=int,default=369)
    p.add_argument("--seed-count",type=int,default=2)
    p.add_argument("--signal-families",nargs="+",default=["markov","switching"])
    p.add_argument("--noise-levels",type=float,nargs="+",default=[.15])
    p.add_argument("--delays",type=int,nargs="+",default=[2])
    p.add_argument("--pair-visit-budget",type=int,default=75)
    p.add_argument("--train-sequences",type=int,default=6)
    p.add_argument("--validation-sequences",type=int,default=3)
    p.add_argument("--test-sequences",type=int,default=3)
    p.add_argument("--sequence-length",type=int,default=24)
    p.add_argument("--burn-in",type=int,default=6)
    p.add_argument("--switching-block",type=int,default=6)
    p.add_argument("--json-output",type=Path)
    p.add_argument("--csv-output",type=Path)
    a=p.parse_args(argv)
    try:
        if not 1<=a.seed_count<=12: raise ValueError("seed-count must be 1-12")
        cfg=TrainingConfig(
            epochs_budgets=tuple(a.epochs_budgets),
            seeds=tuple(range(a.seed_start,a.seed_start+a.seed_count)),
            signal_families=tuple(a.signal_families),noise_levels=tuple(a.noise_levels),
            delays=tuple(a.delays),pair_visit_budget=a.pair_visit_budget,
            train_sequences=a.train_sequences,validation_sequences=a.validation_sequences,
            test_sequences=a.test_sequences,sequence_length=a.sequence_length,
            burn_in=a.burn_in,switching_block=a.switching_block,
        )
        result=run_training(cfg)
    except ValueError as err:
        p.error(str(err))
    output=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if a.json_output:
        a.json_output.parent.mkdir(parents=True,exist_ok=True)
        a.json_output.write_text(output,encoding="utf-8")
        print("Saved Phase 13 report: "+str(a.json_output))
    else:
        print(output,end="")
    if a.csv_output:
        write_csv(result,a.csv_output)
        print("Saved Phase 13 CSV: "+str(a.csv_output),file=sys.stderr)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
