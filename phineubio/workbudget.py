"""Phase 14: training-work proxy caps and burst-corrupted temporal memory.

A declared work-score sums *non-equivalent* symbolic event categories with
unit weights. It permits reproducible epoch allocation but is NOT actual FLOPs,
CPU instructions, fully matched training compute, energy, or hardware parity.
Independent validation selects hyperparameters; final test is untouched.
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
from types import SimpleNamespace

from .compute import ComputeConfig, architecture_costs, match_recurrent_widths
from .fairness import FAMILIES, MODELS, READOUTS, TRAINED
from .temporal import make_sequences
from .training import TrainingConfig, generate_training_sequences, fit_and_test, effort


SIGNALS = FAMILIES + ("switching", "burst")


@dataclass(frozen=True)
class WorkBudgetConfig:
    work_caps: tuple[int, ...] = (120000, 240000)
    seeds: tuple[int, ...] = (369, 370)
    signal_families: tuple[str, ...] = ("markov", "burst")
    noise_levels: tuple[float, ...] = (.15,)
    delays: tuple[int, ...] = (2,)
    pair_visit_budget: int = 75
    train_sequences: int = 6
    validation_sequences: int = 3
    test_sequences: int = 3
    sequence_length: int = 24
    burn_in: int = 6
    max_epochs: int = 8
    learning_rates: tuple[float, ...] = (.03, .1)
    gradient_clip: float = 5.
    stay_probability: float = .85
    period: int = 6
    switching_block: int = 6
    burst_block: int = 4

    def __post_init__(self):
        if (not self.work_caps or len(self.work_caps)>8 or
            any(type(x) is not int or not 1<=x<=100000000 for x in self.work_caps) or
            self.work_caps != tuple(sorted(set(self.work_caps)))):
            raise ValueError("work caps must be sorted distinct positive integers")
        if not self.signal_families or len(set(self.signal_families))!=len(self.signal_families) or any(s not in SIGNALS for s in self.signal_families):
            raise ValueError("invalid signal families")
        if type(self.burst_block) is not int or not 2<=self.burst_block<=self.sequence_length:
            raise ValueError("burst_block must fit sequence_length")
        if type(self.max_epochs) is not int or not 1<=self.max_epochs<=100:
            raise ValueError("max_epochs must be in [1,100]")
        if len(self.work_caps)*len(self.seeds)*len(self.signal_families)*len(self.noise_levels)*len(self.delays)>100:
            raise ValueError("no more than 100 experiment cells")
        TrainingConfig(
            epochs_budgets=(1,), seeds=self.seeds,
            signal_families=tuple("markov" if x=="burst" else x for x in self.signal_families
                                  if x!="burst") or ("markov",),
            noise_levels=self.noise_levels, delays=self.delays,
            pair_visit_budget=self.pair_visit_budget,
            train_sequences=self.train_sequences, validation_sequences=self.validation_sequences,
            test_sequences=self.test_sequences, sequence_length=self.sequence_length,
            burn_in=self.burn_in, learning_rates=self.learning_rates,
            gradient_clip=self.gradient_clip, stay_probability=self.stay_probability,
            period=self.period, switching_block=self.switching_block,
        )


def generate_work_sequences(family, count, length, seed, stay, period, noise,
                            switching_block, burst_block):
    """Burst corruption changes *observed* bits in complete contiguous blocks.

    The clean Markov stream is generated first. Reusing the same seed across
    noise levels shares both clean bits and block-wise uniform noise draws.
    A block is either entirely flipped or entirely unchanged.
    """
    if family != "burst":
        return generate_training_sequences(
            family,count,length,seed,stay,period,noise,switching_block)
    clean_streams=make_sequences(count,length,seed,stay,0.0)
    rng=random.Random(seed ^ 0x14F31B)
    result=[]
    for clean,_ in clean_streams:
        observed=list(clean)
        for start in range(0,length,burst_block):
            flip=int(rng.random()<noise)
            for t in range(start,min(length,start+burst_block)):
                observed[t] ^= flip
        result.append((clean,observed))
    return result


def work_profile(name, costs, train_sequences, validation_sequences, length,
                 start, candidate_count):
    """Enumerate an intentionally narrow and explicit symbolic work-score.

    Categories are **not** equivalent CPU operations. The score adds one unit
    per counted event as a transparent allocation convention; candidates have
    the same rate-search width but may use very different optimizer schedules.
    """
    if name not in MODELS or candidate_count<1 or not 0<=start<length:
        raise ValueError("invalid work profile request")
    if min(train_sequences,validation_sequences)<1:
        raise ValueError("nonempty independent splits required")
    samples=train_sequences*(length-start)
    val_samples=validation_sequences*(length-start)
    if name=="majority":
        return {
            "shared_feature_events":0,"per_candidate_initialization_slots":0,
            "per_candidate_validation_terms":0,
            "per_candidate_epoch_components":{},
            "per_candidate_epoch_units":0,
            "candidate_count":0,
            "majority_training_target_reads":samples,
        }
    width=costs[name]["readout_feature_width"]
    params=costs[name]["trainable_parameter_slots"]
    n=costs[name]["hidden_width"]
    common=0
    if name in ("spiking_lif","tanh_reservoir"):
        # Fixed recurrence is extracted once for train and validation; LIF
        # counts matrix-row visits here, not multiplication instructions.
        common=n*n*length*(train_sequences+validation_sequences)
    components={}
    if name=="trainable_gru":
        components["recurrent_forward_pair_terms"]=3*n*n*length*train_sequences
        components["recurrent_backward_pair_terms"]=6*n*n*length*train_sequences
    elif name=="trainable_elman":
        components["recurrent_forward_pair_terms"]=n*n*length*train_sequences
        components["recurrent_backward_pair_terms"]=2*n*n*length*train_sequences
    else:
        components["recurrent_forward_pair_terms"]=0
        components["recurrent_backward_pair_terms"]=0
    if name in TRAINED:
        components["readout_scalar_terms"]=n*(length+2*(length-start))*train_sequences
        components["optimizer_parameter_slots"]=params*train_sequences
        # Network validation traverses the whole observed sequence.
        val_terms=(costs[name]["recurrent_pair_visits_per_step"]+n)*length*validation_sequences
    else:
        components["readout_scalar_terms"]=3*width*samples
        components["optimizer_parameter_slots"]=params*samples
        val_terms=width*val_samples
    return {
        "shared_feature_events":common,
        "per_candidate_initialization_slots":params,
        "per_candidate_validation_terms":val_terms,
        "per_candidate_epoch_components":components,
        "per_candidate_epoch_units":sum(components.values()),
        "candidate_count":candidate_count,
        "majority_training_target_reads":0,
    }


def schedule_epochs(name, profile, cap, max_epochs):
    """Choose the largest whole epoch count whose modeled score fits cap."""
    if type(cap) is not int or cap<1 or type(max_epochs) is not int or max_epochs<1:
        raise ValueError("positive integer cap and max_epochs required")
    if name=="majority":
        return {
            "epochs":0,"modeled_selection_units":profile["majority_training_target_reads"],
            "unused_cap":None,"cap_utilization":None,
            "shared_feature_events":0,
            "per_candidate_epoch_components":{},
            "selection_proxy_components":{},
            "budget_applies":False,
        }
    k=profile["candidate_count"]
    overhead=profile["shared_feature_events"]+k*(
        profile["per_candidate_initialization_slots"]+
        profile["per_candidate_validation_terms"])
    per_epoch=k*profile["per_candidate_epoch_units"]
    if per_epoch<=0:
        raise ValueError("zero per-epoch work score")
    room=cap-overhead
    count=min(max_epochs,room//per_epoch) if room>=0 else 0
    if count<1:
        raise ValueError("work cap insufficient for even one epoch of "+name+
                         "; minimum "+str(overhead+per_epoch))
    used=overhead+count*per_epoch
    return {
        "epochs":count,"modeled_selection_units":used,
        "unused_cap":cap-used,"cap_utilization":used/cap,
        "shared_feature_events":profile["shared_feature_events"],
        "per_candidate_epoch_components":profile["per_candidate_epoch_components"],
        "selection_proxy_components":{
            "one_time_feature_events":profile["shared_feature_events"],
            "candidate_initialization_slots":k*profile["per_candidate_initialization_slots"],
            "candidate_validation_terms":k*profile["per_candidate_validation_terms"],
            "candidate_training_units":count*per_epoch,
        },
        "budget_applies":True,
    }


def run_work_budget(config=None):
    cfg=config or WorkBudgetConfig()
    widths=match_recurrent_widths(cfg.pair_visit_budget)
    cost=architecture_costs(widths,max(cfg.delays)+1)
    start=max(cfg.burn_in,max(cfg.delays))
    profiles={
        name:work_profile(name,cost,cfg.train_sequences,cfg.validation_sequences,
                          cfg.sequence_length,start,len(cfg.learning_rates))
        for name in MODELS
    }
    # Plan epoch allocations *before* generating test streams.
    allocations={
        str(cap):{name:schedule_epochs(name,profiles[name],cap,cfg.max_epochs)
                  for name in MODELS}
        for cap in cfg.work_caps
    }
    runs=[]
    for family in cfg.signal_families:
        for seed in cfg.seeds:
            for noise in cfg.noise_levels:
                data={
                    key:generate_work_sequences(
                        family,count,cfg.sequence_length,seed*1009+offset,
                        cfg.stay_probability,cfg.period,noise,
                        cfg.switching_block,cfg.burst_block)
                    for key,count,offset in (
                        ("train",cfg.train_sequences,101),
                        ("validation",cfg.validation_sequences,303),
                        ("test",cfg.test_sequences,202),
                    )
                }
                for delay in cfg.delays:
                    for cap in cfg.work_caps:
                        scores,selected,training={},{},{}
                        for name in MODELS:
                            chosen=allocations[str(cap)][name]
                            opts=SimpleNamespace(
                                epochs=chosen["epochs"],learning_rates=cfg.learning_rates,
                                gradient_clip=cfg.gradient_clip,
                            )
                            accuracy,details,select_ns,eval_ns=fit_and_test(
                                name,opts,seed,delay,data["train"],data["validation"],
                                data["test"],start,widths,max(cfg.delays)+1)
                            scores[name]=accuracy
                            selected[name]=details
                            # Majority is a single training-set vote with no optimizer.
                            stats=effort(name,max(1,chosen["epochs"]),len(cfg.learning_rates),
                                         cfg.train_sequences,cfg.sequence_length,start)
                            training[name]={
                                **stats,
                                "selection_elapsed_ns":select_ns,
                                "heldout_test_evaluation_ns":eval_ns,
                            }
                        runs.append({
                            "signal_family":family,"seed":seed,"noise":noise,
                            "delay":delay,"work_cap":cap,
                            "samples":{
                                "train":cfg.train_sequences*(cfg.sequence_length-start),
                                "validation":cfg.validation_sequences*(cfg.sequence_length-start),
                                "test":cfg.test_sequences*(cfg.sequence_length-start),
                            },
                            "accuracy":scores,"selection":selected,"training_effort":training,
                        })
    summary=[]
    for family in cfg.signal_families:
        for noise in cfg.noise_levels:
            for delay in cfg.delays:
                for cap in cfg.work_caps:
                    group=[r for r in runs if
                           (r["signal_family"],r["noise"],r["delay"],r["work_cap"])==
                           (family,noise,delay,cap)]
                    summary.append({
                        "signal_family":family,"noise":noise,"delay":delay,"work_cap":cap,
                        "paired_seeds":len(group),
                        "accuracy":{
                            model:{
                                "mean":statistics.mean(r["accuracy"][model] for r in group),
                                "population_stdev":statistics.pstdev(r["accuracy"][model] for r in group),
                            } for model in MODELS
                        },
                        "median_selection_elapsed_ns":{
                            model:statistics.median(
                                r["training_effort"][model]["selection_elapsed_ns"]
                                for r in group)
                            for model in MODELS
                        },
                        "paired_spiking_minus_gru":statistics.mean(
                            r["accuracy"]["spiking_lif"]-r["accuracy"]["trainable_gru"]
                            for r in group),
                    })
    return {
        "schema_version":"14.0",
        "project":"PhiNeuBio",
        "substrate":"software_simulation",
        "task":"proxy_budgeted_temporal_training_comparison",
        "configuration":asdict(cfg),
        "architecture_costs":cost,
        "work_profiles":profiles,
        "preselected_allocations":allocations,
        "protocol":{
            "budget":"Predeclared unit-weight sum of non-equivalent symbolic events. Model epoch counts are chosen before tests from estimated selection costs.",
            "honesty":"The proxy does NOT equalize CPU instructions, FLOPs, electricity, actual wall time, optimizer difficulty, or total hardware use.",
            "selection":"Same learning-rate grid; train-only optimization and validation-only selection; test labels never choose budgets or rates.",
            "splits":"Independent seeded complete training, validation and test streams, paired across all models and work caps.",
            "burst":"Clean sticky Markov signal with contiguous observation corruption blocks; clean and noise draws nested across noise levels.",
            "cost_components":"Recurrent matrix terms, backward terms, readout scalar terms, optimizer parameter slots, one-time fixed-feature traversal, candidate validation terms.",
            "majority":"Non-learning training majority is retained but not placed under neural training-work budget.",
            "timing":"Phase 13 measured Python selection wall time, reported separately from symbolic score and test evaluation.",
        },
        "runs":runs,"summary":summary,
        "limitations":[
            "Uniform event weights are arbitrary and distinct event types have different machine costs. This is NOT actual compute parity.",
            "Fixed reservoirs extract recurrent features once; GRU and Elman train recurrence via full BPTT.",
            "All candidate rates are charged independently even when one rate wins; logistic-readout models reuse extracted features.",
            "Optimizer epochs and exact actual Python training time differ across architectures; unused cap is always reported.",
            "Synthetic signals and small seed counts cannot establish general superiority, consciousness, or biological tissue effects.",
            "No electrical energy, instruction counter or hardware power measurement.",
        ],
    }


def write_csv(report,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    cols=("signal_family","seed","noise","delay","work_cap","model","epochs",
          "modeled_selection_units","unused_cap","test_accuracy","optimizer_updates",
          "selection_elapsed_ns")
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=cols);writer.writeheader()
        for run in report["runs"]:
            allocation=report["preselected_allocations"][str(run["work_cap"])]
            for name in MODELS:
                budget=allocation[name]
                effort=run["training_effort"][name]
                writer.writerow({
                    "signal_family":run["signal_family"],"seed":run["seed"],
                    "noise":run["noise"],"delay":run["delay"],"work_cap":run["work_cap"],
                    "model":name,"epochs":budget["epochs"],
                    "modeled_selection_units":budget["modeled_selection_units"],
                    "unused_cap":"" if budget["unused_cap"] is None else budget["unused_cap"],
                    "test_accuracy":run["accuracy"][name],
                    "optimizer_updates":effort["optimizer_updates"],
                    "selection_elapsed_ns":effort["selection_elapsed_ns"],
                })


def main(argv=None):
    parser=argparse.ArgumentParser(description="PhiNeuBio Phase 14 modeled training budget study")
    parser.add_argument("--work-caps",type=int,nargs="+",default=[120000,240000])
    parser.add_argument("--max-epochs",type=int,default=8)
    parser.add_argument("--seed-start",type=int,default=369)
    parser.add_argument("--seed-count",type=int,default=2)
    parser.add_argument("--signal-families",nargs="+",default=["markov","burst"])
    parser.add_argument("--noise-levels",type=float,nargs="+",default=[.15])
    parser.add_argument("--delays",type=int,nargs="+",default=[2])
    parser.add_argument("--pair-visit-budget",type=int,default=75)
    parser.add_argument("--train-sequences",type=int,default=6)
    parser.add_argument("--validation-sequences",type=int,default=3)
    parser.add_argument("--test-sequences",type=int,default=3)
    parser.add_argument("--sequence-length",type=int,default=24)
    parser.add_argument("--burn-in",type=int,default=6)
    parser.add_argument("--burst-block",type=int,default=4)
    parser.add_argument("--json-output",type=Path)
    parser.add_argument("--csv-output",type=Path)
    a=parser.parse_args(argv)
    try:
        if not 1<=a.seed_count<=12:
            raise ValueError("seed-count must be 1-12")
        cfg=WorkBudgetConfig(
            work_caps=tuple(a.work_caps),max_epochs=a.max_epochs,
            seeds=tuple(range(a.seed_start,a.seed_start+a.seed_count)),
            signal_families=tuple(a.signal_families),noise_levels=tuple(a.noise_levels),
            delays=tuple(a.delays),pair_visit_budget=a.pair_visit_budget,
            train_sequences=a.train_sequences,validation_sequences=a.validation_sequences,
            test_sequences=a.test_sequences,sequence_length=a.sequence_length,
            burn_in=a.burn_in,burst_block=a.burst_block,
        )
        report=run_work_budget(cfg)
    except ValueError as error:
        parser.error(str(error))
    output=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if a.json_output:
        a.json_output.parent.mkdir(parents=True,exist_ok=True)
        a.json_output.write_text(output,encoding="utf-8")
        print("Saved Phase 14 report: "+str(a.json_output))
    else:
        print(output,end="")
    if a.csv_output:
        write_csv(report,a.csv_output)
        print("Saved Phase 14 CSV: "+str(a.csv_output),file=sys.stderr)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
