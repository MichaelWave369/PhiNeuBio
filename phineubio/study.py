"""Phase 10: preregistered sample-size study, paired tests, multiplicity control.

This module runs Phase 8 models on nested training budgets while holding every
seed's validation/test/probe streams fixed. Comparisons are predeclared; the
audit uses whole independent seeds, not individual time steps.

No biological tissue, physical energy measurement, or hardware parity claim.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .audit import COMPARISONS, MIN_INTERVAL_SEEDS, paired_interval
from .fairness import FAMILIES, MODELS, FairnessConfig, run_fairness


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...] = (369, 370)
    train_sizes: tuple[int, ...] = (4, 8)
    signal_families: tuple[str, ...] = ("iid", "markov")
    delays: tuple[int, ...] = (2,)
    noise_levels: tuple[float, ...] = (0.15,)
    stay_probability: float = 0.85
    period: int = 6
    validation_sequences: int = 4
    test_sequences: int = 4
    probe_sequences: int = 2
    sequence_length: int = 24
    burn_in: int = 6
    epochs: int = 6
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    parameter_budget: int = 95
    inference_repeats: int = 1
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 369

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 12 or len(set(self.seeds)) != len(self.seeds) or any(type(x) is not int for x in self.seeds):
            raise ValueError("seeds must be 1-12 unique integers")
        if not self.train_sizes or len(self.train_sizes) > 8 or any(type(x) is not int or x < 1 for x in self.train_sizes):
            raise ValueError("train_sizes must be positive integers")
        if tuple(sorted(set(self.train_sizes))) != self.train_sizes:
            raise ValueError("train_sizes must be sorted and distinct")
        if not self.signal_families or len(set(self.signal_families)) != len(self.signal_families) or any(x not in FAMILIES for x in self.signal_families):
            raise ValueError("signal_families must be distinct supported families")
        if not self.delays or len(set(self.delays)) != len(self.delays) or any(type(x) is not int or x < 1 for x in self.delays):
            raise ValueError("delays must be distinct positive integers")
        if not self.noise_levels or len(set(self.noise_levels)) != len(self.noise_levels) or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= .5 for x in self.noise_levels):
            raise ValueError("noise_levels must be distinct finite values in [0,.5]")
        if not 0.5 <= self.stay_probability <= 1 or not math.isfinite(self.stay_probability):
            raise ValueError("stay_probability must be finite in [.5,1]")
        if self.sequence_length < 3 or not 2 <= self.period <= self.sequence_length:
            raise ValueError("period or sequence_length invalid")
        if max(self.delays) >= self.sequence_length or not 0 <= self.burn_in < self.sequence_length:
            raise ValueError("delay and burn-in must fit inside sequence_length")
        if min(self.validation_sequences, self.test_sequences, self.probe_sequences) < 1:
            raise ValueError("all held-out splits need at least one sequence")
        if not 1 <= self.epochs <= 100 or not 1 <= self.inference_repeats <= 5:
            raise ValueError("epochs and repeats outside allowed range")
        if not self.learning_rates or len(set(self.learning_rates)) != len(self.learning_rates) or len(self.learning_rates) > 12 or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x <= 1 for x in self.learning_rates):
            raise ValueError("learning_rates invalid")
        if type(self.parameter_budget) is not int or not 11 <= self.parameter_budget <= 1500:
            raise ValueError("parameter_budget invalid")
        if type(self.bootstrap_seed) is not int or type(self.bootstrap_resamples) is not int or not 200 <= self.bootstrap_resamples <= 20000:
            raise ValueError("bootstrap settings invalid")
        if self.condition_count > 200:
            raise ValueError("study exceeds 200 model-training condition cells; split into multiple preregistered studies")

    @property
    def condition_count(self) -> int:
        return math.prod((len(self.seeds),len(self.train_sizes),len(self.signal_families),
                          len(self.delays),len(self.noise_levels)))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_plan(path: Path) -> tuple[StudyConfig, dict]:
    """Reject unknown or mutated plan structure; configs are immutable dataclasses."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "research_question", "configuration", "comparisons"}:
        raise ValueError("plan must contain exactly schema_version, research_question, configuration and comparisons")
    if plan["schema_version"] != "10.0" or not isinstance(plan["research_question"], str) or not plan["research_question"].strip():
        raise ValueError("invalid study plan version or question")
    if plan["comparisons"] != [list(pair) for pair in COMPARISONS]:
        raise ValueError("the plan must retain the four predeclared comparisons")
    cfg = plan["configuration"]
    if not isinstance(cfg, dict) or set(cfg) != {f.name for f in fields(StudyConfig)}:
        raise ValueError("configuration fields do not match StudyConfig")
    arrays = {"seeds", "train_sizes", "signal_families", "delays", "noise_levels", "learning_rates"}
    normalized = {name: tuple(value) if name in arrays and isinstance(value, list) else value for name,value in cfg.items()}
    return StudyConfig(**normalized), plan


def exact_two_sided_sign_p(positive: int, negative: int, ties: int = 0) -> float | None:
    """Exact binomial sign test for directional consistency across independent seeds.

    Ties contribute no direction. This tests P(win)=P(loss), not the mean
    difference, and is not a general null test of neural architecture value.
    """
    if any(type(x) is not int or x < 0 for x in (positive,negative,ties)):
        raise ValueError("win/loss/tie counts must be nonnegative integers")
    n = positive + negative
    if n < MIN_INTERVAL_SEEDS:
        return None
    tail = sum(math.comb(n,k) for k in range(min(positive,negative)+1))
    return min(1.0, 2 * tail / (2 ** n))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni familywise correction for predeclared contrasts."""
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("valid nonempty p values required")
    order = sorted(range(len(p_values)), key=lambda i: (p_values[i], i))
    corrected = [0.0]*len(p_values)
    prev = 0.0
    total = len(p_values)
    for rank,i in enumerate(order):
        prev = max(prev,min(1.0, p_values[i] * (total-rank)))
        corrected[i] = prev
    return corrected


def _paired_group(records: list[dict], config: StudyConfig, size: int, family: str,
                  noise: float, delay: int) -> dict:
    group = [r for r in records if r["train_sequences"]==size and r["signal_family"]==family
             and r["noise"]==noise and r["delay"]==delay]
    if set(r["seed"] for r in group) != set(config.seeds) or len(group) != len(config.seeds):
        raise RuntimeError("missing, duplicate or unexpected seed for condition")
    group.sort(key=lambda r:r["seed"])
    if len({r["samples"]["test"] for r in group}) != 1:
        raise RuntimeError("unequal held-out sample counts across seeds")
    contrasts = []
    for i,(a,b) in enumerate(COMPARISONS):
        deltas = [r["metrics"][a]-r["metrics"][b] for r in group]
        effect = paired_interval(deltas, bootstrap_seed=config.bootstrap_seed+i+
                                 len(records)*31+size*997+delay, resamples=config.bootstrap_resamples)
        p = exact_two_sided_sign_p(effect["positive_seed_count"],effect["negative_seed_count"],effect["tie_seed_count"])
        contrasts.append({
            "first":a, "second":b, "per_seed":[{"seed":r["seed"],"difference":d} for r,d in zip(group,deltas)],
            "effect":effect, "exact_sign_p":p, "holm_adjusted_sign_p":None,
        })
    return {
        "train_sequences":size, "signal_family":family, "noise":noise, "delay":delay,
        "paired_seed_count":len(group),
        "test_samples_per_seed":group[0]["samples"]["test"],
        "mean_heldout_accuracy":{model:statistics.mean(r["metrics"][model] for r in group) for model in MODELS},
        "contrasts":contrasts,
    }


def summarize_study(config: StudyConfig, records: list[dict]) -> list[dict]:
    """Audit a complete preregistered result grid and correct the full test family."""
    expected = config.condition_count
    if len(records) != expected:
        raise ValueError("incomplete study: expected "+str(expected)+" records")
    keys = [(r["train_sequences"],r["signal_family"],r["noise"],r["delay"],r["seed"])
            for r in records]
    if len(set(keys)) != expected:
        raise ValueError("duplicate training-budget/seed/condition record")
    if any(not all(isinstance(r.get("metrics"),dict) and
                   type(r["metrics"].get(model)) in (int,float) and
                   math.isfinite(r["metrics"][model]) and 0 <= r["metrics"][model] <= 1
                   for model in MODELS) for r in records):
        raise ValueError("invalid model metric")
    summaries = [
        _paired_group(records,config,size,family,noise,delay)
        for size in config.train_sizes
        for family in config.signal_families
        for noise in config.noise_levels
        for delay in config.delays
    ]
    all_tests = [contrast for cell in summaries for contrast in cell["contrasts"]]
    # Missing/inconclusive tests are retained in the declared family as p=1.
    adjusted = holm_adjust([1.0 if x["exact_sign_p"] is None else x["exact_sign_p"] for x in all_tests])
    for x,value in zip(all_tests,adjusted):
        x["holm_adjusted_sign_p"] = None if x["exact_sign_p"] is None else value
    return summaries


def run_study(config: StudyConfig | None = None, plan: dict | None = None) -> dict:
    cfg = config or StudyConfig()
    if plan is None:
        plan = {
            "schema_version":"10.0",
            "research_question":"How do seed-paired memory effects vary with nested training sample budgets?",
            "configuration":asdict(cfg),
            "comparisons":[list(pair) for pair in COMPARISONS],
        }
    if _canonical(plan["configuration"]) != _canonical(asdict(cfg)):
        raise ValueError("plan and requested study settings disagree")
    records = []
    for size in cfg.train_sizes:
        fair = FairnessConfig(
            seeds=cfg.seeds, delays=cfg.delays, noise_levels=cfg.noise_levels,
            signal_families=cfg.signal_families, stay_probability=cfg.stay_probability,
            period=cfg.period, train_sequences=size,
            validation_sequences=cfg.validation_sequences, test_sequences=cfg.test_sequences,
            probe_sequences=cfg.probe_sequences, sequence_length=cfg.sequence_length,
            burn_in=cfg.burn_in, epochs=cfg.epochs, learning_rates=cfg.learning_rates,
            parameter_budget=cfg.parameter_budget, inference_repeats=cfg.inference_repeats,
        )
        result = run_fairness(fair)
        for row in result["runs"]:
            records.append({
                "train_sequences":size, "signal_family":row["signal_family"],
                "noise":row["noise"],"delay":row["delay"],"seed":row["seed"],
                "samples":row["samples"],
                "metrics":row["metrics"],
                "validation_selection":{
                    name:{
                        "selected_learning_rate":entry["selected_learning_rate"],
                        "validation_cross_entropy":entry["validation_cross_entropy"],
                        "trainable_parameters":entry["trainable_parameters"],
                    } for name,entry in row["selection"].items()
                },
            })
    return {
        "schema_version":"10.0",
        "project":"PhiNeuBio",
        "substrate":"software_simulation",
        "task":"preregistered_sample_efficiency_study",
        "plan_sha256":_digest(plan),
        "plan":plan,
        "protocol":{
            "paired_train_nesting":"Each seed uses a common prefix of pseudorandom training sequences across budgets",
            "fixed_heldout":"Validation, test and inference probe sequence seeds and sizes remain fixed across training budgets",
            "model_selection":"Learning rates selected on independent validation split per model and budget",
            "hypotheses":"All four model pairs across every declared budget, signal, noise and delay",
            "sign_test":"Two-sided exact binomial sign test on seed wins vs losses; tests directional symmetry, not mean accuracy difference",
            "holm":"Step-down correction across the entire predeclared family of contrasts; untestable contrasts conservatively count as p=1",
            "bootstrap":"Seed-paired 95% percentile confidence intervals are descriptive and not multiplicity-adjusted",
            "hardware":"Model training and profiling run in Phase 8 implementation, but no machine-specific timings are used to rank models in this study",
        },
        "records":records,
        "summary":summarize_study(cfg,records),
        "limitations":[
            "No living cells, biological measurements, medicine or consciousness claims.",
            "Training size sweeps reuse nested sample prefixes; tests and validation are held fixed but finite and noisy.",
            "Sign test tests paired directional wins, not mean-effect null, and assumes independent exchangeable seeds under its null.",
            "Holm adjustments apply to sign-test p-values across declared contrasts, not bootstrap interval widths.",
            "Small seed counts provide very low power, and p>0.05 cannot establish equivalence.",
            "Model budgets are still not equal compute, equal parameter or equally tuned architecture searches.",
            "This is synthetic data and a pure-Python implementation; no physical energy metrics or real-world generalization.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        fields=["train_sequences","signal_family","noise","delay","first","second",
                "mean_difference","bootstrap_low","bootstrap_high","sign_p",
                "holm_adjusted_sign_p","paired_seeds"]
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for cell in report["summary"]:
            for contrast in cell["contrasts"]:
                eff=contrast["effect"];bounds=eff["bootstrap_95_interval"]
                writer.writerow({
                    "train_sequences":cell["train_sequences"],"signal_family":cell["signal_family"],
                    "noise":cell["noise"],"delay":cell["delay"],
                    "first":contrast["first"],"second":contrast["second"],
                    "mean_difference":eff["mean_difference"],
                    "bootstrap_low":"" if bounds is None else bounds[0],
                    "bootstrap_high":"" if bounds is None else bounds[1],
                    "sign_p":"" if contrast["exact_sign_p"] is None else contrast["exact_sign_p"],
                    "holm_adjusted_sign_p":"" if contrast["holm_adjusted_sign_p"] is None else contrast["holm_adjusted_sign_p"],
                    "paired_seeds":eff["paired_seed_count"],
                })


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="PhiNeuBio Phase 10 preregistered sample-size study")
    parser.add_argument("--plan",type=Path,help="Use an existing locked study plan JSON")
    parser.add_argument("--seed-start",type=int)
    parser.add_argument("--seed-count",type=int)
    parser.add_argument("--training-sizes",type=int,nargs="+")
    parser.add_argument("--signal-families",nargs="+")
    parser.add_argument("--delays",type=int,nargs="+")
    parser.add_argument("--noise-levels",type=float,nargs="+")
    parser.add_argument("--train-epochs",type=int)
    parser.add_argument("--sequence-length",type=int)
    parser.add_argument("--burn-in",type=int)
    parser.add_argument("--validation-sequences",type=int)
    parser.add_argument("--test-sequences",type=int)
    parser.add_argument("--probe-sequences",type=int)
    parser.add_argument("--parameter-budget",type=int)
    parser.add_argument("--resamples",type=int)
    parser.add_argument("--output",type=Path,help="Write JSON report; otherwise stdout")
    parser.add_argument("--csv-output",type=Path)
    args=parser.parse_args(argv)
    override_names=("seed_start","seed_count","training_sizes","signal_families","delays",
                    "noise_levels","train_epochs","sequence_length","burn_in",
                    "validation_sequences","test_sequences","probe_sequences","parameter_budget","resamples")
    if args.plan and any(getattr(args,k) is not None for k in override_names):
        parser.error("--plan cannot be combined with experiment-setting overrides")
    try:
        if args.plan:
            cfg,plan=load_plan(args.plan)
        else:
            defaults=StudyConfig()
            values=asdict(defaults)
            if args.seed_count is not None or args.seed_start is not None:
                count=args.seed_count if args.seed_count is not None else len(defaults.seeds)
                start=args.seed_start if args.seed_start is not None else defaults.seeds[0]
                values["seeds"]=tuple(range(start,start+count)) if 1 <= count <= 12 else ()
            for flag,attribute in [("training_sizes","train_sizes"),("signal_families","signal_families"),
                                    ("delays","delays"),("noise_levels","noise_levels")]:
                if getattr(args,flag) is not None:
                    values[attribute]=tuple(getattr(args,flag))
            for flag,attribute in [("train_epochs","epochs"),("sequence_length","sequence_length"),
                                    ("burn_in","burn_in"),("validation_sequences","validation_sequences"),
                                    ("test_sequences","test_sequences"),("probe_sequences","probe_sequences"),
                                    ("parameter_budget","parameter_budget"),("resamples","bootstrap_resamples")]:
                if getattr(args,flag) is not None:
                    values[attribute]=getattr(args,flag)
            cfg=StudyConfig(**values)
            plan=None
        report=run_study(cfg,plan)
    except (ValueError,OSError,TypeError,KeyError,json.JSONDecodeError) as error:
        parser.error(str(error))
    payload=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding="utf-8")
        print("Saved Phase 10 study: "+str(args.output))
    else:
        print(payload,end="")
    if args.csv_output:
        write_csv(report,args.csv_output)
        print("Saved study CSV: "+str(args.csv_output),file=__import__("sys").stderr)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
