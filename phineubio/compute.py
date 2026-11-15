"""Phase 11: recurrent-loop-budget benchmarking with independent CPU inference probes.

The deterministic budget is the number of recurrent matrix-entry visits per
input time step. It is NOT an exact FLOP count or physical energy measurement.
GRU uses three gates; LIF's dense Python loop visits zero-weight entries.
All candidates select learning rates on validation, never test or probe data.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from time import get_clock_info, perf_counter_ns

from .competition import TrainableRNN
from .fairness import (
    FAMILIES, MODELS, READOUTS, TRAINED, FairnessConfig,
    generate_sequences, select_readout, select_trainable,
)
from .gru import TrainableGRU
from .reservoir import Reservoir, ReservoirConfig
from .temporal import TanhReservoir, extract_features


RECURRENT = ("trainable_gru", "trainable_elman", "spiking_lif", "tanh_reservoir")


@dataclass(frozen=True)
class ComputeConfig:
    seeds: tuple[int, ...] = (369, 370)
    signal_families: tuple[str, ...] = ("markov",)
    noise_levels: tuple[float, ...] = (0.15,)
    delays: tuple[int, ...] = (2,)
    pair_visit_budget: int = 108
    train_sequences: int = 6
    validation_sequences: int = 3
    test_sequences: int = 3
    probe_sequences: int = 3
    sequence_length: int = 24
    burn_in: int = 6
    epochs: int = 5
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    stay_probability: float = 0.85
    period: int = 6
    inference_repeats: int = 5

    def __post_init__(self) -> None:
        if type(self.pair_visit_budget) is not int or not 3 <= self.pair_visit_budget <= 3072:
            raise ValueError("pair_visit_budget must be an integer in [3,3072]")
        if type(self.inference_repeats) is not int or not 1 <= self.inference_repeats <= 10:
            raise ValueError("inference_repeats must be 1-10")
        if len(self.seeds) * len(self.signal_families) * len(self.noise_levels) * len(self.delays) > 40:
            raise ValueError("study exceeds 40 paired compute conditions; use a smaller grid")
        # Reuse the complete, tested protocol validation including independent splits.
        FairnessConfig(
            seeds=self.seeds, signal_families=self.signal_families,
            noise_levels=self.noise_levels, delays=self.delays,
            stay_probability=self.stay_probability, period=self.period,
            train_sequences=self.train_sequences,
            validation_sequences=self.validation_sequences,
            test_sequences=self.test_sequences,
            probe_sequences=self.probe_sequences,
            sequence_length=self.sequence_length, burn_in=self.burn_in,
            epochs=self.epochs, learning_rates=self.learning_rates,
            inference_repeats=self.inference_repeats,
        )


def match_recurrent_widths(pair_visit_budget: int) -> dict[str, dict]:
    """Largest width (at most 32) that stays under a *structural* loop cap.

    GRU scans three n-by-n matrices; each other recurrent implementation scans
    one. FLOPs and actual CPU instructions per entry are intentionally NOT equal.
    """
    if type(pair_visit_budget) is not int or not 3 <= pair_visit_budget <= 3072:
        raise ValueError("unsupported recurrent pair-visit budget")
    profiles = {}
    for name in RECURRENT:
        gates = 3 if name == "trainable_gru" else 1
        n = min(32, math.isqrt(pair_visit_budget // gates))
        if n < 1:
            raise ValueError("budget too small for GRU")
        visits = gates * n * n
        profiles[name] = {
            "hidden_width": n,
            "recurrent_matrices_visited": gates,
            "recurrent_pair_visits_per_step": visits,
            "budget_cap": pair_visit_budget,
            "budget_utilization": visits / pair_visit_budget,
        }
    return profiles


def architecture_costs(widths: dict[str, dict], history_width: int) -> dict[str, dict]:
    """Symbolic operation *categories*, never assert these are measured FLOPs."""
    profiles = {}
    for name in RECURRENT:
        n = widths[name]["hidden_width"]
        profile = dict(widths[name])
        if name == "trainable_gru":
            trainable = 3*n*n + 7*n + 1
            fixed = 0
            output = n
            threshold_checks = 0
            nonlinearities = 3*n + 1
        elif name == "trainable_elman":
            trainable = n*n + 3*n + 1
            fixed = 0
            output = n
            threshold_checks = 0
            nonlinearities = n + 1
        elif name == "spiking_lif":
            trainable = 2*n + 1
            fixed = n*n + n
            output = 2*n
            threshold_checks = n
            nonlinearities = 1  # logistic readout; the LIF has threshold checks
        else:
            trainable = n + 1
            fixed = n*n + 2*n
            output = n
            threshold_checks = 0
            nonlinearities = n + 1
        profile.update({
            "trainable_parameter_slots": trainable,
            "fixed_parameter_slots": fixed,
            "total_parameter_slots": trainable + fixed,
            "readout_feature_width": output,
            "nonlinear_function_calls_per_step_proxy": nonlinearities,
            "lif_threshold_comparisons_per_step_proxy": threshold_checks,
            "matched_on_recurrent_pair_visits_only": True,
        })
        profiles[name] = profile
    for name, width in (("history_window", history_width),
                        ("delayed_tap", 1), ("majority", 0)):
        profiles[name] = {
            "hidden_width": 0,
            "recurrent_matrices_visited": 0,
            "recurrent_pair_visits_per_step": 0,
            "budget_cap": None,
            "budget_utilization": None,
            "trainable_parameter_slots": width+1 if width else 0,
            "fixed_parameter_slots": 0,
            "total_parameter_slots": width+1 if width else 0,
            "readout_feature_width": width,
            "nonlinear_function_calls_per_step_proxy": 1 if width else 0,
            "lif_threshold_comparisons_per_step_proxy": 0,
            "matched_on_recurrent_pair_visits_only": False,
        }
    return profiles


def _runner(name: str, model, seed: int, delay: int, probe, start: int, size: int):
    """Construct inference state outside the timed region and reuse the model.

    The pre-instantiated reservoir state resets per sequence. All models
    process the identical observed probe streams with no target labels used.
    """
    network = None
    if name == "spiking_lif":
        network = Reservoir(ReservoirConfig(seed=seed, neurons=size))
    elif name == "tanh_reservoir":
        network = TanhReservoir(neurons=size, seed=seed)

    def execute():
        predictions = 0
        for _clean, observed in probe:
            if name in TRAINED:
                probabilities = model.forward(observed)[1]
                predictions += sum(int(value >= 0.5) for value in probabilities[start:])
            elif network is not None:
                network.reset()
                for t, bit in enumerate(observed):
                    feature = network.step(float(bit))
                    if t >= start:
                        predictions += model.predict(feature)
            elif name == "history_window":
                for t in range(start, len(observed)):
                    feature = tuple(float(observed[t-k]) for k in range(size))
                    predictions += model.predict(feature)
            elif name == "delayed_tap":
                for t in range(start, len(observed)):
                    predictions += model.predict((float(observed[t-delay]),))
            else:
                for t in range(start, len(observed)):
                    predictions += int(model)
        return predictions
    return execute


def benchmark_probes(runners: dict[str, object], repeats: int, seed: int,
                     samples_per_repeat: int) -> tuple[dict, list[list[str]]]:
    """Seeded round-by-round model-order randomization; no training or test calls."""
    if set(runners) != set(MODELS) or not 1 <= repeats <= 10 or samples_per_repeat < 1:
        raise ValueError("incomplete runners or invalid profiling configuration")
    # Warm-up each model once before collecting measurements.
    for name in MODELS:
        runners[name]()
    measurements = {name: [] for name in MODELS}
    result_checksums = {name: [] for name in MODELS}
    order_rng = random.Random(seed)
    round_order = []
    for _ in range(repeats):
        order = list(MODELS)
        order_rng.shuffle(order)
        round_order.append(order)
        for name in order:
            gc.collect()  # outside timed region
            before = perf_counter_ns()
            checksum = runners[name]()
            elapsed = max(1, perf_counter_ns() - before)
            measurements[name].append(elapsed)
            result_checksums[name].append(checksum)
    profiles = {}
    for name in MODELS:
        if len(set(result_checksums[name])) != 1:
            raise RuntimeError("model inference changed across repeated identical probes")
        profiles[name] = {
            "elapsed_ns": measurements[name],
            "median_elapsed_ns": statistics.median(measurements[name]),
            "retained_samples_per_repeat": samples_per_repeat,
            "prediction_checksum": result_checksums[name][0],
        }
    return profiles, round_order


def _one_condition(cfg: ComputeConfig, family: str, seed: int, noise: float,
                   delay: int, widths: dict) -> dict:
    start = max(cfg.burn_in, max(cfg.delays))
    streams = {
        label: generate_sequences(family, count, cfg.sequence_length,
                                  seed*1009 + offset, cfg.stay_probability,
                                  cfg.period, noise)
        for label, count, offset in (
            ("train", cfg.train_sequences, 101),
            ("validation", cfg.validation_sequences, 303),
            ("test", cfg.test_sequences, 202),
            ("probe", cfg.probe_sequences, 404),
        )
    }
    fair_cfg = FairnessConfig(
        seeds=(seed,), delays=cfg.delays, noise_levels=(noise,),
        signal_families=(family,), stay_probability=cfg.stay_probability,
        period=cfg.period, train_sequences=cfg.train_sequences,
        validation_sequences=cfg.validation_sequences,
        test_sequences=cfg.test_sequences,
        probe_sequences=cfg.probe_sequences,
        sequence_length=cfg.sequence_length, burn_in=cfg.burn_in,
        epochs=cfg.epochs, learning_rates=cfg.learning_rates,
        inference_repeats=cfg.inference_repeats,
    )
    history_width = max(cfg.delays) + 1
    models, selection, accuracy = {}, {}, {}
    for name in MODELS:
        if name in TRAINED:
            model, info = select_trainable(name, fair_cfg, seed, delay,
                                           streams["train"], streams["validation"],
                                           start, widths)
            test_accuracy = model.accuracy(streams["test"], delay, start)
        elif name in READOUTS:
            size = widths[name]["hidden_width"] if name in RECURRENT else history_width
            model, info = select_readout(name, fair_cfg, seed, delay,
                                         streams["train"], streams["validation"],
                                         start, size)
            inputs, targets = extract_features(name, streams["test"], delay, start,
                                               size, size, seed)
            test_accuracy = model.score(inputs, targets)
        else:
            labels = [clean[t-delay] for clean, _ in streams["train"]
                      for t in range(start, cfg.sequence_length)]
            model = int(2*sum(labels) >= len(labels))
            targets = [clean[t-delay] for clean, _ in streams["test"]
                       for t in range(start, cfg.sequence_length)]
            test_accuracy = sum(y == model for y in targets) / len(targets)
            info = {"selected_learning_rate": None,
                    "validation_cross_entropy": None,
                    "trainable_parameters": 0}
        models[name] = model
        selection[name] = info
        accuracy[name] = test_accuracy
    runners = {
        name: _runner(name, model, seed, delay, streams["probe"], start,
                      widths[name]["hidden_width"] if name in RECURRENT
                      else history_width if name=="history_window" else 1)
        for name, model in models.items()
    }
    times, orders = benchmark_probes(
        runners, cfg.inference_repeats,
        seed=seed*1013 + delay*37 + int(noise*1000),
        samples_per_repeat=cfg.probe_sequences*(cfg.sequence_length-start),
    )
    return {
        "signal_family": family, "seed": seed, "noise": noise, "delay": delay,
        "samples": {
            "train": cfg.train_sequences*(cfg.sequence_length-start),
            "validation": cfg.validation_sequences*(cfg.sequence_length-start),
            "test": cfg.test_sequences*(cfg.sequence_length-start),
            "probe": cfg.probe_sequences*(cfg.sequence_length-start),
        },
        "accuracy": accuracy,
        "selected_validation": selection,
        "inference": times,
        "round_order": orders,
    }


def run_compute(config: ComputeConfig | None = None) -> dict:
    cfg = config or ComputeConfig()
    widths = match_recurrent_widths(cfg.pair_visit_budget)
    costs = architecture_costs(widths, max(cfg.delays)+1)
    runs = [
        _one_condition(cfg, family, seed, noise, delay, widths)
        for family in cfg.signal_families
        for seed in cfg.seeds
        for noise in cfg.noise_levels
        for delay in cfg.delays
    ]
    summaries = []
    for family in cfg.signal_families:
        for noise in cfg.noise_levels:
            for delay in cfg.delays:
                rows = [row for row in runs if row["signal_family"]==family
                        and row["noise"]==noise and row["delay"]==delay]
                summaries.append({
                    "signal_family": family, "noise": noise, "delay": delay,
                    "seeds": len(rows),
                    "accuracy": {
                        name: {
                            "mean": statistics.mean(r["accuracy"][name] for r in rows),
                            "population_stdev": statistics.pstdev(r["accuracy"][name] for r in rows),
                        } for name in MODELS
                    },
                    "inference_median_ns_across_seed_medians": {
                        name: statistics.median(r["inference"][name]["median_elapsed_ns"] for r in rows)
                        for name in MODELS
                    },
                    "paired_accuracy_deltas": {
                        "spiking_minus_gru": statistics.mean(r["accuracy"]["spiking_lif"]-r["accuracy"]["trainable_gru"] for r in rows),
                        "spiking_minus_tanh": statistics.mean(r["accuracy"]["spiking_lif"]-r["accuracy"]["tanh_reservoir"] for r in rows),
                    },
                })
    return {
        "schema_version": "11.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "recurrent_pair_budget_cpu_probe",
        "configuration": asdict(cfg),
        "recurrent_widths": widths,
        "structural_cost_profiles": costs,
        "runtime_environment": {
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "os_family": platform.system(),
            "machine_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "timer_resolution_seconds": get_clock_info("perf_counter").resolution,
        },
        "protocol": {
            "structural_match": "Cap on recurrent matrix-entry visits per input timestep only; GRU has 3 matrices, other recurrent models 1",
            "structural_match_limits": "Different nonlinearities, arithmetic, thresholds, conditionals and Python implementations: not exact equal FLOPs, instructions or energy",
            "storage": "Total, fixed and trainable parameter slots reported independently; no equal parameter claim",
            "selection": "All trainable systems and fixed readouts select learning rate using only separate validation split",
            "held_out": "All seven models share paired complete test streams; test labels never select hyperparameters",
            "probe": "Separate input-only probe sequences; training, test and validation excluded from timing",
            "timing": "Preconstructed model/recurrent state, one untimed warmup per model; seeded randomized model order per round, perf_counter_ns measurements",
            "rounds": "Per-round order recorded. gc.collect outside timer before each measured inference.",
            "scope": "End-to-end Python forward inference, including burn-in and readout, but not training or architecture/model construction",
            "environment": "All models in one Python process and CPU execution environment; results machine/load/order sensitive",
        },
        "runs": runs,
        "summary": summaries,
        "limitations": [
            "A matched structural recurrent pair-visit budget is only a proxy; actual CPU instructions, FLOPs and branching differ.",
            "GRU and Elman train recurrent weights while the LIF and tanh reservoir only train readouts; training-compute budgets are not matched.",
            "Explicit memory baselines and majority do not have recurrent pair visits and are deliberately unmatched conventional controls.",
            "Actual elapsed Python timings may change under load, interpreter and machine. They are not watts, joules or hardware energy efficiency.",
            "Only a small seed sweep and synthetic binary signals; neither statistical significance nor broad superiority is established.",
            "No living neural tissue, medical intervention or evidence of consciousness.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:
        fields = ["signal_family","seed","noise","delay","model","test_accuracy",
                  "recurrent_pair_visits_per_step","trainable_parameter_slots",
                  "fixed_parameter_slots","median_probe_elapsed_ns"]
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for run in report["runs"]:
            for name in MODELS:
                cost=report["structural_cost_profiles"][name]
                writer.writerow({
                    "signal_family":run["signal_family"],"seed":run["seed"],
                    "noise":run["noise"],"delay":run["delay"],"model":name,
                    "test_accuracy":run["accuracy"][name],
                    "recurrent_pair_visits_per_step":cost["recurrent_pair_visits_per_step"],
                    "trainable_parameter_slots":cost["trainable_parameter_slots"],
                    "fixed_parameter_slots":cost["fixed_parameter_slots"],
                    "median_probe_elapsed_ns":run["inference"][name]["median_elapsed_ns"],
                })


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="PhiNeuBio Phase 11 recurrent-work and CPU budget lab")
    parser.add_argument("--seed-start",type=int,default=369)
    parser.add_argument("--seed-count",type=int,default=2)
    parser.add_argument("--signal-families",nargs="+",default=["markov"])
    parser.add_argument("--noise-levels",type=float,nargs="+",default=[0.15])
    parser.add_argument("--delays",type=int,nargs="+",default=[2])
    parser.add_argument("--pair-visit-budget",type=int,default=108)
    parser.add_argument("--train-sequences",type=int,default=6)
    parser.add_argument("--validation-sequences",type=int,default=3)
    parser.add_argument("--test-sequences",type=int,default=3)
    parser.add_argument("--probe-sequences",type=int,default=3)
    parser.add_argument("--sequence-length",type=int,default=24)
    parser.add_argument("--burn-in",type=int,default=6)
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--inference-repeats",type=int,default=5)
    parser.add_argument("--json-output",type=Path)
    parser.add_argument("--csv-output",type=Path)
    args=parser.parse_args(argv)
    try:
        if not 1 <= args.seed_count <= 20:
            raise ValueError("seed-count must be in [1,20]")
        cfg=ComputeConfig(
            seeds=tuple(range(args.seed_start,args.seed_start+args.seed_count)),
            signal_families=tuple(args.signal_families),
            noise_levels=tuple(args.noise_levels),
            delays=tuple(args.delays),
            pair_visit_budget=args.pair_visit_budget,
            train_sequences=args.train_sequences,
            validation_sequences=args.validation_sequences,
            test_sequences=args.test_sequences,
            probe_sequences=args.probe_sequences,
            sequence_length=args.sequence_length,
            burn_in=args.burn_in,epochs=args.epochs,
            inference_repeats=args.inference_repeats,
        )
        report=run_compute(cfg)
    except ValueError as error:
        parser.error(str(error))
    text=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(text,encoding="utf-8")
        print("Saved compute report: "+str(args.json_output))
    else:
        print(text,end="")
    if args.csv_output:
        write_csv(report,args.csv_output)
        print("Saved CPU comparison CSV: "+str(args.csv_output),file=sys.stderr)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
