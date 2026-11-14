"""Phase 8: independently validation-tuned digital memory models and three signal families.

No model or candidate uses final test labels to choose learning rate. Timing is
measured on an independent, unlabeled inference probe rather than the test set.
Everything here is software simulation, not biological computing.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
import platform
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns, get_clock_info

from .competition import TrainableRNN
from .gru import TrainableGRU
from .readout import LogisticReadout
from .temporal import extract_features, make_sequences
from .validation import validation_loss, widths_under_budget


TRAINED = ("trainable_gru", "trainable_elman")
READOUTS = ("spiking_lif", "tanh_reservoir", "history_window", "delayed_tap")
MODELS = TRAINED + READOUTS + ("majority",)
FAMILIES = ("markov", "periodic", "iid")


@dataclass(frozen=True)
class FairnessConfig:
    seeds: tuple[int, ...] = (369, 370)
    delays: tuple[int, ...] = (1, 3)
    noise_levels: tuple[float, ...] = (0.0, 0.15)
    signal_families: tuple[str, ...] = FAMILIES
    stay_probability: float = 0.85
    period: int = 6
    train_sequences: int = 8
    validation_sequences: int = 4
    test_sequences: int = 4
    probe_sequences: int = 3
    sequence_length: int = 24
    burn_in: int = 6
    epochs: int = 6
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    gradient_clip: float = 5.0
    parameter_budget: int = 95
    inference_repeats: int = 3

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 20 or len(set(self.seeds)) != len(self.seeds) or any(type(x) is not int for x in self.seeds):
            raise ValueError("seeds must be distinct integers")
        if not self.delays or len(self.delays) > 20 or len(set(self.delays)) != len(self.delays) or any(type(x) is not int or x < 1 for x in self.delays):
            raise ValueError("delays must be distinct positive integers")
        if not self.noise_levels or len(self.noise_levels) > 10 or len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise levels must be distinct")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 0.5 for x in self.noise_levels):
            raise ValueError("noise levels must be finite in [0,0.5]")
        if not self.signal_families or len(set(self.signal_families)) != len(self.signal_families) or any(family not in FAMILIES for family in self.signal_families):
            raise ValueError("unknown or duplicated signal family")
        if not math.isfinite(self.stay_probability) or not 0.5 <= self.stay_probability <= 1:
            raise ValueError("stay probability must be in [0.5,1]")
        if self.sequence_length < 3 or not 2 <= self.period <= self.sequence_length:
            raise ValueError("period must fit inside sequence_length")
        if min(self.train_sequences, self.validation_sequences, self.test_sequences, self.probe_sequences) < 1:
            raise ValueError("each split requires sequences")
        if not 0 <= self.burn_in < self.sequence_length or max(self.delays) >= self.sequence_length:
            raise ValueError("burn-in and delays must fit the sequence")
        if not 1 <= self.epochs <= 1000 or not 1 <= self.inference_repeats <= 10:
            raise ValueError("epochs and inference_repeats out of range")
        if not self.learning_rates or len(self.learning_rates) > 12 or len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("learning rates must be distinct")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 < x <= 1 for x in self.learning_rates):
            raise ValueError("learning rates must be finite in (0,1]")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient clip must be finite and positive")
        if type(self.parameter_budget) is not int or not 11 <= self.parameter_budget <= 1500:
            raise ValueError("unsupported parameter budget")


def generate_sequences(
    family: str, count: int, length: int, seed: int, stay: float, period: int, noise: float
) -> list[tuple[list[int], list[int]]]:
    """Distinct deterministic streams. Noise probability only changes observed bits."""
    if family == "markov":
        return make_sequences(count, length, seed, stay, noise)
    if family == "iid":
        rng = random.Random(seed)
        sequences = []
        for _ in range(count):
            clean = [rng.randrange(2) for _ in range(length)]
            observed = [bit ^ int(rng.random() < noise) for bit in clean]
            sequences.append((clean, observed))
        return sequences
    if family != "periodic":
        raise ValueError("unknown signal family")
    rng = random.Random(seed)
    sequences = []
    for _ in range(count):
        template = [rng.randrange(2) for _ in range(period)]
        phase = rng.randrange(period)
        clean = [template[(phase + t) % period] for t in range(length)]
        observed = [bit ^ int(rng.random() < noise) for bit in clean]
        sequences.append((clean, observed))
    return sequences


def select_trainable(name, cfg, seed, delay, train, validation, start, sizes):
    cls = TrainableGRU if name == "trainable_gru" else TrainableRNN
    best, winning = float("inf"), None
    attempts = []
    for rate in cfg.learning_rates:
        net = cls(sizes[name]["hidden_width"], seed)
        curve = net.train(train, delay, start, cfg.epochs, rate,
                          cfg.gradient_clip, seed * 1009 + delay * 307)
        score = validation_loss(net, validation, delay, start)
        attempts.append({"learning_rate": rate, "validation_cross_entropy": score})
        if score < best:
            best, winning = score, (net, rate, curve)
    net, rate, curve = winning
    return net, {
        "selected_learning_rate": rate,
        "validation_cross_entropy": best,
        "candidate_validation_scores": attempts,
        "training_loss_by_epoch": curve,
        "trainable_parameters": net.parameter_count,
        "hidden_width": net.neurons,
    }


def select_readout(name, cfg, seed, delay, train, validation, start, width):
    train_x, train_y = extract_features(name, train, delay, start, width, width, seed)
    val_x, val_y = extract_features(name, validation, delay, start, width, width, seed)
    order = list(range(len(train_y)))
    random.Random(seed * 1009 + delay * 307).shuffle(order)
    best, winning = float("inf"), None
    attempts = []
    for rate in cfg.learning_rates:
        model = LogisticReadout()
        model.fit([train_x[i] for i in order], [train_y[i] for i in order],
                  epochs=cfg.epochs, learning_rate=rate)
        score = -sum(
            y * math.log(max(1e-12, min(1 - 1e-12, model.predict_proba(x))))
            + (1 - y) * math.log(max(1e-12, min(1 - 1e-12, 1 - model.predict_proba(x))))
            for x, y in zip(val_x, val_y)
        ) / len(val_y)
        attempts.append({"learning_rate": rate, "validation_cross_entropy": score})
        if score < best:
            best, winning = score, (model, rate)
    model, rate = winning
    return model, {
        "selected_learning_rate": rate,
        "validation_cross_entropy": best,
        "candidate_validation_scores": attempts,
        "trainable_parameters": len(model.weights) + 1,
        "feature_width": len(model.weights),
    }


def profile_inference(name, model, cfg, seed, delay, probe, start, width):
    """Repeated test-independent inference. Includes feature extraction for baselines."""
    timings = []
    count = len(probe) * (cfg.sequence_length - start)
    for _ in range(cfg.inference_repeats):
        gc.collect()
        begin = perf_counter_ns()
        if name in TRAINED:
            for _, observed in probe:
                model.forward(observed)
        elif name in READOUTS:
            features, _ = extract_features(name, probe, delay, start, width, width, seed)
            for row in features:
                model.predict(row)
        else:
            for _clean, observed in probe:
                for t in range(start, len(observed)):
                    pass  # A constant-prediction majority baseline has no feature calculation.
        elapsed = perf_counter_ns() - begin
        timings.append(max(1, elapsed))
    return {
        "elapsed_ns": timings,
        "median_elapsed_ns": statistics.median(timings),
        "retained_samples_per_repeat": count,
        "scope": "Python end-to-end inference; baseline includes feature extraction, trained RNN forward includes burn-in",
    }


def run_fairness(config: FairnessConfig | None = None) -> dict:
    cfg = config or FairnessConfig()
    sizes = widths_under_budget(cfg.parameter_budget)
    width = sizes["trainable_elman"]["hidden_width"]
    start = max(cfg.burn_in, max(cfg.delays))
    rows = []
    for family in cfg.signal_families:
        for seed in cfg.seeds:
            for noise in cfg.noise_levels:
                streams = {
                    "train": generate_sequences(family, cfg.train_sequences, cfg.sequence_length, seed * 1009 + 101, cfg.stay_probability, cfg.period, noise),
                    "validation": generate_sequences(family, cfg.validation_sequences, cfg.sequence_length, seed * 1009 + 303, cfg.stay_probability, cfg.period, noise),
                    "test": generate_sequences(family, cfg.test_sequences, cfg.sequence_length, seed * 1009 + 202, cfg.stay_probability, cfg.period, noise),
                    "probe": generate_sequences(family, cfg.probe_sequences, cfg.sequence_length, seed * 1009 + 404, cfg.stay_probability, cfg.period, noise),
                }
                for delay in cfg.delays:
                    metrics, selections, timings = {}, {}, {}
                    for name in MODELS:
                        if name in TRAINED:
                            model, selection = select_trainable(
                                name, cfg, seed, delay, streams["train"],
                                streams["validation"], start, sizes,
                            )
                            accuracy = model.accuracy(streams["test"], delay, start)
                        elif name in READOUTS:
                            model, selection = select_readout(
                                name, cfg, seed, delay, streams["train"],
                                streams["validation"], start, width,
                            )
                            test_x, test_y = extract_features(
                                name, streams["test"], delay, start, width, width, seed
                            )
                            accuracy = model.score(test_x, test_y)
                        else:
                            training_labels = [
                                clean[t - delay] for clean, _ in streams["train"]
                                for t in range(start, cfg.sequence_length)
                            ]
                            majority = int(sum(training_labels) * 2 >= len(training_labels))
                            targets = [
                                clean[t - delay] for clean, _ in streams["test"]
                                for t in range(start, cfg.sequence_length)
                            ]
                            accuracy = sum(target == majority for target in targets) / len(targets)
                            model = majority
                            selection = {
                                "selected_learning_rate": None,
                                "validation_cross_entropy": None,
                                "candidate_validation_scores": [],
                                "trainable_parameters": 0,
                            }
                        metrics[name] = accuracy
                        selections[name] = selection
                        timings[name] = profile_inference(
                            name, model, cfg, seed, delay, streams["probe"], start, width
                        )
                    rows.append({
                        "signal_family": family, "seed": seed, "noise": noise, "delay": delay,
                        "samples": {
                            "train": cfg.train_sequences * (cfg.sequence_length - start),
                            "validation": cfg.validation_sequences * (cfg.sequence_length - start),
                            "test": cfg.test_sequences * (cfg.sequence_length - start),
                            "profiling_probe": cfg.probe_sequences * (cfg.sequence_length - start),
                        },
                        "metrics": metrics, "selection": selections, "inference_profiling": timings,
                    })
    summaries = []
    for family in cfg.signal_families:
        for noise in cfg.noise_levels:
            for delay in cfg.delays:
                chosen = [row for row in rows if row["signal_family"] == family and row["noise"] == noise and row["delay"] == delay]
                summaries.append({
                    "signal_family": family, "noise": noise, "delay": delay,
                    "seeds": len(chosen),
                    "metrics": {
                        model: {
                            "mean": statistics.mean(row["metrics"][model] for row in chosen),
                            "population_stdev": statistics.pstdev(row["metrics"][model] for row in chosen),
                        } for model in MODELS
                    },
                    "paired_deltas": {
                        "spiking_minus_gru": statistics.mean(row["metrics"]["spiking_lif"] - row["metrics"]["trainable_gru"] for row in chosen),
                        "spiking_minus_history": statistics.mean(row["metrics"]["spiking_lif"] - row["metrics"]["history_window"] for row in chosen),
                    },
                })
    return {
        "schema_version": "8.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "validation_tuned_multisignal_comparison",
        "configuration": asdict(cfg),
        "runtime_environment": {
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "os_family": platform.system(),
            "machine_architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "timer_resolution_seconds": get_clock_info("perf_counter").resolution,
        },
        "parameter_budget": sizes,
        "models": {
            "trainable_gru": "Trained GRU; validation-selected rate; full BPTT",
            "trainable_elman": "Trained Elman tanh RNN; validation-selected rate; full BPTT",
            "spiking_lif": "Fixed LIF reservoir; validation-selected logistic readout",
            "tanh_reservoir": "Fixed tanh reservoir; validation-selected logistic readout",
            "history_window": "Explicit input-history window; validation-selected logistic readout",
            "delayed_tap": "Explicit observed delayed bit; validation-selected logistic readout",
            "majority": "Training-set majority label, no hyperparameters",
        },
        "protocol": {
            "splits": "Four independent seeded sequence streams: train, validation, test, profiling_probe",
            "test_usage": "Test accuracy evaluated once per selected model. Probe timing uses separate streams and never selects hyperparameters.",
            "selection": "All trainable models, including reservoir logistic readouts, select learning rate on validation binary cross-entropy only",
            "signals": "markov: sticky first-order binary process; periodic: random repeated binary template with random phase per sequence; iid: independent fair random input bits",
            "paired": "Same signals, corruption and time indices across models within each family/seed/noise/delay",
            "parameters": "GRU and Elman capped trainable parameters; reservoir fixed matrices, different dimensions and nonmatched operations",
            "profiling": "Independent probe, repeated perf_counter_ns timings on the current Python runtime; median reported, not energy or standardized speed",
            "tuning": "Small common learning-rate grid; training schedules and candidate architecture capacities differ",
        },
        "runs": rows,
        "summary": summaries,
        "limitations": [
            "Digital simulation only; no actual neurons, neurological treatment or consciousness evidence.",
            "All learning-rate-tunable baselines now use independent validation, but models are NOT equal parameter, equal compute, or equally hyperparameter tuned.",
            "Periodic templates may differ in complexity across seeds. Three toy signal families do not imply general task superiority.",
            "Testing on independent sequences reduces leakage; small seed sweeps provide only descriptive variation.",
            "Python profiling uses serial execution, different runtime implementations and independent probe inputs. It does not establish hardware speed or energy efficiency.",
            "No physical biological hardware interfaces or experimental wet-lab controls are included.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["signal_family", "seed", "noise", "delay", "train_samples", "validation_samples", "test_samples"]
        fields += [f"{model}_accuracy" for model in MODELS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            data = {key: row[key] for key in ("signal_family", "seed", "noise", "delay")}
            data.update({f"{key}_samples": row["samples"][key] for key in ("train", "validation", "test")})
            data.update({f"{model}_accuracy": row["metrics"][model] for model in MODELS})
            writer.writerow(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio Phase 8: fairness and multiple signals")
    parser.add_argument("--seed-start", type=int, default=369)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.15])
    parser.add_argument("--signal-families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--validation-sequences", type=int, default=4)
    parser.add_argument("--test-sequences", type=int, default=4)
    parser.add_argument("--probe-sequences", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--burn-in", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.03, 0.1])
    parser.add_argument("--parameter-budget", type=int, default=95)
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = FairnessConfig(
            seeds=tuple(args.seed_start + i for i in range(args.seed_count)),
            delays=tuple(args.delays), noise_levels=tuple(args.noise_levels),
            signal_families=tuple(args.signal_families),
            train_sequences=args.train_sequences,
            validation_sequences=args.validation_sequences,
            test_sequences=args.test_sequences, probe_sequences=args.probe_sequences,
            sequence_length=args.sequence_length, burn_in=args.burn_in,
            epochs=args.epochs, learning_rates=tuple(args.learning_rates),
            parameter_budget=args.parameter_budget, inference_repeats=args.inference_repeats,
        )
        report = run_fairness(cfg)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
        print("Saved Phase 8 report: " + str(args.json_output))
    else:
        print(payload, end="")
    if args.csv_output:
        write_csv(report, args.csv_output)
        print("Saved CSV: " + str(args.csv_output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
