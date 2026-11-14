"""Reproducible correlated-signal memory comparison with instrumented Python profiling.

The LIF and tanh comparators are FIXED random recurrent reservoirs with
trainable logistic readouts. The tanh network is NOT a fully-trained RNN.
Profiling numbers are local Python-process observations, not energy readings.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns
import tracemalloc
from typing import Sequence

from .readout import LogisticReadout
from .reservoir import Reservoir, ReservoirConfig


MODELS = ("spiking_lif", "tanh_reservoir", "history_window", "delayed_tap", "majority")


@dataclass(frozen=True)
class TemporalConfig:
    seeds: tuple[int, ...] = (369, 370, 371)
    delays: tuple[int, ...] = (1, 3, 6)
    noise_levels: tuple[float, ...] = (0.0, 0.15)
    stay_probability: float = 0.85
    train_sequences: int = 12
    test_sequences: int = 6
    sequence_length: int = 32
    burn_in: int = 8
    neurons: int = 16
    epochs: int = 10
    learning_rate: float = 0.035
    window_size: int | None = None

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 30 or any(type(i) is not int for i in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain 1-30 distinct integers")
        if not self.delays or len(self.delays) > 30 or any(type(i) is not int or i < 1 for i in self.delays) or len(set(self.delays)) != len(self.delays):
            raise ValueError("delays must contain 1-30 distinct positive integers")
        if not self.noise_levels or len(self.noise_levels) > 12 or len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise_levels must be 1-12 distinct values")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 0.5 for x in self.noise_levels):
            raise ValueError("noise probabilities must be finite in [0, 0.5]")
        if not isinstance(self.stay_probability, (int, float)) or not math.isfinite(self.stay_probability) or not 0.5 <= self.stay_probability <= 1:
            raise ValueError("stay_probability must be finite in [0.5, 1]")
        if self.train_sequences < 1 or self.test_sequences < 1 or self.sequence_length < 2:
            raise ValueError("sequence counts and lengths must be positive")
        if not 0 <= self.burn_in < self.sequence_length or max(self.delays) >= self.sequence_length:
            raise ValueError("sequence_length must exceed burn_in and delays")
        if self.neurons < 1 or self.epochs < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("neurons, epochs and learning rate must be positive")
        if self.window_size is not None and (type(self.window_size) is not int or self.window_size < max(self.delays) + 1 or self.window_size > self.sequence_length):
            raise ValueError("window_size must include all delayed target inputs and fit the sequence")

    @property
    def history_length(self) -> int:
        return self.window_size if self.window_size is not None else max(self.delays) + 1


class TanhReservoir:
    """Fixed, randomly connected conventional recurrent tanh state model.

    This is an echo-state-style reservoir, not an RNN trained via
    backpropagation-through-time. Only its separate readout is fitted.
    """

    def __init__(self, neurons: int, seed: int) -> None:
        if neurons < 1:
            raise ValueError("neurons must be positive")
        rng = random.Random(seed)
        self.neurons = neurons
        self.in_weights = [rng.uniform(-0.7, 0.7) for _ in range(neurons)]
        self.bias = [rng.uniform(-0.12, 0.12) for _ in range(neurons)]
        scale = 0.7 / math.sqrt(neurons)
        self.weights = [
            [rng.uniform(-scale, scale) for _ in range(neurons)]
            for _ in range(neurons)
        ]
        self.reset()

    def reset(self) -> None:
        self.state = [0.0] * self.neurons

    def step(self, bit: float) -> tuple[float, ...]:
        if not isinstance(bit, (int, float)) or not math.isfinite(bit) or not 0 <= bit <= 1:
            raise ValueError("bit must be a finite number in [0, 1]")
        centered = 2 * bit - 1
        previous = self.state
        self.state = [
            math.tanh(
                self.bias[i] + self.in_weights[i] * centered
                + sum(w * value for w, value in zip(row, previous))
            )
            for i, row in enumerate(self.weights)
        ]
        return tuple(self.state)


def make_sequences(
    count: int, length: int, seed: int, stay_probability: float, noise: float
) -> list[tuple[list[int], list[int]]]:
    """Generate paired clean Markov chains and noisy observations.

    All noise levels share the same clean streams and uniform draws. Independent
    train/test seeds prevent shared temporal sequences between splits.
    """
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        bit = rng.randrange(2)
        clean = [bit]
        for _ in range(length - 1):
            # Consume a draw even when stay_probability is 0.5 or 1.0.
            if rng.random() >= stay_probability:
                bit ^= 1
            clean.append(bit)
        observed = [bit ^ int(rng.random() < noise) for bit in clean]
        result.append((clean, observed))
    return result


def extract_features(
    name: str,
    sequences: list[tuple[list[int], list[int]]],
    delay: int,
    start: int,
    history_length: int,
    neurons: int,
    seed: int,
) -> tuple[list[tuple[float, ...]], list[int]]:
    """Use identical target labels and retained time indices for every model."""
    reservoir = None
    if name == "spiking_lif":
        reservoir = Reservoir(ReservoirConfig(seed=seed, neurons=neurons))
    elif name == "tanh_reservoir":
        reservoir = TanhReservoir(neurons=neurons, seed=seed)
    elif name not in MODELS:
        raise ValueError("unknown model " + name)
    examples = []
    labels = []
    for clean, observed in sequences:
        if reservoir is not None:
            reservoir.reset()
        for t, bit in enumerate(observed):
            recurrent_state = reservoir.step(float(bit)) if reservoir is not None else ()
            if t < start:
                continue
            if name in ("spiking_lif", "tanh_reservoir"):
                features = recurrent_state
            elif name == "history_window":
                features = tuple(float(observed[t - offset]) for offset in range(history_length))
            elif name == "delayed_tap":
                features = (float(observed[t - delay]),)
            else:
                features = ()
            examples.append(features)
            labels.append(clean[t - delay])
    return examples, labels


def _evaluate(
    name: str,
    config: TemporalConfig,
    seed: int,
    delay: int,
    train: list[tuple[list[int], list[int]]],
    test: list[tuple[list[int], list[int]]],
) -> dict:
    # Profiling includes model instantiation and feature extraction for both
    # splits, fitting, then held-out evaluation. It excludes input generation,
    # JSON serialization, file I/O, and the startup of the Python interpreter.
    gc.collect()
    tracemalloc.start()
    start = perf_counter_ns()
    # A full, unpadded history is required for every model.
    start_index = max(config.burn_in, config.history_length - 1, max(config.delays))
    train_x, train_y = extract_features(name, train, delay, start_index, config.history_length, config.neurons, seed)
    test_x, test_y = extract_features(name, test, delay, start_index, config.history_length, config.neurons, seed)
    extracted = perf_counter_ns()
    if name == "majority":
        majority_label = int(sum(train_y) * 2 >= len(train_y))
        fitted = perf_counter_ns()
        accuracy = sum(y == majority_label for y in test_y) / len(test_y)
    else:
        # One fixed shuffle and optimizer schedule for each trainable readout.
        order = list(range(len(train_y)))
        random.Random(seed * 1009 + delay * 307).shuffle(order)
        readout = LogisticReadout()
        readout.fit(
            [train_x[i] for i in order], [train_y[i] for i in order],
            epochs=config.epochs, learning_rate=config.learning_rate
        )
        fitted = perf_counter_ns()
        accuracy = readout.score(test_x, test_y)
    finished = perf_counter_ns()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "accuracy": accuracy,
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "profiling": {
            "extraction_ns": extracted - start,
            "fit_ns": fitted - extracted,
            "evaluation_ns": finished - fitted,
            "total_ns": finished - start,
            "peak_traced_python_bytes": peak,
        },
    }


def run_temporal(config: TemporalConfig | None = None) -> dict:
    cfg = config or TemporalConfig()
    if cfg.history_length > cfg.sequence_length:
        raise ValueError("history window exceeds sequence length")
    runs = []
    for seed in cfg.seeds:
        for noise in cfg.noise_levels:
            train = make_sequences(cfg.train_sequences, cfg.sequence_length, seed * 1009 + 101, cfg.stay_probability, noise)
            test = make_sequences(cfg.test_sequences, cfg.sequence_length, seed * 1009 + 202, cfg.stay_probability, noise)
            for delay in cfg.delays:
                measurements = {
                    name: _evaluate(name, cfg, seed, delay, train, test) for name in MODELS
                }
                train_counts = {x["train_samples"] for x in measurements.values()}
                test_counts = {x["test_samples"] for x in measurements.values()}
                if len(train_counts) != 1 or len(test_counts) != 1:
                    raise RuntimeError("different sample counts for paired model comparison")
                runs.append({
                    "seed": seed, "noise": noise, "delay": delay,
                    "samples": {"train": train_counts.pop(), "test": test_counts.pop()},
                    "metrics": {name: measurements[name]["accuracy"] for name in MODELS},
                    "profiling": {name: measurements[name]["profiling"] for name in MODELS},
                })
    summary = []
    for noise in cfg.noise_levels:
        for delay in cfg.delays:
            selected = [run for run in runs if run["noise"] == noise and run["delay"] == delay]
            metrics = {
                name: {
                    "mean": statistics.mean(run["metrics"][name] for run in selected),
                    "population_stdev": statistics.pstdev(run["metrics"][name] for run in selected)
                }
                for name in MODELS
            }
            summary.append({
                "noise": noise, "delay": delay, "seeds": len(selected),
                "metrics": metrics,
                "paired_deltas": {
                    "lif_minus_tanh": statistics.mean(run["metrics"]["spiking_lif"] - run["metrics"]["tanh_reservoir"] for run in selected),
                    "lif_minus_window": statistics.mean(run["metrics"]["spiking_lif"] - run["metrics"]["history_window"] for run in selected),
                },
            })
    return {
        "schema_version": "5.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "temporal_signal_memory_profile",
        "configuration": {**asdict(cfg), "window_size": cfg.history_length},
        "models": {
            "spiking_lif": "Fixed recurrent LIF reservoir and trained logistic readout",
            "tanh_reservoir": "Fixed recurrent tanh echo-state-style reservoir and trained logistic readout (NOT fully trained RNN)",
            "history_window": "Conventional explicit observed-history vector with trained logistic readout",
            "delayed_tap": "Explicit observed input at target delay with trained logistic readout",
            "majority": "Training-label majority without input or a trained readout",
        },
        "protocol": {
            "signal": "First-order binary Markov stream: previous bit retained with stay_probability",
            "labels": "Clean signal at time t - delay, observations corrupted by independent bit flips",
            "held_out": "Separate complete training and test Markov streams for each seed; never fit on test data",
            "pairing": "All models receive identical signal/noise data and time indices for each seed/noise/delay",
            "index_start": max(cfg.burn_in, cfg.history_length - 1, max(cfg.delays)),
            "learning": "Only logistic readouts trained; both recurrent state-transition matrices stay fixed",
            "profiling": "perf_counter_ns and tracemalloc peak within each evaluated model's extraction, fitting and inference",
            "profiling_includes": "Model construction, feature extraction, readout training and held-out scoring",
            "profiling_excludes": "Input-sequence generation, interpreter startup, file I/O, serialization and external processes",
            "timings_deterministic": False,
            "memory_scope": "Peak traced Python memory allocations, not resident set size or complete native allocations",
        },
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Numerical simulation only; no living neural tissue, clinical system or awareness test.",
            "Tanh comparison is a fixed-weight recurrent reservoir, not a fully trained gradient-based RNN.",
            "Architectures do not share equal trainable parameters, operations, or tuned hyperparameters.",
            "Measured timings and traced Python allocations depend on machine load, interpreter, and execution order; do not treat as portable rankings.",
            "Temporally correlated signals allow denoising unlike independent random bit streams, but this simple task is not representative of real-world reasoning.",
            "Small seed sweeps give descriptive variation, not confidence intervals or statistically established superiority.",
            "Runtime and Python allocations cannot establish electrical energy efficiency.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "noise", "delay", "train_samples", "test_samples"]
        fields += [f"{name}_accuracy" for name in MODELS]
        fields += [f"{name}_total_ns" for name in MODELS]
        fields += [f"{name}_peak_traced_python_bytes" for name in MODELS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            values = {
                "seed": row["seed"], "noise": row["noise"], "delay": row["delay"],
                "train_samples": row["samples"]["train"], "test_samples": row["samples"]["test"],
            }
            for name in MODELS:
                values[f"{name}_accuracy"] = row["metrics"][name]
                values[f"{name}_total_ns"] = row["profiling"][name]["total_ns"]
                values[f"{name}_peak_traced_python_bytes"] = row["profiling"][name]["peak_traced_python_bytes"]
            writer.writerow(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio Phase 5: temporal signal and Python profiling")
    parser.add_argument("--seed-start", type=int, default=369)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.15])
    parser.add_argument("--stay-probability", type=float, default=0.85)
    parser.add_argument("--train-sequences", type=int, default=12)
    parser.add_argument("--test-sequences", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = TemporalConfig(
            seeds=tuple(args.seed_start + i for i in range(args.seed_count)),
            delays=tuple(args.delays), noise_levels=tuple(args.noise_levels),
            stay_probability=args.stay_probability,
            train_sequences=args.train_sequences, test_sequences=args.test_sequences,
            sequence_length=args.sequence_length, burn_in=args.burn_in,
            neurons=args.neurons, epochs=args.epochs, window_size=args.window_size,
        )
        report = run_temporal(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
        print("Saved temporal profile report: " + str(args.json_output))
    else:
        print(payload, end="")
    if args.csv_output:
        write_csv(report, args.csv_output)
        print("Saved CSV: " + str(args.csv_output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
