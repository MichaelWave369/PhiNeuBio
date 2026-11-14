"""Phase 6: trained tanh Elman RNN versus fixed digital reservoirs and explicit memory.

A small from-scratch recurrent neural network uses full-sequence backpropagation
through time (BPTT). All models see the same paired Markov sequences. This is a
software-only experimental comparison, not biological computing or clinical AI.
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

from .temporal import TemporalConfig, _evaluate, make_sequences


BASELINES = ("spiking_lif", "tanh_reservoir", "history_window", "delayed_tap", "majority")
MODELS = ("trained_rnn",) + BASELINES


@dataclass(frozen=True)
class CompetitionConfig:
    seeds: tuple[int, ...] = (369, 370)
    delays: tuple[int, ...] = (1, 3)
    noise_levels: tuple[float, ...] = (0.0, 0.15)
    stay_probability: float = 0.85
    train_sequences: int = 12
    test_sequences: int = 6
    sequence_length: int = 32
    burn_in: int = 8
    neurons: int = 8
    epochs: int = 8
    learning_rate: float = 0.05
    gradient_clip: float = 5.0

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 20 or any(type(x) is not int for x in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be 1-20 distinct integers")
        if not self.delays or len(self.delays) > 20 or any(type(x) is not int or x < 1 for x in self.delays) or len(set(self.delays)) != len(self.delays):
            raise ValueError("delays must be 1-20 distinct positive integers")
        if not self.noise_levels or len(self.noise_levels) > 10 or len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise_levels must be distinct")
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 0.5 for x in self.noise_levels):
            raise ValueError("noise levels must be in [0, 0.5]")
        if not math.isfinite(self.stay_probability) or not 0.5 <= self.stay_probability <= 1:
            raise ValueError("stay_probability must be in [0.5, 1]")
        if self.train_sequences < 1 or self.test_sequences < 1 or self.neurons < 1 or self.neurons > 128:
            raise ValueError("sequence counts and neurons must be positive (neurons <=128)")
        if self.sequence_length < 2 or not 0 <= self.burn_in < self.sequence_length or max(self.delays) >= self.sequence_length:
            raise ValueError("delays and burn_in must fit inside the sequence")
        if self.epochs < 1 or self.epochs > 1000:
            raise ValueError("epochs must be in [1, 1000]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError("learning_rate must be finite in (0, 1]")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite positive")


class TrainableRNN:
    """Scalar-input tanh Elman RNN + sigmoid readout.

    Hidden transition and output weights learn via exact full-sequence BPTT.
    State resets for each independent input sequence. Gradient clipping is
    applied to the global L2 norm once per training sequence.
    """

    def __init__(self, neurons: int, seed: int) -> None:
        if neurons < 1:
            raise ValueError("neurons must be positive")
        rng = random.Random(seed)
        self.neurons = neurons
        self.wx = [rng.uniform(-0.35, 0.35) for _ in range(neurons)]
        self.wh = [
            [rng.uniform(-0.6, 0.6) / math.sqrt(neurons) for _ in range(neurons)]
            for _ in range(neurons)
        ]
        self.bh = [0.0] * neurons
        self.wy = [rng.uniform(-0.3, 0.3) for _ in range(neurons)]
        self.by = 0.0

    @property
    def parameter_count(self) -> int:
        return self.neurons * self.neurons + 3 * self.neurons + 1

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = max(-40.0, min(40.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def forward(self, observed: Sequence[int]) -> tuple[list[list[float]], list[float]]:
        h = [0.0] * self.neurons
        states = [[0.0] * self.neurons]
        probabilities = []
        for bit in observed:
            if bit not in (0, 1):
                raise ValueError("observed sequence must contain binary bits")
            x = 2 * bit - 1
            h = [
                math.tanh(
                    self.bh[i] + self.wx[i] * x
                    + sum(w * value for w, value in zip(row, h))
                )
                for i, row in enumerate(self.wh)
            ]
            states.append(h)
            probabilities.append(self._sigmoid(self.by + sum(w * z for w, z in zip(self.wy, h))))
        return states, probabilities

    def loss_and_gradients(
        self, observed: Sequence[int], clean: Sequence[int], delay: int, start: int
    ) -> tuple[float, dict]:
        """Full BPTT with average binary cross-entropy on the retained steps."""
        if len(observed) != len(clean) or not 0 <= start < len(clean) or delay < 1 or start < delay:
            raise ValueError("invalid sequence lengths or label window")
        states, probabilities = self.forward(observed)
        n, size = self.neurons, len(clean)
        valid = size - start
        gradients = {
            "wx": [0.0] * n,
            "wh": [[0.0] * n for _ in range(n)],
            "bh": [0.0] * n,
            "wy": [0.0] * n,
            "by": 0.0,
        }
        loss = 0.0
        future_dh = [0.0] * n
        for t in range(size - 1, -1, -1):
            p = probabilities[t]
            h = states[t + 1]
            dh = future_dh[:]
            if t >= start:
                y = clean[t - delay]
                p_safe = max(1e-12, min(1 - 1e-12, p))
                loss -= (y * math.log(p_safe) + (1 - y) * math.log(1 - p_safe)) / valid
                dlogit = (p - y) / valid
                gradients["by"] += dlogit
                for i in range(n):
                    gradients["wy"][i] += dlogit * h[i]
                    dh[i] += dlogit * self.wy[i]
            dz = [dh[i] * (1 - h[i] * h[i]) for i in range(n)]
            previous = states[t]
            x = 2 * observed[t] - 1
            for i in range(n):
                gradients["wx"][i] += dz[i] * x
                gradients["bh"][i] += dz[i]
                for j in range(n):
                    gradients["wh"][i][j] += dz[i] * previous[j]
            # dL/dh_previous = Wh.T @ dL/dz, through all preceding steps.
            future_dh = [
                sum(self.wh[i][j] * dz[i] for i in range(n))
                for j in range(n)
            ]
        return loss, gradients

    def train_sequence(self, observed: Sequence[int], clean: Sequence[int], delay: int, start: int,
                       learning_rate: float, clip: float) -> float:
        loss, gradients = self.loss_and_gradients(observed, clean, delay, start)
        values = gradients["wx"] + gradients["bh"] + gradients["wy"] + [gradients["by"]]
        values += [item for row in gradients["wh"] for item in row]
        norm = math.sqrt(sum(value * value for value in values))
        factor = min(1.0, clip / norm) if norm else 1.0
        step = learning_rate * factor
        for i in range(self.neurons):
            self.wx[i] -= step * gradients["wx"][i]
            self.bh[i] -= step * gradients["bh"][i]
            self.wy[i] -= step * gradients["wy"][i]
            for j in range(self.neurons):
                self.wh[i][j] -= step * gradients["wh"][i][j]
        self.by -= step * gradients["by"]
        return loss

    def train(self, sequences: Sequence[tuple[list[int], list[int]]], delay: int,
              start: int, epochs: int, learning_rate: float, clip: float, seed: int) -> list[float]:
        rng = random.Random(seed)
        order = list(range(len(sequences)))
        curve = []
        for _ in range(epochs):
            rng.shuffle(order)
            losses = []
            for i in order:
                clean, observed = sequences[i]
                losses.append(self.train_sequence(observed, clean, delay, start, learning_rate, clip))
            curve.append(statistics.mean(losses))
        return curve

    def accuracy(self, sequences: Sequence[tuple[list[int], list[int]]], delay: int, start: int) -> float:
        correct = total = 0
        for clean, observed in sequences:
            _, probabilities = self.forward(observed)
            for t in range(start, len(clean)):
                correct += int((probabilities[t] >= 0.5) == bool(clean[t - delay]))
                total += 1
        return correct / total


def _trained_measure(
    config: CompetitionConfig, seed: int, delay: int,
    train: list[tuple[list[int], list[int]]],
    test: list[tuple[list[int], list[int]]],
) -> dict:
    start_index = max(config.burn_in, max(config.delays))
    gc.collect()
    tracemalloc.start()
    try:
        start = perf_counter_ns()
        network = TrainableRNN(config.neurons, seed)
        initialized = perf_counter_ns()
        before = network.accuracy(test, delay, start_index)
        loss_curve = network.train(
            train, delay, start_index, config.epochs, config.learning_rate,
            config.gradient_clip, seed * 1009 + delay * 307,
        )
        fitted = perf_counter_ns()
        after = network.accuracy(test, delay, start_index)
        finished = perf_counter_ns()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "accuracy": after,
        "before_test_accuracy": before,
        "training_loss_by_epoch": loss_curve,
        "trainable_parameters": network.parameter_count,
        "train_samples": config.train_sequences * (config.sequence_length - start_index),
        "test_samples": config.test_sequences * (config.sequence_length - start_index),
        "profiling": {
            "initialization_and_initial_eval_ns": initialized - start,
            "training_including_initial_eval_ns": fitted - initialized,
            "evaluation_ns": finished - fitted,
            "total_ns": finished - start,
            "peak_traced_python_bytes": peak,
        },
    }


def run_competition(config: CompetitionConfig | None = None) -> dict:
    cfg = config or CompetitionConfig()
    start_index = max(cfg.burn_in, max(cfg.delays))
    baseline_cfg = TemporalConfig(
        seeds=cfg.seeds, delays=cfg.delays, noise_levels=cfg.noise_levels,
        stay_probability=cfg.stay_probability,
        train_sequences=cfg.train_sequences, test_sequences=cfg.test_sequences,
        sequence_length=cfg.sequence_length, burn_in=cfg.burn_in,
        neurons=cfg.neurons, epochs=cfg.epochs, learning_rate=cfg.learning_rate,
    )
    runs = []
    for seed in cfg.seeds:
        for noise in cfg.noise_levels:
            train = make_sequences(
                cfg.train_sequences, cfg.sequence_length, seed * 1009 + 101, cfg.stay_probability, noise
            )
            test = make_sequences(
                cfg.test_sequences, cfg.sequence_length, seed * 1009 + 202, cfg.stay_probability, noise
            )
            for delay in cfg.delays:
                trained = _trained_measure(cfg, seed, delay, train, test)
                baseline = {model: _evaluate(model, baseline_cfg, seed, delay, train, test) for model in BASELINES}
                counts = {(trained["train_samples"], trained["test_samples"])}
                counts.update((m["train_samples"], m["test_samples"]) for m in baseline.values())
                if len(counts) != 1:
                    raise RuntimeError("models used unequal sample counts")
                metrics = {"trained_rnn": trained["accuracy"]}
                metrics.update({model: baseline[model]["accuracy"] for model in BASELINES})
                profiling = {"trained_rnn": trained["profiling"]}
                profiling.update({model: baseline[model]["profiling"] for model in BASELINES})
                runs.append({
                    "seed": seed, "noise": noise, "delay": delay,
                    "samples": {"train": trained["train_samples"], "test": trained["test_samples"]},
                    "metrics": metrics,
                    "profiling": profiling,
                    "trained_rnn": {
                        "before_test_accuracy": trained["before_test_accuracy"],
                        "training_loss_by_epoch": trained["training_loss_by_epoch"],
                        "trainable_parameters": trained["trainable_parameters"],
                    },
                })
    summary = []
    for noise in cfg.noise_levels:
        for delay in cfg.delays:
            paired = [r for r in runs if r["noise"] == noise and r["delay"] == delay]
            summary.append({
                "noise": noise, "delay": delay, "seeds": len(paired),
                "metrics": {
                    model: {
                        "mean": statistics.mean(r["metrics"][model] for r in paired),
                        "population_stdev": statistics.pstdev(r["metrics"][model] for r in paired),
                    }
                    for model in MODELS
                },
                "paired_deltas": {
                    "trained_rnn_minus_lif": statistics.mean(
                        r["metrics"]["trained_rnn"] - r["metrics"]["spiking_lif"] for r in paired
                    ),
                    "trained_rnn_minus_window": statistics.mean(
                        r["metrics"]["trained_rnn"] - r["metrics"]["history_window"] for r in paired
                    ),
                },
            })
    return {
        "schema_version": "6.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "trainable_recurrent_memory_comparison",
        "configuration": asdict(cfg),
        "models": {
            "trained_rnn": "Trainable tanh Elman RNN with full-sequence BPTT and gradient clipping",
            "spiking_lif": "Fixed LIF reservoir with trained logistic readout",
            "tanh_reservoir": "Fixed tanh reservoir with trained logistic readout",
            "history_window": "Explicit software memory window with trained logistic readout",
            "delayed_tap": "Observed bit at exact delay with trained logistic readout",
            "majority": "Training label majority",
        },
        "protocol": {
            "data": "Seeded first-order Markov clean streams with paired independently flipped observations",
            "split": "Independent train/test sequences; same inputs and targets for every model",
            "sample_start_index": start_index,
            "gradient": "Full-sequence BPTT, per-sequence cross-entropy averaged over retained output steps, global L2 gradient clipping",
            "trainable_recurrent_weights": "Only trained_rnn changes recurrent weights; fixed reservoirs train readouts only",
            "training": "Sequence-wise SGD across seeded epochs; no test examples used in optimization",
            "profile": "perf_counter_ns and tracemalloc (Python allocations); measurements not deterministic or energy",
            "parameter_parity": "Not established. Trained RNN parameter_count includes input, recurrent, hidden bias, output, and output bias",
        },
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Purely digital simulation, not living neurons or biological intelligence.",
            "This baseline is a trainable Elman RNN, not a GRU or LSTM.",
            "Model hidden sizes are matched, not total parameters, compute budgets, or optimally tuned training.",
            "The trained RNN performs sequence-wise SGD and full BPTT; fixed reservoirs fit readouts on shuffled time-step features. Training schedules differ.",
            "Measured Python wall time and traced allocations are hardware/load-dependent, not electrical energy or fair architecture-speed rankings.",
            "Small seed sweeps and toy Markov sequences do not support broad generalization or superiority claims.",
            "Network learning, classification performance, and neuron-like activity do not demonstrate consciousness.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["seed", "noise", "delay", "train_samples", "test_samples"]
        columns += [f"{model}_accuracy" for model in MODELS]
        columns += [f"{model}_total_ns" for model in MODELS]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["runs"]:
            values = {
                "seed": row["seed"], "noise": row["noise"], "delay": row["delay"],
                "train_samples": row["samples"]["train"], "test_samples": row["samples"]["test"],
            }
            for model in MODELS:
                values[f"{model}_accuracy"] = row["metrics"][model]
                values[f"{model}_total_ns"] = row["profiling"][model]["total_ns"]
            writer.writerow(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio Phase 6: trainable recurrent network competition")
    parser.add_argument("--seed-start", type=int, default=369)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.15])
    parser.add_argument("--stay-probability", type=float, default=0.85)
    parser.add_argument("--train-sequences", type=int, default=12)
    parser.add_argument("--test-sequences", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = CompetitionConfig(
            seeds=tuple(args.seed_start + i for i in range(args.seed_count)),
            delays=tuple(args.delays), noise_levels=tuple(args.noise_levels),
            stay_probability=args.stay_probability,
            train_sequences=args.train_sequences, test_sequences=args.test_sequences,
            sequence_length=args.sequence_length, burn_in=args.burn_in,
            neurons=args.neurons, epochs=args.epochs, learning_rate=args.learning_rate,
        )
        report = run_competition(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
        print("Saved Phase 6 report: " + str(args.json_output))
    else:
        print(payload, end="")
    if args.csv_output:
        write_csv(report, args.csv_output)
        print("Saved CSV: " + str(args.csv_output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
