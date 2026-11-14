"""Controlled, repeated-seed tests of digital spiking-reservoir memory.

Compares identical train/test input streams across fixed recurrent and
nonrecurrent spiking reservoirs, a current-input model, a tapped-delay model,
and a majority-class baseline. Nothing here measures biological tissue.
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
from typing import Sequence

from .readout import LogisticReadout
from .reservoir import Reservoir, ReservoirConfig


MODEL_NAMES = (
    "spiking_recurrent",
    "spiking_no_recurrence",
    "current_input",
    "delayed_input_tap",
    "majority",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...] = (369, 370, 371)
    delays: tuple[int, ...] = (1, 2, 4, 8)
    train_sequences: int = 24
    test_sequences: int = 12
    sequence_length: int = 40
    burn_in: int = 8
    neurons: int = 24
    epochs: int = 12
    learning_rate: float = 0.035
    flip_probability: float = 0.0

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 30 or any(type(s) is not int for s in self.seeds):
            raise ValueError("seeds must be 1-30 integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not self.delays or len(self.delays) > 30 or any(type(d) is not int or d < 1 for d in self.delays):
            raise ValueError("delays must be 1-30 positive integers")
        if len(set(self.delays)) != len(self.delays):
            raise ValueError("delays must be unique")
        if self.train_sequences < 1 or self.test_sequences < 1:
            raise ValueError("sequence counts must be positive")
        if self.sequence_length < 2 or not 0 <= self.burn_in < self.sequence_length:
            raise ValueError("invalid sequence_length or burn_in")
        if max(self.delays) >= self.sequence_length:
            raise ValueError("each delay must be shorter than sequence_length")
        if self.neurons < 1 or self.epochs < 1:
            raise ValueError("neurons and epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.flip_probability) or not 0 <= self.flip_probability <= 0.5:
            raise ValueError("flip_probability must be finite and in [0, 0.5]")


def _make_sequences(
    count: int, length: int, seed: int, flip_probability: float
) -> list[tuple[list[int], list[int]]]:
    """Generate pairs (clean target source, observed possibly flipped input).

    The held-out set gets a distinct random stream and is never fitted.
    """
    rng = random.Random(seed)
    result: list[tuple[list[int], list[int]]] = []
    for _ in range(count):
        clean = [rng.randrange(2) for _ in range(length)]
        observed = [
            bit ^ int(rng.random() < flip_probability) for bit in clean
        ]
        result.append((clean, observed))
    return result


def _extract(
    sequences: list[tuple[list[int], list[int]]],
    delay: int,
    burn_in: int,
    recurrent: Reservoir,
    nonrecurrent: Reservoir,
) -> tuple[dict[str, list[tuple[float, ...]]], list[int]]:
    models = {name: [] for name in MODEL_NAMES if name != "majority"}
    targets: list[int] = []
    for clean, observed in sequences:
        recurrent.reset()
        nonrecurrent.reset()
        for t, value in enumerate(observed):
            a = recurrent.step(float(value))
            b = nonrecurrent.step(float(value))
            if t >= max(delay, burn_in):
                models["spiking_recurrent"].append(a)
                models["spiking_no_recurrence"].append(b)
                models["current_input"].append((float(observed[t]),))
                models["delayed_input_tap"].append((float(observed[t - delay]),))
                targets.append(clean[t - delay])
    return models, targets


def _accuracy(
    train_x: list[tuple[float, ...]],
    train_y: list[int],
    test_x: list[tuple[float, ...]],
    test_y: list[int],
    config: BenchmarkConfig,
    shuffle_seed: int,
) -> float:
    # All models use exactly the same sample order and training hyperparameters.
    order = list(range(len(train_y)))
    random.Random(shuffle_seed).shuffle(order)
    readout = LogisticReadout()
    readout.fit(
        [train_x[i] for i in order],
        [train_y[i] for i in order],
        epochs=config.epochs,
        learning_rate=config.learning_rate,
    )
    return readout.score(test_x, test_y)


def run_benchmark(config: BenchmarkConfig | None = None) -> dict:
    cfg = config or BenchmarkConfig()
    runs = []
    for seed in cfg.seeds:
        # Use one paired set per seed across all delays and model types.
        train = _make_sequences(cfg.train_sequences, cfg.sequence_length, seed * 1009 + 101, cfg.flip_probability)
        test = _make_sequences(cfg.test_sequences, cfg.sequence_length, seed * 1009 + 202, cfg.flip_probability)
        recurrent = Reservoir(ReservoirConfig(neurons=cfg.neurons, seed=seed))
        nonrecurrent = Reservoir(ReservoirConfig(neurons=cfg.neurons, seed=seed, recurrent_gain=0.0))
        for delay in cfg.delays:
            train_features, train_targets = _extract(train, delay, cfg.burn_in, recurrent, nonrecurrent)
            test_features, test_targets = _extract(test, delay, cfg.burn_in, recurrent, nonrecurrent)
            common_order_seed = seed * 1009 + delay * 307
            metrics = {
                name: _accuracy(
                    train_features[name], train_targets,
                    test_features[name], test_targets,
                    cfg, common_order_seed,
                )
                for name in MODEL_NAMES if name != "majority"
            }
            majority_label = int(sum(train_targets) * 2 >= len(train_targets))
            metrics["majority"] = sum(y == majority_label for y in test_targets) / len(test_targets)
            runs.append({
                "seed": seed,
                "delay": delay,
                "samples": {"train": len(train_targets), "test": len(test_targets)},
                "metrics": metrics,
            })

    summary = []
    for delay in cfg.delays:
        selected = [run for run in runs if run["delay"] == delay]
        averages = {}
        for model in MODEL_NAMES:
            values = [run["metrics"][model] for run in selected]
            averages[model] = {
                "mean": statistics.mean(values),
                "population_stdev": statistics.pstdev(values),
            }
        summary.append({
            "delay": delay,
            "seeds": len(selected),
            "metrics": averages,
            "paired_deltas": {
                "recurrent_minus_no_recurrence": statistics.mean(
                    row["metrics"]["spiking_recurrent"] - row["metrics"]["spiking_no_recurrence"]
                    for row in selected
                ),
                "recurrent_minus_delayed_input_tap": statistics.mean(
                    row["metrics"]["spiking_recurrent"] - row["metrics"]["delayed_input_tap"]
                    for row in selected
                ),
            },
        })
    return {
        "schema_version": "3.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "seed_delay_memory_comparison",
        "configuration": asdict(cfg),
        "models": {
            "spiking_recurrent": "Fixed recurrent LIF reservoir with trained logistic readout",
            "spiking_no_recurrence": "Same LIF neuron configuration and seed, recurrent gain set to zero",
            "current_input": "Logistic model trained on only the current observed bit; no external memory",
            "delayed_input_tap": "Logistic model explicitly given the observed bit from delay steps ago",
            "majority": "Predict the majority training label; no input features",
        },
        "protocol": {
            "split": "Independent train and test pseudorandom sequences for each seed",
            "paired_inputs": "Same sequences, labels, and observed bit flips for every model at a seed and delay",
            "noise": "Input bit flips are applied to observed features, not to the clean reference label",
            "reset": "Reservoir state reset between independent sequences",
            "trainable_parameters": "Only logistic readout weights; recurrent synapses fixed",
            "sample_window": "Samples with t >= max(delay, burn_in)",
            "summary_dispersion": "population standard deviation over seeds; descriptive only",
        },
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Software-only spiking-neuron simulation; no living tissue or biological hardware.",
            "All tasks use independent identically distributed random bits and short fixed sequences.",
            "Explicit delayed-input tap is a conventional memory control with privileged access to older observed inputs.",
            "No-recurrence LIF neurons still have leaky membrane state and may retain short history.",
            "A fixed input flip probability is not an exhaustive noise or robustness evaluation.",
            "Population standard deviation over a small seed sweep is not a confidence interval.",
            "Classification accuracy does not measure energy efficiency or consciousness.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    """Write each seed-delay pair and model metric in a portable CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "seed", "delay", "train_samples", "test_samples", *MODEL_NAMES
        ])
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({
                "seed": row["seed"],
                "delay": row["delay"],
                "train_samples": row["samples"]["train"],
                "test_samples": row["samples"]["test"],
                **row["metrics"],
            })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio: controlled memory benchmark suite")
    parser.add_argument("--seed-start", type=int, default=369)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--train-sequences", type=int, default=24)
    parser.add_argument("--test-sequences", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--flip-probability", type=float, default=0.0)
    parser.add_argument("--json-output", type=Path, help="Write a JSON report instead of stdout")
    parser.add_argument("--csv-output", type=Path, help="Additionally save per-seed results to CSV")
    args = parser.parse_args(argv)
    try:
        cfg = BenchmarkConfig(
            seeds=tuple(args.seed_start + i for i in range(args.seed_count)),
            delays=tuple(args.delays),
            train_sequences=args.train_sequences,
            test_sequences=args.test_sequences,
            sequence_length=args.sequence_length,
            burn_in=args.burn_in, neurons=args.neurons, epochs=args.epochs,
            flip_probability=args.flip_probability,
        )
        report = run_benchmark(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
        print("Saved benchmark report: " + str(args.json_output))
    else:
        print(payload, end="")
    if args.csv_output:
        write_csv(report, args.csv_output)
        print("Saved CSV: " + str(args.csv_output), file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
