"""Paired noise × delay × seed comparisons with an explicit memory-window control.

This is digital simulation. No physiological measurements or energy claims.
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
from typing import Sequence

from .benchmarks import _extract, _make_sequences, _accuracy
from .reservoir import Reservoir, ReservoirConfig


MODELS = (
    "spiking_recurrent",
    "spiking_no_recurrence",
    "current_input",
    "history_window",
    "delayed_input_tap",
    "majority",
)
TRAINABLE = tuple(name for name in MODELS if name != "majority")


@dataclass(frozen=True)
class RobustnessConfig:
    seeds: tuple[int, ...] = (369, 370, 371)
    delays: tuple[int, ...] = (1, 2, 4, 8)
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.25)
    train_sequences: int = 16
    test_sequences: int = 8
    sequence_length: int = 36
    burn_in: int = 8
    neurons: int = 20
    epochs: int = 10
    learning_rate: float = 0.035
    window_size: int | None = None

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 30 or any(type(x) is not int for x in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must contain 1-30 unique integers")
        if not self.delays or len(self.delays) > 30 or any(type(x) is not int or x < 1 for x in self.delays) or len(set(self.delays)) != len(self.delays):
            raise ValueError("delays must contain 1-30 unique positive integers")
        if not self.noise_levels or len(self.noise_levels) > 12:
            raise ValueError("noise_levels must contain 1-12 entries")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 0.5 for x in self.noise_levels):
            raise ValueError("noise must be finite and in [0, 0.5]")
        if len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise_levels must be unique")
        if self.train_sequences < 1 or self.test_sequences < 1:
            raise ValueError("sequence counts must be positive")
        if self.sequence_length < 2 or not 0 <= self.burn_in < self.sequence_length or max(self.delays) >= self.sequence_length:
            raise ValueError("sequence_length must exceed all delays and burn_in")
        if self.neurons < 1 or self.epochs < 1 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("neurons, epochs and learning_rate must be positive")
        if self.window_size is not None and (type(self.window_size) is not int or self.window_size < max(self.delays) + 1 or self.window_size > self.sequence_length):
            raise ValueError("window_size must include all tested delays and fit inside the sequence")

    @property
    def history_length(self) -> int:
        return self.window_size if self.window_size is not None else max(self.delays) + 1


def _window_features(sequences: list[tuple[list[int], list[int]]], delay: int, burn_in: int, width: int) -> list[tuple[float, ...]]:
    """Observed values [t, t-1, ...] padded with zeros at the sequence start.

    The padded prefix is excluded whenever t >= max(delay, burn_in, width - 1);
    see _paired_features, which uses the same stricter window for *all* models.
    """
    samples = []
    for _, observed in sequences:
        for t in range(max(delay, burn_in, width - 1), len(observed)):
            samples.append(tuple(float(observed[t - offset]) for offset in range(width)))
    return samples


def _paired_features(
    sequences: list[tuple[list[int], list[int]]],
    delay: int,
    burn_in: int,
    width: int,
    recurrent: Reservoir,
    nonrecurrent: Reservoir,
) -> tuple[dict[str, list[tuple[float, ...]]], list[int]]:
    """Require complete history for every model to prevent unequal sampling."""
    effective_burn_in = max(burn_in, width - 1)
    extracted, labels = _extract(sequences, delay, effective_burn_in, recurrent, nonrecurrent)
    extracted["history_window"] = _window_features(sequences, delay, burn_in, width)
    assert len(extracted["history_window"]) == len(labels)
    return extracted, labels


def _work_proxies(cfg: RobustnessConfig, train_samples: int, test_samples: int, width: int) -> dict:
    """Deterministic *counts*, not FLOPs, measured execution time, or joules.

    Both current LIF implementations scan all n*n connection slots per step,
    including the no-recurrence case. This is a property of the Python code.
    """
    feature_widths = {
        "spiking_recurrent": cfg.neurons * 2,
        "spiking_no_recurrence": cfg.neurons * 2,
        "current_input": 1,
        "history_window": width,
        "delayed_input_tap": 1,
        "majority": 0,
    }
    total_steps = (cfg.train_sequences + cfg.test_sequences) * cfg.sequence_length
    return {
        model: {
            "feature_dimensions": dim,
            "trainable_readout_parameters": dim + 1 if model != "majority" else 0,
            "readout_coefficient_update_slots": cfg.epochs * train_samples * dim if model != "majority" else 0,
            "reservoir_neuron_state_updates": total_steps * cfg.neurons if model.startswith("spiking_") else 0,
            "dense_recurrent_pair_scans": total_steps * cfg.neurons**2 if model.startswith("spiking_") else 0,
            "feature_values_per_test_set": test_samples * dim,
        }
        for model, dim in feature_widths.items()
    }


def run_robustness(config: RobustnessConfig | None = None) -> dict:
    cfg = config or RobustnessConfig()
    width = cfg.history_length
    # All models and delays use the *same* sample interval; no prefix-padding
    # advantage or differing numbers of test examples.
    if width > cfg.sequence_length:
        raise ValueError("history window cannot exceed sequence_length")
    runs: list[dict] = []
    for seed in cfg.seeds:
        recurrent = Reservoir(ReservoirConfig(seed=seed, neurons=cfg.neurons))
        nonrecurrent = Reservoir(ReservoirConfig(seed=seed, neurons=cfg.neurons, recurrent_gain=0.0))
        for noise in cfg.noise_levels:
            # Independent sequence sets; clean bit streams remain identical across
            # noise levels because RNG draws occur regardless of flip probability.
            train = _make_sequences(cfg.train_sequences, cfg.sequence_length, seed * 1009 + 101, noise)
            test = _make_sequences(cfg.test_sequences, cfg.sequence_length, seed * 1009 + 202, noise)
            for delay in cfg.delays:
                train_x, train_y = _paired_features(train, delay, cfg.burn_in, width, recurrent, nonrecurrent)
                test_x, test_y = _paired_features(test, delay, cfg.burn_in, width, recurrent, nonrecurrent)
                # Reuse the Phase 3 readout implementation and identical train
                # sample ordering for every architecture at a given seed/delay.
                order_seed = seed * 1009 + delay * 307
                metrics = {
                    name: _accuracy(train_x[name], train_y, test_x[name], test_y, cfg, order_seed)
                    for name in TRAINABLE
                }
                majority_label = int(sum(train_y) * 2 >= len(train_y))
                metrics["majority"] = sum(y == majority_label for y in test_y) / len(test_y)
                runs.append({
                    "seed": seed, "noise": noise, "delay": delay,
                    "samples": {"train": len(train_y), "test": len(test_y)},
                    "metrics": metrics,
                    "work_proxies": _work_proxies(cfg, len(train_y), len(test_y), width),
                })

    summary: list[dict] = []
    for noise in cfg.noise_levels:
        for delay in cfg.delays:
            selected = [r for r in runs if r["noise"] == noise and r["delay"] == delay]
            model_metrics = {}
            for name in MODELS:
                values = [r["metrics"][name] for r in selected]
                model_metrics[name] = {
                    "mean": statistics.mean(values),
                    "population_stdev": statistics.pstdev(values),
                }
            summary.append({
                "noise": noise, "delay": delay, "seeds": len(selected),
                "metrics": model_metrics,
                "paired_deltas": {
                    "recurrent_minus_no_recurrence": statistics.mean(
                        r["metrics"]["spiking_recurrent"] - r["metrics"]["spiking_no_recurrence"] for r in selected
                    ),
                    "recurrent_minus_history_window": statistics.mean(
                        r["metrics"]["spiking_recurrent"] - r["metrics"]["history_window"] for r in selected
                    ),
                },
            })
    return {
        "schema_version": "4.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "noise_delay_memory_robustness",
        "configuration": {**asdict(cfg), "window_size": width},
        "models": {
            "spiking_recurrent": "Fixed recurrent LIF network + trainable logistic readout",
            "spiking_no_recurrence": "Same LIF implementation with recurrent gain zero; membrane leak still retains state",
            "current_input": "Logistic readout of current observed bit; no external memory",
            "history_window": "Conventional logistic readout with explicit observed-input shift register of length window_size",
            "delayed_input_tap": "Logistic readout of observed bit from exactly delay steps earlier",
            "majority": "Predict majority label from training set",
        },
        "protocol": {
            "held_out_split": "Independent random sequences, not frames from training sequences",
            "paired_trials": "Same clean inputs, noise mask, target labels, sample interval, and sample order for all models",
            "noise_coupling": "Same pseudorandom noise draws for each level; masks nested as probability increases",
            "target": "Predict clean bit at time t-delay from observed potentially noisy sequence",
            "training": "Only logistic readouts trained; recurrent synapses held fixed",
            "effective_start_index": max(cfg.burn_in, width - 1, max(cfg.delays)),
            "work_proxy_definition": "Analytical counts based on implemented loop dimensions; not FLOPs, wall time or physical energy",
            "dispersion_definition": "Population standard deviation of held-out accuracy across seeds; no confidence interval",
            "theoretical_single_observation_accuracy": "For IID fair bits flipped independently at p<=0.5, Bayes accuracy from the corresponding observed bit is 1-p",
        },
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Pure digital model, not living neurons or biological measurement.",
            "A sliding history window explicitly stores past inputs and may outperform the reservoir with much less compute.",
            "No-recurrence LIF still has leaky state; dense recurrent pair scans are performed even when their weights are zero.",
            "The work proxies count selected operations/slots but are not benchmarked FLOPs, time, energy consumption, or equal-parameter comparisons.",
            "Random independent fair input bits cannot be denoised from surrounding bits; they lack temporal redundancy.",
            "Population standard deviation over selected random seeds does not establish statistical significance.",
            "Classification accuracy does not demonstrate awareness, consciousness, or medical utility.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "noise", "delay", "train_samples", "test_samples", *MODELS])
        writer.writeheader()
        for run in report["runs"]:
            writer.writerow({
                "seed": run["seed"], "noise": run["noise"], "delay": run["delay"],
                "train_samples": run["samples"]["train"], "test_samples": run["samples"]["test"],
                **run["metrics"],
            })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio Phase 4: paired noise/delay/seed robustness")
    parser.add_argument("--seed-start", type=int, default=369)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.1, 0.25])
    parser.add_argument("--train-sequences", type=int, default=16)
    parser.add_argument("--test-sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=36)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--neurons", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--json-output", type=Path, help="Save report to JSON instead of stdout")
    parser.add_argument("--csv-output", type=Path, help="Save per-run results to CSV")
    args = parser.parse_args(argv)
    try:
        cfg = RobustnessConfig(
            seeds=tuple(args.seed_start + i for i in range(args.seed_count)),
            delays=tuple(args.delays), noise_levels=tuple(args.noise_levels),
            train_sequences=args.train_sequences, test_sequences=args.test_sequences,
            sequence_length=args.sequence_length, burn_in=args.burn_in,
            neurons=args.neurons, epochs=args.epochs, window_size=args.window_size,
        )
        result = run_robustness(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(data, encoding="utf-8")
        print("Saved robustness report: " + str(args.json_output))
    else:
        print(data, end="")
    if args.csv_output:
        write_csv(result, args.csv_output)
        print("Saved CSV: " + str(args.csv_output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
