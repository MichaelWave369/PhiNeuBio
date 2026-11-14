"""Phase 7: separate train/validation/test splits for GRU versus Elman RNN.

Only train and validation splits influence architecture learning-rate selection.
Test sequences are evaluated once after model selection. Historical fixed
reservoir baselines retain their existing untuned readout protocol.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

from .competition import TrainableRNN
from .gru import TrainableGRU
from .temporal import TemporalConfig, _evaluate, make_sequences


TRAINED = ("trainable_gru", "trainable_elman")
BASELINES = ("spiking_lif", "tanh_reservoir", "history_window", "delayed_tap", "majority")
MODELS = TRAINED + BASELINES


@dataclass(frozen=True)
class ValidationConfig:
    seeds: tuple[int, ...] = (369, 370)
    delays: tuple[int, ...] = (1, 3)
    noise_levels: tuple[float, ...] = (0.0, 0.15)
    stay_probability: float = 0.85
    train_sequences: int = 8
    validation_sequences: int = 4
    test_sequences: int = 4
    sequence_length: int = 24
    burn_in: int = 6
    epochs: int = 6
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    gradient_clip: float = 5.0
    parameter_budget: int = 95

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 20 or any(type(x) is not int for x in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be 1-20 unique integers")
        if not self.delays or len(self.delays) > 20 or any(type(x) is not int or x < 1 for x in self.delays) or len(set(self.delays)) != len(self.delays):
            raise ValueError("delays must be 1-20 unique positive integers")
        if not self.noise_levels or len(self.noise_levels) > 10 or len(set(self.noise_levels)) != len(self.noise_levels):
            raise ValueError("noise levels must be distinct")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 0.5 for x in self.noise_levels):
            raise ValueError("noise levels must be finite and in [0, 0.5]")
        if not math.isfinite(self.stay_probability) or not 0.5 <= self.stay_probability <= 1:
            raise ValueError("stay_probability must be finite in [0.5, 1]")
        if min(self.train_sequences, self.validation_sequences, self.test_sequences) < 1:
            raise ValueError("each data split requires at least one complete sequence")
        if self.sequence_length < 2 or max(self.delays) >= self.sequence_length or not 0 <= self.burn_in < self.sequence_length:
            raise ValueError("delays and burn-in must fit inside sequence length")
        if self.epochs < 1 or self.epochs > 1000:
            raise ValueError("epochs must be between 1 and 1000")
        if not self.learning_rates or len(self.learning_rates) > 12 or len(set(self.learning_rates)) != len(self.learning_rates):
            raise ValueError("learning_rates must contain 1-12 distinct candidates")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 < x <= 1 for x in self.learning_rates):
            raise ValueError("learning rates must be finite in (0, 1]")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient clipping must be finite and positive")
        if type(self.parameter_budget) is not int or not 11 <= self.parameter_budget <= 1500:
            raise ValueError("parameter_budget must be an integer in [11, 1500]")


def widths_under_budget(budget: int) -> dict[str, dict[str, int]]:
    """Largest hidden width <=32 whose trainable weights fit under the cap."""
    if type(budget) is not int or not 11 <= budget <= 1500:
        raise ValueError("budget out of supported range")
    sizes = {
        "trainable_gru": (lambda n: 3 * n * n + 7 * n + 1),
        "trainable_elman": (lambda n: n * n + 3 * n + 1),
    }
    result = {}
    for name, count in sizes.items():
        width = max(n for n in range(1, 33) if count(n) <= budget)
        result[name] = {
            "hidden_width": width,
            "trainable_parameters": count(width),
            "budget_cap": budget,
        }
    return result


def validation_loss(network, sequences, delay: int, start: int) -> float:
    """Mean held-out validation binary cross-entropy. No model updates."""
    if not sequences:
        raise ValueError("validation data cannot be empty")
    loss = 0.0
    examples = 0
    for clean, observed in sequences:
        if not 0 <= start < len(clean) or start < delay or len(clean) != len(observed):
            raise ValueError("invalid validation sample window")
        probabilities = network.forward(observed)[1]
        for t in range(start, len(clean)):
            p = max(1e-12, min(1 - 1e-12, probabilities[t]))
            y = clean[t - delay]
            loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
            examples += 1
    return loss / examples


def select_and_test(
    model_name: str,
    config: ValidationConfig,
    seed: int,
    delay: int,
    train: list[tuple[list[int], list[int]]],
    validation: list[tuple[list[int], list[int]]],
    test: list[tuple[list[int], list[int]]],
) -> dict:
    """Fit candidates and choose by validation loss before *any* test call."""
    if model_name not in TRAINED:
        raise ValueError("unknown trainable model")
    dimensions = widths_under_budget(config.parameter_budget)[model_name]
    cls = TrainableGRU if model_name == "trainable_gru" else TrainableRNN
    start = max(config.burn_in, max(config.delays))
    start_time = perf_counter_ns()
    candidates = []
    selected = None
    best_loss = float("inf")
    for learning_rate in config.learning_rates:
        net = cls(dimensions["hidden_width"], seed)
        curve = net.train(
            train, delay, start, config.epochs, learning_rate,
            config.gradient_clip, seed * 1009 + delay * 307
        )
        loss = validation_loss(net, validation, delay, start)
        candidates.append({
            "learning_rate": learning_rate,
            "validation_cross_entropy": loss,
            "last_epoch_training_loss": curve[-1],
        })
        # First model wins an exact tie. Candidate order is predeclared.
        if loss < best_loss:
            best_loss, selected = loss, (net, learning_rate, curve)
    selection_ns = perf_counter_ns() - start_time
    network, chosen_lr, curve = selected
    # The test split has not been accessed by this function before this line.
    test_start = perf_counter_ns()
    accuracy = network.accuracy(test, delay, start)
    test_ns = perf_counter_ns() - test_start
    return {
        "test_accuracy": accuracy,
        "selected_learning_rate": chosen_lr,
        "validation_cross_entropy": best_loss,
        "candidate_validation_scores": candidates,
        "training_loss_by_epoch": curve,
        "trainable_parameters": network.parameter_count,
        "hidden_width": dimensions["hidden_width"],
        "selection_elapsed_ns": selection_ns,
        "one_test_evaluation_ns": test_ns,
    }


def run_validation(config: ValidationConfig | None = None) -> dict:
    cfg = config or ValidationConfig()
    sizes = widths_under_budget(cfg.parameter_budget)
    start = max(cfg.burn_in, max(cfg.delays))
    # Readout-only baselines are kept untuned; sizes and optimizers differ.
    baseline_cfg = TemporalConfig(
        seeds=cfg.seeds, delays=cfg.delays, noise_levels=cfg.noise_levels,
        stay_probability=cfg.stay_probability,
        train_sequences=cfg.train_sequences, test_sequences=cfg.test_sequences,
        sequence_length=cfg.sequence_length, burn_in=cfg.burn_in,
        neurons=sizes["trainable_elman"]["hidden_width"], epochs=cfg.epochs,
        learning_rate=cfg.learning_rates[0],
    )
    runs = []
    for seed in cfg.seeds:
        for noise in cfg.noise_levels:
            train = make_sequences(cfg.train_sequences, cfg.sequence_length, seed * 1009 + 101, cfg.stay_probability, noise)
            validation = make_sequences(cfg.validation_sequences, cfg.sequence_length, seed * 1009 + 303, cfg.stay_probability, noise)
            test = make_sequences(cfg.test_sequences, cfg.sequence_length, seed * 1009 + 202, cfg.stay_probability, noise)
            for delay in cfg.delays:
                chosen = {name: select_and_test(name, cfg, seed, delay, train, validation, test) for name in TRAINED}
                other = {name: _evaluate(name, baseline_cfg, seed, delay, train, test) for name in BASELINES}
                metrics = {name: chosen[name]["test_accuracy"] for name in TRAINED}
                metrics.update({name: other[name]["accuracy"] for name in BASELINES})
                counts = {(len(train) * (cfg.sequence_length - start), len(test) * (cfg.sequence_length - start))}
                counts.update((item["train_samples"], item["test_samples"]) for item in other.values())
                if len(counts) != 1:
                    raise RuntimeError("inconsistent held-out sample window")
                runs.append({
                    "seed": seed, "noise": noise, "delay": delay,
                    "samples": {
                        "train": len(train) * (cfg.sequence_length - start),
                        "validation": len(validation) * (cfg.sequence_length - start),
                        "test": len(test) * (cfg.sequence_length - start),
                    },
                    "metrics": metrics,
                    "selection": chosen,
                    "baseline_profiling": {name: other[name]["profiling"] for name in BASELINES},
                })
    summaries = []
    for noise in cfg.noise_levels:
        for delay in cfg.delays:
            group = [row for row in runs if row["noise"] == noise and row["delay"] == delay]
            summaries.append({
                "noise": noise, "delay": delay, "seeds": len(group),
                "metrics": {
                    model: {
                        "mean": statistics.mean(r["metrics"][model] for r in group),
                        "population_stdev": statistics.pstdev(r["metrics"][model] for r in group),
                    } for model in MODELS
                },
                "paired_deltas": {
                    "gru_minus_elman": statistics.mean(
                        r["metrics"]["trainable_gru"] - r["metrics"]["trainable_elman"] for r in group),
                    "gru_minus_spiking_lif": statistics.mean(
                        r["metrics"]["trainable_gru"] - r["metrics"]["spiking_lif"] for r in group),
                },
            })
    return {
        "schema_version": "7.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "validation_selected_gru_competition",
        "configuration": asdict(cfg),
        "parameter_budgets": sizes,
        "models": {
            "trainable_gru": "Trainable GRU with reset/update gates, full BPTT, per-sequence SGD",
            "trainable_elman": "Trainable tanh Elman RNN, full BPTT, per-sequence SGD",
            "spiking_lif": "Fixed spiking LIF reservoir and trainable logistic readout (untuned)",
            "tanh_reservoir": "Fixed recurrent tanh reservoir and logistic readout (untuned)",
            "history_window": "Explicit observed-input history window and logistic readout (untuned)",
            "delayed_tap": "Exact observed delayed bit and logistic readout (untuned)",
            "majority": "Most frequent training label",
        },
        "protocol": {
            "split": "Three independently seeded complete sequence streams per seed/noise level: train, validation and test",
            "test_usage": "Trainable models select learning rate using validation BCE only; test accuracy evaluated once afterwards",
            "baseline_tuning": "Fixed reservoir/explicit-memory baselines use an untuned inherited readout protocol",
            "window": "Every model uses time indices from max(burn_in, max(delays)) onward",
            "training": "Full BPTT and sequence-wise SGD for GRU and Elman; global gradient clipping",
            "budget": "Cap applies to trainable GRU/Elman parameter counts; largest hidden widths <=32 fitting cap; not exact parameter equality",
            "repeatability": "Clean/noisy sequences, model initialization, training and validation selection are seeded; durations are not",
            "profiling": "Candidate-selection and single-test-evaluation wall elapsed nanoseconds for trainable models, not standardized energy or matched GPU performance",
        },
        "runs": runs,
        "summary": summaries,
        "limitations": [
            "Pure digital simulation, no living biological tissue or consciousness evidence.",
            "Trainable GRU and Elman have capped, not exactly matched, trainable parameter counts and differing operation counts.",
            "Fixed spiking and tanh reservoirs have different fixed parameter/computation budgets and no validation tuning; architecture comparison is exploratory.",
            "Validation selects among a small predeclared learning-rate grid; no independent architecture/hyperparameter optimization study.",
            "Seed-sweep dispersion is descriptive, not a confidence interval, and short Markov tasks cannot establish general intelligence.",
            "Real Python elapsed time depends on CPU, interpreter, order and system load; it is not measured energy.",
        ],
    }


def write_csv(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["seed", "noise", "delay", "train_samples", "validation_samples", "test_samples"]
        columns += [f"{model}_test_accuracy" for model in MODELS]
        columns += ["gru_selected_learning_rate", "elman_selected_learning_rate"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in report["runs"]:
            values = {
                "seed": run["seed"], "noise": run["noise"], "delay": run["delay"],
                "train_samples": run["samples"]["train"],
                "validation_samples": run["samples"]["validation"],
                "test_samples": run["samples"]["test"],
                **{f"{model}_test_accuracy": run["metrics"][model] for model in MODELS},
                "gru_selected_learning_rate": run["selection"]["trainable_gru"]["selected_learning_rate"],
                "elman_selected_learning_rate": run["selection"]["trainable_elman"]["selected_learning_rate"],
            }
            writer.writerow(values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio Phase 7: validation-selected GRU experiment")
    parser.add_argument("--seed-start", type=int, default=369)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.15])
    parser.add_argument("--train-sequences", type=int, default=8)
    parser.add_argument("--validation-sequences", type=int, default=4)
    parser.add_argument("--test-sequences", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--burn-in", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.03, 0.1])
    parser.add_argument("--parameter-budget", type=int, default=95)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = ValidationConfig(
            seeds=tuple(args.seed_start + i for i in range(args.seed_count)),
            delays=tuple(args.delays), noise_levels=tuple(args.noise_levels),
            train_sequences=args.train_sequences, validation_sequences=args.validation_sequences,
            test_sequences=args.test_sequences, sequence_length=args.sequence_length,
            burn_in=args.burn_in, epochs=args.epochs,
            learning_rates=tuple(args.learning_rates), parameter_budget=args.parameter_budget,
        )
        report = run_validation(cfg)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload, encoding="utf-8")
        print("Saved Phase 7 report: " + str(args.json_output))
    else:
        print(payload, end="")
    if args.csv_output:
        write_csv(report, args.csv_output)
        print("Saved CSV: " + str(args.csv_output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
