"""Command-line entry point for the PhiNeuBio Phase 1 benchmark."""

import argparse
import json
from typing import Sequence

from .experiment import ExperimentConfig, run_experiment
from .reservoir import ReservoirConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio: digital LIF reservoir memory benchmark")
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--neurons", type=int, default=32)
    parser.add_argument("--train-sequences", type=int, default=36)
    parser.add_argument("--test-sequences", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=36)
    parser.add_argument("--burn-in", type=int, default=6)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig(
            seed=args.seed, train_sequences=args.train_sequences,
            test_sequences=args.test_sequences, sequence_length=args.sequence_length,
            burn_in=args.burn_in, delay=args.delay, epochs=args.epochs,
            reservoir=ReservoirConfig(seed=args.seed, neurons=args.neurons)
        )
        report = run_experiment(config)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
