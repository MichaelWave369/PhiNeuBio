# ΦNeuBio

**PhiNeuBio** is an open-source, software-first research toolkit for bio-inspired computation, spiking neural networks, and measurable closed-loop learning.

**Current scope:** simulated neurons only. This repository does not operate living tissue, establish consciousness, or offer medical technology.

## Phase 1: a runnable spiking-neural-reservoir experiment

The first prototype implements seeded leaky integrate-and-fire (LIF) neurons, sparse recurrent connectivity, and a trained logistic readout. A delayed-bit task tests whether reservoir state retains information from earlier input. It reports held-out accuracy alongside a majority-class baseline. These are small illustrative benchmarks, **not evidence of superior performance, biological intelligence, or consciousness**.

Requirements: Python 3.10+; no third-party runtime dependencies.

Run the experiment:

    python -m phineubio

Try a smaller, faster experiment:

    python -m phineubio --train-sequences 12 --test-sequences 6 --delay 2 --epochs 8

Run tests:

    python -m unittest discover -s tests -v

The command writes a machine-readable JSON report to stdout, including the seed, configuration, sample counts, baseline, and held-out test accuracy. Re-running with identical arguments and Python environment produces identical reports.

## Research principles

1. **Reproduce first.** Record random seeds, experiment configuration, and held-out results.
2. **Separate mechanisms from claims.** A spike simulator is not a culture of biological neurons; task accuracy is not evidence of awareness.
3. **Use independent comparisons.** Report trivial baselines, sample counts, and limitations, including negative results.
4. **Protect future experiments.** Any real biological hardware interface must be explicitly optional and independently safety- and ethics-reviewed.
5. **Integrate by contract.** PhiStream, PhiMemory9, and CompAware can consume versioned reports without coupling their runtime to this repository.

See [Architecture](docs/ARCHITECTURE.md), [Experiment protocol](docs/EXPERIMENTS.md), [Integrations](docs/INTEGRATIONS.md), and [Research and ethics boundaries](docs/ETHICS.md).

## Status

- Phase 1: software-only simulation and reproducible delayed-memory benchmark
- Phase 2 (proposed): feedback-control task, feature ablations, and baseline comparisons
- Phase 3 (research only): opt-in external-hardware adapters following safety review

Licensed under [MIT](LICENSE).