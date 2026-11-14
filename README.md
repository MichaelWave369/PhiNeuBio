# ΦNeuBio

**PhiNeuBio** is an open-source, software-first research toolkit for bio-inspired computation, spiking neural networks, and measurable closed-loop learning.

**Current scope:** simulated neurons only. This repository does not operate living tissue, establish consciousness, or offer medical technology.

## Phase 1: spiking reservoir memory

The first prototype implements seeded leaky integrate-and-fire (LIF) neurons, sparse recurrent connectivity, and a trained logistic readout. A delayed-bit task tests whether reservoir state retains information from earlier input.

Run:

    python -m phineubio

Try a smaller experiment:

    python -m phineubio --train-sequences 12 --test-sequences 6 --delay 2 --epochs 8

## Phase 2: reward-feedback experiment and activity dashboard

A new experiment presents binary cues to the simulated reservoir. A logistic readout selects an action, receives a binary reward, and updates its weights. The digital reservoir's recurrent synapses **remain fixed**. Because the task has only two deterministic choices, reward effectively reveals the correct answer; this is **not general reinforcement learning**.

Generate a fully reproducible report with per-neuron simulated activity:

    python -m phineubio.feedback --output dashboard/feedback-report.json

Launch the offline dashboard from the repository root:

    python -m http.server 8000

Open [ΦNeuBio Activity Lab](http://localhost:8000/dashboard/) in your browser. It replays the saved simulated neuron spike raster, choices, rewards, and cumulative reward curve. No cloud connection is required. Alternatively, open `dashboard/index.html` directly and use **Choose JSON report**.

The viewer is an **animated trace replay**, not a live recording of biological neurons. For protocol and interpretation, read [Phase 2 experiment notes](docs/FEEDBACK.md).

## Phase 3: controlled memory benchmarks

Compare recurrent spiking models against nonrecurrent controls, simple logistic baselines, and explicit digital delay memory across multiple seeds, delay lengths, and noisy inputs. Results are **descriptive comparisons**, not proof of biological advantage.

    python -m phineubio.benchmarks --json-output dashboard/benchmark-report.json --csv-output benchmark-results.csv
    python -m http.server 8000

Open [Memory Benchmark Lab](http://localhost:8000/dashboard/benchmark.html) to explore accuracy-by-delay curves and seed variability. You can also choose a JSON report directly from the HTML file picker.

Read [Phase 3 protocol and limitations](docs/BENCHMARKS.md). The tapped-delay control has explicit memory and is *supposed* to be a tough comparison.

## Phase 4: robustness and computational-work comparisons

Challenge the spiking reservoir with a conventional sliding input-history window and independently noisy bits. All six models receive paired train/test data across random seeds, noise levels and input delays.

    python -m phineubio.robustness --json-output dashboard/robustness-report.json --csv-output robustness-results.csv
    python -m http.server 8000

Open [Robustness Lab](http://localhost:8000/dashboard/robustness.html) for interactive accuracy-versus-delay curves, noise filters, individual seeded results, and **analytical work descriptors**. These are not CPU benchmarks, electricity measurements or a biological experiment.

Read [Phase 4 research protocol](docs/ROBUSTNESS.md).

## Run tests

Requires Python 3.10+, no third-party runtime packages.

    python -m unittest discover -s tests -v

GitHub Actions validates tests and all four JSON experiment entry points with Python 3.10 and 3.12.

## Research principles

1. **Reproduce first.** Record random seeds, experiment configuration, held-out results, and limitations.
2. **Separate mechanisms from claims.** A spike simulator is not a culture of biological neurons; task accuracy is not evidence of awareness.
3. **Use independent comparisons.** Report trivial baselines, sample counts, and limitations, including negative results.
4. **Protect future experiments.** Any biological hardware adapter must be explicitly optional and independently safety- and ethics-reviewed.
5. **Integrate by contract.** PhiStream, PhiMemory9, and CompAware can consume versioned reports without coupling their runtimes.

See [Architecture](docs/ARCHITECTURE.md), [Robustness study](docs/ROBUSTNESS.md), [Delayed-memory protocol](docs/EXPERIMENTS.md), [Integrations](docs/INTEGRATIONS.md), [Feedback protocol](docs/FEEDBACK.md), [Memory benchmarks](docs/BENCHMARKS.md), and [Research/ethics boundaries](docs/ETHICS.md).

## Status

- Phase 1: software-only spiking reservoir and delayed-memory benchmark
- Phase 2: closed-loop binary-feedback learning, JSON traces, and local activity dashboard
- Phase 3: paired seed sweeps, stronger baselines, controlled noise and delay benchmarks
- Phase 4: paired noise sweeps, conventional input-window controls, and analytical computational-work descriptors
- Phase 5 (proposed): real runtime profiling, RNN comparators, temporally redundant signals
- Research-only future: opt-in external biological hardware adapters following safety review

Licensed under [MIT](LICENSE).
