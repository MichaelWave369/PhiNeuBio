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

## Phase 5: correlated signals and local Python profiling

A seeded Markov signal adds temporal structure missing from the Phase 3/4 IID-bit experiments. Compare a **fixed recurrent tanh network**, fixed recurrent spiking LIF network, conventional history window, delayed-input tap, and majority baseline. The recurrent networks are **not fully trained RNNs**; only their readouts learn.

    python -m phineubio.temporal --json-output dashboard/temporal-report.json --csv-output temporal-results.csv
    python -m http.server 8000

Open [Temporal Signal Lab](http://localhost:8000/dashboard/temporal.html) to compare descriptive accuracy and **actual Python-process elapsed time and traced allocations**. These are environment-dependent observations, not electrical energy measurements or optimized performance rankings.

Read the [Phase 5 protocol and profiler limitations](docs/TEMPORAL.md).

## Phase 6: trainable recurrent competition

A from-scratch **Elman tanh recurrent neural network** now trains recurrent, input and output weights using full-sequence backpropagation through time (BPTT), sequence-wise SGD, and gradient clipping. We compare it with the fixed LIF spiking reservoir, fixed tanh reservoir, history-window and delayed-tap controls on identical held-out correlated sequences.

    python -m phineubio.competition --json-output dashboard/competition-report.json --csv-output competition-results.csv
    python -m http.server 8000

Open [Adaptive Competition Lab](http://localhost:8000/dashboard/competition.html). It displays test accuracy, model comparisons, training-loss curves and measured Python profiling. Recurrent network training is now real, but this is **not** a GRU, and budgets are not parameter-matched.

Read the [Phase 6 research protocol](docs/COMPETITION.md), including the gradient-check test and limitations.

## Phase 7: trainable GRU and validation-first research

A pure-Python **trainable GRU** now competes with the Phase 6 trainable Elman RNN, fixed LIF and tanh reservoirs, and explicit software-memory controls. Both trainable models select learning rates using **independent validation sequences**, then run one final test evaluation. Their exact parameter counts are reported under a shared cap (not exact parity).

    python -m phineubio.validation --json-output dashboard/validation-report.json --csv-output validation-results.csv
    python -m http.server 8000

Open [Validation Lab](http://localhost:8000/dashboard/validation.html) to compare held-out accuracy, selected hyperparameters and parameter budgets. Read the [Phase 7 methodology](docs/VALIDATION.md).

## Phase 8: validation fairness and multiple signals

**All trainable readouts** now get a validation-selected learning rate. That includes fixed spiking LIF reservoirs, fixed tanh reservoirs, explicit history-window memory, delayed-input controls, and the GRU/Elman models. The new second signal family uses periodic binary templates, alongside first-order Markov streams. The final test remains separate from all tuning.

    python -m phineubio.fairness --json-output dashboard/fairness-report.json --csv-output fairness-results.csv
    python -m http.server 8000

Open [Fairness Lab](http://localhost:8000/dashboard/fairness.html) for signal/noise/delay comparisons, selected validation parameters, and **repeated inference timing on an independent profiling-only dataset**. Models are not equal compute or equal size; these diagnostics are not energy measurements.

Read the [Phase 8 protocol](docs/FAIRNESS.md).

## Phase 9: seed-paired evidence audit

A third signal family of IID fair random bits now acts as a no-temporal-memory control. The independent **Evidence Audit** validates saved Phase 8 reports, predeclares four paired architecture contrasts, summarizes win/loss counts across random seeds, and adds reproducible seed-level bootstrap intervals **only when five or more seeds are available**. No model training is rerun during the audit.

    python -m phineubio.fairness --seed-count 5 --signal-families iid markov periodic --json-output dashboard/fairness-report.json
    python -m phineubio.audit --input dashboard/fairness-report.json --output dashboard/audit-report.json --csv-output audit-effects.csv
    python -m http.server 8000

Open [Evidence Audit Lab](http://localhost:8000/dashboard/audit.html). The Phase 8 generator records anonymous runtime characteristics for provenance (OS family, interpreter, architecture and CPU count), but no hostnames, usernames, paths or serial numbers. The audit reports exploratory differences, **not statistically proven superiority** or energy efficiency.

Read the [Phase 9 statistical limitations and protocol](docs/AUDIT.md).

## Phase 10: sample efficiency, frozen protocol, and multiplicity awareness

Phase 10 adds a committed study plan that repeats the Phase 8 seven-model experiment at multiple **nested training-set sizes**, with the same independent validation/test streams for each seed. It predeclares all four model comparisons and reports seed-paired bootstrap intervals plus exact directional sign tests with **Holm correction across the entire declared family**.

    python -m phineubio.study --plan study-plan.json --output dashboard/study-report.json --csv-output study-effects.csv
    python -m http.server 8000

Open [Study Protocol Lab](http://localhost:8000/dashboard/study.html) to inspect sample-efficiency curves, unadjusted descriptive intervals and adjusted p-values. The committed plan runs 40 seed-budget-condition experiments and is a CPU-intensive research experiment; CI uses a small smoke case.

Read [Phase 10 study methodology and limits](docs/STUDY.md). The study excludes nondeterministic hardware timings from its reproducible report and **still does not establish equal computational budget, energy efficiency, or biological advantage**.

## Phase 11: constrained recurrent work and independent CPU probes

A **recurrent matrix-entry visit budget** now sets hidden sizes for trained GRU, trained Elman, fixed spiking LIF and fixed tanh recurrence. Each is evaluated with paired train/validation/test data and repeated **CPU inference** on a separate probe set. Probe model order is randomized by round, and the raw times, parameter slots and architecture work descriptors are retained.

    python -m phineubio.compute --json-output dashboard/compute-report.json --csv-output compute-results.csv
    python -m http.server 8000

Open [Compute Budget Lab](http://localhost:8000/dashboard/compute.html) to inspect held-out accuracy, recurrent visits per step, trained/fixed weight slots and observed CPU timings.

Read [Phase 11 research methodology](docs/COMPUTE.md). Structural recurrent-loop visits are **not equal FLOPs, CPU instructions, training compute or energy use**. The conventional explicit-memory baselines deliberately remain unmatched.

## Phase 12: multiple recurrent budgets and operation ledgers

The [Budget Sweep Lab](http://localhost:8000/dashboard/sweep.html) compares several recurrent matrix-visit caps on identical, independently seeded train/validation/test/probe streams. Along with paired held-out accuracy and CPU timing, the new **work ledger** separately reports dense recurrent products, readout products, nonlinear evaluations, LIF spike-triggered weight additions, nonzero active edges and threshold checks. These categories are **not** exact hardware instructions, standardized FLOPs or physical energy measurements.

    python -m phineubio.sweep --json-output dashboard/sweep-report.json --csv-output sweep-results.csv
    python -m http.server 8000

Open the link above or load the saved JSON locally. Read [Phase 12 methodology](docs/SWEEP.md). The descriptive structural Pareto screen compares recurrent models only; conventional explicit-memory controls remain separate.

## Phase 13: learning-cost accounting and nonstationary signals

[Learning Cost Lab](http://localhost:8000/dashboard/training.html) compares seven architectures across training-epoch budgets and a new switching Markov signal. Every trained candidate receives equal **retained target presentations**, while the report explicitly records unequal sequence-wise BPTT and pointwise readout optimizer update counts, local candidate-selection time and independently held-out test accuracy.

    python -m phineubio.training --json-output dashboard/training-report.json --csv-output training-results.csv
    python -m http.server 8000

Open the Lab URL, or load the generated JSON locally. Selection timing includes initialization, feature extraction, optimizer training and validation scoring. It is **not** pure gradient processing time, equal compute, electrical energy or evidence of biological tissue learning.

Read [Phase 13 research methodology](docs/TRAINING.md).

## Run tests

Requires Python 3.10+, no third-party runtime packages.

    python -m unittest discover -s tests -v

GitHub Actions validates tests and all eight prior simulation entry points, the Phase 9 audit, Phase 10 study Phase 11 compute benchmark and Phase 12 budget sweep and Phase 13 training-cost study with Python 3.10 and 3.12.

## Research principles

1. **Reproduce first.** Record random seeds, experiment configuration, held-out results, and limitations.
2. **Separate mechanisms from claims.** A spike simulator is not a culture of biological neurons; task accuracy is not evidence of awareness.
3. **Use independent comparisons.** Report trivial baselines, sample counts, and limitations, including negative results.
4. **Protect future experiments.** Any biological hardware adapter must be explicitly optional and independently safety- and ethics-reviewed.
5. **Integrate by contract.** PhiStream, PhiMemory9, and CompAware can consume versioned reports without coupling their runtimes.

See [Architecture](docs/ARCHITECTURE.md), [Robustness study](docs/ROBUSTNESS.md), [Temporal research](docs/TEMPORAL.md), [Adaptive competition](docs/COMPETITION.md), [Independent validation](docs/VALIDATION.md), [Phase 8 fairness](docs/FAIRNESS.md), [Phase 9 evidence audit](docs/AUDIT.md), [Phase 10 study protocol](docs/STUDY.md), [Phase 11 compute protocol](docs/COMPUTE.md), [Phase 12 budget sweep](docs/SWEEP.md), [Phase 13 learning-cost protocol](docs/TRAINING.md), [Delayed-memory protocol](docs/EXPERIMENTS.md), [Integrations](docs/INTEGRATIONS.md), [Feedback protocol](docs/FEEDBACK.md), [Memory benchmarks](docs/BENCHMARKS.md), and [Research/ethics boundaries](docs/ETHICS.md).

## Status

- Phase 1: software-only spiking reservoir and delayed-memory benchmark
- Phase 2: closed-loop binary-feedback learning, JSON traces, and local activity dashboard
- Phase 3: paired seed sweeps, stronger baselines, controlled noise and delay benchmarks
- Phase 4: paired noise sweeps, conventional input-window controls, and analytical computational-work descriptors
- Phase 5: temporally correlated signals, fixed tanh recurrence comparison, and measured Python profiling
- Phase 6: trainable recurrent Elman RNN with BPTT, loss curves, and paired held-out comparisons
- Phase 7: trainable GRU with gradient checks, independent validation selection, and parameter-capped comparisons
- Phase 8: independent validation of fixed readouts, periodic signals, repeated independent-probe timing
- Phase 9: IID negative control, complete-grid audit, seed-level exploratory bootstrap uncertainty and local runtime provenance
- Phase 10: a predeclared sample-efficiency plan, paired sign tests and family-wide Holm correction
- Phase 11: recurrent-loop visit budgets, paired accuracy and randomized independent CPU inference probes
- Phase 12: multi-budget sweeps, auditable LIF activity and separate algorithmic work ledgers
- Phase 13: matched target-exposure budgets, explicit optimizer-update counts, local selection timing and synthetic switching signals
- Phase 14 (proposed): instrumented compute parity and broader temporal tasks
- Research-only future: opt-in external biological hardware adapters following safety review

Licensed under [MIT](LICENSE).
