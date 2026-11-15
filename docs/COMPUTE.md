# Phase 11: structural compute budgets and local CPU inference probes

PhiNeuBio compares different software memory architectures on the **same complete train/validation/test/probe streams**. This phase adds a limited but explicit **recurrent-pair-visit budget**, separating deterministic structural accounting from actual CPU wall-clock timing.

## What is and isn't matched

All four recurrent systems use the largest hidden width (up to 32) that fits the predeclared **matrix-entry visit cap**:

- Trainable GRU: three n × n recurrent gate matrices scanned per input step, so proxy = 3n².
- Trainable Elman RNN: one n × n recurrent matrix, proxy = n².
- Fixed LIF spiking reservoir: one n × n adjacency scan (including zero weights), proxy = n².
- Fixed tanh reservoir: one n × n recurrent matrix, proxy = n².

The default structural cap is **108** visits/step, corresponding to 6-unit GRU (108 visits) and 10-unit Elman/LIF/tanh systems (100 visits each). The exact model widths, fixed weight slots, trainable weight slots, readout feature sizes, nonlinearities and LIF thresholds are recorded.

**Structural pair visits are NOT equal FLOPs or actual CPU instructions.** A GRU matrix entry performs different work from an active-spike check; LIF traversal visits zero weights; nonlinearities and Python branch overhead differ. Trained recurrent networks optimize all their recurrent weights; fixed reservoirs only fit a logistic readout. Training compute is **not** matched.

Explicit history windows and delayed-input taps have **zero recurrent pair visits** by design. They remain unmatched but important conventional baselines; it would be misleading to force them into a recurrent compute budget.

## Experimental protocol

- Each condition has independent, seeded training, validation, final test and inference-probe sequences.
- Same input stream, noise mask, target delay, held-out sample index and learning-rate candidates for every model.
- No test labels or probe targets are used for hyperparameter selection. Rates are selected by independent validation loss, including the fixed reservoirs.
- One final held-out test accuracy reading per model.
- Inference models and fixed reservoir matrices are constructed **before** timing.
- Each model receives one unmeasured warm-up. Repeated rounds shuffle the **entire seven-model execution order** by a reproducible random seed. Garbage collection occurs before, outside each timed pass. The exact round orders and raw times are exported.
- Timing uses `perf_counter_ns` in the **same Python process** and CPU environment. Every pass includes complete sequence forward computation, warm-up timesteps in each sequence and readout prediction for retained timesteps. The reported probe sample denominator counts retained timesteps, so raw median elapsed nanoseconds is the more transparent measure.

Repetitions reduce some order bias but **cannot** eliminate ambient CPU load, thermal effects, Python overhead or algorithmic differences. No watts, joules, per-model FLOPs, GPU benchmarks or biological neuron measurements are claimed. Memory allocation, physical power consumption and train-time parity remain future work.

## Run locally

Standard-library Python 3.10+, no paid dependencies:

    python -m phineubio.compute --json-output dashboard/compute-report.json --csv-output compute-results.csv

For a longer local CPU comparison, vary the number of seeds and rounds:

    python -m phineubio.compute --seed-count 5 --inference-repeats 10 --pair-visit-budget 108 --json-output dashboard/compute-report.json --csv-output compute-results.csv

Then serve the repository and open the dashboard:

    python -m http.server 8000

Open http://localhost:8000/dashboard/compute.html or load a JSON file via the file picker.

Quick CI-scale smoke test:

    python -m phineubio.compute --seed-count 1 --signal-families iid --noise-levels 0 --delays 1 --pair-visit-budget 75 --train-sequences 3 --validation-sequences 2 --test-sequences 2 --probe-sequences 2 --sequence-length 12 --burn-in 3 --epochs 2 --inference-repeats 2 --json-output dashboard/compute-report.json --csv-output compute-results.csv

## Scientific limits and next research work

This phase *constrains* one recurrent-loop complexity proxy, **not** exact computational work. It improves transparency but is not an architecture-speed championship. The experiment is software-only, with binary toy tasks and small seed sweeps; negative results remain valid research data. Next: instruction- or hardware-counter-based measurements where accessible, equal training-parameter and optimization budgets, statistical confidence across hardware replicates and separately metered electrical energy if actual physical hardware instrumentation becomes available.
