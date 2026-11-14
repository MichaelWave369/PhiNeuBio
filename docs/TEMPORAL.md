# Phase 5: temporal structure and measured Python computational costs

## Scientific question

How does a simulated spiking-neural reservoir compare with a conventional recurrent tanh state model and explicit software memory when the inputs are **temporally correlated**?

**Everything is software.** No tissue, experimental biological data, clinical claim, or consciousness inference.

## Input generation

The clean bit stream is a first-order Markov process. A bit stays unchanged with configurable probability `stay_probability` (default 0.85), otherwise it flips. Each input observation is independently flipped with the configured noise rate. Labels are the *clean* signal from `delay` steps earlier.

Train/test are independent complete streams for each seed. The same underlying clean streams and nested random noise draws are used for all noise levels. Each model gets **the identical sequence, labels, and retained time indices** for each seed/noise/delay; the readouts use the same shuffled sample order, epoch count, and learning rate.

All samples before `t = max(burn_in, window_size - 1, max(delays))` are discarded. Models do not see targets as features, except for the deliberate delayed observed-input tap that uses an explicitly stored **noisy** past input. Models never receive clean target bits as inputs.

## Five comparators

- **Spiking LIF:** fixed recurrent leaky integrate-and-fire neurons with trained logistic readout of potentials and spikes.
- **Conventional tanh reservoir:** fixed random recurrent tanh state-transition network (echo-state-style) with trained logistic readout. This is **not a fully trained RNN**: it does not use backpropagation-through-time or gradient updates of recurrent weights.
- **History window:** logistic classifier given a stored history of the last W observed bits. W defaults to `max(delays) + 1`.
- **Delayed tap:** logistic classifier given the observed bit at exactly the requested delay. This is a strong **explicit memory** control, not a fair no-memory reference.
- **Majority:** predict most common label in training data.

No architecture has been hyperparameter-tuned to achieve comparable parameter counts or compute budgets; results are exploratory.

## Real measurements, carefully scoped

Timing uses `time.perf_counter_ns()` and memory tracking uses `tracemalloc`. They cover **Python model construction, train/test feature extraction, readout training, and held-out scoring for each model**. Input generation, interpreter startup, JSON serialization, and file I/O are outside the measurement.

The report records extraction, fit, evaluation, total nanoseconds and peak traced Python allocation bytes. Profiling is sequential on the running Python process, with garbage collection before each model and tracemalloc restarted for each. Measurements are real observations but subject to run-order, interpreter, garbage collector, hardware, thermal state and other workloads.

A measured **Python traced-allocation peak is not RSS**, and `perf_counter_ns` is not CPU-only time. Neither is a measurement of energy or material efficiency. CI checks that measurements exist and are internally consistent, **never asserts exact wall-clock values or rankings**. Repeated local hardware runs are needed for credible speed comparisons.

## Run locally

Requires Python 3.10+; no external dependencies.

    python -m phineubio.temporal --json-output dashboard/temporal-report.json --csv-output temporal-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/temporal.html. Or open the HTML directly and select the report.

Quick CI-like experiment:

    python -m phineubio.temporal --seed-count 1 --delays 1 2 --noise-levels 0 0.2 --train-sequences 3 --test-sequences 2 --sequence-length 12 --neurons 5 --epochs 3

## Important limits and next experiments

This phase adds a **fixed recurrent tanh comparator**, not a fully trained RNN. To investigate advantages responsibly, follow up with a trained GRU or RNN baseline, matched model budgets, broad seed sweeps, multiple sequence distributions, pre-registered comparisons, and repeated on-device profiling. Preserve negative results. Research performance does not imply consciousness.
