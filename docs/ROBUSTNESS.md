# Phase 4: paired noise × delay × seed study

## Goal

Test a falsifiable question: **does a recurrent digital spiking reservoir retain information usefully when compared to explicit, conventional storage under controlled noise?** This is a software-only model study, not a biological experiment.

## Models

1. `spiking_recurrent`: fixed LIF reservoir, trained logistic readout of voltage and spike vectors.
2. `spiking_no_recurrence`: identical seeded LIF implementation with recurrent gain zero. Membrane leak is still stateful.
3. `current_input`: logistic classifier trained on only the current observed bit.
4. `history_window`: logistic classifier trained on an ordinary sliding window of observed bits `[t, t-1, ..., t-W+1]`. Window width defaults to max tested delay + 1.
5. `delayed_input_tap`: logistic classifier explicitly given the observed bit from the requested delay.
6. `majority`: majority label on the train split.

**Important:** These are not equal-parameter or equal-compute models. The history window and delayed tap have explicit conventional memory, a fair *functionality* control but a different resource budget. The no-recurrence reservoir still has leaky neuron-state memory and is not an entirely stateless baseline.

## Experimental protocol

Each `seed` defines train/test random bit streams. The *clean* target at time t is the bit at t-delay. Independently flipped noisy *observed* inputs feed all models. Noise levels are coupled using identical random draws: raising the flip probability yields nested masks. All candidates are evaluated on the same sequence-level train/test partition, labels, sample window, and readout shuffle ordering.

Every delay and model uses samples at times `t >= max(burn_in, window_size - 1, max(delays))`. This deliberately discards an identical initial section so a wider history-window model cannot get different test samples. Inputs are independent fair bits: there is no redundancy for genuine denoising across time. The theoretical best classification accuracy from the corresponding noisy bit alone is `1 - flip_probability` in expectation (for p≤0.5). Do not mistake random fluctuations above it for generalization.

## Reproduce

From the repository root, with Python 3.10+ (standard library only):

    python -m phineubio.robustness --json-output dashboard/robustness-report.json --csv-output robustness-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/robustness.html. Or open the HTML directly and choose the JSON report.

Short run:

    python -m phineubio.robustness --seed-count 2 --delays 1 3 --noise-levels 0 0.25 --train-sequences 6 --test-sequences 3 --sequence-length 16 --burn-in 4 --neurons 8 --epochs 5

The report records per-run held-out accuracy for all six models, seed-level variability, paired accuracy deltas, and architecture-derived work proxies. CSV has one row per seed × noise × delay.

## What work proxies actually mean

- `feature_dimensions`: per-sample input vector width to each logistic readout.
- `trainable_readout_parameters`: logistic weights plus intercept.
- `readout_coefficient_update_slots`: epochs × train samples × readout width, not a counted FLOP total.
- `reservoir_neuron_state_updates`: number of simulated neuron step updates over train and test inputs (the extraction phase).
- `dense_recurrent_pair_scans`: for the current Python implementation, number of entries traversed by the inner adjacency matrix scans, including zero-weight edges. **Even the no-recurrence implementation traverses these zero entries**.
- `feature_values_per_test_set`: number of scalars in the materialized test feature vectors.

These are deterministic architecture/work *descriptors*. They are not instrumented CPU instructions, exact floating-point operation counts, actual runtimes, memory peaks, or joules. Comparing two models by one count alone is not a hardware-efficiency conclusion.

## Caveats and next steps

- No lived biology, clinical use, or consciousness test.
- This phase does **not** yet benchmark equal-parameter RNNs or wall-clock time.
- A full statistical comparison would involve predeclared hypotheses, more seeds and tasks, confidence intervals, and controls for hyperparameter tuning.
- Next: richer temporally redundant signals, a standard recurrent neural-network comparison, and separately measured runtime / memory footprints.

Negative results count as results. Publish those too.
