# Phase 6: trainable recurrent competition

## Why it matters

The previous PhiNeuBio tanh model was a **fixed-weight reservoir**. It only trained a logistic readout. Phase 6 introduces a genuinely trainable recurrent transition matrix through a from-scratch *Elman tanh RNN* using full-sequence backpropagation through time (BPTT).

This is a digital machine-learning experiment, **not living-neuron computing**, and does not test consciousness.

## Models and data

- **Trainable RNN:** single scalar observed-bit input, tanh hidden layer, trainable recurrent matrix and input weights, trainable sigmoid output. Full-sequence BPTT, sequence-wise SGD, mean binary cross-entropy on retained target steps, and global gradient-norm clipping.
- **Spiking LIF:** fixed recurrent spiking reservoir and trained logistic readout.
- **Tanh reservoir:** fixed recurrent tanh transition network and trained logistic readout.
- **History window:** conventional explicit history with logistic readout.
- **Delayed tap:** conventional stored bit at the exact delay.
- **Majority:** constant prediction of the training label majority.

All models receive the **same** seeded Markov signal, independent input-bit flips, clean target bits, and completely separate train/test sequences. All use the same retained time indices, with initial samples discarded up to `max(burn_in, max(delays))` (which covers the default history window).

The hidden width is the same for spiking, fixed tanh, and trained RNN. **Total parameters, arithmetic operations, effective capacity, optimization schedules, and wall-clock budgets are not matched.** The trained RNN optimizes a sequence loss. The fixed reservoirs fit their readouts to individually shuffled timestep features; this may advantage either approach. Do not interpret the comparison as a proof of hardware superiority.

## Reproduce

Requires Python 3.10+, standard library only:

    python -m phineubio.competition --json-output dashboard/competition-report.json --csv-output competition-results.csv
    python -m http.server 8000

Open `http://localhost:8000/dashboard/competition.html` to inspect the results and training-loss curve.

Quick test:

    python -m phineubio.competition --seed-count 1 --delays 1 2 --noise-levels 0 --train-sequences 4 --test-sequences 3 --sequence-length 14 --burn-in 3 --neurons 4 --epochs 3

## Metrics

The report includes held-out accuracy, per-epoch training cross-entropy for the trainable RNN, the RNN's initial held-out accuracy (never used for optimization), total parameter count for the trainable RNN, and `perf_counter_ns` / `tracemalloc` local profiling alongside legacy comparators.

An independent finite-difference test checks the gradient of a recurrent weight against the analytical BPTT gradient. A second test verifies that recurrent weights genuinely change during training.

**Note:** The trainable RNN training-timing bucket includes the initial held-out evaluation, and other models' training-timing buckets are different. Treat timings as experimental diagnostics, not matched latency benchmarks. Accuracy is seeded and reproducible; times and traced allocations are not guaranteed stable.

## Next research priorities

1. Add fully trained GRU/LSTM controls (with explicit optimizer and budget parity).
2. Match parameter counts, tune learning rates on separate validation sequences, and evaluate on a frozen test split.
3. Profile repeated local runs with randomized evaluation order and separate preprocessing/training/inference stages.
4. Add multi-task nonbinary signals and sample-complexity curves, publishing negative findings too.
