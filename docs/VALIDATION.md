# Phase 7: gated recurrence and independent validation

Phase 7 introduces a **trainable GRU** built in pure Python with full-sequence backpropagation through time (BPTT). The GRU's input, reset gate, update gate, candidate and output weights all learn. Six finite-difference checks compare numerical and analytical gate gradients.

## Three disjoint sequence splits

The experiment generates entirely separate seeded Markov sequences for training, validation and testing. The same three splits serve each candidate architecture. For both trainable models, Phase 7 predeclares a small list of learning rates, trains each candidate on **training only**, scores candidate binary cross-entropy on **validation only**, and selects the lowest validation-loss candidate. Only then is the selected model evaluated against the independent test sequences.

This avoids tuning on the final test set, but **does not** resolve variance from tiny datasets or make comparisons fully matched.

## Comparison models and budget

- **Trainable GRU** with reset/update gates, candidate state, sigmoid readout, exact full BPTT and clipped gradients.
- **Trainable Elman RNN** from Phase 6 with full BPTT.
- **Fixed spiking LIF reservoir** with logistic readout.
- **Fixed tanh reservoir** with logistic readout.
- **Conventional history window** with logistic readout.
- **Explicit delayed tap** with logistic readout.
- **Majority-class constant predictor**.

A cap sets the maximum number of trainable parameters for **GRU and Elman only**. Each uses the largest hidden width (at most 32) fitting the cap:

- Elman: n² + 3n + 1 parameters.
- GRU: 3n² + 7n + 1 parameters.

Default cap 95 gives Elman width 8 with 89 trainable parameters, versus GRU width 4 with 77 trainable parameters. The widths and exact parameter counts are documented in every report. The comparison is **budget-capped, not strictly equal-parameter or equal-compute**. Fixed baselines are not validation tuned; this is an exploratory functionality comparison, not a fair architecture championship.

## Run

From the repository root, Python 3.10+, standard library only:

    python -m phineubio.validation --json-output dashboard/validation-report.json --csv-output validation-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/validation.html, or open the HTML directly and select a generated JSON report.

For a faster run:

    python -m phineubio.validation --seed-count 1 --delays 1 --noise-levels 0 --train-sequences 4 --validation-sequences 3 --test-sequences 2 --sequence-length 12 --burn-in 3 --epochs 2 --parameter-budget 55

## Measurements and cautions

The JSON records the full predeclared learning-rate grid, per-candidate validation losses, selected rates, independent test accuracy, hidden widths, trainable parameter counts, and per-model elapsed Python profiling. The final test is used only after the candidate is selected, and the selected model is not retrained on the test set.

The two trainable systems optimize per-sequence cross entropy using SGD. Fixed reservoir baselines still use an inherited logistic readout protocol that is **not validation tuned**. Their parameter/computational budgets differ. Machine-specific elapsed times are not energy measurements. The experiment concerns **digital simulations** and says nothing about consciousness, biological cell learning, or medical efficacy.

Next: validation-tune fixed reservoir readouts, run on a broader set of tasks, enforce cost parity more tightly, and evaluate on larger seed sweeps with confidence intervals and repeated hardware profiling. Negative results remain publishable results.
