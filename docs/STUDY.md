# Phase 10: predeclared sample-efficiency study

## What is new

PhiNeuBio Phase 10 is a **study protocol** rather than another neural architecture. It runs the Phase 8 seven-model competition at multiple, nested training-data budgets, freezes the test and validation streams per seed, and examines predeclared comparisons across the full condition grid.

It supplies a committed `study-plan.json` and a SHA-256 of the canonical plan in the result, making protocol changes detectable. The plan is auditable, **not cryptographically immutable**; a different plan has a different fingerprint. No code rewrites or training on the test set are part of the audit.

**Study question:** How do seed-paired architecture differences change as training sequences increase while validation and test data remain fixed?

## How training-data sizes are paired

For each signal family, noise, delay and random seed, the generator starts from the same random seed at each budget. A larger training set contains the same sequence prefix as a smaller one. Validation, test and inference-probe seeds are fixed and generated independently from the training set, with equal sequence lengths and held-out sample windows at all budgets.

All seven models receive the same sequences within a condition. Each trainable model or readout chooses its learning rate on validation data only, using the same predeclared candidate grid as earlier phases. Models remain different in parameter count, fixed weights, training schedules, and compute: **this does not make the architectures equal-budget**.

## Predeclared questions and uncertainty

The four fixed contrasts are Spiking LIF vs GRU, Spiking LIF vs fixed tanh, Spiking LIF vs explicit history memory, and GRU vs explicit history memory. These are evaluated in every training-size × signal × noise × delay cell. Positive effects mean the first named model had higher held-out test accuracy.

There are two distinct uncertainty/statistical quantities:

1. **Exploratory 95% percentile bootstrap interval** around the *mean paired held-out accuracy difference*, resampling whole random seeds. No interval is produced below five seeds. This interval is **not** corrected for multiplicity or guaranteed to have 95% coverage on a tiny sample.
2. **Exact two-sided sign-test p-value** based on how many seeds favored each model (ties excluded), not on effect magnitudes. For fewer than five *nontied* seed differences, no test p-value is reported. Under the null, each independent seed must be equally likely to favor either model. **Holm-Bonferroni adjustment** is applied to the full predeclared comparison family across all training sizes, signals, noises and delays. Untestable contrasts conservatively occupy a slot with p=1 when applying correction.

A small adjusted p-value is evidence only against an equal *directional win probability* under the test assumptions. It is **not** proof of computational, biological or general-purpose superiority. A large p-value never proves equivalence. All samples are synthetic; sign tests assume effectively independent seed replicates.

## Reproduce

Python 3.10+, no paid or runtime dependencies. The committed study plan defines **10 seeds**, **2 training-set sizes**, **2 signal families**, **1 noise level** and **1 delay**, for 40 seed-budget-condition experiments. This is a more involved CPU workload than CI.

Run the committed plan:

    python -m phineubio.study --plan study-plan.json --output dashboard/study-report.json --csv-output study-effects.csv

To launch the browser dashboard:

    python -m http.server 8000

Open http://localhost:8000/dashboard/study.html or choose a JSON file from local disk.

Fast CI-like smoke run, using a separate automatically described protocol:

    python -m phineubio.study --seed-count 1 --training-sizes 2 3 --signal-families iid --delays 1 --noise-levels 0 --train-epochs 2 --sequence-length 12 --burn-in 3 --validation-sequences 2 --test-sequences 2 --probe-sequences 1 --parameter-budget 55 --resamples 200 --output dashboard/study-report.json --csv-output study-effects.csv

The resulting report contains every seed-level held-out result, selected validation settings, plan fingerprint, paired means, seed wins/losses/ties, bootstrap intervals when allowed, exact sign p-values and family-wide Holm corrections. It intentionally **does not** include nondeterministic runtime observations; Phase 8 reports remain the place for exploratory Python inference profiling.

## Important limits

- These are **digital simulations**, not living neurons or medical measurements. Classification does not demonstrate consciousness.
- Reusing a small final test set across training budgets makes its finite noise common across budgets, not eliminated. Predeclared comparisons reduce selection bias but do not guarantee power.
- Ten seeds and two training budgets remain exploratory, especially after correction across many comparisons.
- No hardware energy measurements are included, and network implementations and computational costs remain unequal.
- Python timings from earlier phases are machine-specific and are not suitable evidence of real-world power efficiency.
- The source code version must also be recorded by a researcher when presenting results outside the repository; the plan fingerprint alone does not establish which revision of code generated it.

Future work: broader tasks, larger independent datasets, matched operations or measured hardware budgets, and validated inference/energy profiling across devices.
