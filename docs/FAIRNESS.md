# Phase 8: validation fairness, new signals, and repeated independent inference probes

## Research question

When independent validation is also applied to the **fixed spiking and tanh reservoir readouts**, do the conclusions change across more than one temporal signal distribution? Phase 7 applied validation selection to GRU/Elman only. Phase 8 closes that particular experimental gap without claiming perfect computational parity.

## Common data, separate decisions

Each configured seed, noise level and signal family generates four **separate complete sequence sets**: training, validation, test and profiling probe. They are generated using different seeds (101, 303, 202, 404), with identical streams provided to every model within a condition. Models train on training sequences; tune learning rate on validation sequences; and assess held-out test accuracy once *after* selection. The profiling probe is separate from the final test and never enters optimization or hyperparameter selection.

Three signal families are available:

- **Markov:** first-order binary process, with a configurable probability that a bit stays the same across consecutive steps.
- **Periodic:** random binary template of configurable length with independent random phase per sequence. The clean template repeats; noise flips each observed bit independently.
- **IID:** independent fair random bits; provides a control without temporal structure or context-based denoising information.

The periodic template can be trivially simple in an individual seeded run; all three families are toy problems, not representations of living tissue or natural language.

## Compared models

- Trainable GRU and trainable Elman RNN. Both train recurrent weights using full BPTT, with candidate learning rates chosen by *validation cross entropy*.
- Fixed spiking LIF and fixed tanh recurrent reservoirs. Recurrent matrices **stay fixed**; their logistic readout learning rates now also receive validation-based selection.
- Explicit observed-history window and exact delayed tap. Their logistic readout rates are likewise selected on validation only.
- Constant training-set-majority baseline. No tuneable learning rate.

The explicit history and delayed tap are intentionally strong controls. All models retain identical time indices, starting from max(burn_in, max(delays)).

## Budget honesty

The configured parameter cap applies to trainable GRU/Elman only; a fixed reservoir may carry many additional nontrainable weights. The tuned readout parameter counts are separately recorded. This is **not equal-parameter, equal-compute or universally fair architecture competition**, although validation selection is more consistent.

## Profiling

The Phase 8 report now includes optional machine provenance metadata: Python version, OS family, architecture, logical CPU count and timer resolution. No usernames, hostnames, or hardware serial numbers are included.


Each already-selected model performs repeated inference on an *independent profiling probe*, not the final test. The report stores each elapsed reading, their median, and the number of retained samples. Profiling includes digital network forward computation or reservoir feature generation plus readout prediction. It excludes training. Python timings are environment-specific and strongly dependent on implementation differences, model order and system load. They **do not measure energy**. Their proper role is diagnostic and reproducible on the same hardware, not establishing universal speed rankings.

## Run

Python 3.10+; no runtime dependencies:

    python -m phineubio.fairness --json-output dashboard/fairness-report.json --csv-output fairness-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/fairness.html or choose the JSON report from the local HTML file.

Fast smoke test:

    python -m phineubio.fairness --seed-count 1 --delays 1 --noise-levels 0 --signal-families periodic --train-sequences 3 --validation-sequences 2 --test-sequences 2 --probe-sequences 1 --sequence-length 12 --burn-in 3 --epochs 2 --parameter-budget 55 --inference-repeats 2

## Next rigor milestones

Tune fixed reservoir sizes/connectivity on an independent validation protocol, collect broader benchmark tasks, compare seed-paired confidence intervals, and run repeated hardware profiling with order randomization. Eventually separate training/inference energy measurements would require external hardware tools. No learning behavior here is evidence for consciousness or biological neuron experimentation.
