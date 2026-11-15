# Phase 9: seed-paired evidence audit

PhiNeuBio Phase 9 adds a third **IID random bit** signal family to the Markov and periodic signals, then audits saved Phase 8 reports without retraining or repeating CPU profiling.

The audit predeclares four comparisons: spiking LIF against GRU, fixed tanh, and explicit history; and GRU against explicit history. Every condition uses the same seed set. The validator checks for duplicate, missing, nonfinite, or out-of-grid entries and mismatched sample counts.

For each seed, the effect is held-out accuracy(first) minus accuracy(second). The audit computes mean, median, population standard deviation, wins/losses/ties, and, if there are at least **five independent seeds**, a deterministic **95% percentile bootstrap interval** over seed-paired effects. It does not treat thousands of correlated frames as independent experiments. Fewer than five seeds yields **no interval**. Even five seeds is insufficient for strong inference: these intervals are exploratory, may have poor coverage, and do not correct for the multiple comparisons across many conditions.

The complete source report is fingerprinted using a canonical JSON SHA-256 digest. It also copies optional runtime information recorded when the original experiment ran. The audit does not assert hardware or energy advantages and does not infer biological learning or awareness.

## Reproduce

From the repository root, using Python 3.10+ and the standard library only:

    python -m phineubio.fairness --seed-count 5 --delays 1 3 --noise-levels 0 0.15 --signal-families iid markov periodic --json-output dashboard/fairness-report.json

    python -m phineubio.audit --input dashboard/fairness-report.json --output dashboard/audit-report.json --csv-output audit-effects.csv

    python -m http.server 8000

Open http://localhost:8000/dashboard/audit.html. A five-seed run across the full signal grid can take longer than CI smoke tests; use the same command with one signal and delay for a quick first pass.

## Interpretation and limitations

- IID fair bits carry no redundancy across neighboring time steps and serve as a negative control for claims of denoising from context.
- Bootstrap intervals resample **whole independent seeds**, not timesteps. The 95% label is a descriptive percentile interval and not a guarantee of actual coverage.
- Four pairwise contrasts across multiple noise, signal and delay conditions are reported without multiplicity adjustment; do not cherry-pick apparent winners.
- Different architectures still have unequal fixed parameter counts, compute costs and tuning search spaces. A positive performance difference is not proof of general superiority.
- Timing and source runtime metadata are diagnostic only; no energy readings, living neural tissue or consciousness claims are made.

Next: larger preregistered tasks, paired confidence intervals with better coverage, tighter compute budgets, randomized execution order and independently calibrated hardware profiling.
