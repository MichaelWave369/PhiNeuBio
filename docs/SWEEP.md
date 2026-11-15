# Phase 12: multi-budget CPU experiments and explicit operation ledgers

Phase 12 extends the Phase 11 recurrent-visit comparison to several structural budgets using the **same independent seeded training, validation, test and profiling-probe streams** at every budget.

## Method

For each budget (default 27, 75, 108), all seven model families train with independent validation-based learning-rate selection and produce held-out test accuracy plus randomized-order Python CPU probe timings. Recurrent systems use the largest width that fits the matrix-visit limit (GRU scans three matrices, Elman/fixed tanh/LIF one). Explicit history, delayed tap and majority remain deliberately unmatched controls.

The new operation ledger records **distinct events** that must not be collapsed into a fictitious universal FLOP count:

- Dense recurrent scalar products for GRU, Elman and fixed tanh.
- Scalar products in output/readout dot operations.
- Defined nonlinear evaluations and LIF threshold comparisons.
- For LIF, an **untimed replay of identical probe signals** counts conditional weight additions for active previous spikes, including zero-weight entries, as done by the Python implementation. It separately counts nonzero active edges and emitted spikes.
- Trainable and fixed parameter slots, inherited from Phase 11.

LIF event counting does **not** read target labels, perform training or enter the timed measurement. Independent tests enumerate spike events directly and confirm these counts. The ledger is a model-algorithm accounting system, **not hardware performance counters**. It does not measure CPU instructions, standardized FLOPs, memory bandwidth, watts or joules.

The report includes a structural **Pareto screen** for recurrent models only, showing options not dominated simultaneously in mean held-out accuracy and matrix visits. This is descriptive, not a significance test; CPU timing and training costs are not involved in this frontier.

## Run on your machine

Python 3.10+, standard library only:

    python -m phineubio.sweep --json-output dashboard/sweep-report.json --csv-output sweep-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/sweep.html or load the generated JSON locally.

Longer run:

    python -m phineubio.sweep --budgets 27 75 108 192 --seed-count 5 --signal-families iid markov periodic --inference-repeats 5 --json-output dashboard/sweep-report.json --csv-output sweep-results.csv

Quick smoke:

    python -m phineubio.sweep --budgets 12 27 --seed-count 1 --signal-families iid --noise-levels 0 --delays 1 --train-sequences 3 --validation-sequences 2 --test-sequences 2 --probe-sequences 1 --sequence-length 12 --burn-in 3 --epochs 2 --inference-repeats 2 --json-output dashboard/sweep-report.json --csv-output sweep-results.csv

## Limitations

The trained GRU/Elman networks learn recurrent weights while LIF/tanh reservoir transitions remain fixed. Training budget, parameter budget, and full instruction count are **not** matched. Phase 12's visit caps constrain one part of the computation only. Python timing measurements are machine-dependent and are not energy estimates. Synthetic signals and small seed sets do not establish real-world advantage. No living neural tissue or consciousness claims are made. Future work should broaden tasks, match end-to-end training cost, and add optional, real external hardware counters or power meters.
