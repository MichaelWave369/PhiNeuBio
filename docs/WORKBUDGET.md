# Phase 14: symbolic training-work budgets and burst-noise memory

PhiNeuBio Phase 14 allocates training epochs using a **declared symbolic work-score cap** rather than using the same epoch count across architectures. This is an intermediate methodological experiment, **not real end-to-end compute parity**.

## Defined work-score

Each architecture has a selection-work profile composed of (a) one-time fixed-reservoir feature extraction, (b) candidate model setup, (c) candidate validation forward computation, and (d) candidate per-epoch training.

The GRU profile assigns 3 n-squared recurrent forward terms and 6 n-squared backward terms per full-sequence timestep. The Elman RNN uses n-squared and 2 n-squared terms. Fixed reservoirs reuse one extraction across learning-rate candidates, and readout SGD gets explicit readout scalar-term and parameter-update slot counts.

The score is the **sum of distinct events weighted equally by an arbitrary value of one**. These events have different actual instruction costs. Therefore, similar scores DO NOT mean equal FLOPs, CPU instructions, training wall time, memory bandwidth, power or energy. The complete component ledger and unused caps are exported instead of claiming exact compute parity.

For each declared cap, the program chooses the largest full integer epoch count between one and the configured maximum that fits its modeled selection-work score for every trainable architecture. The choice happens before test data is generated. If any model cannot fit even one full epoch under the cap, the program fails explicitly. The non-learning majority control is a separate unbudgeted reference.

## Correlated burst-noise challenge

The new burst signal family begins with a clean seeded Markov chain. For each contiguous block of configurable length, one uniform draw determines whether every observed bit in that block is corrupted. The clean signal is unchanged. Reusing a seed across different corruption probabilities preserves clean streams and nested noise masks. This is a synthetic test for contiguous bad observations, not physical sensor or biological tissue data.

## Data and model selection

Training, validation, and final test streams use independent seeds. All architectures receive identical complete sequences and delayed targets within each condition. Each trained architecture evaluates the same learning-rate candidate grid, selects only by validation loss, then evaluates held-out test accuracy once. The report separates symbolic score, unused work cap, optimizer-update counts, locally measured Python candidate-selection time, and final test evaluation time.

## Run locally

Python 3.10+ with no dependencies beyond the standard library:

    python -m phineubio.workbudget --json-output dashboard/workbudget-report.json --csv-output workbudget-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/workbudget.html, or use the page's local JSON file picker.

Fast CI-style smoke:

    python -m phineubio.workbudget --work-caps 10000 25000 --max-epochs 3 --seed-count 1 --signal-families burst --noise-levels 0 --delays 1 --pair-visit-budget 27 --train-sequences 3 --validation-sequences 2 --test-sequences 2 --sequence-length 12 --burn-in 3 --burst-block 3 --json-output dashboard/workbudget-report.json --csv-output workbudget-results.csv

## Limits

Proxy weights are arbitrary. Epoch allocation may leave unused score due to granularity or a common maximum-epoch ceiling; both are visible. Fixed reservoirs fit only readouts, while GRU/Elman train recurrent matrices. Small synthetic studies cannot establish general superiority or statistical significance. Time is measured inside Python, not with hardware counters or an energy meter. No real biological cells, medical claims or consciousness conclusions are involved.

Actual hardware parity would require calibrated CPU instruction or performance counters, controlled system load, memory accounting and independently repeated experiments on real devices.
