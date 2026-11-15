# Phase 13: learning-budget comparison and switching signals

PhiNeuBio Phase 13 measures what learning-rate selection actually costs in the software experiment, alongside independent held-out accuracy. It uses the same per-seed training, validation and final test sequences at every epoch budget.

## What is matched

For a fixed training set and epoch budget, the trained GRU and Elman models and the fixed-reservoir logistic readouts receive the **same number of retained target presentations per candidate learning rate**. Candidate counts and epochs are predeclared.

This does **not** imply equivalent computational work. GRU and Elman use one sequence-wise BPTT update for each sequence, while readout SGD uses one optimizer update for every retained timestep. The versioned report records *both* target presentations and optimizer updates for every model, plus the recurrent BPTT processing steps. Fixed reservoirs never train their recurrent matrices. Parameter slots are carried over from Phase 11.

## Selection and timing

For each budget/model, select a learning rate on independent **validation** data, using the same rate grid. Only then read final **test** targets. A perf_counter_ns timer surrounds the entire candidate-selection process, **including initialization, training feature extraction, optimization and validation scoring**. A different timer measures final test evaluation. Neither represents pure gradient-update time, CPU instructions, electrical energy or hardware efficiency.

The same train/validation/test streams are reused when changing the epoch budget so the comparison is paired. Running the experiment again reproduces seeded accuracy and selected hyperparameters, not wall times.

## Switching-signal task

The new synthetic *switching* family alternates blocks with 95% and 55% bit persistence. It generates clean sequences first and corrupts only observed bits, preserving nested noise masks at fixed seeds. This tests a kind of temporal nonstationarity but is **not biological data**.

## Run

Python 3.10+, standard library only:

    python -m phineubio.training --json-output dashboard/training-report.json --csv-output training-results.csv
    python -m http.server 8000

Open http://localhost:8000/dashboard/training.html, or choose the JSON file from the page.

Fast CI-scale example:

    python -m phineubio.training --epochs-budgets 1 2 --seed-count 1 --signal-families switching --noise-levels 0 --delays 1 --pair-visit-budget 27 --train-sequences 3 --validation-sequences 2 --test-sequences 2 --sequence-length 12 --burn-in 3 --switching-block 3 --json-output dashboard/training-report.json --csv-output training-results.csv

## Limitations

Matching target presentations is *not* equal training compute, FLOPs, memory traffic, runtime, SGD update count, or watts. Small toy signal studies cannot establish broad neural advantage or consciousness. No physical biological tissue or electrical energy measurements are made. Next improvements should include calibrated training operation counts and independent hardware profilers if available.
