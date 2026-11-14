# Phase 3: repeated-seed, controlled digital memory benchmarks

## Why this exists

An impressive-looking simulated spike raster does not establish that recurrence provides a useful memory mechanism. These benchmarks compare a **fixed recurrent leaky integrate-and-fire (LIF) reservoir** against transparent, conventional controls across multiple input delays and random seeds.

Everything is a digital software model. No biological data, living cells, tissue interfaces, or consciousness measurement are involved.

## Protocol and baselines

Each run generates independent training and testing **sequences**, rather than splitting frames from a common sequence. A clean random bit stream supplies reference labels. A possibly noisy observed bit stream feeds all candidate models. For prediction at time t and chosen delay d, the target is clean[t - d]. Burn-in removes the initial state transient. Network states reset between sequences.

Models:

- **Spiking recurrent:** fixed LIF reservoir, trained logistic readout of voltages and spikes.
- **Spiking no-recurrence:** same seeded LIF neurons but recurrent gain set to zero. Note: leaky neuron state can still encode history.
- **Current input:** logistic classifier on only observed[t], without a memory mechanism.
- **Delayed input tap:** logistic classifier directly given observed[t - d]. This deliberately uses conventional explicit storage and is a strong memory control.
- **Majority:** most common label in the training set, evaluated on held-out labels.

All trainable classifiers share the same label set, training-sample shuffle order, number of epochs, and learning rate. Noise flips observed input bits independently, not the clean reference labels. Models see the exact same observed sequences for each seed and delay. Recurrent synapses stay fixed; only readout weights adapt.

## Reproduce

Run from repository root (Python 3.10+; standard library only):

    python -m phineubio.benchmarks --json-output dashboard/benchmark-report.json --csv-output benchmark-results.csv

Optional faster check:

    python -m phineubio.benchmarks --seed-count 2 --delays 1 2 --train-sequences 8 --test-sequences 4 --neurons 12 --epochs 6

Noise trial:

    python -m phineubio.benchmarks --flip-probability 0.15 --json-output dashboard/benchmark-report.json

Serve the repository with:

    python -m http.server 8000

Open http://localhost:8000/dashboard/benchmark.html. You can also open the HTML locally and choose a JSON report. The dashboard shows mean held-out accuracy by delay, seed variation, and explicit model-control comparisons.

## Interpreting the numbers

- **Recurrent vs no-recurrence:** an ablation of recurrent *connections*, not all neuron memory.
- **Recurrent vs current-input:** tests benefit of having historical state, but is not a fair comparison with an input-memory-equipped conventional system.
- **Recurrent vs delayed-input tap:** compares learned encoding of history against an explicitly stored older bit. Expect the tap to dominate at low noise.
- The test set only changes random sequences, not task distribution. Short seed sweeps give descriptive means and population standard deviations, not confidence intervals.
- Controlled accuracy alone cannot prove biological plausibility, energy efficiency, scalability, intelligence, or consciousness.

### Research questions for Phase 4

1. Does recurrence help under longer delays, accounting for network size and compute?
2. Do trends hold over 20+ seeds and multiple noise levels?
3. How does a tapped input-history window and a conventional RNN compare at equal parameter count?
4. Can we collect operation-count and wall-clock measurements without claiming physical energy data?

**Publish negative results too.** We are testing hypotheses, not creating a marketing graphic.
