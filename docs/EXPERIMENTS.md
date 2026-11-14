# Experiment protocol: delayed-bit memory

## Research question

Can state from a small recurrent spiking simulation support decoding a binary signal presented a fixed number of steps earlier?

## Protocol

1. Generate training and held-out test sequences using distinct deterministic pseudorandom seeds.
2. Feed each sequence through an identically seeded fixed LIF reservoir, resetting state between sequences.
3. After burn-in, label the feature vector at time t using the input bit at time t - delay.
4. Fit a logistic readout on training vectors only.
5. Record training and held-out accuracy and a majority-class baseline selected on training labels.
6. Save the JSON report with exact parameters. A single run is exploratory; replicate over seeds before making comparative conclusions.

## What the benchmark does not show

It does not show learning inside biological neurons, reinforcement learning, cognitive understanding, state-of-the-art memory performance, efficiency compared to GPUs, or consciousness. Its readout learns from labels; the reservoir connections remain fixed.

## Next experiments

Add a direct-input readout baseline, delay curves, seed-sweep mean and dispersion, explicit train/test data checks, and closed-loop task feedback. Make these separate reviewed changes so the initial protocol remains inspectable.