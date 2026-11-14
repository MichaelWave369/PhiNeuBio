# Phase 2: closed-loop feedback and neuron activity viewer

## Goal

Demonstrate a simple closed loop: **cue → simulated spiking network → action → reward → readout update**.

This is a software-only demonstration of **reward-guided readout adaptation**, not evidence of human neuronal learning or general reinforcement learning.

## Run

From the repository root:

    python -m phineubio.feedback --output dashboard/feedback-report.json
    python -m http.server 8000

Open http://localhost:8000/dashboard/ in a browser. The dashboard automatically loads the local report. Alternatively open dashboard/index.html directly and select a JSON report using the file picker.

Try another seed or network size:

    python -m phineubio.feedback --seed 42 --neurons 64 --train-trials 200 --output dashboard/feedback-report.json

The dashboard animates an **already recorded simulated trace**. It is not a live wet-lab stream or a physical neural recording.

## The task

Each trial starts with a reset simulated reservoir. The environment supplies cue 0 or 1, and the fixed recurrent network emits spike and membrane-potential features. A logistic readout selects action 0 or 1. The environment awards reward 1 for matching the cue, otherwise 0. The readout updates based on the action and the reward: in a two-option deterministic task a zero reward reveals that the unchosen action would have succeeded.

This is **not** a general reward-learning algorithm. Because only two possible cues exist, held-out trial order does not test novel contexts. Recurrent synapses do not change.

## Measurements

- Before and after evaluation: balanced cue orders held out from the training trial order, greedy actions, no exploration, no updates.
- Training reward rate, first-half and second-half training reward rate, expected random-action baseline (0.5).
- Up to trace_limit training trials: cue, action, binary reward, exploration flag, decision probability, spike indices, voltages, and cumulative reward rate.
- Configuration, data schema version, and limitations embedded in the JSON.

If a network scores well, it means **the trainable readout** learned this simple binary mapping. It does not indicate neural tissue, plastic recurrent synapses, self-awareness, or superiority to conventional models.

## Next research questions

Evaluate a range of held-out noise levels and delayed cues, compare to direct-input logistic baselines, run seed sweeps, include confusion matrices, and separate training/assessment conditions. Extend feedback only with falsifiable predictions and honest controls.