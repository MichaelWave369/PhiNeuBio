"""Minimal trainable GRU with full-sequence BPTT, pure Python standard library.

The GRU is a *digital* recurrent neural network, not a model of biological
tissue. Recurrent, input, gate and output parameters are all trainable.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Sequence


class TrainableGRU:
    """Scalar binary input, tanh candidate, sigmoid gates and sigmoid output.

    h[t] = z[t] * h[t-1] + (1-z[t]) * candidate[t].
    Gates use the previous hidden state. Each independent sequence starts with
    zeros; gradients propagate through the *full* input sequence.
    """

    def __init__(self, neurons: int, seed: int) -> None:
        if type(neurons) is not int or neurons < 1:
            raise ValueError("neurons must be a positive integer")
        self.neurons = neurons
        rng = random.Random(seed)
        n = neurons
        self.wzx = [rng.uniform(-0.35, 0.35) for _ in range(n)]
        self.wrx = [rng.uniform(-0.35, 0.35) for _ in range(n)]
        self.wnx = [rng.uniform(-0.35, 0.35) for _ in range(n)]
        self.uz = [[rng.uniform(-0.5, 0.5) / math.sqrt(n) for _ in range(n)] for _ in range(n)]
        self.ur = [[rng.uniform(-0.5, 0.5) / math.sqrt(n) for _ in range(n)] for _ in range(n)]
        self.un = [[rng.uniform(-0.5, 0.5) / math.sqrt(n) for _ in range(n)] for _ in range(n)]
        self.bz = [0.0] * n
        self.br = [0.0] * n
        self.bn = [0.0] * n
        self.wy = [rng.uniform(-0.3, 0.3) for _ in range(n)]
        self.by = 0.0

    @property
    def parameter_count(self) -> int:
        return 3 * self.neurons * self.neurons + 7 * self.neurons + 1

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-40.0, min(40.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, observed: Sequence[int]) -> tuple[list[dict], list[float]]:
        h = [0.0] * self.neurons
        caches = []
        probabilities = []
        for bit in observed:
            if bit not in (0, 1):
                raise ValueError("sequence inputs must be binary")
            x = 2 * bit - 1
            previous = h
            z = [self._sigmoid(self.wzx[i] * x + self.bz[i] +
                 sum(w * v for w, v in zip(self.uz[i], previous)))
                 for i in range(self.neurons)]
            r = [self._sigmoid(self.wrx[i] * x + self.br[i] +
                 sum(w * v for w, v in zip(self.ur[i], previous)))
                 for i in range(self.neurons)]
            reset_previous = [r[i] * previous[i] for i in range(self.neurons)]
            candidate = [math.tanh(self.wnx[i] * x + self.bn[i] +
                         sum(w * v for w, v in zip(self.un[i], reset_previous)))
                         for i in range(self.neurons)]
            h = [z[i] * previous[i] + (1 - z[i]) * candidate[i]
                 for i in range(self.neurons)]
            caches.append({"x": x, "previous": previous, "z": z, "r": r,
                           "reset_previous": reset_previous, "candidate": candidate, "h": h})
            probabilities.append(self._sigmoid(self.by + sum(a * b for a, b in zip(self.wy, h))))
        return caches, probabilities

    def loss_and_gradients(
        self, observed: Sequence[int], clean: Sequence[int], delay: int, start: int
    ) -> tuple[float, dict]:
        if len(observed) != len(clean) or not 0 <= start < len(clean) or delay < 1 or start < delay:
            raise ValueError("invalid sequence, target delay, or sample window")
        caches, probabilities = self.forward(observed)
        n = self.neurons
        gradients = {
            name: [[0.0] * n for _ in range(n)] if name in ("uz", "ur", "un") else [0.0] * n
            for name in ("wzx", "wrx", "wnx", "uz", "ur", "un", "bz", "br", "bn", "wy")
        }
        gradients["by"] = 0.0
        loss = 0.0
        retained = len(clean) - start
        future_dh = [0.0] * n

        for t in range(len(clean) - 1, -1, -1):
            cache = caches[t]
            h, previous = cache["h"], cache["previous"]
            z, r, candidate = cache["z"], cache["r"], cache["candidate"]
            x, reset_previous = cache["x"], cache["reset_previous"]
            dh = future_dh[:]

            if t >= start:
                label = clean[t - delay]
                p = probabilities[t]
                stable = max(1e-12, min(1 - 1e-12, p))
                loss -= (label * math.log(stable) + (1 - label) * math.log(1 - stable)) / retained
                dout = (p - label) / retained
                gradients["by"] += dout
                for i in range(n):
                    gradients["wy"][i] += dout * h[i]
                    dh[i] += dout * self.wy[i]

            dz = [dh[i] * (previous[i] - candidate[i]) for i in range(n)]
            dcandidate = [dh[i] * (1 - z[i]) for i in range(n)]
            dprevious = [dh[i] * z[i] for i in range(n)]

            da_n = [dcandidate[i] * (1 - candidate[i] ** 2) for i in range(n)]
            for i in range(n):
                gradients["wnx"][i] += da_n[i] * x
                gradients["bn"][i] += da_n[i]
                for j in range(n):
                    gradients["un"][i][j] += da_n[i] * reset_previous[j]
            d_reset_previous = [sum(self.un[i][j] * da_n[i] for i in range(n)) for j in range(n)]
            dr = [d_reset_previous[i] * previous[i] for i in range(n)]
            for i in range(n):
                dprevious[i] += d_reset_previous[i] * r[i]

            da_r = [dr[i] * r[i] * (1 - r[i]) for i in range(n)]
            da_z = [dz[i] * z[i] * (1 - z[i]) for i in range(n)]
            for i in range(n):
                gradients["wrx"][i] += da_r[i] * x
                gradients["wzx"][i] += da_z[i] * x
                gradients["br"][i] += da_r[i]
                gradients["bz"][i] += da_z[i]
                for j in range(n):
                    gradients["ur"][i][j] += da_r[i] * previous[j]
                    gradients["uz"][i][j] += da_z[i] * previous[j]
            for j in range(n):
                dprevious[j] += sum(
                    self.ur[i][j] * da_r[i] + self.uz[i][j] * da_z[i]
                    for i in range(n)
                )
            future_dh = dprevious
        return loss, gradients

    def train_sequence(
        self, observed: Sequence[int], clean: Sequence[int], delay: int, start: int,
        learning_rate: float, clip: float
    ) -> float:
        if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(clip) or clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive and finite")
        loss, gradients = self.loss_and_gradients(observed, clean, delay, start)
        values = []
        for name, grads in gradients.items():
            if name == "by":
                values.append(grads)
            elif name in ("uz", "ur", "un"):
                values.extend(v for row in grads for v in row)
            else:
                values.extend(grads)
        norm = math.sqrt(sum(v * v for v in values))
        scale = learning_rate * (min(1.0, clip / norm) if norm else 1.0)
        for name, grads in gradients.items():
            current = getattr(self, name)
            if name == "by":
                self.by -= scale * grads
            elif name in ("uz", "ur", "un"):
                for i in range(self.neurons):
                    for j in range(self.neurons):
                        current[i][j] -= scale * grads[i][j]
            else:
                for i in range(self.neurons):
                    current[i] -= scale * grads[i]
        return loss

    def train(
        self, sequences: Sequence[tuple[list[int], list[int]]], delay: int,
        start: int, epochs: int, learning_rate: float, clip: float, seed: int
    ) -> list[float]:
        if not sequences or epochs < 1:
            raise ValueError("nonempty training sequences and positive epochs required")
        rng = random.Random(seed)
        order = list(range(len(sequences)))
        curve = []
        for _ in range(epochs):
            rng.shuffle(order)
            losses = []
            for i in order:
                clean, observed = sequences[i]
                losses.append(self.train_sequence(observed, clean, delay, start, learning_rate, clip))
            curve.append(statistics.mean(losses))
        return curve

    def accuracy(
        self, sequences: Sequence[tuple[list[int], list[int]]], delay: int, start: int
    ) -> float:
        if not sequences:
            raise ValueError("evaluation sequences required")
        correct = total = 0
        for clean, observed in sequences:
            _, probabilities = self.forward(observed)
            if not 0 <= start < len(clean) or start < delay:
                raise ValueError("invalid sample window")
            for t in range(start, len(clean)):
                correct += int((probabilities[t] >= 0.5) == bool(clean[t - delay]))
                total += 1
        return correct / total
