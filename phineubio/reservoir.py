"""Seeded leaky integrate-and-fire (LIF) reservoir, simulated in software.

This is a numerical model, not a model of a particular human tissue culture.
"""

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class ReservoirConfig:
    neurons: int = 32
    leak: float = 0.82
    threshold: float = 1.0
    input_gain: float = 1.35
    recurrent_gain: float = 0.45
    density: float = 0.18
    inhibitory_fraction: float = 0.25
    seed: int = 369

    def __post_init__(self) -> None:
        if self.neurons < 1:
            raise ValueError("neurons must be positive")
        if not 0 <= self.leak < 1:
            raise ValueError("leak must be in [0, 1)")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if not math.isfinite(self.input_gain) or self.input_gain < 0:
            raise ValueError("input_gain must be finite and nonnegative")
        if not math.isfinite(self.recurrent_gain) or self.recurrent_gain < 0:
            raise ValueError("recurrent_gain must be finite and nonnegative")
        if not 0 <= self.density <= 1:
            raise ValueError("density must be in [0, 1]")
        if not 0 <= self.inhibitory_fraction <= 1:
            raise ValueError("inhibitory_fraction must be in [0, 1]")


class Reservoir:
    """Stateful synchronous discrete-time LIF network.

    Each time step uses the *previous* spike vector for recurrent input.
    Spiking neurons reset to zero. The emitted feature vector concatenates
    membrane potentials and binary spike states for a linear readout.
    """

    def __init__(self, config: ReservoirConfig | None = None) -> None:
        self.config = config or ReservoirConfig()
        rng = random.Random(self.config.seed)
        n = self.config.neurons
        inhibitory = set(rng.sample(range(n), int(n * self.config.inhibitory_fraction)))
        self.input_weights = [rng.uniform(0.7, 1.3) * self.config.input_gain for _ in range(n)]
        scale = self.config.recurrent_gain / math.sqrt(max(1, n * self.config.density))
        self.weights = [
            [
                (rng.uniform(0.2, 1.0) * scale * (-1 if source in inhibitory else 1))
                if target != source and rng.random() < self.config.density
                else 0.0
                for source in range(n)
            ]
            for target in range(n)
        ]
        self.reset()

    def reset(self) -> None:
        self.voltages = [0.0] * self.config.neurons
        self.spikes = [0] * self.config.neurons

    def step(self, stimulus: float) -> tuple[float, ...]:
        """Advance one step for a finite scalar stimulus in [0, 1]."""
        if not isinstance(stimulus, (int, float)) or not math.isfinite(stimulus):
            raise ValueError("stimulus must be a finite number")
        if not 0.0 <= stimulus <= 1.0:
            raise ValueError("stimulus must be in [0, 1]")

        previous_spikes = self.spikes
        new_voltages: list[float] = []
        new_spikes: list[int] = []
        for i, row in enumerate(self.weights):
            current = stimulus * self.input_weights[i]
            current += sum(weight for weight, active in zip(row, previous_spikes) if active)
            voltage = self.config.leak * self.voltages[i] + current
            spiked = int(voltage >= self.config.threshold)
            new_spikes.append(spiked)
            new_voltages.append(0.0 if spiked else max(-2.0, voltage))
        self.voltages = new_voltages
        self.spikes = new_spikes
        return tuple(self.voltages) + tuple(float(s) for s in self.spikes)

    @property
    def feature_count(self) -> int:
        return self.config.neurons * 2
