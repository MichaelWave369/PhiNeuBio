"""Closed-loop, reward-driven learning on a simulated spiking reservoir.

A two-action environment gives only binary reward to the learner. In this
deliberately simple deterministic task, observing reward reveals the correct
action: an incorrect choice implies that the *other* action was correct.
The learner trains its readout only. The reservoir weights remain fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Sequence

from .reservoir import Reservoir, ReservoirConfig


@dataclass(frozen=True)
class FeedbackConfig:
    seed: int = 369
    neurons: int = 32
    train_trials: int = 160
    evaluation_trials: int = 60
    exploration_rate: float = 0.2
    learning_rate: float = 0.08
    trace_limit: int = 200

    def __post_init__(self) -> None:
        if self.neurons < 1:
            raise ValueError("neurons must be positive")
        if self.train_trials < 2 or self.evaluation_trials < 2:
            raise ValueError("trial counts must be at least 2")
        if not math.isfinite(self.exploration_rate) or not 0 <= self.exploration_rate <= 1:
            raise ValueError("exploration_rate must be finite and in [0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.trace_limit < 0:
            raise ValueError("trace_limit cannot be negative")


class BinaryRewardAgent:
    """Online logistic readout, updated from two-action reward feedback only."""

    def __init__(self, feature_count: int, learning_rate: float) -> None:
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.weights = [0.0] * feature_count
        self.bias = 0.0
        self.learning_rate = learning_rate

    def probability(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature width mismatch")
        activation = self.bias + sum(w * x for w, x in zip(self.weights, features))
        activation = max(-40.0, min(40.0, activation))
        return 1.0 / (1.0 + math.exp(-activation))

    def select(self, features: Sequence[float], rng: random.Random, exploration: float) -> tuple[int, bool, float]:
        probability = self.probability(features)
        explored = rng.random() < exploration
        action = rng.randrange(2) if explored else int(probability >= 0.5)
        return action, explored, probability

    def update(self, features: Sequence[float], action: int, reward: int) -> None:
        if action not in (0, 1) or reward not in (0, 1):
            raise ValueError("binary action and reward are required")
        # In this two-action deterministic environment, a failed action
        # establishes the other action as correct. Not general RL.
        inferred_target = action if reward else 1 - action
        error = inferred_target - self.probability(features)
        self.bias += self.learning_rate * error
        for i, value in enumerate(features):
            self.weights[i] += self.learning_rate * error * value


def _balanced_cues(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    cues = [i % 2 for i in range(count)]
    rng.shuffle(cues)
    return cues


def _state_for_cue(reservoir: Reservoir, cue: int) -> tuple[float, ...]:
    reservoir.reset()
    return reservoir.step(float(cue))


def _evaluate(agent: BinaryRewardAgent, reservoir: Reservoir, cues: Sequence[int]) -> float:
    correct = 0
    for cue in cues:
        features = _state_for_cue(reservoir, cue)
        decision = int(agent.probability(features) >= 0.5)
        correct += int(decision == cue)
    return correct / len(cues)


def run_feedback(config: FeedbackConfig | None = None) -> dict:
    """Run a seeded cue-action task and return a versioned, replayable report.

    Each trial resets the simulated reservoir and presents a binary cue.
    The agent observes the reservoir's spikes/voltages, chooses an action,
    receives 0/1 reward, and updates its readout using only that feedback.
    Evaluations use the same held-out cue order before and after training,
    with exploration off and no weight updates.
    """
    cfg = config or FeedbackConfig()
    reservoir = Reservoir(ReservoirConfig(seed=cfg.seed, neurons=cfg.neurons))
    agent = BinaryRewardAgent(reservoir.feature_count, cfg.learning_rate)
    train_cues = _balanced_cues(cfg.train_trials, cfg.seed + 101)
    evaluation_cues = _balanced_cues(cfg.evaluation_trials, cfg.seed + 202)
    rng = random.Random(cfg.seed + 303)

    before = _evaluate(agent, reservoir, evaluation_cues)
    rewards: list[int] = []
    trace: list[dict] = []
    for index, cue in enumerate(train_cues):
        features = _state_for_cue(reservoir, cue)
        action, explored, probability = agent.select(features, rng, cfg.exploration_rate)
        reward = int(action == cue)
        rewards.append(reward)
        if index < cfg.trace_limit:
            trace.append({
                "trial": index + 1,
                "cue": cue,
                "action": action,
                "reward": reward,
                "explored": explored,
                "probability_action_1": round(probability, 6),
                "spike_indices": [i for i, fired in enumerate(reservoir.spikes) if fired],
                "voltages": [round(v, 5) for v in reservoir.voltages],
                "cumulative_reward_rate": round(sum(rewards) / len(rewards), 6),
            })
        agent.update(features, action, reward)

    after = _evaluate(agent, reservoir, evaluation_cues)
    half = len(rewards) // 2
    return {
        "schema_version": "2.0",
        "project": "PhiNeuBio",
        "substrate": "software_simulation",
        "task": "binary_cue_reward_feedback",
        "configuration": asdict(cfg),
        "neuron_count": cfg.neurons,
        "evaluation": {
            "held_out_trial_count": len(evaluation_cues),
            "same_cues_before_and_after": True,
            "exploration_off": True,
        },
        "metrics": {
            "before_accuracy": before,
            "after_accuracy": after,
            "random_action_expected_accuracy": 0.5,
            "training_reward_rate": sum(rewards) / len(rewards),
            "first_half_training_reward_rate": sum(rewards[:half]) / half,
            "second_half_training_reward_rate": sum(rewards[half:]) / (len(rewards) - half),
        },
        "trace": trace,
        "limitations": [
            "Digital simulation only. No living neurons or biological-hardware connection.",
            "Only the readout learns; simulated recurrent connections do not adapt.",
            "Two deterministic actions make the correct target inferable from binary reward. This is not general reinforcement learning.",
            "Evaluation repeats the same two possible cues in a held-out trial order; it does not test unseen contexts.",
            "No claim of consciousness, brain equivalence, superior efficiency, or clinical utility.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PhiNeuBio: simulated binary reward feedback experiment")
    parser.add_argument("--seed", type=int, default=369)
    parser.add_argument("--neurons", type=int, default=32)
    parser.add_argument("--train-trials", type=int, default=160)
    parser.add_argument("--evaluation-trials", type=int, default=60)
    parser.add_argument("--exploration-rate", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--trace-limit", type=int, default=200)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path, instead of stdout")
    args = parser.parse_args(argv)
    try:
        config = FeedbackConfig(
            seed=args.seed, neurons=args.neurons, train_trials=args.train_trials,
            evaluation_trials=args.evaluation_trials, exploration_rate=args.exploration_rate,
            learning_rate=args.learning_rate, trace_limit=args.trace_limit,
        )
        report = run_feedback(config)
    except ValueError as exc:
        parser.error(str(exc))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print("Saved feedback report: " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
