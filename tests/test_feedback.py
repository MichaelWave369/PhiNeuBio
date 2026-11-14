import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from phineubio.feedback import BinaryRewardAgent, FeedbackConfig, main, run_feedback


class FeedbackTests(unittest.TestCase):
    def test_reproducible_learning_report(self):
        cfg = FeedbackConfig(seed=21, neurons=12, train_trials=100, evaluation_trials=40)
        report = run_feedback(cfg)
        self.assertEqual(report, run_feedback(cfg))
        self.assertEqual(report["schema_version"], "2.0")
        self.assertEqual(report["evaluation"]["held_out_trial_count"], 40)
        self.assertEqual(report["metrics"]["before_accuracy"], 0.5)
        self.assertGreater(report["metrics"]["after_accuracy"], 0.5)
        self.assertEqual(len(report["trace"]), 100)
        self.assertTrue(all(len(frame["voltages"]) == 12 for frame in report["trace"]))

    def test_training_disabled_trace_still_evaluates(self):
        cfg = FeedbackConfig(seed=1, neurons=8, train_trials=80, evaluation_trials=20, trace_limit=0)
        report = run_feedback(cfg)
        self.assertEqual(report["trace"], [])
        self.assertIn("after_accuracy", report["metrics"])

    def test_feedback_agent_binary_rule(self):
        agent = BinaryRewardAgent(2, 0.1)
        before = agent.probability([1.0, 0.0])
        agent.update([1.0, 0.0], action=0, reward=0)
        self.assertGreater(agent.probability([1.0, 0.0]), before)
        with self.assertRaises(ValueError):
            agent.update([1.0, 0.0], action=0, reward=3)

    def test_invalid_config(self):
        for kwargs in [
            {"train_trials": 1}, {"evaluation_trials": 1},
            {"neurons": 0}, {"exploration_rate": -0.1},
            {"exploration_rate": 2}, {"learning_rate": 0}, {"trace_limit": -1}
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FeedbackConfig(**kwargs)

    def test_cli_creates_json_for_dashboard(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "feedback.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--neurons", "8", "--train-trials", "16",
                             "--evaluation-trials", "10", "--output", str(output)])
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["task"], "binary_cue_reward_feedback")
            self.assertEqual(report["neuron_count"], 8)


if __name__ == "__main__":
    unittest.main()
