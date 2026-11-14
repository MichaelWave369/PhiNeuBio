import unittest

from phineubio.experiment import ExperimentConfig, dataset, run_experiment
from phineubio.reservoir import Reservoir, ReservoirConfig


class ExperimentTests(unittest.TestCase):
    def test_repeated_run_has_identical_report(self):
        config = ExperimentConfig(
            seed=7, train_sequences=4, test_sequences=3,
            sequence_length=16, burn_in=3, delay=2, epochs=3,
            reservoir=ReservoirConfig(seed=7, neurons=10)
        )
        first = run_experiment(config)
        self.assertEqual(first, run_experiment(config))
        self.assertEqual(first["samples"]["train"], 4 * 13)
        self.assertEqual(first["samples"]["test"], 3 * 13)
        for value in first["metrics"].values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(first["substrate"], "software_simulation")

    def test_dataset_seeds_are_distinct(self):
        reservoir = Reservoir(ReservoirConfig(neurons=8))
        a = dataset(reservoir, sequence_count=5, sequence_length=20, burn_in=3, delay=2, seed=101)
        b = dataset(reservoir, sequence_count=5, sequence_length=20, burn_in=3, delay=2, seed=202)
        self.assertNotEqual(a[1], b[1])

    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(delay=20, sequence_length=10)


if __name__ == "__main__":
    unittest.main()
