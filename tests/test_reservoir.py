import math
import unittest

from phineubio.reservoir import Reservoir, ReservoirConfig


class ReservoirTests(unittest.TestCase):
    def test_reproducible_after_reset(self):
        model = Reservoir(ReservoirConfig(neurons=12, seed=22))
        inputs = [0, 1, 1, 0, 1, 0]
        first = [model.step(bit) for bit in inputs]
        model.reset()
        second = [model.step(bit) for bit in inputs]
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 24)

    def test_seed_controls_connectivity(self):
        a = Reservoir(ReservoirConfig(neurons=16, seed=1))
        b = Reservoir(ReservoirConfig(neurons=16, seed=1))
        c = Reservoir(ReservoirConfig(neurons=16, seed=2))
        self.assertEqual(a.weights, b.weights)
        self.assertNotEqual(a.weights, c.weights)

    def test_invalid_stimuli(self):
        model = Reservoir()
        for value in [-1, 1.01, math.nan, math.inf, "yes"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                model.step(value)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ReservoirConfig(neurons=0)
        with self.assertRaises(ValueError):
            ReservoirConfig(leak=1)
        with self.assertRaises(ValueError):
            ReservoirConfig(density=-0.1)


if __name__ == "__main__":
    unittest.main()
