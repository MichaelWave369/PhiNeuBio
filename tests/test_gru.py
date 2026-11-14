"""Numerical gate-gradient checks and deterministic learning tests for Phase 7."""

import math
import unittest

from phineubio.gru import TrainableGRU
from phineubio.temporal import make_sequences


class GRUTests(unittest.TestCase):
    def test_shapes_and_parameter_count(self):
        model = TrainableGRU(4, 123)
        states, predictions = model.forward([0, 1, 0])
        self.assertEqual(len(states), 3)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(states[0]["h"]), 4)
        self.assertEqual(model.parameter_count, 3 * 16 + 7 * 4 + 1)
        self.assertTrue(all(0 < p < 1 for p in predictions))

    def test_gate_and_candidate_gradients_match_finite_difference(self):
        model = TrainableGRU(3, 67)
        inputs = [1, 0, 0, 1, 1, 0]
        clean = [1, 0, 1, 1, 0, 1]
        _, gradients = model.loss_and_gradients(inputs, clean, delay=2, start=3)
        eps = 1e-5
        for name, i, j in [("uz", 0, 1), ("ur", 1, 0), ("un", 2, 1),
                           ("wzx", 0, None), ("wrx", 1, None), ("wnx", 2, None)]:
            with self.subTest(name=name):
                weights = getattr(model, name)
                if j is None:
                    old = weights[i]
                    weights[i] = old + eps
                    high = model.loss_and_gradients(inputs, clean, 2, 3)[0]
                    weights[i] = old - eps
                    low = model.loss_and_gradients(inputs, clean, 2, 3)[0]
                    weights[i] = old
                    analytic = gradients[name][i]
                else:
                    old = weights[i][j]
                    weights[i][j] = old + eps
                    high = model.loss_and_gradients(inputs, clean, 2, 3)[0]
                    weights[i][j] = old - eps
                    low = model.loss_and_gradients(inputs, clean, 2, 3)[0]
                    weights[i][j] = old
                    analytic = gradients[name][i][j]
                self.assertAlmostEqual(analytic, (high - low) / (2 * eps), places=5)

    def test_recurrent_and_gates_update(self):
        model = TrainableGRU(3, 12)
        before = [row[:] for row in model.un], [row[:] for row in model.uz]
        data = [1, 1, 0, 1, 0, 0, 1, 1]
        result = model.train_sequence(data, data, 1, 3, 0.1, 5.0)
        self.assertTrue(math.isfinite(result))
        self.assertNotEqual(before[0], model.un)
        self.assertNotEqual(before[1], model.uz)

    def test_training_is_seeded_and_reproducible(self):
        data = make_sequences(5, 15, 21, 0.9, 0.1)
        a, b = TrainableGRU(4, 20), TrainableGRU(4, 20)
        curve = a.train(data, 2, 4, 3, 0.1, 5.0, 37)
        self.assertEqual(curve, b.train(data, 2, 4, 3, 0.1, 5.0, 37))
        self.assertEqual(a.un, b.un)
        self.assertTrue(0 <= a.accuracy(data, 2, 4) <= 1)

    def test_invalid_samples_and_configuration(self):
        with self.assertRaises(ValueError):
            TrainableGRU(0, 1)
        g = TrainableGRU(3, 1)
        with self.assertRaises(ValueError):
            g.forward([0, 2])
        with self.assertRaises(ValueError):
            g.loss_and_gradients([0, 1], [1], 1, 1)
        with self.assertRaises(ValueError):
            g.train_sequence([0, 1], [1, 0], 1, 1, -0.1, 5.0)


if __name__ == "__main__":
    unittest.main()
