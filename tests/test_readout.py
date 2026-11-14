import unittest

from phineubio.readout import LogisticReadout


class ReadoutTests(unittest.TestCase):
    def test_can_fit_separable_data(self):
        data = [(0.0,), (1.0,)] * 20
        labels = [0, 1] * 20
        model = LogisticReadout()
        model.fit(data, labels, epochs=12, learning_rate=0.1)
        self.assertGreater(model.score(data, labels), 0.95)

    def test_unfitted_readout_raises(self):
        with self.assertRaises(ValueError):
            LogisticReadout().predict([1.0])

    def test_mismatched_inputs_raise(self):
        with self.assertRaises(ValueError):
            LogisticReadout().fit([(1.0,)], [1, 0])


if __name__ == "__main__":
    unittest.main()
