"""IID control validates independent random bits and nested noise masks."""

import unittest

from phineubio.fairness import FairnessConfig, generate_sequences


class IIDSignalTests(unittest.TestCase):
    def test_iid_is_repeatable_and_noise_masks_nested(self):
        clean = generate_sequences("iid", 8, 50, 369, 0.85, 6, 0)
        low = generate_sequences("iid", 8, 50, 369, 0.85, 6, 0.1)
        high = generate_sequences("iid", 8, 50, 369, 0.85, 6, 0.4)
        self.assertEqual(clean, generate_sequences("iid", 8, 50, 369, 0.85, 6, 0))
        for (truth, observed), (_, low_obs), (_, high_obs) in zip(clean, low, high):
            self.assertEqual(truth, observed)
            self.assertTrue(all(not(a != b and a == c) for a,b,c in zip(truth,low_obs,high_obs)))
        self.assertGreater(sum(a != b for clean_bits, _ in clean for a,b in zip(clean_bits,clean_bits[1:])),100)

    def test_fairness_config_accepts_iid(self):
        cfg = FairnessConfig(signal_families=("iid",), seeds=(369,), delays=(1,))
        self.assertEqual(cfg.signal_families, ("iid",))


if __name__ == "__main__":
    unittest.main()
