import unittest
import numpy as np
from reproduce_saved_scores import metrics, bootstrap


class MetricsTest(unittest.TestCase):
    def test_perfect_and_endpoint(self):
        m = metrics([0, 1], [0, 1])
        self.assertEqual(m['auroc'], 1)
        self.assertEqual(m['ece'], 0)
        self.assertEqual(metrics([0, 0], [1, 1])['ece'], 1)

    def test_reversal_and_ties(self):
        self.assertEqual(metrics([0, 1], [1, 0])['diagnostic_inverted_auroc'], 1)
        self.assertEqual(metrics([0, 1], [.5, .5])['auroc'], .5)

    def test_invalid(self):
        with self.assertRaises(ValueError): metrics([0, 1], [0, np.nan])
        with self.assertRaises(ValueError): metrics([0, 2], [0, 1])

    def test_single_class(self):
        self.assertIsNone(metrics([0, 0], [.2, .3])['auroc'])

    def test_deterministic_ci(self):
        y, p = [0, 0, 1, 1], [.1, .6, .4, .9]
        self.assertEqual(bootstrap(y, p, 20, 42), bootstrap(y, p, 20, 42))


if __name__ == '__main__': unittest.main()
