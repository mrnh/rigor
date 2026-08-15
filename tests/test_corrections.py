import unittest

from rigor import corrections as corr


class TestBonferroni(unittest.TestCase):
    def test_adjusted_alpha_is_alpha_over_m(self):
        result = corr.bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05)
        self.assertAlmostEqual(result.adjusted_alpha, 0.0125, places=9)

    def test_rejects_only_below_adjusted_alpha(self):
        result = corr.bonferroni([0.001, 0.03, 0.5], alpha=0.05)
        # adjusted alpha = 0.05/3 = 0.01667
        self.assertEqual(result.reject, [True, False, False])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            corr.bonferroni([])


class TestBenjaminiHochberg(unittest.TestCase):
    def test_less_conservative_than_bonferroni(self):
        # A batch where BH should reject at least as many as Bonferroni.
        p_values = [0.001, 0.008, 0.02, 0.04, 0.15, 0.6]
        bh = corr.benjamini_hochberg(p_values, alpha=0.05)
        bonf = corr.bonferroni(p_values, alpha=0.05)
        self.assertGreaterEqual(sum(bh.reject), sum(bonf.reject))

    def test_known_textbook_example(self):
        # 5 p-values, alpha=0.05: sorted [0.001, 0.008, 0.039, 0.041, 0.042]
        # BH thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
        # p_(1)=0.001 <= 0.01 ok; p_(2)=0.008 <= 0.02 ok; p_(3)=0.039 <= 0.03? no
        # p_(4)=0.041 <= 0.04? no; p_(5)=0.042 <= 0.05? yes -> largest passing rank is 5
        # so ALL are rejected (step-up: once the largest rank passes, everything below it is too)
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042]
        result = corr.benjamini_hochberg(p_values, alpha=0.05)
        self.assertEqual(result.reject, [True, True, True, True, True])

    def test_nothing_significant_rejects_nothing(self):
        result = corr.benjamini_hochberg([0.5, 0.6, 0.7], alpha=0.05)
        self.assertEqual(result.reject, [False, False, False])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            corr.benjamini_hochberg([])


if __name__ == "__main__":
    unittest.main()
