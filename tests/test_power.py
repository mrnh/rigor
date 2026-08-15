"""Power/sample-size tests. The main structural guarantee tested here is
that sample_size_* and power_* are true numerical inverses of each other
(sample_size solves for n via bisection over power, so this isn't
comparing two independently-derived formulas -- it's checking the solver
actually found where they cross). Alongside that, a loose sanity check
against the widely-cited Cohen (1988) reference case anchors the whole
family to the literature, with a tolerance wide enough to allow for the
normal-vs-noncentral-t approximation gap this module documents.
"""
import math
import unittest

from rigor import power as pw


class TestTwoSampleTPowerSampleSizeConsistency(unittest.TestCase):
    def test_power_increases_with_n(self):
        low = pw.power_two_sample_t_test(n=10, effect_size=0.5)
        high = pw.power_two_sample_t_test(n=200, effect_size=0.5)
        self.assertLess(low, high)

    def test_power_is_alpha_at_zero_effect_size(self):
        # With no true effect, "power" to reject is just the false-positive rate.
        power = pw.power_two_sample_t_test(n=100, effect_size=0.0, alpha=0.05)
        self.assertAlmostEqual(power, 0.05, places=6)

    def test_sample_size_meets_but_just_barely_exceeds_target_power(self):
        n = pw.sample_size_two_sample_t_test(effect_size=0.5, alpha=0.05, power=0.8)
        self.assertGreaterEqual(pw.power_two_sample_t_test(n, 0.5), 0.8 - 1e-6)
        self.assertLess(pw.power_two_sample_t_test(n - 1, 0.5), 0.8)

    def test_matches_cohen_1988_reference_case_within_tolerance(self):
        # The single most-cited number in power analysis: d=0.5 (medium),
        # alpha=.05 two-tailed, power=.80 -> classic answer is n~=64 per
        # group (e.g. Cohen's own tables, and modern exact-noncentral-t
        # tools like G*Power/R's pwr package all land in the 63-64 range).
        # This module's normal approximation should land close, not exact.
        n = pw.sample_size_two_sample_t_test(effect_size=0.5, alpha=0.05, power=0.8)
        self.assertTrue(55 <= n <= 75, f"n={n}, expected roughly 64")

    def test_zero_effect_size_raises(self):
        with self.assertRaises(ValueError):
            pw.sample_size_two_sample_t_test(effect_size=0.0)

    def test_larger_effect_needs_smaller_sample(self):
        n_small_effect = pw.sample_size_two_sample_t_test(effect_size=0.2, power=0.8)
        n_large_effect = pw.sample_size_two_sample_t_test(effect_size=0.8, power=0.8)
        self.assertGreater(n_small_effect, n_large_effect)


class TestOneSamplePower(unittest.TestCase):
    def test_consistency(self):
        n = pw.sample_size_one_sample_t_test(effect_size=0.4, power=0.9)
        self.assertGreaterEqual(pw.power_one_sample_t_test(n, 0.4), 0.9 - 1e-6)
        self.assertLess(pw.power_one_sample_t_test(n - 1, 0.4), 0.9)


class TestTwoProportionPower(unittest.TestCase):
    def test_power_increases_with_n(self):
        low = pw.power_two_proportion_z_test(n=20, p1=0.5, p2=0.3)
        high = pw.power_two_proportion_z_test(n=500, p1=0.5, p2=0.3)
        self.assertLess(low, high)

    def test_sample_size_consistency(self):
        n = pw.sample_size_two_proportion_z_test(p1=0.5, p2=0.4, alpha=0.05, power=0.8)
        self.assertGreaterEqual(pw.power_two_proportion_z_test(n, 0.5, 0.4), 0.8 - 1e-6)
        self.assertLess(pw.power_two_proportion_z_test(n - 1, 0.5, 0.4), 0.8)

    def test_identical_proportions_raise(self):
        with self.assertRaises(ValueError):
            pw.sample_size_two_proportion_z_test(p1=0.3, p2=0.3)

    def test_larger_gap_needs_smaller_sample(self):
        n_small_gap = pw.sample_size_two_proportion_z_test(p1=0.51, p2=0.50, power=0.8)
        n_large_gap = pw.sample_size_two_proportion_z_test(p1=0.7, p2=0.3, power=0.8)
        self.assertGreater(n_small_gap, n_large_gap)


if __name__ == "__main__":
    unittest.main()
