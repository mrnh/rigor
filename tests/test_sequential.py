"""Sequential (mSPRT) testing tests. The one structural guarantee that
actually matters here -- more than any single p-value being "right" --
is the always-valid property itself: checking the test after every new
observation must not inflate the false-positive rate the way naively
re-running a fixed-sample test does. test_always_valid_guarantee_*
checks that directly by simulation, the same way test_power.py checks
sample_size/power are true inverses rather than trusting the formula by
inspection alone.
"""
import math
import random
import unittest

from rigor import sequential as seq


class TestAlwaysValidPValue(unittest.TestCase):
    def test_zero_effect_gives_p_one(self):
        lam, p = seq.always_valid_p_value(0.0, se=1.0, tau=0.05)
        self.assertLess(lam, 1.0)
        self.assertEqual(p, 1.0)

    def test_p_value_is_min_of_one_and_inverse_lambda(self):
        for theta, se, tau in [(0.3, 0.1, 0.05), (-2.0, 0.5, 1.0), (0.01, 0.2, 0.3)]:
            lam, p = seq.always_valid_p_value(theta, se, tau)
            self.assertAlmostEqual(p, min(1.0, 1.0 / lam), places=9)

    def test_more_evidence_shrinks_p_value(self):
        # Same standardized effect, smaller se (more information) -> more
        # extreme evidence -> smaller always-valid p-value.
        _, p_early = seq.always_valid_p_value(0.5, se=0.5, tau=0.2)
        _, p_later = seq.always_valid_p_value(0.5, se=0.1, tau=0.2)
        self.assertGreater(p_early, p_later)

    def test_nonpositive_se_raises(self):
        with self.assertRaises(ValueError):
            seq.always_valid_p_value(0.1, se=0.0, tau=0.1)

    def test_nonpositive_tau_raises(self):
        with self.assertRaises(ValueError):
            seq.always_valid_p_value(0.1, se=0.1, tau=0.0)


class TestSequentialTwoSampleMeanTest(unittest.TestCase):
    def test_strong_effect_is_significant(self):
        rng = random.Random(0)
        a = [rng.gauss(0.0, 1.0) for _ in range(300)]
        b = [rng.gauss(0.8, 1.0) for _ in range(300)]
        result = seq.sequential_two_sample_mean_test(a, b, tau=0.3)
        self.assertTrue(result.reject_null(0.05))
        self.assertLess(result.p_value, 0.01)
        self.assertAlmostEqual(result.theta_hat, sum(b) / len(b) - sum(a) / len(a), places=9)

    def test_no_effect_not_significant(self):
        rng = random.Random(1)
        a = [rng.gauss(0.0, 1.0) for _ in range(50)]
        b = [rng.gauss(0.0, 1.0) for _ in range(50)]
        result = seq.sequential_two_sample_mean_test(a, b, tau=0.3)
        self.assertFalse(result.reject_null(0.05))

    def test_small_n_warns(self):
        result = seq.sequential_two_sample_mean_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], tau=1.0)
        self.assertTrue(any("small" in w for w in result.warnings))

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            seq.sequential_two_sample_mean_test([1.0], [1.0, 2.0], tau=1.0)

    def test_zero_variance_identical_groups_gives_no_evidence_not_crash(self):
        # Regression test: this used to raise ValueError("se must be
        # positive") instead of handling the degenerate case the way
        # inference._safe_ratio does for the classical t-test.
        result = seq.sequential_two_sample_mean_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], tau=1.0)
        self.assertEqual(result.se, 0.0)
        self.assertEqual(result.z, 0.0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.reject_null(0.05))
        self.assertTrue(any("se=0" in w for w in result.warnings))

    def test_zero_variance_different_groups_stays_conservative_not_certain(self):
        # Same zero-variance crash, but with theta_hat != 0: every
        # observation in a differs from every observation in b with no
        # within-group spread at all. Tempting to read this as maximal
        # evidence, but for a test meant to be checked after every new
        # observation, a degenerate zero-variance estimate this early is
        # exactly the kind of small-sample noise the always-valid
        # guarantee has to stay robust to -- see _zero_se_result. z is
        # still reported honestly (it *is* an infinite standardized
        # gap), but p_value/reject_null must not follow it to certainty.
        result = seq.sequential_two_sample_mean_test([5.0, 5.0, 5.0], [9.0, 9.0, 9.0], tau=1.0)
        self.assertEqual(result.se, 0.0)
        self.assertEqual(result.z, math.inf)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.reject_null(0.05))


class TestSequentialTwoProportionTest(unittest.TestCase):
    def test_strong_effect_is_significant(self):
        result = seq.sequential_two_proportion_test(600, 1000, 400, 1000, tau=0.05)
        self.assertTrue(result.reject_null(0.05))
        self.assertAlmostEqual(result.theta_hat, -0.2, places=9)

    def test_identical_proportions_not_significant(self):
        result = seq.sequential_two_proportion_test(500, 1000, 500, 1000, tau=0.05)
        self.assertEqual(result.p_value, 1.0)

    def test_zero_variance_no_separation_returns_p_one_not_crash(self):
        result = seq.sequential_two_proportion_test(0, 1, 0, 1, tau=0.05)
        self.assertEqual(result.p_value, 1.0)
        self.assertTrue(any("se=0" in w for w in result.warnings))

    def test_complete_separation_at_small_n_stays_conservative_not_certain(self):
        # se=0 here comes from complete separation (0/5 vs 5/5). It's
        # tempting to treat this as the strongest possible result (a
        # one-shot Fisher's exact test on the same table would indeed
        # call it significant, p~=0.008) -- but this is a *sequential*
        # test, checked after every new observation, and complete
        # separation at n=1 per arm happens under the null purely by
        # chance far too often (~40%, verified by simulation during
        # development) to treat as proof. p_value/reject_null must stay
        # conservative; only z (a plain descriptive fact about the
        # observed gap) reflects the separation.
        result = seq.sequential_two_proportion_test(0, 5, 5, 5, tau=0.05)
        self.assertEqual(result.se, 0.0)
        self.assertEqual(result.theta_hat, 1.0)
        self.assertEqual(result.z, math.inf)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.reject_null(0.05))

    def test_out_of_range_successes_raises(self):
        with self.assertRaises(ValueError):
            seq.sequential_two_proportion_test(11, 10, 5, 10, tau=0.05)

    def test_low_count_warns(self):
        result = seq.sequential_two_proportion_test(1, 10, 8, 10, tau=0.05)
        self.assertTrue(any("below 5" in w for w in result.warnings))


class TestAlwaysValidGuaranteeUnderRepeatedPeeking(unittest.TestCase):
    def test_stopping_at_first_significant_look_still_bounds_false_positive_rate(self):
        # The whole point of this module: unlike a naive re-run of a
        # fixed-sample test at every look (see naive_peeking_inflation,
        # which is *expected* to blow past alpha), checking the
        # always-valid p-value after every new observation and stopping
        # at the first p < alpha must not exceed alpha's false-positive
        # rate under H0, for any stopping time. Simulated directly here
        # rather than just trusting the closed-form derivation.
        rng = random.Random(42)
        trials = 1500
        n_looks = 20
        alpha = 0.05
        tau = 0.05
        true_p = 0.3  # same in both arms -- H0 true
        false_positives = 0
        for _ in range(trials):
            s1 = s2 = 0
            rejected = False
            for k in range(1, n_looks + 1):
                s1 += 1 if rng.random() < true_p else 0
                s2 += 1 if rng.random() < true_p else 0
                result = seq.sequential_two_proportion_test(s1, k, s2, k, tau=tau)
                if result.p_value < alpha:
                    rejected = True
                    break
            if rejected:
                false_positives += 1
        rate = false_positives / trials
        # mSPRT is typically conservative (true rate well under the
        # nominal bound), so this is a loose upper-bound check, not a
        # tight equality -- the property being tested is "does not
        # exceed alpha," not "is exactly alpha."
        self.assertLessEqual(rate, alpha)


class TestNaivePeekingInflation(unittest.TestCase):
    def test_single_look_matches_nominal_alpha(self):
        # With only one look, "peeking" hasn't happened yet -- the naive
        # and correct false-positive rates should coincide.
        out = seq.naive_peeking_inflation(n_looks=1, alpha=0.05, trials=4000, seed=7)
        self.assertAlmostEqual(out["estimated_true_alpha"], 0.05, delta=0.02)

    def test_repeated_looks_inflate_false_positive_rate(self):
        out = seq.naive_peeking_inflation(n_looks=10, alpha=0.05, trials=4000, seed=7)
        self.assertGreater(out["estimated_true_alpha"], 0.10)

    def test_more_looks_inflate_further(self):
        few = seq.naive_peeking_inflation(n_looks=5, alpha=0.05, trials=3000, seed=1)
        many = seq.naive_peeking_inflation(n_looks=40, alpha=0.05, trials=3000, seed=1)
        self.assertLess(few["estimated_true_alpha"], many["estimated_true_alpha"])

    def test_is_deterministic_given_seed(self):
        a = seq.naive_peeking_inflation(n_looks=5, trials=500, seed=99)
        b = seq.naive_peeking_inflation(n_looks=5, trials=500, seed=99)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
