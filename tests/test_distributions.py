"""Verify the distribution primitives against closed-form identities.

Rather than trust that a remembered numerical-recipes algorithm was
transcribed correctly, each distribution is checked against a case where
independent math gives an exact closed-form answer:

- t distribution with df=1 is exactly the standard Cauchy distribution.
- chi-squared with df=2 is exactly Exponential(rate=1/2), which has a
  closed-form CDF.
- A squared t(df) variate is exactly distributed F(1, df) — this ties
  the t and F implementations to each other independently of either
  one's internal algorithm.

If these hold to high precision, the incomplete gamma/beta machinery
underneath is correct, regardless of how confident anyone is about
having recalled the coefficients right.
"""
import math
import unittest

from rigor import distributions as d


class TestNormal(unittest.TestCase):
    def test_standard_normal_median_is_zero(self):
        self.assertAlmostEqual(d.normal_cdf(0.0), 0.5, places=12)

    def test_round_trip_cdf_ppf(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            self.assertAlmostEqual(d.normal_cdf(d.normal_ppf(p)), p, places=9)

    def test_known_critical_value_1_96(self):
        # The textbook 95% two-tailed normal critical value.
        self.assertAlmostEqual(d.normal_ppf(0.975), 1.959963985, places=6)


class TestTDistributionAgainstCauchy(unittest.TestCase):
    """df=1 Student's t is exactly the standard Cauchy distribution:
    CDF(x) = 1/2 + atan(x)/pi."""

    def test_matches_cauchy_cdf_across_range(self):
        for x in (-5.0, -1.5, -0.3, 0.0, 0.3, 1.5, 5.0):
            expected = 0.5 + math.atan(x) / math.pi
            self.assertAlmostEqual(d.t_cdf(x, df=1), expected, places=9)

    def test_ppf_round_trip(self):
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            x = d.t_ppf(p, df=1)
            self.assertAlmostEqual(d.t_cdf(x, df=1), p, places=6)

    def test_converges_to_normal_for_large_df(self):
        for x in (-2.0, -0.5, 0.5, 2.0):
            self.assertAlmostEqual(d.t_cdf(x, df=200000), d.normal_cdf(x), places=3)

    def test_known_textbook_critical_value_df10(self):
        # Standard two-tailed 5% critical value for df=10, in every t-table.
        self.assertAlmostEqual(d.t_critical(0.05, df=10), 2.228, places=2)


class TestChiSquaredAgainstExponential(unittest.TestCase):
    """df=2 chi-squared is exactly Exponential(rate=1/2): CDF(x) = 1 - exp(-x/2)."""

    def test_matches_exponential_cdf(self):
        for x in (0.1, 1.0, 3.0, 10.0):
            expected = 1.0 - math.exp(-x / 2.0)
            self.assertAlmostEqual(d.chi2_cdf(x, df=2), expected, places=9)

    def test_cdf_is_zero_at_zero(self):
        self.assertEqual(d.chi2_cdf(0.0, df=2), 0.0)

    def test_ppf_round_trip(self):
        for p in (0.05, 0.5, 0.95):
            x = d.chi2_ppf(p, df=5)
            self.assertAlmostEqual(d.chi2_cdf(x, df=5), p, places=6)

    def test_known_textbook_critical_value_df5(self):
        self.assertAlmostEqual(d.chi2_critical(0.05, df=5), 11.070, places=2)


class TestFDistributionTiesToT(unittest.TestCase):
    """If T ~ t(df), then T^2 ~ F(1, df). Ties the F implementation to
    the t implementation independently of either one's own algorithm."""

    def test_squared_t_matches_f(self):
        for df in (3, 10, 30):
            for t in (0.5, 1.5, 2.5):
                lhs = 2 * d.t_cdf(t, df) - 1  # P(|T| <= t)
                rhs = d.f_cdf(t * t, 1, df)   # P(F <= t^2)
                self.assertAlmostEqual(lhs, rhs, places=6, msg=(df, t))

    def test_ppf_round_trip(self):
        for p in (0.1, 0.5, 0.9):
            x = d.f_ppf(p, 3, 20)
            self.assertAlmostEqual(d.f_cdf(x, 3, 20), p, places=6)


if __name__ == "__main__":
    unittest.main()
