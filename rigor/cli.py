"""rigor's CLI — classical statistical inference, verified and citable.

LLM agents (and people) doing data analysis constantly need small,
well-defined statistical answers -- is this difference significant, how
big is the effect, how many samples do I need -- and constantly get them
subtly wrong: the wrong test for the data shape, a one-tailed/two-tailed
mix-up, a forgotten multiple-comparisons correction, a p-value treated
as effect size. This computes the actual answer instead, with the
formula used and its assumptions stated, rather than a number recalled
from training data.

Commands (as `rigor ...` once installed, or `python3 rigor.py ...` run
straight from a checkout without installing):
    rigor ttest one-sample --data 1,2,3 --mu0 0
    rigor ttest two-sample --a 1,2,3 --b 4,5,6 [--equal-var]
    rigor ttest paired --a 1,2,3 --b 4,5,6
    rigor ztest one-proportion --successes 45 --n 100 --p0 0.5
    rigor ztest two-proportion --successes1 80 --n1 100 --successes2 20 --n2 100
    rigor chi2 goodness-of-fit --observed 5,8,9 --expected 10,10,10
    rigor chi2 independence --table "10,20;30,40"
    rigor anova --groups "1,2,3|4,5,6|7,8,9"
    rigor effect-size cohens-d --a 1,2,3 --b 4,5,6
    rigor effect-size cohens-h --p1 0.6 --p2 0.3
    rigor effect-size cramers-v --chi2 10 --n 100 --rows 2 --cols 2
    rigor power ttest-2samp --effect-size 0.5 --power 0.8      # -> required n
    rigor power ttest-2samp --effect-size 0.5 --n 64           # -> achieved power
    rigor power proportion --p1 0.5 --p2 0.4 --power 0.8
    rigor correct bonferroni --p 0.01,0.02,0.03,0.04
    rigor correct bh --p 0.01,0.02,0.03,0.04
"""
import argparse
import math
import sys

from rigor import corrections, effect_size, inference, power


def _floats(csv: str):
    return [float(x) for x in csv.split(",") if x.strip() != ""]


def _print_result(result: inference.TestResult, alpha: float) -> None:
    print(f"{result.name}")
    print(f"  statistic = {result.statistic:.6g}")
    if result.df is not None:
        df_str = f"df = {result.df:.4g}"
        if result.df2 is not None:
            df_str += f", {result.df2:.4g}"
        print(f"  {df_str}")
    print(f"  p-value   = {result.p_value:.6g}")
    print(f"  reject H0 at alpha={alpha}: {result.reject_null(alpha)}")
    if result.confidence_interval:
        lo, hi = result.confidence_interval
        pct = int(result.confidence_level * 100)
        print(f"  {pct}% CI    = ({lo:.6g}, {hi:.6g})")
    print(f"  citation  = {result.citation}")
    for w in result.warnings:
        print(f"  warning   : {w}")


def _print_correction(result: corrections.CorrectionResult) -> None:
    print(f"{result.method} (alpha={result.alpha})")
    if result.adjusted_alpha is not None:
        print(f"  adjusted alpha = {result.adjusted_alpha:.6g}")
    for p, rej in zip(result.p_values, result.reject):
        print(f"  p={p:.6g}  {'REJECT (significant)' if rej else 'fail to reject'}")
    print(f"  citation = {result.citation}")


def cmd_ttest(args) -> int:
    if args.ttest_kind == "one-sample":
        result = inference.one_sample_t_test(_floats(args.data), args.mu0)
    elif args.ttest_kind == "two-sample":
        result = inference.two_sample_t_test(_floats(args.a), _floats(args.b), equal_var=args.equal_var)
    else:  # paired
        result = inference.paired_t_test(_floats(args.a), _floats(args.b))
    _print_result(result, args.alpha)
    return 0


def cmd_ztest(args) -> int:
    if args.ztest_kind == "one-proportion":
        result = inference.one_proportion_z_test(args.successes, args.n, args.p0)
    else:  # two-proportion
        result = inference.two_proportion_z_test(args.successes1, args.n1, args.successes2, args.n2)
    _print_result(result, args.alpha)
    return 0


def cmd_chi2(args) -> int:
    if args.chi2_kind == "goodness-of-fit":
        result = inference.chi_square_goodness_of_fit(_floats(args.observed), _floats(args.expected))
    else:  # independence
        table = [[float(x) for x in row.split(",")] for row in args.table.split(";")]
        result = inference.chi_square_independence(table)
    _print_result(result, args.alpha)
    return 0


def cmd_anova(args) -> int:
    groups = [_floats(g) for g in args.groups.split("|")]
    result = inference.one_way_anova(*groups)
    _print_result(result, args.alpha)
    return 0


def cmd_effect_size(args) -> int:
    if args.es_kind == "cohens-d":
        value = effect_size.cohens_d(_floats(args.a), _floats(args.b))
        print(f"Cohen's d = {value:.6g}")
    elif args.es_kind == "hedges-g":
        value = effect_size.hedges_g(_floats(args.a), _floats(args.b))
        print(f"Hedges' g = {value:.6g}")
    elif args.es_kind == "cohens-h":
        value = effect_size.cohens_h(args.p1, args.p2)
        print(f"Cohen's h = {value:.6g}")
    else:  # cramers-v
        value = effect_size.cramers_v(args.chi2, args.n, args.rows, args.cols)
        print(f"Cramer's V = {value:.6g}")
    return 0


def cmd_power(args) -> int:
    if args.power_kind == "ttest-2samp":
        if args.power is not None:
            n = power.sample_size_two_sample_t_test(args.effect_size, args.alpha, args.power)
            print(f"Required n per group = {n:.2f} (round up: {math.ceil(n)})")
        else:
            p = power.power_two_sample_t_test(args.n, args.effect_size, args.alpha)
            print(f"Power = {p:.6g}")
    elif args.power_kind == "ttest-1samp":
        if args.power is not None:
            n = power.sample_size_one_sample_t_test(args.effect_size, args.alpha, args.power)
            print(f"Required n = {n:.2f}")
        else:
            p = power.power_one_sample_t_test(args.n, args.effect_size, args.alpha)
            print(f"Power = {p:.6g}")
    else:  # proportion
        if args.power is not None:
            n = power.sample_size_two_proportion_z_test(args.p1, args.p2, args.alpha, args.power)
            print(f"Required n per group = {n:.2f}")
        else:
            p = power.power_two_proportion_z_test(args.n, args.p1, args.p2, args.alpha)
            print(f"Power = {p:.6g}")
    return 0


def cmd_correct(args) -> int:
    p_values = _floats(args.p)
    if args.correct_kind == "bonferroni":
        result = corrections.bonferroni(p_values, args.alpha)
    else:  # bh
        result = corrections.benjamini_hochberg(p_values, args.alpha)
    _print_correction(result)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Classical statistical inference, verified and citable.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ttest = sub.add_parser("ttest", help="t-tests")
    p_ttest.add_argument("ttest_kind", choices=["one-sample", "two-sample", "paired"])
    p_ttest.add_argument("--data", help="comma-separated sample (one-sample)")
    p_ttest.add_argument("--mu0", type=float, default=0.0)
    p_ttest.add_argument("--a", help="comma-separated sample A")
    p_ttest.add_argument("--b", help="comma-separated sample B")
    p_ttest.add_argument("--equal-var", action="store_true", help="use pooled variance instead of Welch's")
    p_ttest.add_argument("--alpha", type=float, default=0.05)
    p_ttest.set_defaults(func=cmd_ttest)

    p_ztest = sub.add_parser("ztest", help="z-tests for proportions")
    p_ztest.add_argument("ztest_kind", choices=["one-proportion", "two-proportion"])
    p_ztest.add_argument("--successes", type=int)
    p_ztest.add_argument("--n", type=int)
    p_ztest.add_argument("--p0", type=float)
    p_ztest.add_argument("--successes1", type=int)
    p_ztest.add_argument("--n1", type=int)
    p_ztest.add_argument("--successes2", type=int)
    p_ztest.add_argument("--n2", type=int)
    p_ztest.add_argument("--alpha", type=float, default=0.05)
    p_ztest.set_defaults(func=cmd_ztest)

    p_chi2 = sub.add_parser("chi2", help="chi-squared tests")
    p_chi2.add_argument("chi2_kind", choices=["goodness-of-fit", "independence"])
    p_chi2.add_argument("--observed", help="comma-separated observed counts")
    p_chi2.add_argument("--expected", help="comma-separated expected counts")
    p_chi2.add_argument("--table", help="rows separated by ';', values by ',' e.g. '10,20;30,40'")
    p_chi2.add_argument("--alpha", type=float, default=0.05)
    p_chi2.set_defaults(func=cmd_chi2)

    p_anova = sub.add_parser("anova", help="one-way ANOVA")
    p_anova.add_argument("--groups", required=True, help="groups separated by '|', values by ',' e.g. '1,2,3|4,5,6'")
    p_anova.add_argument("--alpha", type=float, default=0.05)
    p_anova.set_defaults(func=cmd_anova)

    p_es = sub.add_parser("effect-size", help="effect size measures")
    p_es.add_argument("es_kind", choices=["cohens-d", "hedges-g", "cohens-h", "cramers-v"])
    p_es.add_argument("--a", help="comma-separated sample A")
    p_es.add_argument("--b", help="comma-separated sample B")
    p_es.add_argument("--p1", type=float)
    p_es.add_argument("--p2", type=float)
    p_es.add_argument("--chi2", type=float)
    p_es.add_argument("--n", type=int)
    p_es.add_argument("--rows", type=int)
    p_es.add_argument("--cols", type=int)
    p_es.set_defaults(func=cmd_effect_size)

    p_power = sub.add_parser("power", help="power / sample-size calculators")
    p_power.add_argument("power_kind", choices=["ttest-2samp", "ttest-1samp", "proportion"])
    p_power.add_argument("--effect-size", type=float, dest="effect_size")
    p_power.add_argument("--p1", type=float)
    p_power.add_argument("--p2", type=float)
    p_power.add_argument("--n", type=float, help="observations per group -- given this, compute achieved power")
    p_power.add_argument("--power", type=float, help="target power -- given this, compute required n")
    p_power.add_argument("--alpha", type=float, default=0.05)
    p_power.set_defaults(func=cmd_power)

    p_correct = sub.add_parser("correct", help="multiple-comparisons correction")
    p_correct.add_argument("correct_kind", choices=["bonferroni", "bh"])
    p_correct.add_argument("--p", required=True, help="comma-separated p-values")
    p_correct.add_argument("--alpha", type=float, default=0.05)
    p_correct.set_defaults(func=cmd_correct)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
