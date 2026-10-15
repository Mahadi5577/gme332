import math
from fractions import Fraction

import pytest

from gme332.certificates import exact_primal_certificate_d3, verify_witness_family


@pytest.mark.parametrize("d", [2, 3, 4, 5, 6])
def test_witness_family_is_fully_decomposable(d):
    v = verify_witness_family(d)
    assert v["decomposition_exact"] and v["P_Q_psd_exact"]
    formula = 2 * d * (d - 1) / (2 * (d * d - 1) + math.sqrt(d - 1))
    assert float(v["p_star"]) == pytest.approx(formula, abs=1e-14)


def test_d2_reproduces_ghz3_value():
    assert float(verify_witness_family(2)["p_star"]) == pytest.approx(4 / 7, abs=1e-15)


def test_d3_threshold_and_witness_value_exact():
    v = verify_witness_family(3)
    assert v["p_star"].a == Fraction(96, 127) and v["p_star"].b == Fraction(-6, 127)
    c = v["c_normalised"]                                  # -(4 - sqrt2)/21
    assert c.a == Fraction(-4, 21) and c.b == Fraction(1, 21)


def test_exact_primal_decomposition():
    p = exact_primal_certificate_d3()
    assert p["sum_equals_rho"] and p["all_ok"] and p["complementary_slackness"]


def test_rho_pstar_is_biseparable():
    from gme332.certificates import biseparability_certificate_d3
    r = biseparability_certificate_d3()
    assert r["biseparable"], {k: v for k, v in r["checks"].items() if not v}
