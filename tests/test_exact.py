from fractions import Fraction

from gme332.exact import ExMat, QSqrt, ex_is_psd


def test_qsqrt_arithmetic_and_sign():
    s2 = QSqrt(0, 1, 2)
    assert (s2 * s2).a == 2 and (s2 * s2).b == 0
    x = QSqrt(12, 0, 2) / QSqrt(16, 1, 2)          # 12/(16+sqrt2) = 6(16-sqrt2)/127
    assert x.a == Fraction(96, 127) and x.b == Fraction(-6, 127)
    assert QSqrt(3, -2, 2).sign() == 1             # 3 - 2 sqrt2 > 0
    assert QSqrt(-3, 3, 2).sign() == 1             # -3 + 3 sqrt2 > 0
    assert QSqrt(1, -1, 2).sign() == -1            # 1 - sqrt2 < 0


def test_exact_psd_detects_negative_and_rank_deficient():
    psd = ExMat(2, 2).add_outer({0: QSqrt(1, 0, 2), 1: QSqrt(0, 1, 2)})   # rank one
    assert ex_is_psd(psd)
    indef = ExMat(2, 2).add_sym(0, 1, 1)
    assert not ex_is_psd(indef)
