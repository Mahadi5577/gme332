"""Tests for the v1.1 fixes (science audit 2026-10-15)."""
import os
import subprocess
import sys

import cvxpy as cp
import numpy as np
import pytest

from gme332.config import CUTS, DIMS
from gme332.dynamics import liouvillian
from gme332.exact import QSqrt
from gme332.experiment import malik2016_certificate
from gme332.sdp import SolverError, _solve, family_witness_sdp
from gme332.states import ghz_family_ket, negativity_rate_terms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_qsqrt_perfect_square_is_canonical():
    x = QSqrt(2, -1, 4)                        # 2 - sqrt(4) = 0
    assert x.is_zero() and x.sign() == 0
    assert QSqrt(0, 1, 1).a == 1 and QSqrt(0, 1, 1).b == 0
    assert QSqrt(-3, 1, 4).sign() == -1        # -3 + 2 < 0


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_family_dual_matches_witness_formula(d):
    r = family_witness_sdp(d)
    formula = 2 * d * (d - 1) / (2 * (d * d - 1) + np.sqrt(d - 1))
    assert r["threshold"] == pytest.approx(formula, abs=1e-6)


def test_family_dual_gap_at_d6():
    r = family_witness_sdp(6)
    assert r["c_min"] == pytest.approx(-0.068410, abs=2e-6)
    assert r["threshold"] - 60 / (70 + np.sqrt(5)) > 5e-4


def test_solver_failure_is_raised():
    x = cp.Variable()
    with pytest.raises(SolverError):
        _solve(cp.Problem(cp.Minimize(x), [x >= 1, x <= 0]))


def test_kernel_term_of_initial_rate():
    k = ghz_family_ket(3); rho = np.outer(k, k)
    L = liouvillian(DIMS, 0.05, 0.10, 0.02)
    drho = (L @ rho.reshape(-1)).reshape(18, 18)
    neg, ker = negativity_rate_terms(rho, drho, DIMS, CUTS["AB|C"])
    assert ker == pytest.approx(1 / 60, abs=1e-6)          # +0.016667
    assert neg + ker == pytest.approx(-0.326667, abs=1e-6)


def test_malik_fidelity_uses_achievable_phase_alignment():
    m = malik2016_certificate()
    assert m["F"] == pytest.approx(0.7991, abs=1e-4)
    assert m["residual_phase_deg"] == pytest.approx(-4.06, abs=0.01)
    assert m["g_term_bound"] == pytest.approx(0.0021, abs=1e-4)
    assert m["witness_worst_case"][0.01] < -0.06


def test_notebook_is_generated_from_package():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_notebook.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.slow
def test_witness_sdps_accept_real_input():
    from gme332.sdp import decomposable_witness_value, genuine_multipartite_negativity
    k = ghz_family_ket(3); rho = np.outer(k, k)                # float64 input
    assert decomposable_witness_value(rho, DIMS, real=True)[0] == pytest.approx(-(4 - np.sqrt(2)) / 21, abs=1e-6)
    assert genuine_multipartite_negativity(rho, DIMS, real=True) == pytest.approx(np.sqrt(2) / 3, abs=1e-6)
