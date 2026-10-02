import numpy as np
import pytest

from gme332.config import CUTS, DIMS
from gme332.dynamics import Propagator, liouvillian
from gme332.states import ghz_family_ket, negativity, negativity_right_derivative


@pytest.fixture(scope="module")
def rho0():
    k = ghz_family_ket(3)
    return np.outer(k, k)


def test_initial_negativities(rho0):
    assert negativity(rho0, DIMS, CUTS["A|BC"]) == pytest.approx(np.sqrt(2) / 3, abs=1e-12)
    assert negativity(rho0, DIMS, CUTS["AB|C"]) == pytest.approx(1.0, abs=1e-12)
    assert negativity(rho0, DIMS, CUTS["AC|B"]) == pytest.approx(1.0, abs=1e-12)


def test_one_sided_rate_matches_finite_difference(rho0):
    L = liouvillian(DIMS, 0.05, 0.10, 0.02)
    prop = Propagator(L, rho0.astype(complex))
    drho = (L @ rho0.reshape(-1)).reshape(18, 18)
    for k in CUTS.values():
        rate = negativity_right_derivative(rho0, drho, DIMS, k)
        fd = (negativity(prop(1e-7), DIMS, k) - negativity(rho0, DIMS, k)) / 1e-7
        assert rate == pytest.approx(fd, abs=1e-5)
    assert negativity_right_derivative(rho0, drho, DIMS, CUTS["AB|C"]) == pytest.approx(-0.326667, abs=1e-6)
