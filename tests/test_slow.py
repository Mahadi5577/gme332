"""Numerical cross-checks (CLARABEL SDPs, Qiskit). Run with:  pytest -m slow"""
import numpy as np
import pytest

from gme332.config import DIMS
from gme332.states import ghz_family_ket

pytestmark = pytest.mark.slow
P_STAR = 12 / (16 + np.sqrt(2))


@pytest.fixture(scope="module")
def rho0():
    k = ghz_family_ket(3)
    return np.outer(k, k)


def test_sdp_primal_and_dual_match_exact_threshold(rho0):
    from gme332.sdp import decomposable_witness_value, ppt_mixture_noise_threshold
    assert ppt_mixture_noise_threshold(rho0, DIMS) == pytest.approx(P_STAR, abs=1e-6)
    c, _ = decomposable_witness_value(rho0, DIMS)
    assert c == pytest.approx(-(4 - np.sqrt(2)) / 21, abs=1e-6)


def test_weak_constraint_is_not_a_witness(rho0):
    from gme332.sdp import weak_constraint_pitfall
    pit = weak_constraint_pitfall(rho0, DIMS)
    assert pit["value"] == pytest.approx(-1 / 6, abs=1e-6)
    assert pit["value_on_product_state"] < -0.05


def test_preparation_circuit_prepares_target():
    from qiskit.quantum_info import Statevector
    from gme332.basis import flat_index
    from gme332.device import encoded_register_index, ghz233_circuit
    target = np.zeros(32)
    ket = ghz_family_ket(3)
    for a in range(2):
        for b in range(3):
            for c in range(3):
                target[encoded_register_index(a, b, c)] = ket[flat_index((a, b, c), DIMS)]
    assert np.allclose(Statevector(ghz233_circuit()).data, target, atol=1e-12)
