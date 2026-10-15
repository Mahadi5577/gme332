import pytest

from gme332.experiment import load_malik2016, malik2016_certificate


def test_data_file_is_complete():
    pops, cohs = load_malik2016()
    assert len(pops) == 18 and len(cohs) == 3
    assert sum(pops.values()) == pytest.approx(1.0, abs=1e-9)


def test_reproduces_published_fidelity_and_witness_bound():
    m = malik2016_certificate()
    assert m["F"] == pytest.approx(0.7991, abs=1e-4)       # reported 0.801 +- 0.018
    assert m["witness"] == pytest.approx(-0.0755, abs=5e-5)
    assert m["q_witness"] == pytest.approx(0.576, abs=1e-3)
    assert m["q_fidelity"] == pytest.approx(0.178, abs=1e-3)
