"""Noisy-device simulation: local qubit encoding, 5-CNOT circuit,
FakeSherbrooke (IBM Eagle r3) calibration snapshot, density-matrix GME tests."""
import itertools
from typing import Dict

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from .basis import flat_index
from .certificates import verify_witness_family
from .config import DIMS, RANDOM_SEED
from .sdp import ppt_mixture_noise_threshold
from .states import ghz_family_ket


def encoded_register_index(a, b, c):
    """Little-endian register index for the local encoding A->q0, B->(q1,q2),
    C->(q3,q4); qutrit level j is the two-qubit state |j> (j = 0, 1, 2)."""
    return a + 2 * b + 8 * c


def register_to_party_order(rho_reg):
    """Reorder the 32 x 32 register density matrix to the party basis
    |a, b4, c4> of C^2 (x) C^4 (x) C^4 (index 16 a + 4 b4 + c4)."""
    perm = np.zeros(32, dtype=int)
    for a, b4, c4 in itertools.product(range(2), range(4), range(4)):
        perm[16 * a + 4 * b4 + c4] = a + 2 * b4 + 8 * c4
    return rho_reg[np.ix_(perm, perm)]


def ghz233_circuit() -> QuantumCircuit:
    """Prepares (|000> + |111> + |122>)/sqrt3 in the local encoding with 5 CNOTs:
    RY on q0 (amplitudes 1/sqrt3, sqrt(2/3)), controlled-H q0->q1,
    q2 = q0 AND NOT q1 (two CNOTs), then copy q1->q3 and q2->q4."""
    qc = QuantumCircuit(5)
    qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0)
    qc.ch(0, 1)
    qc.cx(0, 2); qc.cx(1, 2)
    qc.cx(1, 3); qc.cx(2, 4)
    return qc


def device_simulation(shots=8192, noise_margins=True) -> Dict:
    backend = FakeSherbrooke()
    ket     = ghz_family_ket(3)
    ket_reg = np.zeros(32, complex)
    for a, b, c in itertools.product(range(2), range(3), range(3)):
        ket_reg[encoded_register_index(a, b, c)] = ket[flat_index((a, b, c), DIMS)]
    qc = ghz233_circuit()
    assert np.allclose(Statevector(qc).data, ket_reg, atol=1e-12)

    tc   = transpile(qc, backend, optimization_level=3, seed_transpiler=RANDOM_SEED)
    phys = list(tc.layout.final_index_layout())
    tc_dm = tc.copy(); tc_dm.save_density_matrix(qubits=phys)
    sim_dm = AerSimulator.from_backend(backend, method="density_matrix")
    rho_reg = np.asarray(sim_dm.run(tc_dm, seed_simulator=RANDOM_SEED)
                         .result().data()["density_matrix"].data)

    qc_m = qc.copy(); qc_m.measure_all()
    tcm  = transpile(qc_m, backend, optimization_level=3, seed_transpiler=RANDOM_SEED)
    counts = (AerSimulator.from_backend(backend)
              .run(tcm, shots=shots, seed_simulator=RANDOM_SEED).result().get_counts())

    # device parameters actually used (initial layout of the prepared qubits)
    init = list(tc.layout.initial_index_layout())[:5]
    t1 = [backend.qubit_properties(q).t1 * 1e6 for q in init]
    t2 = [backend.qubit_properties(q).t2 * 1e6 for q in init]
    ro = [backend.target["measure"][(q,)].error for q in init]
    ecr_err = [inst.error for qargs, inst in backend.target["ecr"].items()
               if inst is not None and set(qargs) <= set(init)]

    # party-ordered 2x4x4 state and the post-selected 2x3x3 state
    rho_party = register_to_party_order(rho_reg)
    keep = [16 * a + 4 * b + c for a, b, c in itertools.product(range(2), range(3), range(3))]
    rho_proj  = rho_party[np.ix_(keep, keep)]
    p_post    = float(np.real(np.trace(rho_proj)))
    rho_post  = rho_proj / p_post
    ket_party = np.zeros(32); ket_party[keep] = ket
    F_reg  = float(np.real(ket_party @ rho_party @ ket_party))
    F_post = float(np.real(ket @ rho_post @ ket))

    W3 = verify_witness_family(3)["W_numeric"]           # Tr W = 1, exact decomposable
    wit_post = float(np.real(np.trace(W3 @ rho_post)))
    wit_reg  = p_post * wit_post                         # embedded witness on the register
    if noise_margins:                                   # the 2x4x4 SDP dominates the runtime
        q_post = ppt_mixture_noise_threshold(rho_post, DIMS)
        q_reg  = ppt_mixture_noise_threshold(rho_party, [2, 4, 4])
    else:
        q_post = q_reg = float("nan")
    return dict(counts=counts, shots=shots, backend=backend.name,
                depth=tc.depth(), n_ecr=tc.count_ops().get("ecr", 0),
                layout=init, T1_us=t1, T2_us=t2, readout_err=ro, ecr_err=ecr_err,
                F_reg=F_reg, F_post=F_post, p_post=p_post,
                witness_post=wit_post, witness_reg=wit_reg,
                noise_margin_post=q_post, noise_margin_reg=q_reg,
                gme_fidelity_reg=F_reg > 2 / 3, gme_ppt_mixture_reg=q_reg > 0)
