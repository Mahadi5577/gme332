"""gme332 — exact PPT-mixture certification of genuine multipartite entanglement
for the asymmetric (3,3,2) GHZ state (|000> + |111> + |122>)/sqrt3.

Submodules (import what you need; heavy dependencies load lazily):
    basis, exact, certificates   exact arithmetic and certificates (pure Python)
    states, sdp, dynamics        numerical analysis (NumPy, CVXPY, SciPy)
    device                       noisy-device simulation (Qiskit)
    experiment                   re-analysis of Malik et al. (2016) data
    figures, pipeline, cli       figures, timed pipeline, command line
"""

__version__ = "1.1.0"
