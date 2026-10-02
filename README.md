# gme332

Reproducibility code for **"Exact PPT-Mixture Threshold for Certifying Genuine Multipartite Entanglement of the Asymmetric (3,3,2) GHZ State"**.

The target state is

```
|ψ⟩ = (|000⟩ + |111⟩ + |122⟩)/√3  ∈  C² ⊗ C³ ⊗ C³
```

This is the (3,3,2) state realised with photons by Malik *et al.*, Nat. Photon. 10, 248 (2016). It is mixed with white noise, ρ(p) = (1−p)|ψ⟩⟨ψ| + p·𝟙/18.

## Main results reproduced

| Result | Value | Phase |
|---|---|---|
| Fidelity-witness threshold | p\*_F = 6/17 ≈ 35.29 % | 0 |
| PPT-mixture threshold, proven exactly by an explicit witness and an exact PPT-mixture decomposition | p\* = 12/(16+√2) ≈ 68.91 % | 2 |
| 2×d×d family bound (tight for d ≤ 5, not at d = 6) | p\*(d) = 2d(d−1)/(2(d²−1)+√(d−1)) | 2 |
| GME lifetime vs first bipartite sudden death | τ_GME = 6.172 vs τ_ESD = 8.969 | 3 |
| Noisy-device simulation (FakeSherbrooke, 6 ECR) | F = 0.9674, Tr(W₃ρ) = −0.117 | 4 |
| Malik *et al.* data: witness value and extra-noise tolerance | Tr(W₃ρ) ≤ −0.0755; 0.576 vs 0.179 (fidelity test) | 5 |

## Install

```bash
python -m pip install -e ".[dev]"           # loose versions (pyproject.toml)
# or reproduce the exact environment used for the paper:
python -m pip install -r requirements.txt && python -m pip install -e . --no-deps
```

Python ≥ 3.10. `qiskit-ibm-runtime` is needed only for phase 4.

## Use

```bash
gme332 verify                     # exact certificates only (pure rational arithmetic), < 1 s
gme332 run                        # full pipeline, all phases, about 14 min (see below)
gme332 run --phases 2 5           # selected phases
gme332 run --quick                # smoke test: same code paths, fewer points
gme332 run --fig-dir ../TQE_Template   # write the paper figures straight into the manuscript folder
python -m pytest                  # fast tests (about 10 s)
python -m pytest -m slow          # SDP and Qiskit cross-checks (about 15 s)
```

`gme332 run` writes `results/results.txt` (log), `results/timings.json` and `results/figures/` (vector PDF and 300-dpi PNG). The reference output used for the paper is in `results/reference/`.

## Runtime

Measured on AMD Ryzen 5 5500U (6 cores / 12 threads), 5.9 GB RAM, Windows 11, Python 3.12, with the versions in `requirements.txt`.

| Phase | Content | Full run | `--quick` |
|---|---|---|---|
| 0 | fidelity witness | < 0.1 s | < 0.1 s |
| 1 | partial-transpose spectra + Fig. 1 | 2 s | 2 s |
| 2 | exact certificates, SDPs, 2×d×d family (d ≤ 6) + Fig. 2 | 392 s | 11 s |
| 3 | Lindblad dynamics, ESD, N_g(t), τ_GME + Fig. 3 | 96 s | 13 s |
| 3b | γ_corr sweep (6 values) + Fig. 4 | 54 s | 10 s (2 values) |
| 4 | device simulation (transpile, density matrix, 2×4×4 SDP) + Fig. 5 | 297 s | 10 s (noise-margin SDPs skipped) |
| 5 | Malik *et al.* re-analysis | < 0.1 s | < 0.1 s |
| **total** | | **842 s ≈ 14 min** | **≈ 46 s** |

The d = 6 family SDP dominates phase 2, and the 32-dimensional PPT-mixture SDP dominates phase 4. The full pipeline runs within 5.9 GB of RAM. Don't start other large SDPs alongside it. The unreduced d ≥ 5 family SDP runs out of memory on this machine, which is why the code uses the symmetry-reduced formulation.

## Repository layout

```
gme332/
├── src/gme332/
│   ├── basis.py          flat indexing (pure Python)
│   ├── exact.py          exact arithmetic in Q(√m), exact PSD test (pure Python)
│   ├── certificates.py   witness family W_d, exact PPT-mixture decomposition at d = 3
│   ├── states.py         states, partial transpose, negativity, one-sided dN/dt
│   ├── sdp.py            PPT-mixture SDPs, genuine multipartite negativity (CVXPY/CLARABEL)
│   ├── dynamics.py       Liouvillian, exact propagator, ESD times, GME lifetime
│   ├── device.py         local qubit encoding, 5-CNOT circuit, FakeSherbrooke simulation
│   ├── experiment.py     re-analysis of Malik et al. (2016)
│   ├── data/             published density-matrix elements (CSV, with source)
│   ├── figures.py        publication figures
│   ├── pipeline.py       timed phases 0–5 (maps to paper sections)
│   └── cli.py            `gme332` command
├── tests/                pytest suite (fast + `-m slow`)
├── notebooks/GME_Q1.ipynb   self-contained Colab notebook (same code in one cell)
├── results/reference/    reference log and figures used in the paper
├── pyproject.toml, requirements.txt
```

The notebook is kept for Colab users. It needs no installation of this package, and it runs the same pipeline as `gme332 run`.

## Citation

MD. Nurol Amin, *Exact PPT-Mixture Threshold for Certifying Genuine Multipartite Entanglement of the Asymmetric (3,3,2) GHZ State* (manuscript submitted, 2026). Citation metadata is in `CITATION.cff`.

Author: MD. Nurol Amin, Daffodil International University, Dhaka, Bangladesh (amin15-5577@diu.edu.bd).
