# gme332

Reproducibility code for **"Exact PPT-mixture threshold for the asymmetric (3,3,2) GHZ state and analytic witnesses for the 2×d×d family"** (MD. Nurol Amin, manuscript submitted, 2026).

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
| 2×d×d family bound: exact at d = 2, 3, tight for d = 4, 5, suboptimal for d ≥ 6 (primal SDP d ≤ 6, block-reduced dual SDP d ≤ 8) | p\*(d) = 2d(d−1)/(2(d²−1)+√(d−1)) | 2 |
| GME lifetime vs first bipartite sudden death (bracketed, with kernel term of dN/dt) | τ_GME = 6.172 vs τ_ESD = 8.969 | 3 |
| Noisy-device simulation (FakeSherbrooke, 6 ECR) | F = 0.9674, Tr(W₃ρ) = −0.117 | 4 |
| Malik *et al.* data: fidelity (achievable phase alignment), witness value, rounding robustness, extra-noise tolerance | F = 0.7991; Tr(W₃ρ) ≤ −0.0755 (≤ −0.060 if every element is off by 0.01); 0.576 vs 0.178 (fidelity test) | 5 |

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
python -m pytest                  # fast tests (about 5 s)
python -m pytest -m slow          # SDP and Qiskit cross-checks (about 15 s)
python tools/build_notebook.py    # regenerate the Colab notebook from src/ (--check to verify)
```

`gme332 run` writes `results/results.txt` (log), `results/timings.json` and `results/figures/` (vector PDF and 300-dpi PNG). The reference output used for the paper is `results/reference/results_v1.1.0.txt` (`results_v8.txt` is the earlier v1.0 log).

Every SDP checks the solver status: `optimal` is accepted, `optimal_inaccurate` raises a visible `RuntimeWarning`, anything else stops the run with `SolverError`.

## Runtime

Measured on AMD Ryzen 5 5500U (6 cores / 12 threads), 5.9 GB RAM, Windows 11, Python 3.12, with the versions in `requirements.txt`.

| Phase | Content | Full run | `--quick` |
|---|---|---|---|
| 0 | fidelity witness | < 0.1 s | < 0.1 s |
| 1 | partial-transpose spectra + Fig. 1 | 2 s | 2 s |
| 2 | exact certificates, primal SDPs d ≤ 6, dual SDPs d ≤ 8 + Fig. 2 | 302 s | 15 s |
| 3 | Lindblad dynamics, ESD, N_g(t), τ_GME and its bracket + Fig. 3 | 77 s | 20 s |
| 3b | γ_corr sweep (6 values) + Fig. 4 | 44 s | 15 s (2 values) |
| 4 | device simulation (transpile, density matrix, 2×4×4 SDP) + Fig. 5 | 240 s | 12 s (noise-margin SDPs skipped) |
| 5 | Malik *et al.* re-analysis | < 0.1 s | < 0.1 s |
| **total** | | **665 s ≈ 11 min** | **≈ 63 s** |

The d = 6 family SDP dominates phase 2, and the 32-dimensional PPT-mixture SDP dominates phase 4. The full pipeline runs within 5.9 GB of RAM. Don't start other large SDPs alongside it. The 32-dimensional register noise-margin SDP in phase 4 (value 0.789) ends with status `optimal_inaccurate` and is reported with a warning; the paper quotes only the post-selected code-space margin (0.680), whose SDP solves to `optimal`. The unreduced d ≥ 5 family SDP runs out of memory on this machine, which is why the code uses the symmetry-reduced formulation.

## Repository layout

```
gme332/
├── src/gme332/
│   ├── basis.py          flat indexing (pure Python)
│   ├── exact.py          exact arithmetic in Q(√m), exact PSD test (pure Python)
│   ├── certificates.py   witness family W_d, exact PPT-mixture decomposition at d = 3
│   ├── states.py         states, partial transpose, negativity, one-sided dN/dt
│   ├── sdp.py            PPT-mixture SDPs (primal and block-reduced dual), genuine multipartite negativity (CVXPY/CLARABEL)
│   ├── dynamics.py       Liouvillian, exact propagator, ESD times, GME lifetime
│   ├── device.py         local qubit encoding, 5-CNOT circuit, FakeSherbrooke simulation
│   ├── experiment.py     re-analysis of Malik et al. (2016)
│   ├── data/             published density-matrix elements (CSV, with source)
│   ├── figures.py        publication figures
│   ├── pipeline.py       timed phases 0–5 (maps to paper sections)
│   └── cli.py            `gme332` command
├── tests/                pytest suite (fast + `-m slow`)
├── tools/build_notebook.py   generates the notebook from src/ (tested: no drift)
├── notebooks/GME_Q1.ipynb   self-contained Colab notebook (generated; same code in one cell)
├── results/reference/    reference log and figures used in the paper
├── LICENSE (MIT), CITATION.cff, .zenodo.json
├── pyproject.toml, requirements.txt
```

The notebook is kept for Colab users. It is generated from the package sources, needs no installation of this package, and runs the same pipeline as `gme332 run`.

## Citation

MD. Nurol Amin, *Exact PPT-mixture threshold for the asymmetric (3,3,2) GHZ state and analytic witnesses for the 2×d×d family* (manuscript submitted, 2026). Citation metadata is in `CITATION.cff`; an archival DOI will be added with the Zenodo release.

## License

MIT, see `LICENSE`.

Author: MD. Nurol Amin, Daffodil International University, Dhaka, Bangladesh (amin15-5577@diu.edu.bd).
