"""Re-analysis of the photonic (3,3,2) experiment of Malik et al.,
Nat. Photon. 10, 248 (2016), with the exact witness W3.

The authors' OAM labels (A, B in {-1,0,1}, C in {0,1}) map to this package's
labels |a b c> by  a = C,  b = {0:0, 1:1, -1:2}[A],  c = {0:0, -1:1, 1:2}[B].
"""
import csv
import io
from importlib import resources
from typing import Dict, Tuple

import numpy as np

DATA_FILE = "malik2016_extended_data_table1.csv"


def load_malik2016() -> Tuple[Dict, Dict]:
    """Return (populations, coherences) keyed by the authors' (A, B, C) labels."""
    text = resources.files("gme332.data").joinpath(DATA_FILE).read_text(encoding="utf-8")
    rows = csv.DictReader(io.StringIO("".join(l for l in text.splitlines(True)
                                              if not l.startswith("#"))))
    pops, cohs = {}, {}
    for r in rows:
        row = (int(r["row_A"]), int(r["row_B"]), int(r["row_C"]))
        col = (int(r["col_A"]), int(r["col_B"]), int(r["col_C"]))
        val = complex(float(r["re"]), float(r["im"]))
        if r["kind"] == "pop":
            pops[row] = val.real
        else:
            cohs[(row, col)] = val
    return pops, cohs


def malik2016_certificate() -> Dict:
    """Upper bound on Tr(W3 rho_exp) from the published elements; the unmeasured
    coherence rho_{011,022} is bounded by positivity, |rho_ij| <= sqrt(rho_ii rho_jj).
    Local phases are aligned by a diagonal unitary on party B (keeps W3 fully decomposable)."""
    populations, coherences = load_malik2016()
    fA, fB = {0: 0, 1: 1, -1: 2}, {0: 0, -1: 1, 1: 2}
    lab = lambda t: f"{t[2]}{fA[t[0]]}{fB[t[1]]}"
    pop = {lab(k): v for k, v in populations.items()}
    coh = {(lab(a), lab(b)): v for (a, b), v in coherences.items()}
    c1, c2, c3 = abs(coh[("000", "111")]), abs(coh[("000", "122")]), abs(coh[("111", "122")])
    F = (pop["000"] + pop["111"] + pop["122"] + 2 * (c1 + c2 + c3)) / 3
    S8 = ["001", "002", "010", "020", "101", "102", "110", "120"]
    g_term = (np.sqrt(pop["011"]) + np.sqrt(pop["022"])) ** 2 / np.sqrt(2)
    W_t = -2 * (c1 + c2) + sum(pop[s] for s in S8) + g_term + np.sqrt(2) * pop["100"]
    trW = 8 + 2 * np.sqrt(2)
    return dict(F=F, witness_unnorm=W_t, witness=W_t / trW,
                q_witness=-W_t / (-W_t + trW / 18),
                q_fidelity=(F - 2 / 3) / (F - 1 / 18),
                coherence_sum_needed=(W_t + 2 * (c1 + c2)) / 2, coherence_sum=c1 + c2)
