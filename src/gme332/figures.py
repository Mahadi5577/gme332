"""Publication figures (vector PDF + 300-dpi PNG; legends outside axes;
fixed per-cut colours from the Okabe-Ito palette)."""
import itertools
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .config import CUTS

matplotlib.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 9,
    "axes.labelsize" : 9,
    "axes.titlesize" : 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 600,
    "axes.linewidth" : 0.6,
    "lines.linewidth": 1.2,
    "pdf.fonttype"   : 42,
    "ps.fonttype"    : 42,
})

CB = dict(
    blue="#0072B2", orange="#E69F00", green="#009E73",
    red="#D55E00",  purple="#CC79A7", teal="#56B4E9",
    yellow="#F0E442", black="#000000",
)
CUT_COLOR = {"A|BC": CB["blue"], "AB|C": CB["orange"], "AC|B": CB["green"]}


def _save(fig, fig_dir, name):
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(fig_dir, name + ".png"), bbox_inches="tight", dpi=300)
    if not matplotlib.is_interactive():      # script run: free the canvas
        plt.close(fig)


def _clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def fig_ppt_spectra(spectra, fig_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 3.3), layout="constrained")
    for ax, (lbl, eigs) in zip(axes, spectra.items()):
        se = np.sort(eigs); ix = np.arange(len(se))
        ax.bar(ix, se, color=CUT_COLOR[lbl], alpha=0.55, width=0.8)
        neg = se < -1e-12
        ax.bar(ix[neg], se[neg], color=CB["red"], alpha=0.85, width=0.8)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(fr"${lbl}$   ($\min\lambda={se[0]:.4f}$)")
        ax.set_xlabel("Index"); ax.set_ylabel(r"$\lambda$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _clean(ax)
    handles = [mpatches.Patch(color=CB["red"], alpha=0.85, label=r"negative eigenvalues (NPT)"),
               mpatches.Patch(color="0.55", alpha=0.55, label="non-negative eigenvalues")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    _save(fig, fig_dir, "phase1"); plt.show(); return fig


def fig_thresholds(wit, family, fig_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    p = np.linspace(0, 1, 401)
    c = float(wit[3]["c_normalised"]); D = 18
    cF = (2 / 3 - 1) / (12 - 1)                               # W_F / Tr W_F, Tr W_F = 11
    ax1.plot(p * 100, (1 - p) * c + p / D, color=CB["blue"],
             label=r"decomposable witness $W_3$ (exact)")
    ax1.plot(p * 100, (1 - p) * cF + p / D, color=CB["orange"], ls="--",
             label=r"fidelity witness $W_F$ ($\alpha=2/3$)")
    ax1.axhline(0, color="black", lw=0.6)
    ps, pF = float(wit[3]["p_star"]), 6 / 17
    ax1.axvline(ps * 100, color=CB["blue"], lw=0.8, ls=":",
                label=fr"$p^*=12/(16+\sqrt{{2}})\approx{ps*100:.2f}\%$")
    ax1.axvline(pF * 100, color=CB["orange"], lw=0.8, ls=":",
                label=fr"$p^*_F=6/17\approx{pF*100:.2f}\%$")
    ax1.set_xlabel(r"white-noise weight $p$ (%)")
    ax1.set_ylabel(r"$\mathrm{Tr}(W\rho(p))/\mathrm{Tr}\,W$")
    ax1.set_title(r"(a) $2\times3\times3$ state: witness expectation"); _clean(ax1)

    ds = np.array(sorted(family))
    dd = np.linspace(2, ds.max(), 200)
    ax2.plot(dd, 2 * dd * (dd - 1) / (2 * (dd ** 2 - 1) + np.sqrt(dd - 1)), color=CB["blue"],
             label=r"$p^*(d)$ from $W_d$ (proven lower bound)")
    ax2.plot(ds, [family[d] for d in ds], "o", color=CB["blue"], mfc="white", ms=6,
             label="PPT-mixture SDP optimum")
    ax2.plot(dd, 2 * dd / (2 * dd ** 2 - 1), color=CB["orange"], ls="--",
             label=r"$p^*_F(d)=2d/(2d^2-1)$")
    ax2.set_xlabel(r"local dimension $d$"); ax2.set_ylabel(r"threshold $p^*$")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title(r"(b) $2\times d\times d$ family"); _clean(ax2)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=3, frameon=False)
    _save(fig, fig_dir, "phase2"); plt.show(); return fig


def fig_dynamics(times, per_cut, esd, t_ng, ng, tau_gme, params, fig_dir=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True,
                                   layout="constrained")
    for lbl, arr in per_cut.items():
        ax1.plot(times, arr, color=CUT_COLOR[lbl], ls="--" if lbl == "AC|B" else "-",
                 label=fr"$\mathcal{{N}}_{{{lbl}}}$  ($\tau_{{\rm ESD}}={esd[lbl]:.3f}$)")
        ax1.axvline(esd[lbl], color=CUT_COLOR[lbl], lw=0.7, ls=":")
    ax1.set_ylabel(r"negativity $\mathcal{N}_k(t)$")
    g1, gp, gc = params
    ax1.set_title(fr"Lindblad dynamics ($\gamma_1={g1}$, $\gamma_\phi={gp}$, "
                  fr"$\gamma_{{\rm corr}}={gc}$)"); _clean(ax1)
    ax2.plot(t_ng, ng, color="black", marker="o", ms=2.5,
             label=r"genuine multipartite negativity $\mathcal{N}_g(t)$")
    ax2.axvline(tau_gme, color=CB["red"], lw=0.9, ls="--",
                label=fr"$\tau_{{\rm GME}}={tau_gme:.3f}$ (PPT-mixture boundary)")
    ax2.set_xlabel(r"time $t$ (arb. units)"); ax2.set_ylabel(r"$\mathcal{N}_g(t)$")
    ax2.set_ylim(bottom=-0.005); _clean(ax2)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="outside lower center", ncol=2, frameon=False)
    _save(fig, fig_dir, "phase3"); plt.show(); return fig


def fig_gamma_sweep(sweep, fig_dir=None):
    gcs = sorted(sweep)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
    for lbl in CUTS:
        ax.plot(gcs, [sweep[g]["esd"][lbl] for g in gcs], marker="o", ms=4,
                color=CUT_COLOR[lbl], ls="-" if lbl != "AC|B" else "--",
                label=fr"$\tau_{{\rm ESD}}({lbl})$")
    ax.plot(gcs, [sweep[g]["tau_gme"] for g in gcs], marker="s", ms=4, color="black",
            label=r"$\tau_{\rm GME}$ (PPT-mixture criterion)")
    ax.set_xlabel(r"correlated dephasing rate $\gamma_{\rm corr}$")
    ax.set_ylabel(r"lifetime"); _clean(ax)
    fig.legend(loc="outside lower center", ncol=2, frameon=False)
    _save(fig, fig_dir, "phase3b"); plt.show(); return fig


def fig_device(hw, fig_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), width_ratios=[2.2, 1],
                                   layout="constrained")
    shots = hw["shots"]; probs = {}
    leak = 0
    for key, cnt in hw["counts"].items():
        k = int(key.replace(" ", ""), 2)
        q = [(k >> i) & 1 for i in range(5)]
        b, c = q[1] + 2 * q[2], q[3] + 2 * q[4]
        if b == 3 or c == 3:
            leak += cnt; continue
        probs[(q[0], b, c)] = probs.get((q[0], b, c), 0) + cnt / shots
    labels = ["".join(map(str, s)) for s in itertools.product(range(2), range(3), range(3))]
    vals = [probs.get(tuple(int(x) for x in s), 0) for s in labels] + [leak / shots]
    cols = [CB["blue"] if s in ("000", "111", "122") else "0.6" for s in labels] + [CB["red"]]
    ax1.bar(range(len(vals)), vals, color=cols, width=0.75)
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels(labels + ["leak"], rotation=60, fontsize=6.5)
    ax1.set_ylabel("probability"); ax1.set_xlabel(r"measured $|abc\rangle$")
    ax1.set_title(f"{hw['shots']} shots, {hw['backend']} noise model"); _clean(ax1)
    ax2.axis("off")
    rows = [("register fidelity $F$", f"{hw['F_reg']:.4f}"),
            ("post-selected fidelity", f"{hw['F_post']:.4f}"),
            ("post-selection prob.", f"{hw['p_post']:.4f}"),
            (r"$\mathrm{Tr}(W_3\rho)$ (register)", f"{hw['witness_reg']:+.4f}"),
            ("extra white noise (register)", f"{hw['noise_margin_reg']:.3f}"),
            ("ECR gates / depth", f"{hw['n_ecr']} / {hw['depth']}")]
    for i, (k, v) in enumerate(rows):
        ax2.text(0.0, 0.95 - 0.14 * i, k, transform=ax2.transAxes, fontsize=8, color="0.3")
        ax2.text(1.0, 0.95 - 0.14 * i, v, transform=ax2.transAxes, fontsize=8,
                 ha="right", fontweight="bold")
    handles = [mpatches.Patch(color=CB["blue"], label="target components"),
               mpatches.Patch(color="0.6", label="other code states"),
               mpatches.Patch(color=CB["red"], label="leakage (qutrit level 3)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    _save(fig, fig_dir, "phase4"); plt.show(); return fig
