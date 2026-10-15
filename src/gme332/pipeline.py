"""The reproducibility pipeline, split into timed phases.

Phase  Content                                                       Paper
0      fidelity witness, alpha = 2/3, p*_F = 6/17                    Sec. III
1      partial-transpose spectra                                     Fig. 1
2      exact witness family, exact primal certificate, SDPs          Sec. IV-V, Fig. 2, Table 1
3      Lindblad dynamics, ESD times, N_g(t), GME lifetime            Sec. VII, Fig. 3
3b     gamma_corr sweep                                              Sec. VII-D, Fig. 4
4      noisy-device simulation (FakeSherbrooke)                      Sec. VIII, Fig. 5, Table 3
5      re-analysis of the Malik et al. (2016) photonic data          Sec. VI, Table 2
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np

from . import figures
from .certificates import (biseparability_certificate_d3, exact_primal_certificate_d3,
                           verify_witness_family)
from .config import CUTS, DIMS
from .dynamics import Propagator, collapse_operators, esd_time, gme_lifetime, liouvillian
from .experiment import malik2016_certificate
from .sdp import (decomposable_witness_value, family_threshold_sdp, family_witness_sdp,
                  genuine_multipartite_negativity, ppt_mixture_noise_threshold,
                  weak_constraint_pitfall)
from .states import (ghz_family_ket, negativity, negativity_rate_terms,
                     negativity_right_derivative, partial_transpose)

ALL_PHASES = ("0", "1", "2", "3", "3b", "4", "5")


@dataclass
class Settings:
    gamma_1: float = 0.05
    gamma_phi: float = 0.10
    gamma_corr: float = 0.02
    t_max: float = 30.0
    gamma_corr_sweep: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)
    family_dims: Tuple[int, ...] = (2, 3, 4, 5, 6)
    dual_dims: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    ng_points: int = 41
    shots: int = 8192
    noise_margins: bool = True
    fig_dir: str = "results/figures"

    @classmethod
    def quick(cls, **kw):
        """Smoke-test settings: same code paths, fewer points (minutes, not tens of minutes)."""
        return cls(gamma_corr_sweep=(0.0, 0.08), family_dims=(2, 3, 4), dual_dims=(2, 3, 4), ng_points=6,
                   shots=2048, noise_margins=False, **kw)


@dataclass
class Context:
    settings: Settings
    out: Dict = field(default_factory=dict)
    timings: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.ket = ghz_family_ket(3)
        self.rho0 = np.outer(self.ket, self.ket)


def sep(title: str, width: int = 72):
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")


def phase0(ctx: Context):
    sep("Phase 0 - Fidelity witness")
    ket, D = ctx.ket, 18
    alphas = {}
    for lbl, k in CUTS.items():
        rest = [p for p in range(3) if p not in k]
        M = ket.reshape(DIMS).transpose(list(k) + rest).reshape(int(np.prod([DIMS[p] for p in k])), -1)
        alphas[lbl] = float(np.linalg.svd(M, compute_uv=False)[0] ** 2)
    alpha = max(alphas.values()); pF = D * (1 - alpha) / (D - 1)
    print("  largest squared Schmidt coefficient per cut:",
          {k: round(v, 10) for k, v in alphas.items()})
    print(f"  alpha = {alpha:.10f} (2/3)   p*_F = D(1-alpha)/(D-1) = {pF:.10f} (6/17)")
    ctx.out.update(alpha=alpha, pF=pF)


def phase1(ctx: Context):
    sep("Phase 1 - Partial-transpose spectra")
    spectra = {lbl: np.linalg.eigvalsh(partial_transpose(ctx.rho0, DIMS, k)) for lbl, k in CUTS.items()}
    for lbl, ev in spectra.items():
        print(f"  {lbl}: min eigenvalue = {ev[0]:+.8f}   negativity = {-ev[ev < 0].sum():.8f}")
    figures.fig_ppt_spectra(spectra, ctx.settings.fig_dir)


def phase2(ctx: Context):
    sep("Phase 2 - PPT-mixture criterion: exact witness, exact primal, SDP")
    s, rho0 = ctx.settings, ctx.rho0
    wit = {}
    for d in sorted(set((2, 3, 4, 5, 6)) | set(s.family_dims)):
        wit[d] = v = verify_witness_family(d)
        print(f"  d={d}: W_d = P_k + T_k(Q_k) exact: {v['decomposition_exact']}, "
              f"P_k,Q_k PSD exact: {v['P_Q_psd_exact']}, Tr W~ = {v['trace_W']}, "
              f"Tr(W~ rho_psi) = {v['tr_W_rho_psi']}, p* = {float(v['p_star']):.10f}")
    print(f"  d=3: c = Tr(W rho_psi)/Tr W = {wit[3]['c_normalised']}  (= -(4-sqrt2)/21)")
    print(f"       p* = {wit[3]['p_star']}  (= 12/(16+sqrt2) = 6(16-sqrt2)/127)")
    primal = exact_primal_certificate_d3()
    print(f"  exact primal PPT-mixture decomposition at p*: sum exact = {primal['sum_equals_rho']}, "
          f"all six PSD conditions exact = {primal['all_ok']}, "
          f"complementary slackness = {primal['complementary_slackness']}")
    bis = biseparability_certificate_d3()
    print(f"  exact biseparable decomposition of rho(p*): {bis['biseparable']} "
          f"({sum(bis['checks'].values())}/{len(bis['checks'])} conditions) -> true GME threshold = p*")
    q_num = ppt_mixture_noise_threshold(rho0, DIMS)
    c_num, _ = decomposable_witness_value(rho0, DIMS)
    print(f"  numerical primal threshold = {q_num:.10f}; numerical dual c_min = {c_num:.10f}")
    pit = weak_constraint_pitfall(rho0, DIMS)
    print(f"  pitfall check: program with only W + T_k(W) >= 0 gives {pit['value']:.6f} "
          f"(naive threshold {pit['naive_threshold']:.4f}) but its optimiser has "
          f"expectation {pit['value_on_product_state']:+.6f} on a product state -> not a witness")
    family = {}
    for d in s.family_dims:
        family[d] = family_threshold_sdp(d)
        print(f"  family d={d}: SDP threshold {family[d]:.10f}   W_d formula "
              f"{float(wit[d]['p_star']):.10f}   diff {family[d] - float(wit[d]['p_star']):+.1e}")
    D = lambda d: 2 * d * d
    for d in s.family_dims:                         # optimal c_min implied by the primal threshold
        c_from_primal = -family[d] / (D(d) * (1 - family[d]))
        print(f"  family d={d}: c_min implied by primal = {c_from_primal:.6f}   "
              f"W_d value Tr(W_d rho)/Tr W_d = {float(wit[d]['c_normalised']) if d in wit else float('nan'):.6f}")
    dual = {}
    for d in s.dual_dims:
        dual[d] = r = family_witness_sdp(d)
        pw = 2 * d * (d - 1) / (2 * (d * d - 1) + (d - 1) ** 0.5)
        cw = -2 * (d - 1) / d / (4 * (d - 1) + 2 * (d - 1) ** 0.5)
        print(f"  family dual d={d}: c_min = {r['c_min']:.6f} (W_d: {cw:.6f})   threshold {r['threshold']:.6f} "
              f"(W_d formula {pw:.6f}, diff {r['threshold'] - pw:+.1e})")
    figures.fig_thresholds(wit, family, s.fig_dir)
    ctx.out.update(biseparability=bis, witness=wit, primal=primal, q_num=q_num, c_num=c_num, pitfall=pit, family=family,
                   family_dual=dual)


def phase3(ctx: Context):
    import qutip as qt
    sep("Phase 3 - Lindblad dynamics")
    s, rho0, D = ctx.settings, ctx.rho0, 18
    L = liouvillian(DIMS, s.gamma_1, s.gamma_phi, s.gamma_corr)
    prop = Propagator(L, rho0.astype(complex))
    drho = (L @ rho0.reshape(-1)).reshape(D, D)
    rates = {}
    for lbl, k in CUTS.items():
        rates[lbl] = negativity_right_derivative(rho0, drho, DIMS, k)
        neg_term, ker_term = negativity_rate_terms(rho0, drho, DIMS, k)
        h = 1e-7
        fd = (negativity(prop(h), DIMS, k) - negativity(rho0, DIMS, k)) / h
        print(f"  dN/dt|0+ {lbl}: {rates[lbl]:+.6f}   (finite difference {fd:+.6f}); "
              f"negative-space term {neg_term:+.6f}, kernel term {ker_term:+.6f}, "
              f"overstatement without kernel {abs(neg_term) / abs(rates[lbl]) - 1:.1%}")
    times = np.linspace(0, 14, 561)
    per_cut = {lbl: np.array([negativity(prop(t), DIMS, k) for t in times]) for lbl, k in CUTS.items()}
    mono = {lbl: bool(np.max(np.diff(a)) <= 1e-12) for lbl, a in per_cut.items()}
    esd = {lbl: esd_time(prop, DIMS, k, s.t_max) for lbl, k in CUTS.items()}
    print("  monotone per-cut decay on 561-point grid:", mono)
    print("  exact ESD times:", {k: round(v, 4) for k, v in esd.items()})
    qt_ops = [qt.Qobj(C, dims=[DIMS, DIMS])
              for C in collapse_operators(DIMS, s.gamma_1, s.gamma_phi, s.gamma_corr)]
    t_chk = np.linspace(0, 10, 11)
    res = qt.mesolve(qt.Qobj(np.zeros((D, D)), dims=[DIMS, DIMS]),
                     qt.Qobj(rho0, dims=[DIMS, DIMS]), t_chk, qt_ops,
                     options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000})
    dev = max(np.abs(st.full() - prop(t)).max() for st, t in zip(res.states, t_chk))
    print(f"  QuTiP mesolve cross-check: max |rho_qutip - rho_exact| = {dev:.1e}")
    t_ng = np.linspace(0, 10, s.ng_points)
    ng = np.array([genuine_multipartite_negativity(prop(t), DIMS, real=True) for t in t_ng])
    tau = gme_lifetime(prop, DIMS, esd["A|BC"])
    n_min = min(negativity(rho0, DIMS, k) for k in CUTS.values())
    print(f"  N_g(0) = {ng[0]:.6f}   (bound min_k N_k = {n_min:.6f})   "
          f"GME lifetime (PPT-mixture boundary) tau_GME = {tau:.4f}")
    t_lo, t_hi = round(tau, 2) - 0.01, round(tau, 2) + 0.01
    v_lo = decomposable_witness_value(prop(t_lo), DIMS, real=True)[0]
    v_hi = decomposable_witness_value(prop(t_hi), DIMS, real=True)[0]
    print(f"  tau_GME bracket: witness value {v_lo:+.1e} at t = {t_lo:.2f}, {v_hi:+.1e} at t = {t_hi:.2f}")
    figures.fig_dynamics(times, per_cut, esd, t_ng, ng, tau,
                         (s.gamma_1, s.gamma_phi, s.gamma_corr), s.fig_dir)
    ctx.out.update(rates=rates, esd=esd, monotone=mono, qutip_dev=dev, ng=(t_ng, ng), tau_gme=tau)


def phase3b(ctx: Context):
    sep("Phase 3b - gamma_corr sweep")
    s, rho0 = ctx.settings, ctx.rho0
    sweep = {}
    for gc in s.gamma_corr_sweep:
        pr = Propagator(liouvillian(DIMS, s.gamma_1, s.gamma_phi, gc), rho0.astype(complex))
        e = {lbl: esd_time(pr, DIMS, k, s.t_max) for lbl, k in CUTS.items()}
        sweep[gc] = dict(esd=e, tau_gme=gme_lifetime(pr, DIMS, e["A|BC"]))
        print(f"  gamma_corr={gc:.3f}: ESD A|BC={e['A|BC']:.4f}  AB|C={e['AB|C']:.4f}  "
              f"AC|B={e['AC|B']:.4f}   tau_GME={sweep[gc]['tau_gme']:.4f}")
    g0, g1 = min(sweep), max(sweep)
    if g1 > g0:
        red = {lbl: 1 - sweep[g1]["esd"][lbl] / sweep[g0]["esd"][lbl] for lbl in CUTS}
        red["tau_GME"] = 1 - sweep[g1]["tau_gme"] / sweep[g0]["tau_gme"]
        print(f"  reduction from gamma_corr={g0} to {g1}: " +
              ", ".join(f"{k} {v:.1%}" for k, v in red.items()))
    figures.fig_gamma_sweep(sweep, s.fig_dir)
    ctx.out["sweep"] = sweep


def phase4(ctx: Context):
    from .device import device_simulation          # heavy import (qiskit) only when needed
    sep("Phase 4 - Noisy-device simulation (FakeSherbrooke, local encoding)")
    hw = device_simulation(ctx.settings.shots, ctx.settings.noise_margins)
    print(f"  backend {hw['backend']}, physical qubits {hw['layout']}, "
          f"ECR count {hw['n_ecr']}, depth {hw['depth']}")
    print(f"  T1 (us) {np.round(hw['T1_us'], 1)}  T2 (us) {np.round(hw['T2_us'], 1)}")
    print(f"  readout errors {np.round(hw['readout_err'], 4)}  ECR errors {np.round(hw['ecr_err'], 4)}")
    print(f"  register fidelity F = {hw['F_reg']:.4f} (> 2/3: {hw['gme_fidelity_reg']})")
    print(f"  post-selection probability = {hw['p_post']:.4f}, post-selected fidelity = {hw['F_post']:.4f}")
    print(f"  Tr(W_3 rho) register = {hw['witness_reg']:+.5f}, post-selected = {hw['witness_post']:+.5f}")
    print(f"  extra white noise tolerated before PPT mixture: register {hw['noise_margin_reg']:.4f}, "
          f"post-selected {hw['noise_margin_post']:.4f}")
    figures.fig_device(hw, ctx.settings.fig_dir)
    ctx.out["device"] = hw


def phase5(ctx: Context):
    sep("Phase 5 - Re-analysis of the photonic (3,3,2) experiment (Malik et al. 2016)")
    m = malik2016_certificate()
    ph = m["coherence_phases_deg"]
    print(f"  coherence phases (deg): {ph[0]:.1f}, {ph[1]:.1f}, {ph[2]:.1f}; moduli |rho_000,111| = "
          f"{m['coherence_moduli'][0]:.3f}, |rho_000,122| = {m['coherence_moduli'][1]:.3f}")
    print(f"  fidelity with the two GHZ coherences aligned = {m['F']:.4f}  (reported 0.801 +- 0.018); "
          f"residual phase of rho_111,122 = {m['residual_phase_deg']:.2f} deg")
    print(f"  positivity bound on the unmeasured-coherence term: {m['g_term_bound']:.4f}")
    print(f"  Tr(W~3 rho_exp) <= {m['witness_unnorm']:.4f}   normalised Tr(W3 rho_exp) <= {m['witness']:.4f}")
    print("  worst case if every element is shifted by delta: " +
          ", ".join(f"delta={dl}: {v:+.4f}" for dl, v in m["witness_worst_case"].items()))
    print(f"  extra white noise tolerated: witness {m['q_witness']:.3f}, fidelity test {m['q_fidelity']:.3f}")
    print(f"  witness negative unless |rho_000,111| + |rho_000,122| < {m['coherence_sum_needed']:.4f} "
          f"(measured {m['coherence_sum']:.4f})")
    ctx.out["malik2016"] = m


PHASES = {"0": phase0, "1": phase1, "2": phase2, "3": phase3, "3b": phase3b,
          "4": phase4, "5": phase5}


def run_pipeline(phases: Iterable[str] = ALL_PHASES, settings: Settings = None) -> Context:
    """Run the selected phases in order and record the wall-clock time of each."""
    ctx = Context(settings or Settings())
    for name in phases:
        t0 = time.perf_counter()
        PHASES[name](ctx)
        ctx.timings[name] = time.perf_counter() - t0
    sep("Timing (wall clock)")
    for name, sec in ctx.timings.items():
        print(f"  phase {name:>2}: {sec:8.1f} s")
    print(f"  total   : {sum(ctx.timings.values()):8.1f} s")
    return ctx
