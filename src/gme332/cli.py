"""Command-line interface.

    gme332 verify                       exact certificates only (seconds)
    gme332 run                          full pipeline, all phases
    gme332 run --phases 2 5             selected phases
    gme332 run --quick                  smoke test with fewer points
    gme332 run --fig-dir ../TQE_Template --log results/results.txt
"""
import argparse
import json
import os
import sys
import warnings


class _Tee:
    """Duplicate stdout into a log file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _versions() -> str:
    import importlib.metadata as md
    pkgs = ["numpy", "scipy", "cvxpy", "clarabel", "qutip", "qiskit", "qiskit-aer",
            "qiskit-ibm-runtime", "matplotlib"]
    out = []
    for p in pkgs:
        try:
            out.append(f"{p} {md.version(p)}")
        except md.PackageNotFoundError:
            out.append(f"{p} (not installed)")
    return " | ".join(out)


def cmd_verify(_args) -> int:
    from .certificates import (biseparability_certificate_d3, exact_primal_certificate_d3,
                               verify_witness_family)
    ok = True
    for d in range(2, 7):
        v = verify_witness_family(d)
        good = v["decomposition_exact"] and v["P_Q_psd_exact"]
        ok &= good
        print(f"d={d}: witness W_d fully decomposable (exact) = {good}; "
              f"p*(d) = {float(v['p_star']):.10f}")
    p = exact_primal_certificate_d3()
    ok &= p["all_ok"] and p["complementary_slackness"]
    print(f"d=3: exact PPT-mixture decomposition of rho(p*) valid = {p['all_ok']}, "
          f"complementary slackness = {p['complementary_slackness']}, p* = {p['p_star']}")
    b = biseparability_certificate_d3()
    ok &= b["biseparable"]
    print(f"d=3: rho(p*) biseparable (exact; true GME threshold = p*) = {b['biseparable']}")
    print("ALL EXACT CERTIFICATES VERIFIED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


def cmd_run(args) -> int:
    import matplotlib
    matplotlib.use("Agg")                     # headless; figures are written to disk
    from .pipeline import ALL_PHASES, Settings, run_pipeline

    if args.log:
        sys.stdout = _Tee(args.log)
    print(_versions())
    settings = (Settings.quick if args.quick else Settings)(fig_dir=args.fig_dir)
    phases = args.phases or list(ALL_PHASES)
    unknown = [p for p in phases if p not in ALL_PHASES]
    if unknown:
        print(f"unknown phase(s) {unknown}; choose from {ALL_PHASES}")
        return 2
    ctx = run_pipeline(phases, settings)
    if args.timings:
        os.makedirs(os.path.dirname(os.path.abspath(args.timings)), exist_ok=True)
        with open(args.timings, "w", encoding="utf-8") as fh:
            json.dump({"quick": args.quick, "seconds": ctx.timings}, fh, indent=2)
    return 0


def main(argv=None) -> int:
    # Hide only library deprecation noise; solver-accuracy warnings stay visible.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(prog="gme332", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check the exact certificates (fast)")
    run = sub.add_parser("run", help="run the reproducibility pipeline")
    run.add_argument("--phases", nargs="+", metavar="P",
                     help="subset of phases: 0 1 2 3 3b 4 5 (default: all)")
    run.add_argument("--quick", action="store_true", help="smoke-test settings")
    run.add_argument("--fig-dir", default="results/figures", help="figure output directory")
    run.add_argument("--log", default="results/results.txt", help="text log ('' to disable)")
    run.add_argument("--timings", default="results/timings.json", help="per-phase timings (JSON)")
    args = parser.parse_args(argv)
    return {"verify": cmd_verify, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
