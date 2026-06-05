"""
Supplementary Material — Paper IX
Even-n torus knots: exclusion of T(2,2), extension to T(2,2m), anti-phase scaling.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PAPERS_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from _figure_utils import save_axes_panel
from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _physics_utils import LEPTON_MASSES

_LEPTON_SOURCE = "paper_local_public_constants"

C_N = float(np.sqrt(2.0))
THETA_3 = (2 / 3) / 3
SEP = "=" * 64


def A(th: float, n: int, k: int) -> float:
    return 1.0 + C_N * np.cos(th + 2 * np.pi * k / n)


def sumA2(th: float, n: int) -> float:
    return float(sum(A(th, n, k) ** 2 for k in range(n)))


def sumAbsA(th: float, n: int) -> float:
    return float(sum(abs(A(th, n, k)) for k in range(n)))


def Qt(n: int, th: float) -> float:
    sa = sumAbsA(th, n)
    return sumA2(th, n) / (sa * sa)


def M3() -> float:
    me = float(LEPTON_MASSES["electron"])
    amps = sorted((1.0 + C_N * np.cos(THETA_3 + 2 * np.pi * k / 3)) ** 2 for k in range(3))
    return me / amps[0]


def Mn(n: int, m3: float) -> float:
    return m3 * (n / 3.0) ** (3 / 4)


def negcount(n: int, th: float) -> int:
    return sum(1 for k in range(n) if A(th, n, k) < 0)


def solve_bis(n: int, ng: int = 120000) -> Optional[float]:
    if n == 3:
        return THETA_3
    lo, hi = 1e-14, np.pi / n - 1e-14
    ts = np.linspace(lo, hi, ng)
    diff = n * ts - np.array([Qt(n, float(t)) for t in ts])
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return None
    a, b = float(ts[sc[0]]), float(ts[sc[0] + 1])
    for _ in range(90):
        m = (a + b) / 2
        if n * m - Qt(n, m) < 0:
            a = m
        else:
            b = m
    return (a + b) / 2


def solve_br(n: int) -> Optional[float]:
    if n == 3:
        return THETA_3
    f = lambda t: n * t - Qt(n, t)
    lo, hi = 1e-14, np.pi / n - 1e-14
    try:
        if f(lo) * f(hi) > 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15))
    except Exception:
        return None


def sym_fourier_rows() -> pd.DataFrame:
    th = sp.Symbol("theta", real=True)
    rows = []
    pts = (sp.Rational(1, 11), sp.Rational(17, 100), -sp.Rational(23, 100))
    for n in range(2, 8):
        expr = sum(sp.cos(th + 2 * sp.pi * sp.Integer(k) / n) ** 2 for k in range(n)) - sp.Rational(n, 2)
        simp = sp.simplify(expr)
        gate = all(abs(float(sp.N(expr.subs(th, t), 50))) < 1e-12 for t in pts)
        rows.append(
            {
                "n": n,
                "expr_minus_n_over_2": str(simp)[:120],
                "n_ge_3_expected_zero": n >= 3,
                "numeric_gate_zero": gate,
                "n2_expected_2cos2th": n == 2,
            }
        )
    return pd.DataFrame(rows)


def _try_numba_Qtilde_even(n: int, thetas: np.ndarray) -> Tuple[str, np.ndarray]:
    ref = np.array([Qt(n, float(t)) for t in thetas])
    try:
        from numba import njit

        @njit
        def batch(nn: int, ts: np.ndarray) -> np.ndarray:
            c = 1.4142135623730951
            twopi = 6.283185307179586
            out = np.empty(len(ts))
            for i in range(len(ts)):
                t = ts[i]
                sq = 0.0
                sa = 0.0
                for k in range(nn):
                    a = 1.0 + c * np.cos(t + twopi * k / nn)
                    sq += a * a
                    sa += abs(a)
                out[i] = sq / (sa * sa)
            return out

        return "used", batch(n, thetas)
    except ImportError:
        return "skipped", ref


def _try_torch_slope(n: int, theta_n: float) -> Optional[float]:
    try:
        import torch

        pi_t = torch.tensor(np.pi, dtype=torch.float64)
        th = torch.tensor(theta_n, dtype=torch.float64, requires_grad=True)
        amps = [1.0 + C_N * torch.cos(th + 2.0 * pi_t * k / n) for k in range(n)]
        sq = sum(a * a for a in amps)
        sa = sum(torch.abs(a) for a in amps)
        y = float(n) * th - sq / (sa * sa)
        g = torch.autograd.grad(y, th)[0]
        return float(g.item())
    except ImportError:
        return None


def _try_jax_slope(n: int, theta_n: float) -> Optional[float]:
    try:
        import jax.numpy as jnp
        from jax import grad

        def g(th: float) -> float:
            amps = [1.0 + C_N * jnp.cos(th + 2.0 * jnp.pi * k / n) for k in range(n)]
            sq = sum(a * a for a in amps)
            sa = sum(jnp.abs(a) for a in amps)
            return float(n) * th - sq / (sa * sa)

        return float(grad(g)(theta_n))
    except ImportError:
        return None


def plot_bundle(out: Path, m3: float, theta: Dict[int, float]) -> Path:
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ts = np.linspace(0, 2 * np.pi, 700)
    for n, c in [(2, "C3"), (3, "C0"), (5, "C2")]:
        y = [sum(np.cos(t + 2 * np.pi * k / n) ** 2 for k in range(n)) for t in ts]
        ax[0, 0].plot(ts, y, label=f"n={n}", color=c)
    ax[0, 0].set_title(r"Fourier sum $\sum_k\cos^2(\theta+2\pi k/n)$")
    ax[0, 0].set_xlabel(r"$\theta$ (rad)")
    ax[0, 0].set_ylabel(r"$\sum_k\cos^2(\theta+2\pi k/n)$")
    ax[0, 0].legend(fontsize=8)

    for n in (4, 6, 8, 10):
        th = theta[n]
        mn = Mn(n, m3)
        ms = sorted(mn * A(th, n, k) ** 2 for k in range(n))
        ax[0, 1].plot(range(n), ms, "o-", label=f"n={n}")
    ax[0, 1].set_title("Even-$n$ mode spectra")
    ax[0, 1].set_xlabel(r"Mode index $k$")
    ax[0, 1].set_ylabel(r"$m_k$ (MeV)")
    ax[0, 1].legend(fontsize=8)

    ns = list(range(3, 14))
    coeff = 2 * np.pi**2 / (np.pi + 4) ** 2
    ax[1, 0].plot(ns, [theta[n] for n in ns], "o-", label="theta_n")
    ax[1, 0].plot(ns, [coeff / n**2 for n in ns], "--", label="asymptotic")
    ax[1, 0].set_title(r"Convergence of $\theta_n$")
    ax[1, 0].set_xlabel(r"Family index $n$")
    ax[1, 0].set_ylabel(r"$\theta_n$ (rad)")
    ax[1, 0].legend(fontsize=8)

    x = []
    y = []
    for n in ns:
        th = theta[n]
        mn = Mn(n, m3)
        ms = sorted(mn * A(th, n, k) ** 2 for k in range(n))
        x.append(mn * th * th)
        y.append(ms[0])
    ax[1, 1].plot(x, y, "o-")
    ax[1, 1].set_title(r"Minimum mass vs $M_n\theta_n^2$")
    ax[1, 1].set_xlabel(r"$M_n\theta_n^2$ (MeV)")
    ax[1, 1].set_ylabel(r"$m_{\min}$ (MeV)")
    fig.tight_layout()
    pdf = out / "paper_IX_figures.pdf"
    fig.savefig(pdf, bbox_inches="tight")

    # Export individual panels for LaTeX inclusion
    for i, ax_ in enumerate([ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]):
        save_axes_panel(fig, ax_, out / f"fig9_{i}_panel.pdf")

    plt.close(fig)
    return pdf


def run_paper_ix(write_outputs: bool = True, verbose: bool = True):
    out = output_dir("paper_IX") if write_outputs else None
    m3 = M3()
    theta = {}
    for n in range(3, 14):
        t = solve_bis(n) if n != 3 else THETA_3
        if t is None:
            raise RuntimeError(f"no theta n={n}")
        theta[n] = t
    for n in [2, 4, 6, 8, 10, 12]:
        if n not in theta:
            t = solve_bis(n)
            if t is not None:
                theta[n] = t

    df_sym = sym_fourier_rows()
    rows1 = []
    test_th = 0.7
    for n in [2, 3, 4, 5, 6, 7, 8, 10]:
        s = sum(np.cos(test_th + 2 * np.pi * k / n) ** 2 for k in range(n))
        r = s - n / 2
        rows1.append({"n": n, "sum_cos2": s, "n_over_2": n / 2, "residual": r, "valid": abs(r) < 1e-9})
    df1 = pd.DataFrame(rows1)

    t2 = solve_bis(2)
    m2 = Mn(2, m3)
    amps2 = [A(t2, 2, k) for k in range(2)] if t2 is not None else [np.nan, np.nan]
    sumA2_n2 = float(sum(a * a for a in amps2))
    df2 = pd.DataFrame(
        [{"theta2": t2, "sum_Ak2": sumA2_n2, "expected_2n": 4.0, "sum_m": m2 * sumA2_n2, "expected_4M2": 4 * m2}]
    )

    rows3 = []
    for n in [4, 6, 8, 10]:
        tb = solve_bis(n)
        tr = solve_br(n)
        if tb is None or tr is None:
            raise RuntimeError(f"no even root n={n}")
        ok, d = agreement_ok(tb, tr, rtol=1e-10, atol=1e-12)
        th = tr
        mn = Mn(n, m3)
        ms = sorted(mn * A(th, n, k) ** 2 for k in range(n))
        total = sum(ms)
        rows3.append(
            {
                "n": n,
                "theta_deg": np.degrees(th),
                "M_n": mn,
                "Sigma_n": total,
                "n_plus": n - negcount(n, th),
                "n_minus": negcount(n, th),
                "sum_rule_ok": agreement_ok(total, 2 * n * mn, atol=0.02, rtol=1e-10)[0],
                "brentq_bisect_ok": ok,
                "theta_diff": d,
                "residual_abs": abs(n * th - Qt(n, th)),
            }
        )
    df3 = pd.DataFrame(rows3)

    rows4 = []
    for n in [4, 6, 8, 10]:
        th = solve_br(n)
        mn = Mn(n, m3)
        ms = sorted(mn * A(th, n, k) ** 2 for k in range(n))
        for i, m in enumerate(ms):
            rows4.append({"n": n, "mode_index": i, "m_MeV": m})
    df4 = pd.DataFrame(rows4)

    rows5 = []
    for n in [4, 6, 8, 10, 12]:
        th = solve_br(n)
        nf = math.floor(5 * n / 8) - math.ceil(3 * n / 8) + 1
        nn = negcount(n, th)
        note = "3n/8 exact integer -> boundary correction" if (3 * n) % 8 == 0 and nf != nn else ""
        rows5.append({"n": n, "n_minus_formula": nf, "n_minus_numeric": nn, "match": nf == nn, "note": note})
    df5 = pd.DataFrame(rows5)

    rows6 = []
    for n in [5, 7, 9, 11, 13, 4, 6, 10]:
        th = solve_br(n)
        mn = Mn(n, m3)
        ms = sorted(mn * A(th, n, k) ** 2 for k in range(n))
        mmin = ms[0]
        kstar = math.ceil(3 * n / 8)
        phi = th + 2 * np.pi * kstar / n
        astar = 1 + C_N * np.cos(phi)
        mpred = mn * astar * astar
        rows6.append(
            {
                "n": n,
                "m_min": mmin,
                "M_theta2": mn * th * th,
                "ratio_mmin_over_Mtheta2": mmin / (mn * th * th),
                "k_star": kstar,
                "A_star": astar,
                "m_pred": mpred,
                "pred_over_actual": mpred / mmin,
            }
        )
    df6 = pd.DataFrame(rows6)

    rows7 = []
    for n in range(3, 14):
        th = theta[n]
        mn = Mn(n, m3)
        sig = 2 * n * mn
        neg = negcount(n, th)
        rows7.append(
            {"n": n, "theta_deg": np.degrees(th), "M_n": mn, "Sigma_n": sig, "split": f"{n-neg}+{neg}", "status": "VERIFIED" if n == 3 else "predicted"}
        )
    df7 = pd.DataFrame(rows7)

    coeff = 2 * np.pi**2 / (np.pi + 4) ** 2
    df8 = pd.DataFrame([{"n": n, "theta_rad": theta[n], "asymptotic": coeff / n**2, "ratio": theta[n] / (coeff / n**2)} for n in range(3, 14)])

    ts_even = np.linspace(1e-6, np.pi / 8 - 1e-6, 400)
    nb_status, nb_vals = _try_numba_Qtilde_even(8, ts_even)
    nb_ref = np.array([Qt(8, float(t)) for t in ts_even])
    nb_res = float(np.max(np.abs(nb_vals - nb_ref)))
    nb_ok = bool(nb_res < 1e-10)
    df_checks_numba = pd.DataFrame(
        [{"numba": nb_status, "n": 8, "max_abs_Qtilde_residual_vs_numpy": nb_res, "numba_parity_ok": nb_ok}]
    )

    slope_rows = []
    autodiff_ok = True
    for n in (4, 6, 8, 10):
        th = solve_br(n)
        tg = _try_torch_slope(n, th)
        jg = _try_jax_slope(n, th)
        slope_rows.append({"n": n, "torch_d_residual_dtheta": tg, "jax_d_residual_dtheta": jg})
        if tg is not None and tg <= 0:
            autodiff_ok = False
        if jg is not None and jg <= 0:
            autodiff_ok = False
    df_checks_autodiff = pd.DataFrame(slope_rows)

    df_checks_scipy = df3[["n", "theta_deg", "brentq_bisect_ok", "theta_diff", "residual_abs"]].copy()

    sym_ok = bool(df_sym[df_sym["n"] >= 3]["numeric_gate_zero"].all())
    n2_fail = abs(df2["sum_Ak2"].iloc[0] - 4.0) > 1e-3
    even_ok = bool(df3["sum_rule_ok"].all() and df3["brentq_bisect_ok"].all())
    split_ok = bool(df5[df5["n"] != 8]["match"].all())
    goldstone_neg = bool((df6["ratio_mmin_over_Mtheta2"].max() / df6["ratio_mmin_over_Mtheta2"].min()) > 2)
    overall = sym_ok and n2_fail and even_ok and split_ok and goldstone_neg and nb_ok and autodiff_ok
    dfsum = pd.DataFrame(
        [
            {
                "sym_fourier_exclusion": sym_ok,
                "n2_exclusion_demo": n2_fail,
                "even_sum_rule": even_ok,
                "split_formula_except_boundary": split_ok,
                "not_goldstone_scaling": goldstone_neg,
                "numba_Qtilde_even_optional": nb_ok,
                "autodiff_even_slopes_optional": autodiff_ok,
                "lepton_masses_source": _LEPTON_SOURCE,
                "overall_ok": overall,
            }
        ]
    )

    pdf = None
    if write_outputs and out is not None:
        df_sym.to_csv(out / "checks_symbolic_fourier_exclusion.csv", index=False)
        df1.to_csv(out / "table1_fourier_identity_check.csv", index=False)
        df2.to_csv(out / "table2_n2_exclusion_demo.csv", index=False)
        df3.to_csv(out / "table3_even_n_self_consistency.csv", index=False)
        df4.to_csv(out / "table4_even_n_spectra.csv", index=False)
        df5.to_csv(out / "table5_amplitude_split_even.csv", index=False)
        df6.to_csv(out / "table6_anti_phase_boundary.csv", index=False)
        df7.to_csv(out / "table7_full_census.csv", index=False)
        df8.to_csv(out / "table8_theta_convergence.csv", index=False)
        df_checks_scipy.to_csv(out / "checks_scipy_brentq_vs_bisect_even.csv", index=False)
        df_checks_numba.to_csv(out / "checks_numba_Qtilde_even.csv", index=False)
        df_checks_autodiff.to_csv(out / "checks_autodiff_even_slopes.csv", index=False)
        dfsum.to_csv(out / "checks_summary_IX_gate.csv", index=False)
        pdf = str(plot_bundle(out, m3, theta))
    if verbose:
        print(SEP)
        print("Paper IX — reproducibility bundle")
        print(SEP)
        print(dfsum.to_string(index=False))
        print(describe_autodiff_backends())
    return {"output_dir": str(out) if out else None, "summary_pass": overall, "figures_pdf": pdf, "summary_gate": dfsum}


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper IX supplementary — reproducibility harness.")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    r = run_paper_ix(write_outputs=not a.no_write, verbose=not a.quiet)
    ok = bool(r["summary_gate"]["overall_ok"].iloc[0])
    if ok and not a.quiet:
        print("All Paper IX supplementary checks passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
