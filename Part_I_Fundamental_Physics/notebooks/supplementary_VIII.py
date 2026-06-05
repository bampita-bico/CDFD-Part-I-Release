"""
Supplementary Material — Paper VIII
"The CDFT Torus Knot Spectrum: Closing the Four Open Problems of Paper VII"

Paper-local public `run_paper_viii()`: SymPy integral for <|A|> = (pi+4)/(2pi);
SciPy `brentq` self-consistency vs bisection; Pandas tables; Matplotlib figure bundle;
optional Numba `Q_tilde` batch and Torch slope gate.

Outputs: Part_I_Fundamental_Physics/outputs/paper_VIII/
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from _figure_utils import save_axes_panel
from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

from _physics_utils import LEPTON_MASSES

_LEPTON_SOURCE = "paper_local_public_constants"

SEP = "=" * 64
C_N = float(np.sqrt(2.0))
THETA_3 = (2.0 / 3.0) / 3.0


def brannen_amp(theta: float, n: int, k: int, c: float = C_N) -> float:
    return 1.0 + c * np.cos(theta + 2.0 * np.pi * k / n)


def sum_A_sq(theta: float, n: int, c: float = C_N) -> float:
    return float(sum(brannen_amp(theta, n, k, c) ** 2 for k in range(n)))


def sum_abs_A(theta: float, n: int, c: float = C_N) -> float:
    return float(sum(abs(brannen_amp(theta, n, k, c)) for k in range(n)))


def Q_tilde(n: int, theta: float, c: float = C_N) -> float:
    num = sum_A_sq(theta, n, c)
    den = sum_abs_A(theta, n, c)
    return num / (den * den)


def M_3_from_electron() -> float:
    me = float(LEPTON_MASSES["electron"])
    amps_sq = sorted(
        (1.0 + C_N * np.cos(THETA_3 + 2.0 * np.pi * k / 3.0)) ** 2 for k in range(3)
    )
    return me / amps_sq[0]


def M_n_scale(n: int, m3: float) -> float:
    return m3 * (n / 3.0) ** (3.0 / 4.0)


def count_neg(n: int, theta: float) -> int:
    return sum(1 for k in range(n) if brannen_amp(theta, n, k) < 0)


def solve_theta_bisect(n: int, ngrid: int = 120_000) -> Optional[float]:
    if n == 3:
        return THETA_3
    lo, hi = 1e-14, np.pi / n - 1e-14
    thetas = np.linspace(lo, hi, ngrid)
    lhs = n * thetas
    rhs = np.array([Q_tilde(n, float(t)) for t in thetas])
    diff = lhs - rhs
    sc = np.where(np.diff(np.sign(diff)))[0]
    if len(sc) == 0:
        return None
    t_lo, t_hi = float(thetas[sc[0]]), float(thetas[sc[0] + 1])
    for _ in range(90):
        t_mid = (t_lo + t_hi) / 2.0
        if n * t_mid - Q_tilde(n, t_mid) < 0:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return (t_lo + t_hi) / 2.0


def solve_theta_brentq(n: int) -> Optional[float]:
    if n == 3:
        return THETA_3

    def f(t: float) -> float:
        return n * t - Q_tilde(n, t)

    lo, hi = 1e-14, np.pi / n - 1e-14
    try:
        fl, fh = f(lo), f(hi)
        if fl * fh > 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15))
    except ValueError:
        return None


def n_minus_formula(n: int) -> int:
    return math.floor(5 * n / 8) - math.ceil(3 * n / 8) + 1


def _sympy_abs_A_integral() -> pd.DataFrame:
    """Piecewise ∫|1+√2 cos φ| dφ = π+4 (avoid heavy `simplify` on full sum)."""
    phi = sp.Symbol("phi", real=True)
    c = sp.sqrt(2)
    I1 = sp.integrate(1 + c * sp.cos(phi), (phi, 0, 3 * sp.pi / 4))
    I2 = sp.integrate(-(1 + c * sp.cos(phi)), (phi, 3 * sp.pi / 4, 5 * sp.pi / 4))
    I3 = sp.integrate(1 + c * sp.cos(phi), (phi, 5 * sp.pi / 4, 2 * sp.pi))
    I_tot = (I1 + I2 + I3).evalf(80)
    target = (sp.pi + 4).evalf(80)
    num_ok = abs(float(I_tot) - float(target)) < 1e-12
    literal = sp.simplify((I1 + I2 + I3) - (sp.pi + 4)) == 0
    avg_float = (np.pi + 4) / (2 * np.pi)
    asymp = 2.0 / (avg_float**2)
    return pd.DataFrame(
        [
            {
                "I_piecewise_evalf": str(I_tot)[:80],
                "equals_pi_plus_4_nsimplify": literal,
                "numeric_integral_gate": num_ok,
                "asymp_coeff_2_over_avg_sq": asymp,
            }
        ]
    )


def _try_torch_slope(n: int, theta_n: float) -> Optional[float]:
    try:
        import torch

        pi_t = torch.tensor(np.pi, dtype=torch.float64)

        def g(th: "torch.Tensor") -> "torch.Tensor":
            amps = [1.0 + C_N * torch.cos(th + 2.0 * pi_t * k / n) for k in range(n)]
            sq = sum(a * a for a in amps)
            sa = sum(torch.abs(a) for a in amps)
            q = sq / (sa * sa)
            return float(n) * th - q

        th = torch.tensor(theta_n, dtype=torch.float64, requires_grad=True)
        y = g(th)
        gr = torch.autograd.grad(y, th)[0]
        return float(gr.item())
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
            q = sq / (sa * sa)
            return float(n) * th - q

        return float(grad(g)(theta_n))
    except ImportError:
        return None


def _try_numba_Qtilde_grid(n: int, thetas: np.ndarray) -> Optional[np.ndarray]:
    try:
        from numba import njit

        @njit
        def batch(nn: int, ts: np.ndarray) -> np.ndarray:
            c = 1.4142135623730951
            out = np.empty(len(ts))
            twopi = 6.283185307179586
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

        return batch(n, thetas)
    except ImportError:
        return None


def mode_masses_sorted(n: int, theta: float, m_n: float) -> List[float]:
    return sorted(m_n * brannen_amp(theta, n, k) ** 2 for k in range(n))


def _plot_VIII_bundle(
    out: Path,
    m3: float,
    theta_by_n: Dict[int, float],
    th5: float,
    m5: float,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    ns = sorted(theta_by_n.keys())
    ths = [theta_by_n[n] for n in ns]
    coeff = 2 * np.pi**2 / (np.pi + 4) ** 2
    pred = [coeff / n**2 for n in ns]
    axes[0, 0].semilogy(ns, ths, "o-", label=r"$\theta_n$")
    axes[0, 0].semilogy(ns, pred, "--", alpha=0.7, label=r"$2\pi^2/(\pi+4)^2/n^2$")
    axes[0, 0].set_xlabel(r"$n$")
    axes[0, 0].set_ylabel(r"$\theta_n$ (rad)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Self-consistency angles")

    amps5 = [brannen_amp(th5, 5, k) for k in range(5)]
    m5_modes = [m5 * a * a for a in amps5]
    pdg = [("omega", 782.7), ("K*(892)", 891.7), ("phi", 1019.5), ("f0(980)", 980.0), ("K*(1270)", 1270.0)]
    axes[0, 1].bar(range(5), sorted(m5_modes))
    axes[0, 1].set_xticks(range(5))
    axes[0, 1].set_xlabel("sorted mode index")
    axes[0, 1].set_ylabel(r"$m$ (MeV)")
    axes[0, 1].set_title(r"$\mathbb{Z}_5$ spectrum (PDG refs in table)")
    for i, (_, pm) in enumerate(pdg[:3]):
        axes[0, 1].axhline(pm, color="C3", ls=":", lw=0.8, alpha=0.6)

    # Doublet Δm for n=7,9,11
    ax = axes[1, 0]
    for n, color in zip((7, 9, 11), ("C0", "C1", "C2")):
        th = theta_by_n[n]
        mn = M_n_scale(n, m3)
        splits = []
        ks = []
        for k in range(1, n // 2 + 1):
            mk = mn * brannen_amp(th, n, k) ** 2
            mnk = mn * brannen_amp(th, n, n - k) ** 2
            splits.append(abs(mk - mnk))
            ks.append(k)
        ax.plot(ks, splits, "o-", label=f"n={n}", color=color)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$|m_k - m_{n-k}|$ (MeV)")
    ax.legend(fontsize=8)
    ax.set_title("Doublet splittings")

    residuals = [abs(n * theta_by_n[n] - Q_tilde(n, theta_by_n[n])) for n in ns if n >= 4]
    ns_r = [n for n in ns if n >= 4]
    axes[1, 1].semilogy(ns_r, residuals, "s-")
    axes[1, 1].set_xlabel(r"$n$")
    axes[1, 1].set_ylabel(r"$|n\theta_n - \tilde Q_n|$")
    axes[1, 1].set_title("Self-consistency residuals")

    fig.tight_layout()
    pdf = out / "paper_VIII_figures.pdf"
    fig.savefig(pdf, bbox_inches="tight")

    # Export individual panels for LaTeX inclusion
    for i, ax in enumerate([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]):
        save_axes_panel(fig, ax, out / f"fig8_{i}_panel.pdf")

    plt.close(fig)
    return pdf


def run_paper_viii(write_outputs: bool = True, verbose: bool = True) -> Dict[str, object]:
    out: Optional[Path] = output_dir("paper_VIII") if write_outputs else None
    m3 = M_3_from_electron()

    df_sym = _sympy_abs_A_integral()
    sym_ok = bool(df_sym["numeric_integral_gate"].iloc[0])

    ns_main = list(range(3, 22))
    theta_bi: Dict[int, float] = {}
    theta_br: Dict[int, float] = {}
    for n in ns_main:
        tb = solve_theta_bisect(n)
        tr = solve_theta_brentq(n)
        if tb is None or tr is None:
            raise RuntimeError(f"theta solve failed for n={n}")
        theta_bi[n] = tb
        theta_br[n] = tr

    rows1 = []
    for n in ns_main:
        ok, d = agreement_ok(theta_bi[n], theta_br[n], rtol=1e-10, atol=1e-12)
        th = theta_br[n]
        qv = Q_tilde(n, th)
        res = abs(n * th - qv)
        neg = count_neg(n, th)
        rows1.append(
            {
                "n": n,
                "theta_rad": th,
                "theta_deg": np.degrees(th),
                "n_times_theta": n * th,
                "Q_tilde": qv,
                "residual_abs": res,
                "n_plus": n - neg,
                "n_minus": neg,
                "bisect_brentq_agree": ok,
                "theta_diff": d,
            }
        )
    df1 = pd.DataFrame(rows1)

    rows2 = []
    for n in [5, 7, 9, 11, 12, 13, 15, 17, 19, 21]:
        th = theta_br[n]
        nf = n_minus_formula(n)
        nn = count_neg(n, th)
        rows2.append(
            {
                "n": n,
                "n_minus_formula": nf,
                "n_minus_numeric": nn,
                "match": nf == nn,
            }
        )
    df2 = pd.DataFrame(rows2)

    coeff_asymp = 2 * np.pi**2 / (np.pi + 4) ** 2
    rows3 = []
    for n in (5, 7, 9, 11, 13):
        th = theta_br[n]
        pred = coeff_asymp / n**2
        rows3.append(
            {
                "n": n,
                "theta_exact": th,
                "asymptotic": pred,
                "ratio_theta_over_asymp": th / pred,
                "pct_error": abs(th - pred) / th * 100.0,
            }
        )
    df3 = pd.DataFrame(rows3)

    th5 = theta_br[5]
    m5 = M_n_scale(5, m3)
    amps5_raw = [brannen_amp(th5, 5, k) for k in range(5)]
    masses5_tuples = sorted([(m5 * a * a, a) for a in amps5_raw], key=lambda x: x[0])
    pdg_list = [
        ("omega(782)", 782.7),
        ("K*(892)", 891.7),
        ("phi(1020)", 1019.5),
        ("f0(980)", 980.0),
        ("K*(1270)", 1270.0),
    ]
    rows4 = []
    for i, (m, a) in enumerate(masses5_tuples):
        sector = "co-phase" if a >= 0 else "anti-phase"
        best = min((abs(m - pm), name, pm) for name, pm in pdg_list)
        rows4.append(
            {
                "sorted_index": i,
                "A_k": a,
                "m_MeV": m,
                "sector": sector,
                "best_pdg": best[1],
                "best_pdg_MeV": best[2],
                "delta_MeV": best[0],
            }
        )
    df4 = pd.DataFrame(rows4)

    rows5 = []
    for n in (7, 9, 11, 13):
        th = theta_br[n]
        mn = M_n_scale(n, m3)
        masses = mode_masses_sorted(n, th, mn)
        for i, mm in enumerate(masses):
            rows5.append({"n": n, "mode_index": i, "m_MeV": mm, "sum_modes": sum(masses), "two_n_Mn": 2 * n * mn})
    df5 = pd.DataFrame(rows5)

    rows6 = []
    for n in (7, 9, 11):
        th = theta_br[n]
        mn = M_n_scale(n, m3)
        for k in range(1, n // 2 + 1):
            mk = mn * brannen_amp(th, n, k) ** 2
            mnk = mn * brannen_amp(th, n, n - k) ** 2
            rows6.append(
                {
                    "n": n,
                    "k": k,
                    "m_k": mk,
                    "m_n_minus_k": mnk,
                    "abs_split_MeV": abs(mk - mnk),
                }
            )
    df6 = pd.DataFrame(rows6)

    ok_brent_all = bool(df1["bisect_brentq_agree"].all())
    ok_res = bool((df1["residual_abs"] < 1e-8).all())
    ok_split = bool(df2["match"].all())
    ok_sum5 = agreement_ok(sum(m for m, _ in masses5_tuples), 10 * m5)[0]

    ts = np.linspace(1e-6, np.pi / 7 - 1e-6, 500)
    ref = np.array([Q_tilde(7, float(t)) for t in ts])
    nb = _try_numba_Qtilde_grid(7, ts)
    numba_status = "used" if nb is not None else "skipped"
    numba_residual = 0.0
    numba_ok = True
    if nb is not None:
        numba_residual = float(np.max(np.abs(ref - nb)))
        numba_ok = bool(numba_residual < 1e-10)

    slopes_ok = True
    slope_rows = []
    for n in (5, 9, 15, 21):
        gt = _try_torch_slope(n, theta_br[n])
        gj = _try_jax_slope(n, theta_br[n])
        slope_rows.append({"n": n, "torch_d_residual_dtheta": gt, "jax_d_residual_dtheta": gj})
        if gt is not None and gt <= 0:
            slopes_ok = False
        if gj is not None and gj <= 0:
            slopes_ok = False
    df_slopes = pd.DataFrame(slope_rows)
    df_checks_scipy = df1[
        ["n", "theta_rad", "bisect_brentq_agree", "theta_diff", "residual_abs"]
    ].copy()
    df_checks_numba = pd.DataFrame(
        [
            {
                "numba": numba_status,
                "max_abs_Qtilde_residual_vs_numpy": numba_residual,
                "numba_parity_ok": numba_ok,
            }
        ]
    )

    summary_pass = sym_ok and ok_brent_all and ok_res and ok_split and ok_sum5 and numba_ok and slopes_ok

    df_summary = pd.DataFrame(
        [
            {
                "sympy_abs_integral": sym_ok,
                "brentq_vs_bisect_all_n": ok_brent_all,
                "self_consistency_residuals": ok_res,
                "amplitude_split_formula": ok_split,
                "Z5_mass_sum": ok_sum5,
                "numba_Qtilde_optional": numba_ok,
                "autodiff_slopes_positive_optional": slopes_ok,
                "lepton_masses_source": _LEPTON_SOURCE,
                "overall_ok": summary_pass,
            }
        ]
    )

    pdf_path: Optional[str] = None
    if verbose:
        print(SEP)
        print("Paper VIII — reproducibility bundle")
        print(SEP)
        print(df_sym.to_string(index=False))
        print(df_summary.to_string(index=False))
        print(describe_autodiff_backends())

    theta_dict = {n: theta_br[n] for n in ns_main}
    if write_outputs and out is not None:
        df_sym.to_csv(out / "checks_symbolic_abs_A_integral.csv", index=False)
        df1.to_csv(out / "table1_self_consistency_solutions.csv", index=False)
        df2.to_csv(out / "table2_amplitude_split_theorem.csv", index=False)
        df3.to_csv(out / "table3_large_n_asymptotics.csv", index=False)
        df4.to_csv(out / "table4_Z5_mode_spectrum.csv", index=False)
        df5.to_csv(out / "table5_higher_spectra.csv", index=False)
        df6.to_csv(out / "table6_doublet_pairing.csv", index=False)
        df_checks_scipy.to_csv(out / "checks_scipy_brentq_vs_bisect.csv", index=False)
        df_checks_numba.to_csv(out / "checks_numba_Qtilde_grid.csv", index=False)
        df_slopes.to_csv(out / "checks_autodiff_slopes.csv", index=False)
        df_summary.to_csv(out / "checks_summary_VIII_gate.csv", index=False)
        pdf = _plot_VIII_bundle(out, m3, theta_dict, th5, m5)
        pdf_path = str(pdf)
        if verbose:
            print(f"Artifacts → {out}")

    return {
        "output_dir": str(out) if out else None,
        "summary_pass": summary_pass,
        "theta_by_n": theta_dict,
        "figures_pdf": pdf_path,
        "summary_gate": df_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper VIII supplementary — reproducibility harness.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    res = run_paper_viii(write_outputs=not args.no_write, verbose=not args.quiet)
    ok = bool(res["summary_gate"]["overall_ok"].iloc[0])  # type: ignore[index]
    if ok and not args.quiet:
        print(SEP)
        print("All Paper VIII supplementary checks passed.")
        print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
