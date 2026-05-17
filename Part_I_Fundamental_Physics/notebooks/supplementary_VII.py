"""
Supplementary Material — Paper VII
"The Universal Mass Sum Rule: A Theta-Independent Prediction
for All CDFT Torus Knot Families"

Uses paper-local public constants for PDG lepton masses; SymPy for sum A_k^2 = 2n;
SciPy `brentq` for Z₅ self-consistency; Pandas CSV tables; Matplotlib figure bundle;
optional Numba scan and Torch/JAX derivative at the fixed point.

Outputs: Part_I_Fundamental_Physics/outputs/paper_VII/
Notebook: Part_I_Fundamental_Physics/notebooks/paper_VII_fullstack.ipynb
"""
from __future__ import annotations

import argparse
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
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

from _physics_utils import LEPTON_MASSES

_LEPTON_SOURCE = "paper_local_public_constants"

SEP = "=" * 64
C_N = float(np.sqrt(2.0))
Q_3_TARGET = 2.0 / 3.0
THETA_3 = Q_3_TARGET / 3.0


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


def solve_theta_sc_bisect(n: int, ngrid: int = 200_000) -> Optional[float]:
    """First root of n*theta - Q_tilde(n,theta) on (0, pi/n)."""
    if n == 3:
        return THETA_3
    lo, hi = 1e-12, np.pi / n - 1e-12
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


def solve_theta_sc_brentq(n: int) -> Optional[float]:
    if n == 3:
        return THETA_3

    def f(t: float) -> float:
        return n * t - Q_tilde(n, t)

    lo, hi = 1e-14, np.pi / n - 1e-14
    try:
        if f(lo) * f(hi) > 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15))
    except ValueError:
        return None


def _sympy_sum_Ak_squared() -> pd.DataFrame:
    th = sp.Symbol("theta", real=True)
    c = sp.sqrt(2)
    rows = []
    test_pts = (sp.Rational(1, 11), sp.Rational(17, 100), -sp.Rational(23, 100))
    for n in (3, 5, 7, 9):
        s = sum((1 + c * sp.cos(th + 2 * sp.pi * sp.Integer(j) / n)) ** 2 for j in range(n))
        s_simpl = sp.simplify(s)
        literal_ok = s_simpl == 2 * n
        num_ok = all(abs(float(sp.N(s_simpl.subs(th, t) - 2 * n, 50))) < 1e-12 for t in test_pts)
        rows.append(
            {
                "n": n,
                "sum_Ak2_simplified": str(s_simpl)[:100],
                "literal_equals_2n": literal_ok,
                "numeric_gate_2n": num_ok,
            }
        )
    return pd.DataFrame(rows)


def _try_torch_df_z5(theta_root: float) -> Optional[float]:
    try:
        import torch

        pi_t = torch.tensor(np.pi, dtype=torch.float64)

        def f(th: "torch.Tensor") -> "torch.Tensor":
            amps = [
                1.0 + C_N * torch.cos(th + 2.0 * pi_t * k / 5.0) for k in range(5)
            ]
            sq = sum(a * a for a in amps)
            sa = sum(torch.abs(a) for a in amps)
            q = sq / (sa * sa)
            return 5.0 * th - q

        th = torch.tensor(theta_root, dtype=torch.float64, requires_grad=True)
        y = f(th)
        g = torch.autograd.grad(y, th, create_graph=True)[0]
        return float(abs(g.item()))
    except ImportError:
        return None


def _try_jax_df_z5(theta_root: float) -> Optional[float]:
    try:
        import jax.numpy as jnp
        from jax import grad

        def f(th: float) -> float:
            amps = [1.0 + C_N * jnp.cos(th + 2.0 * jnp.pi * k / 5.0) for k in range(5)]
            sq = sum(a * a for a in amps)
            sa = sum(abs(a) for a in amps)
            q = sq / (sa * sa)
            return 5.0 * th - q

        g = grad(f)(theta_root)
        return float(abs(float(g)))
    except ImportError:
        return None


def _numba_z5_root_scan() -> Tuple[str, bool, int]:
    try:
        from numba import njit

        @njit
        def scan(npt: int) -> int:
            c = 1.4142135623730951
            lo = 1e-12
            hi = 3.141592653589793 / 5.0 - 1e-12
            prev = 0.0
            sign_changes = 0
            for i in range(npt):
                t = lo + (hi - lo) * i / (npt - 1)
                sq = 0.0
                sa = 0.0
                for k in range(5):
                    a = 1.0 + c * np.cos(t + 2.0 * 3.141592653589793 * k / 5.0)
                    sq += a * a
                    sa += abs(a)
                q = sq / (sa * sa)
                f = 5.0 * t - q
                if i > 0:
                    if prev * f < 0:
                        sign_changes += 1
                prev = f
            return sign_changes

        sc = scan(500_000)
        return "used", sc >= 1, sc
    except ImportError:
        return "skipped", True, -1


def _plot_VII_bundle(
    out: Path,
    m3: float,
    ns: List[int],
    theta5: float,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    n_arr = np.array(ns, dtype=float)
    sigma = np.array([2.0 * n * M_n_scale(int(n), m3) for n in ns])
    axes[0].loglog(n_arr, sigma, "o-", label=r"$\Sigma_n=2nM_n$")
    ref = sigma[0] * (n_arr / n_arr[0]) ** (7.0 / 4.0)
    axes[0].loglog(n_arr, ref, "--", alpha=0.7, label=r"$\propto n^{7/4}$ anchor")
    axes[0].set_xlabel(r"$n$")
    axes[0].set_ylabel(r"$\Sigma_n$ (MeV)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Mass sum hierarchy")

    amps0 = [brannen_amp(0.0, 5, k) for k in range(5)]
    colors = ["C0" if a >= 0 else "C3" for a in amps0]
    axes[1].bar(range(5), amps0, color=colors)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xticks(range(5))
    axes[1].set_xlabel(r"$k$")
    axes[1].set_ylabel(r"$A_k(\theta=0)$")
    axes[1].set_title(r"$\mathbb{Z}_5$ amplitude split (3+2)")

    ths = np.linspace(1e-6, np.pi / 5 - 1e-6, 800)
    lhs = 5 * ths
    rhs = np.array([Q_tilde(5, float(t)) for t in ths])
    axes[2].plot(ths, lhs, label=r"$5\theta$")
    axes[2].plot(ths, rhs, label=r"$\tilde Q_5(\theta)$")
    axes[2].axvline(theta5, color="k", ls=":", lw=1)
    axes[2].scatter([theta5], [5 * theta5], color="red", zorder=5, s=40)
    axes[2].set_xlabel(r"$\theta$")
    axes[2].legend(fontsize=8)
    axes[2].set_title(r"$\mathbb{Z}_5$ self-consistency fixed point")

    fig.tight_layout()
    pdf = out / "paper_VII_figures.pdf"
    fig.savefig(pdf, bbox_inches="tight")

    # Export individual panels for LaTeX inclusion
    for i, ax in enumerate(axes):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"fig7_{i}_panel.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    plt.close(fig)
    return pdf


def run_paper_vii(write_outputs: bool = True, verbose: bool = True) -> Dict[str, object]:
    out: Optional[Path] = output_dir("paper_VII") if write_outputs else None

    m3 = M_3_from_electron()
    me = float(LEPTON_MASSES["electron"])
    mmu = float(LEPTON_MASSES["muon"])
    mtau = float(LEPTON_MASSES["tau"])
    sum_pdg = me + mmu + mtau
    sigma3_pred = 6.0 * m3

    # Table 1
    rows1 = []
    for th in (0.1, 2.0 / 9.0, 0.25):
        amps = [brannen_amp(th, 3, k) for k in range(3)]
        total = m3 * sum(a * a for a in amps)
        ok, diff = agreement_ok(total, 6.0 * m3)
        rows1.append(
            {
                "n": 3,
                "theta_rad": th,
                "theta_deg": np.degrees(th),
                "sum_m_k_MeV": total,
                "two_n_Mn_MeV": 6.0 * m3,
                "sum_rule_ok": ok,
                "abs_diff_MeV": diff,
            }
        )
    ok_pdg, diff_pdg = agreement_ok(sigma3_pred, sum_pdg, rtol=5e-3, atol=0.5)
    rows1.append(
        {
            "n": 3,
            "theta_rad": float("nan"),
            "theta_deg": float("nan"),
            "sum_m_k_MeV": sum_pdg,
            "two_n_Mn_MeV": sigma3_pred,
            "sum_rule_ok": ok_pdg,
            "abs_diff_MeV": diff_pdg,
        }
    )
    df1 = pd.DataFrame(rows1)

    # Table 2
    ns_hier = list(range(3, 14))
    rows2 = []
    for n in ns_hier:
        mn = M_n_scale(n, m3)
        sn = 2.0 * n * mn
        fac = (n / 3.0) ** (7.0 / 4.0)
        rows2.append(
            {
                "n": n,
                "knot": f"T(2,{n})",
                "M_n_MeV": mn,
                "sum_m_k_MeV": sn,
                "n_pow_7_over_4_factor": fac,
                "verified_n3": n == 3,
            }
        )
    df2 = pd.DataFrame(rows2)

    # Table 3 — Z5 split
    amps_sym = [(brannen_amp(0.0, 5, k), k) for k in range(5)]
    rows3a = [
        {"k": k, "A_k": a, "sign": "+" if a > 0 else "-"} for a, k in sorted(amps_sym, key=lambda x: x[0])
    ]
    thetas_period = np.linspace(0, 2 * np.pi / 5, 100_000, endpoint=False)
    n_neg_vals = [count_neg(5, float(t)) for t in thetas_period]
    n2_frac = sum(1 for nn in n_neg_vals if nn == 2) / len(n_neg_vals)
    n1_frac = sum(1 for nn in n_neg_vals if nn == 1) / len(n_neg_vals)
    n0_frac = 1.0 - n1_frac - n2_frac
    df3a = pd.DataFrame(rows3a)
    df3b = pd.DataFrame(
        [
            {
                "fraction_2_neg": n2_frac,
                "fraction_1_neg": n1_frac,
                "fraction_0_neg": n0_frac,
                "pi_over_20_rad": np.pi / 20,
                "seven_pi_over_20_rad": 7 * np.pi / 20,
            }
        ]
    )

    # Table 4 — Z5 theta-independent
    m5 = M_n_scale(5, m3)
    thetas_test = [0.0, 0.1, 0.5, 1.0, np.pi / 5, 2 * np.pi / 5 - 0.01]
    rows4 = []
    for th in thetas_test:
        amps = [brannen_amp(th, 5, k) for k in range(5)]
        total = m5 * sum(a * a for a in amps)
        ok, _ = agreement_ok(total, 10.0 * m5)
        rows4.append({"theta_rad": th, "sum_m_k_MeV": total, "ten_M5_MeV": 10.0 * m5, "sum_rule_ok": ok})
    q5_samples = np.linspace(1e-6, np.pi / 5 - 1e-6, 50_000)
    q5v = np.array([Q_tilde(5, float(t)) for t in q5_samples])
    q5_min, q5_max = float(q5v.min()), float(q5v.max())
    df4 = pd.DataFrame(rows4)
    df4_meta = pd.DataFrame(
        [
            {
                "M_5_MeV": m5,
                "mean_mass_2M5_MeV": 2.0 * m5,
                "sum_Ak2_exact": 10.0,
                "A_k_min": 1.0 - C_N,
                "A_k_max": 1.0 + C_N,
                "m_k_max_MeV": m5 * (1.0 + C_N) ** 2,
                "Q5_min_scan": q5_min,
                "Q5_max_scan": q5_max,
            }
        ]
    )

    # Table 5 — self-consistency
    th5_bi = solve_theta_sc_bisect(5)
    th5_br = solve_theta_sc_brentq(5)
    if th5_bi is None or th5_br is None:
        raise RuntimeError("Z5 self-consistency root not found")
    ok_th, dth = agreement_ok(th5_bi, th5_br, rtol=1e-10, atol=1e-12)
    res = abs(5.0 * th5_br - Q_tilde(5, th5_br))
    amps5 = [brannen_amp(th5_br, 5, k) for k in range(5)]
    masses5 = sorted([m5 * a * a for a in amps5])
    df5 = pd.DataFrame(
        [
            {
                "theta_bisect_rad": th5_bi,
                "theta_brentq_rad": th5_br,
                "theta_agreement_ok": ok_th,
                "abs_theta_diff": dth,
                "five_theta": 5.0 * th5_br,
                "Q_tilde_at_theta": Q_tilde(5, th5_br),
                "residual_abs": res,
                "sum_masses_MeV": sum(masses5),
                "ten_M5_MeV": 10.0 * m5,
            }
        ]
    )
    df5_modes = pd.DataFrame([{"mode_index": i, "m_MeV": m} for i, m in enumerate(masses5)])

    # Table 6
    rows6 = [
        ("Energy scale M_n (MeV)", f"{M_n_scale(3, m3):.4f}", f"{m5:.4f}"),
        ("Number of modes", "3", "5"),
        ("Brannen c_n", "sqrt(2)", "sqrt(2)"),
        ("Valid domain all A_k>0", "YES (-pi/12,pi/12)", "NO"),
        ("Q_n constant 2/n", "YES Q_3=2/3", f"NO in [{q5_min:.4f},{q5_max:.4f}]"),
        ("sum m_k = 2nM_n", f"YES {6*m3:.2f}", f"YES {10*m5:.2f}"),
        ("theta derivation", "algebraic", "transcendental"),
        ("Sector closed", "YES (Paper V)", "OPEN"),
        ("Mode split", "3 positive A_k", "3+2 or 4+1"),
    ]
    df6 = pd.DataFrame(rows6, columns=["property", "Z3_leptons", "Z5_cinquefoil"])

    # Table 7
    rows7 = []
    for n in ns_hier:
        mn = M_n_scale(n, m3)
        sn = 2.0 * n * mn
        status = "VERIFIED" if n == 3 else "PREDICTED"
        rows7.append({"n": n, "sum_m_k_MeV": sn, "status": status})
    df7 = pd.DataFrame(rows7)

    df_sym = _sympy_sum_Ak_squared()
    sym_ok = bool(df_sym["numeric_gate_2n"].all())

    ok_sum3 = all(
        agreement_ok(m3 * sum(brannen_amp(th, 3, k) ** 2 for k in range(3)), 6.0 * m3)[0]
        for th in (0.1, 2.0 / 9.0, 0.25)
    )
    ok_brent = ok_th and res < 1e-10
    ok_table4 = bool(df4["sum_rule_ok"].all())

    numba_status, numba_ok, sign_changes = _numba_z5_root_scan()
    td = _try_torch_df_z5(th5_br)
    jd = _try_jax_df_z5(th5_br)
    torch_ok = True if td is None else td > 1e-6
    jax_ok = True if jd is None else jd > 1e-6
    has_z5_bracket = sign_changes >= 1 if sign_changes >= 0 else True

    df_checks_scipy = pd.DataFrame(
        [
            {
                "theta_bisect_rad": th5_bi,
                "theta_brentq_rad": th5_br,
                "theta_agreement_ok": ok_th,
                "abs_theta_diff": dth,
                "residual_abs": res,
            }
        ]
    )
    df_checks_numba = pd.DataFrame(
        [
            {
                "numba": numba_status,
                "bracket_found": has_z5_bracket,
                "n_sign_changes": sign_changes,
            }
        ]
    )
    df_checks_autodiff = pd.DataFrame(
        [
            {
                "torch_abs_d_residual_dtheta": td,
                "jax_abs_d_residual_dtheta": jd,
                "torch_gate": torch_ok,
                "jax_gate": jax_ok,
                "backends": describe_autodiff_backends(),
            }
        ]
    )

    summary_pass = (
        sym_ok
        and ok_sum3
        and ok_brent
        and ok_table4
        and numba_ok
        and torch_ok
        and jax_ok
    )

    df_summary = pd.DataFrame(
        [
            {
                "sympy_sum_Ak2_2n": sym_ok,
                "mass_sum_n3_multi_theta": ok_sum3,
                "Z5_brentq_residual": ok_brent,
                "Z5_sum_independent_theta": ok_table4,
                "numba_Z5_bracket_scan": numba_ok,
                "torch_df_optional": torch_ok,
                "jax_df_optional": jax_ok,
                "pdg_sum_vs_6M3_loose": ok_pdg,
                "lepton_masses_source": _LEPTON_SOURCE,
                "overall_ok": summary_pass,
            }
        ]
    )

    pdf_path: Optional[str] = None
    if verbose:
        print(SEP)
        print("Paper VII — reproducibility bundle")
        print(SEP)
        print(df_sym.to_string(index=False))
        print()
        print(df_summary.to_string(index=False))
        print(describe_autodiff_backends())

    if write_outputs and out is not None:
        df_sym.to_csv(out / "checks_symbolic_sum_Ak2.csv", index=False)
        df1.to_csv(out / "table1_sum_rule_verification.csv", index=False)
        df2.to_csv(out / "table2_mass_sum_hierarchy.csv", index=False)
        df3a.to_csv(out / "table3_Z5_amplitudes_theta0.csv", index=False)
        df3b.to_csv(out / "table3_Z5_negative_fractions.csv", index=False)
        df4.to_csv(out / "table4_Z5_theta_independent_sums.csv", index=False)
        df4_meta.to_csv(out / "table4_Z5_bounds_meta.csv", index=False)
        df5.to_csv(out / "table5_Z5_self_consistency.csv", index=False)
        df5_modes.to_csv(out / "table5_Z5_mode_masses.csv", index=False)
        df6.to_csv(out / "table6_Z3_vs_Z5_structure.csv", index=False)
        df7.to_csv(out / "table7_complete_predictions.csv", index=False)
        df_checks_scipy.to_csv(out / "checks_scipy_Z5_brentq_vs_bisect.csv", index=False)
        df_checks_numba.to_csv(out / "checks_numba_Z5_root_scan.csv", index=False)
        df_checks_autodiff.to_csv(out / "checks_autodiff_Z5_fixedpoint.csv", index=False)
        df_summary.to_csv(out / "checks_summary_VII_gate.csv", index=False)
        pdf = _plot_VII_bundle(out, m3, ns_hier, th5_br)
        pdf_path = str(pdf)
        if verbose:
            print(f"Artifacts → {out}")

    return {
        "output_dir": str(out) if out else None,
        "summary_pass": summary_pass,
        "M_3_MeV": m3,
        "figures_pdf": pdf_path,
        "summary_gate": df_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper VII supplementary — reproducibility harness.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    res = run_paper_vii(write_outputs=not args.no_write, verbose=not args.quiet)
    ok = bool(res["summary_gate"]["overall_ok"].iloc[0])  # type: ignore[index]
    if ok and not args.quiet:
        print(SEP)
        print("All Paper VII supplementary checks passed.")
        print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
