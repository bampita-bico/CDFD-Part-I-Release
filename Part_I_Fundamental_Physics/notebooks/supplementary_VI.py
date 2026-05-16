"""
Supplementary Material — Paper VI
"The Universal Torus Knot Hierarchy: Z_n Symmetry, the Cinquefoil Family,
and the General Structure of CDFT Particle Families"

Author: Steve Bico Mujjabi, MD (2026)
ORCID: https://orcid.org/0009-0001-0556-5516

Uses paper-local public helpers for chi, kappa, Brannen/Koide primitives, PDG
lepton masses; SymPy for Z_n Fourier identities and c_n = sqrt(2); SciPy for bounded
optimisation of max_θ min_k A_k; Pandas CSV tables; Matplotlib figure bundle;
optional Numba/sklearn Q₃ flatness scan; optional Torch/JAX gradient of min_k A_k.

Usage (from repository root):
    pip install -r requirements.txt
    pip install -r physics_papers/requirements-fullstack.txt
    python physics_papers/supplementary_VI.py

Notebook: physics_papers/notebooks/paper_VI_fullstack.ipynb
Outputs:  physics_papers/outputs/paper_VI/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PAPERS_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

from _physics_utils import (
    ALPHA_MEASURED,
    CHI_TARGET,
    LEPTON_MASSES,
    brannen_masses,
    fit_brannen_to_leptons,
    kappa_for_chi,
    koide_ratio,
    chi_self_consistency,
)

SEP = "=" * 64
BETA = 1.75
C_N = np.sqrt(2.0)
Q_3_TARGET = 2.0 / 3.0
THETA_3_DERIVED = Q_3_TARGET / 3.0  # Paper V: 2/9 rad


def brannen_amp(theta: float, n: int, k: int, c: float = C_N) -> float:
    return 1.0 + c * np.cos(theta + 2.0 * np.pi * k / n)


def min_k_amp(theta: float, n: int, c: float = C_N) -> float:
    return float(min(brannen_amp(theta, n, k, c) for k in range(n)))


def max_over_theta_min_k_amp(n: int, c: float = C_N) -> Tuple[float, float]:
    """Return (max_θ min_k A_k, argmax_theta) on [0, 2π/n]."""

    upper = 2.0 * np.pi / n

    def neg_min(th: float) -> float:
        return -min_k_amp(th, n, c)

    res = minimize_scalar(neg_min, bounds=(0.0, upper), method="bounded")
    th_star = float(res.x)
    return float(-res.fun), th_star


def sum_A_sq(theta: float, n: int, c: float = C_N) -> float:
    return float(sum(brannen_amp(theta, n, k, c) ** 2 for k in range(n)))


def sum_abs_A(theta: float, n: int, c: float = C_N) -> float:
    return float(sum(abs(brannen_amp(theta, n, k, c)) for k in range(n)))


def Q_tilde(theta: float, n: int, c: float = C_N) -> float:
    """Modified Koide-like ratio using |A_k| in denominator (Paper VI Eq. Q_tilde)."""
    num = sum_A_sq(theta, n, c)
    den = sum_abs_A(theta, n, c)
    return num / (den * den)


def M_3_from_paper_v(m_e_mev: Optional[float] = None) -> float:
    """Trefoil scale M_3 = m_e / A_{e,min}^2 at θ = 2/9 (Paper V)."""
    me = float(m_e_mev if m_e_mev is not None else LEPTON_MASSES["electron"])
    amps_sq = sorted(
        (1.0 + C_N * np.cos(THETA_3_DERIVED + 2.0 * np.pi * k / 3.0)) ** 2 for k in range(3)
    )
    ae_sq = amps_sq[0]
    return me / ae_sq


def M_n_faddeev_niemi(n: int, m3_mev: float) -> float:
    return m3_mev * (n / 3.0) ** (3.0 / 4.0)


def _sympy_zn_identities() -> pd.DataFrame:
    """Z_n cosine sums (Paper VI Sec. 2): symbolic form + numeric substitution gate."""
    th = sp.Symbol("theta", real=True)
    rows = []
    test_pts = (sp.Rational(1, 11), sp.Rational(17, 100), -sp.Rational(23, 100))
    for n in (3, 5, 7, 9):
        s_cos = sum(sp.cos(th + 2 * sp.pi * sp.Integer(j) / n) for j in range(n))
        s_cos2 = sum(sp.cos(th + 2 * sp.pi * sp.Integer(j) / n) ** 2 for j in range(n))
        sc = sp.simplify(s_cos)
        sc2 = sp.simplify(s_cos2)
        # SymPy may not reduce higher-n sums to 0 literally; verify on sample angles.
        sum_cos_zero = all(
            abs(float(sp.N(s_cos.subs(th, t), 50))) < 1e-12 for t in test_pts
        )
        sum_cos_sq_ok = all(
            abs(float(sp.N((s_cos2 - sp.Rational(n, 2)).subs(th, t), 50))) < 1e-12 for t in test_pts
        )
        rows.append(
            {
                "n": n,
                "sum_cos_simplified": str(sc)[:120],
                "sum_cos_is_zero_literal": sc == 0,
                "sum_cos_zero_numeric_gate": sum_cos_zero,
                "sum_cos_sq_simplified": str(sc2)[:120],
                "sum_cos_sq_equals_n_over_2_literal": sp.simplify(sc2 - sp.Rational(n, 2)) == 0,
                "sum_cos_sq_numeric_gate": sum_cos_sq_ok,
            }
        )
    return pd.DataFrame(rows)


def _sympy_cn_max_entropy() -> pd.DataFrame:
    """Var(A) = c^2/2 and c^2/2 = 1 => c = sqrt(2)."""
    c = sp.Symbol("c", positive=True)
    var_expr = c**2 / 2
    eq = sp.Eq(var_expr, 1)
    roots = sp.solve(eq, c)
    pos = [r for r in roots if r.is_real and r > 0]
    return pd.DataFrame(
        [
            {
                "variance_A_equals_c_sq_over_2": str(var_expr),
                "max_entropy_Var_equals_mean_sq": str(eq),
                "positive_root_c": float(pos[0]) if pos else float("nan"),
                "matches_sqrt2": bool(pos and abs(float(pos[0]) - np.sqrt(2)) < 1e-15),
            }
        ]
    )


def _sympy_Q3_algebra() -> pd.DataFrame:
    """sum A_k^2 = 2n and sum A_k = n => Q = 2M n / (M n^2) = 2/n for n=3."""
    n = 3
    c = sp.sqrt(2)
    sum_a2 = n + c**2 * sp.Rational(n, 2)
    sum_a = n  # identity I
    Q = (sum_a2) / (sum_a**2)
    return pd.DataFrame(
        [
            {
                "n": n,
                "sum_Ak_squared_symbolic": str(sp.simplify(sum_a2)),
                "sum_Ak_squared_numeric": float(sum_a2),
                "sum_Ak": int(sum_a),
                "Q_n_from_algebra": float(sp.N(Q)),
                "target_2_over_n": 2.0 / n,
            }
        ]
    )


def _numeric_variance_ratio_gate(ns: Sequence[int], thetas: Sequence[float]) -> Tuple[pd.DataFrame, bool]:
    rows = []
    all_ok = True
    for n in ns:
        for theta in thetas:
            A = [brannen_amp(theta, n, k) for k in range(n)]
            mean_a = float(np.mean(A))
            var_a = float(np.var(A, ddof=0))
            s1 = sum(np.cos(theta + 2.0 * np.pi * k / n) for k in range(n))
            s2 = sum(np.cos(theta + 2.0 * np.pi * k / n) ** 2 for k in range(n))
            ok1, d1 = agreement_ok(s1, 0.0, rtol=0.0, atol=1e-12)
            ok2, d2 = agreement_ok(s2, n / 2.0)
            ok3, d3 = agreement_ok(mean_a, 1.0)
            ok4, d4 = agreement_ok(var_a, C_N**2 / 2.0)
            ok5, d5 = agreement_ok(var_a / (mean_a**2), 1.0)
            row_ok = ok1 and ok2 and ok3 and ok4 and ok5
            all_ok = all_ok and row_ok
            rows.append(
                {
                    "n": n,
                    "theta_rad": theta,
                    "sum_cos_residual": d1,
                    "sum_cos_sq_residual": d2,
                    "mean_A_residual": d3,
                    "var_A_residual": d4,
                    "var_over_mean_sq_residual": d5,
                    "row_ok": row_ok,
                }
            )
    return pd.DataFrame(rows), all_ok


def _table_valid_domain() -> pd.DataFrame:
    rows = []
    for n in (3, 5, 7, 9, 11):
        mx, th_at = max_over_theta_min_k_amp(n)
        valid = mx > 0.0
        rows.append(
            {
                "n": n,
                "max_theta_min_k_Ak": mx,
                "argmax_theta_rad": th_at,
                "valid_domain_all_Ak_positive": valid,
                "matches_paper_table": (
                    (n == 3 and mx > 0.25)
                    or (n == 5 and -0.16 < mx < -0.12)
                    or (n == 7 and mx < -0.2)
                    or (n == 9 and mx < -0.3)
                    or (n == 11 and mx < -0.35)
                ),
            }
        )
    return pd.DataFrame(rows)


def _numba_Q3_scan(theta_lo: float, theta_hi: float, npts: int, M: float) -> Tuple[pd.DataFrame, float]:
    """NumPy vs Numba Koide Q on Z_3 Brannen masses in (theta_lo, theta_hi)."""

    grid = np.linspace(theta_lo, theta_hi, npts)

    def numpy_Qs(grid_: np.ndarray) -> np.ndarray:
        out = np.empty_like(grid_, dtype=np.float64)
        for i, t in enumerate(grid_):
            out[i] = koide_ratio(*brannen_masses(M, float(t)))
        return out

    q_np = numpy_Qs(grid)
    try:
        from numba import njit

        @njit
        def triple_koide(theta: float, mm: float) -> float:
            acc = np.empty(3, dtype=np.float64)
            for k in range(3):
                acc[k] = mm * ((1.0 + np.sqrt(2.0) * np.cos(theta + 2.0 * np.pi * k / 3.0)) ** 2)
            ssum = acc[0] + acc[1] + acc[2]
            rsum = np.sqrt(acc[0]) + np.sqrt(acc[1]) + np.sqrt(acc[2])
            return ssum / (rsum * rsum)

        @njit
        def scan_nb(grid_: np.ndarray, mm: float) -> np.ndarray:
            oo = np.empty(grid_.shape[0], dtype=np.float64)
            for i in range(grid_.shape[0]):
                oo[i] = triple_koide(grid_[i], mm)
            return oo

        q_nb = scan_nb(grid, M)
        nb_mod = "used"
        max_abs = float(np.max(np.abs(q_nb - q_np)))
        try:
            from sklearn.metrics import r2_score

            r2 = float(r2_score(q_np, q_nb))
        except ImportError:
            r2 = float("nan")
    except ImportError:
        q_nb = q_np
        nb_mod = "skipped"
        max_abs = 0.0
        r2 = 1.0

    df = pd.DataFrame(
        [
            {
                "numba_vs_numpy_max_abs_Q_residual": max_abs,
                "r2_sklearn": r2,
                "numba": nb_mod,
            }
        ]
    )
    q_dev = float(np.max(np.abs(q_np - Q_3_TARGET)))
    return df, q_dev


def try_torch_grad_min_amp(theta0: float, n: int) -> Optional[float]:
    try:
        import torch
    except ImportError:
        return None

    th = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    pi_t = torch.tensor(np.pi, dtype=torch.float64)
    vals = torch.stack(
        [
            1.0
            + torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
            * torch.cos(th + 2.0 * pi_t * k / n)
            for k in range(n)
        ]
    )
    m = torch.min(vals)
    g = torch.autograd.grad(m, th, create_graph=False)[0]
    return float(torch.abs(g).item())


def try_jax_grad_min_amp(theta0: float, n: int) -> Optional[float]:
    try:
        import jax.numpy as jnp
        from jax import grad
    except ImportError:
        return None

    def min_amp(th: float) -> float:
        amps = jnp.array(
            [1.0 + jnp.sqrt(2.0) * jnp.cos(th + 2.0 * jnp.pi * k / n) for k in range(n)]
        )
        return jnp.min(amps)

    g = grad(min_amp)(theta0)
    return float(abs(float(g)))


def _plot_VI_bundle(
    out: Path,
    m3_mev: float,
    hierarchy_ns: Sequence[int],
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.95))

    # (a) min_k A_k(theta) for n = 3,5,7,9
    ax0 = axes[0]
    theta_grid = np.linspace(-0.35, 0.35, 800)
    for n, sty in zip((3, 5, 7, 9), ("#3498db", "#e74c3c", "#2ecc71", "#9b59b6")):
        ys = [min_k_amp(t, n) for t in theta_grid]
        ax0.plot(theta_grid, ys, lw=1.35, label=f"$n={n}$", color=sty)
    ax0.axhline(0.0, color="#7f8c8d", ls="--", lw=1.0)
    ax0.axvline(np.pi / 12, color="#95a5a6", ls=":", lw=1.0)
    ax0.axvline(-np.pi / 12, color="#95a5a6", ls=":", lw=1.0)
    ax0.set_xlabel(r"$\theta$ (rad)")
    ax0.set_ylabel(r"$\min_k A_k(\theta)$")
    ax0.set_title("(a) Brannen amplitudes: $c_n=\\sqrt{2}$")
    ax0.legend(fontsize=8, loc="lower right")

    # (b) Faddeev–Niemi hierarchy
    ax1 = axes[1]
    Ms = [M_n_faddeev_niemi(n, m3_mev) for n in hierarchy_ns]
    x = np.arange(len(hierarchy_ns))
    colors = ["#3498db" if n != 5 else "#e67e22" for n in hierarchy_ns]
    ax1.bar(x, Ms, color=colors)
    ax1.set_xticks(x, [str(n) for n in hierarchy_ns])
    ax1.set_xlabel(r"$n$ ($T(2,n)$)")
    ax1.set_ylabel("$M_n$ / MeV")
    ax1.set_title("(b) $M_n = M_3 (n/3)^{3/4}$")

    # (c) Q_tilde for n=3 vs n=5
    ax2 = axes[2]
    th2 = np.linspace(-0.26, 0.26, 500)
    q3 = [Q_tilde(t, 3) for t in th2]
    q5 = [Q_tilde(t, 5) for t in th2]
    ax2.plot(th2, q3, color="#3498db", lw=1.4, label=r"$\tilde Q_3(\theta)$")
    ax2.plot(th2, q5, color="#e74c3c", lw=1.4, label=r"$\tilde Q_5(\theta)$")
    ax2.axhline(2.0 / 3.0, color="#3498db", ls=":", alpha=0.7)
    ax2.axhline(2.0 / 5.0, color="#e74c3c", ls=":", alpha=0.7)
    ax2.set_xlabel(r"$\theta$ (rad)")
    ax2.set_ylabel(r"$\tilde Q_n$")
    ax2.set_title(r"(c) Modified ratio $\sum A_k^2 / (\sum|A_k|)^2$")
    ax2.legend(fontsize=8)

    fig.suptitle(
        "Paper VI — Torus knot Z_n hierarchy — CDFT supplementary (2026)",
        fontsize=11,
        fontweight="bold",
        y=1.05,
    )
    pdf = out / "paper_VI_figures.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight", dpi=150)

    # Export individual panels for LaTeX inclusion
    for i, ax in enumerate([ax0, ax1, ax2]):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"fig6_{i}_panel.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    plt.close(fig)
    return pdf


def run_paper_vi(*, write_outputs: bool = True, verbose: bool = True) -> Dict:
    out: Optional[Path] = output_dir("paper_VI") if write_outputs else None

    m_e_mev = float(LEPTON_MASSES["electron"])
    m_3 = M_3_from_paper_v(m_e_mev)
    m_5 = M_n_faddeev_niemi(5, m_3)

    geom = chi_self_consistency()
    chi = CHI_TARGET
    kappa = kappa_for_chi(chi, BETA)
    fit_ref = fit_brannen_to_leptons()

    # --- Table 1: public constants + variance ratio (universal c_n) ---
    df_sym_id = _sympy_zn_identities()
    df_sym_cn = _sympy_cn_max_entropy()
    df_sym_q3 = _sympy_Q3_algebra()
    test_ns = (3, 5, 7, 9)
    test_thetas = (np.pi / 17, 0.21, -0.11)
    df_var_numeric, ok_var_numeric = _numeric_variance_ratio_gate(test_ns, test_thetas)

    df_table1 = pd.concat(
        [
            df_sym_cn.assign(block="sympy_cn"),
            df_sym_q3.assign(block="sympy_Q3_algebra"),
        ],
        ignore_index=True,
    )

    # --- Table 2: generalised Koide (algebra text + Z_3 public-helper cross-check) ---
    Q_brannen_theta = float(koide_ratio(*brannen_masses(1.0, THETA_3_DERIVED)))
    ok_q3_brannen, _ = agreement_ok(Q_brannen_theta, Q_3_TARGET)

    df_table2 = pd.DataFrame(
        [
            {
                "identity": "sum_m_over_sum_sqrt_m_squared_Z3",
                "Q_at_theta_2_over_9": Q_brannen_theta,
                "target_2_over_3": Q_3_TARGET,
                "agreement": ok_q3_brannen,
            },
            {
                "identity": "public_helper_koide_PDG_fit_theta_deg",
                "value": fit_ref["theta_deg"],
                "koide_Q_fit": fit_ref["koide_Q"],
                "notes": "reference Brannen fit to PDG (Paper II style)",
            },
        ]
    )

    # --- Table 3: valid domain ---
    df_table3 = _table_valid_domain()
    ok_valid_only_n3 = bool(df_table3.loc[df_table3["n"] == 3, "valid_domain_all_Ak_positive"].iloc[0])
    ok_paper_shape = bool(df_table3["matches_paper_table"].all())

    # --- Table 4: Z_3 Koide scan on (-pi/12, pi/12) ---
    theta_max = np.pi / 12.0
    thetas = np.linspace(-theta_max + 1e-3, theta_max - 1e-3, 500)
    q3_list = []
    for th in thetas:
        amps = [brannen_amp(th, 3, k) for k in range(3)]
        if any(a <= 0 for a in amps):
            continue
        masses = [a**2 for a in amps]
        q3_list.append(sum(masses) / (sum(np.sqrt(m) for m in masses)) ** 2)
    q3_dev = float(max(abs(q - Q_3_TARGET) for q in q3_list)) if q3_list else 1.0
    ok_q3_scan = q3_dev <= 1e-12

    df_table4 = pd.DataFrame(
        [
            {
                "theta_band_rad": f"(-pi/12, pi/12) excluding boundaries",
                "n_samples": len(q3_list),
                "max_abs_Q_minus_2_over_3": q3_dev,
                "pass": ok_q3_scan,
            }
        ]
    )

    # --- Table 5: Faddeev–Niemi hierarchy ---
    hierarchy_ns = (3, 5, 7, 9, 11, 13)
    knot_names = {3: "T(2,3)", 5: "T(2,5)", 7: "T(2,7)", 9: "T(2,9)", 11: "T(2,11)", 13: "T(2,13)"}
    rows_h = []
    for n in hierarchy_ns:
        mn = M_n_faddeev_niemi(n, m_3)
        rows_h.append(
            {
                "n": n,
                "knot": knot_names[n],
                "M_n_MeV": mn,
                "ratio_n_over_3_pow_3_over_4": (n / 3.0) ** (3.0 / 4.0),
            }
        )
    df_table5 = pd.DataFrame(rows_h)
    ok_m5, m5_diff = agreement_ok(m_5, 460.4, rtol=2.5e-3, atol=0.15)

    # --- Table 6: knot classification (static) ---
    torus_data = [
        (1, "odd", "unknot", "No", "Z_1"),
        (2, "even", "Hopf link", "No", "Z_2"),
        (3, "odd", "trefoil knot", "Yes", "Z_3"),
        (4, "even", "Solomon link", "No", "Z_4"),
        (5, "odd", "cinquefoil knot", "Yes", "Z_5"),
        (6, "even", "2-component link", "No", "Z_6"),
        (7, "odd", "heptafoil knot", "Yes", "Z_7"),
        (8, "even", "2-component link", "No", "Z_8"),
        (9, "odd", "T(2,9) knot", "Yes", "Z_9"),
    ]
    df_table6 = pd.DataFrame(
        torus_data,
        columns=["n", "parity", "topological_class", "persistent_knot", "symmetry"],
    )

    # --- Table 7: Z_5 obstruction + Q_tilde samples ---
    thetas_full = np.linspace(0.0, 2.0 * np.pi, 20000)
    min_amps_5 = np.array([min_k_amp(t, 5) for t in thetas_full])
    max_min_5 = float(np.max(min_amps_5))
    frac_pos = float(np.mean(min_amps_5 > 0.0))
    th_maxmin = float(thetas_full[int(np.argmax(min_amps_5))])

    theta_5_samples = [0.0, 0.1, 0.3, np.pi / 10, np.pi / 5, np.pi / 3]
    rows_z5 = []
    for th in theta_5_samples:
        amps = [brannen_amp(th, 5, k) for k in range(5)]
        masses = [a**2 for a in amps]
        q5_abs = sum(masses) / (sum(np.sqrt(abs(m)) for m in masses)) ** 2
        rows_z5.append({"theta_rad": th, "min_Ak": min(amps), "Q5_abs_sqrt": q5_abs})
    df_table7a = pd.DataFrame(
        [
            {
                "fraction_theta_period_all_Ak_positive": frac_pos,
                "max_over_theta_min_k_Ak": max_min_5,
                "argmax_theta_rad": th_maxmin,
                "no_connected_valid_domain": max_min_5 < 0.0,
            }
        ]
    )
    df_table7b = pd.DataFrame(rows_z5)

    q5_range = df_table7b["Q5_abs_sqrt"].astype(float)
    ok_q5_nonconst = float(q5_range.max() - q5_range.min()) > 0.01

    # --- Table 8: status snapshot ---
    g_factor = chi * (np.log(8.0 * chi) - BETA) + kappa / chi
    rho0_lit = 1.587e13  # kg/m^3 order-of-magnitude anchor from Papers III–V (Paper VI script legacy)
    df_table8 = pd.DataFrame(
        [
            {"block": "public_helper", "key": "CHI_TARGET", "value": chi},
            {"block": "public_helper", "key": "ALPHA_MEASURED", "value": ALPHA_MEASURED},
            {"block": "public_helper", "key": "kappa_for_chi", "value": kappa},
            {"block": "public_helper", "key": "g_E_norm", "value": g_factor},
            {"block": "paper_VI", "key": "M_3_MeV", "value": m_3},
            {"block": "paper_VI", "key": "M_5_MeV", "value": m_5},
            {"block": "paper_VI", "key": "c_n_universal", "value": C_N},
            {"block": "legacy_anchor", "key": "rho0_kg_m3_order_of_magnitude", "value": rho0_lit},
        ]
    )

    # SciPy max-min cross-check vs dense grid for n=5
    mx_sci, th_sci = max_over_theta_min_k_amp(5)
    mx_grid = float(np.max([min_k_amp(t, 5) for t in np.linspace(0, 2 * np.pi / 5, 50000)]))
    ok_scipy_grid, scipy_diff = agreement_ok(mx_sci, mx_grid, rtol=1e-3, atol=5e-4)

    df_checks_scipy = pd.DataFrame(
        [
            {
                "n": 5,
                "max_min_scipy": mx_sci,
                "max_min_dense_grid": mx_grid,
                "agreement": ok_scipy_grid,
                "abs_diff": scipy_diff,
            }
        ]
    )

    numba_df, q_dev_numba = _numba_Q3_scan(-theta_max + 1e-3, theta_max - 1e-3, 2600, 1.0)
    ok_numba_flat = q_dev_numba <= 1e-10

    tdq = try_torch_grad_min_amp(th_sci, 5)
    jdq = try_jax_grad_min_amp(th_sci, 5)
    # At interior of smooth region, |grad| finite; near kink may be large — gate: optional or small
    torch_ok = True if tdq is None else tdq <= 10.0
    jax_ok = True if jdq is None else jdq <= 10.0

    sym_rows_ok = bool(
        df_sym_id["sum_cos_zero_numeric_gate"].all()
        and df_sym_id["sum_cos_sq_numeric_gate"].all()
        and bool(df_sym_cn.iloc[0]["matches_sqrt2"])
    )

    summary_pass = (
        ok_var_numeric
        and sym_rows_ok
        and ok_q3_brannen
        and ok_valid_only_n3
        and ok_paper_shape
        and ok_q3_scan
        and ok_m5
        and ok_q5_nonconst
        and ok_numba_flat
        and ok_scipy_grid
        and torch_ok
        and jax_ok
    )

    df_summary = pd.DataFrame(
        [
            {
                "numeric_variance_ratio_gate": ok_var_numeric,
                "sympy_fourier_and_cn": sym_rows_ok,
                "public_helper_koide_at_theta_2_9": ok_q3_brannen,
                "valid_domain_only_Z3": ok_valid_only_n3,
                "valid_domain_table_shape": ok_paper_shape,
                "Q3_flat_scan_valid_domain": ok_q3_scan,
                "M5_agreement_460p4_MeV": ok_m5,
                "M5_abs_diff_MeV": m5_diff,
                "Q5_nonconstant_samples": ok_q5_nonconst,
                "numba_Q3_scan_gate": ok_numba_flat,
                "scipy_vs_dense_grid_n5": ok_scipy_grid,
                "torch_grad_min_amp_optional": torch_ok,
                "jax_grad_min_amp_optional": jax_ok,
                "overall_ok": summary_pass,
            }
        ]
    )

    pdf_path_val: Optional[str] = None
    if verbose:
        print(SEP)
        print("Paper VI — paper-local public reproducibility bundle")
        print(SEP)
        print(df_sym_id.to_string(index=False))
        print()
        print(df_sym_cn.to_string(index=False))
        print()
        print(df_sym_q3.to_string(index=False))
        print()
        print(df_var_numeric.to_string(index=False))
        print()
        print(df_table3.to_string(index=False))
        print()
        print(df_table5.to_string(index=False))
        print(df_summary.to_string(index=False))
        print(numba_df.to_string(index=False))
        print(df_checks_scipy.to_string(index=False))
        print(
            pd.DataFrame(
                [{"torch_abs_dmin_dtheta": tdq, "jax_abs_dmin_dtheta": jdq, "backends": describe_autodiff_backends()}]
            ).to_string(index=False)
        )

    if write_outputs and out is not None:
        df_sym_id.to_csv(out / "checks_symbolic_Zn_identities.csv", index=False)
        df_sym_cn.to_csv(out / "checks_symbolic_cn_max_entropy.csv", index=False)
        df_sym_q3.to_csv(out / "checks_symbolic_Q3_algebra.csv", index=False)
        df_var_numeric.to_csv(out / "checks_numeric_variance_ratio.csv", index=False)
        df_table1.to_csv(out / "table1_universal_cn_and_Q3_algebra.csv", index=False)
        df_table2.to_csv(out / "table2_generalised_koide_public_helper.csv", index=False)
        df_table3.to_csv(out / "table3_valid_domain_Zn.csv", index=False)
        df_table4.to_csv(out / "table4_Z3_koide_scan_valid_domain.csv", index=False)
        df_table5.to_csv(out / "table5_faddeev_niemi_hierarchy.csv", index=False)
        df_table6.to_csv(out / "table6_knot_classification.csv", index=False)
        df_table7a.to_csv(out / "table7a_Z5_valid_domain_summary.csv", index=False)
        df_table7b.to_csv(out / "table7b_Z5_Q_samples.csv", index=False)
        df_table8.to_csv(out / "table8_public_helper_status_snapshot.csv", index=False)
        df_checks_scipy.to_csv(out / "checks_scipy_max_min_amps.csv", index=False)
        numba_df.to_csv(out / "checks_numba_Q3_flat.csv", index=False)
        pd.DataFrame(
            [{"torch_abs_dmin_dtheta": tdq, "jax_abs_dmin_dtheta": jdq, "backends": describe_autodiff_backends()}]
        ).to_csv(out / "checks_autodiff_min_amp.csv", index=False)
        df_summary.to_csv(out / "checks_summary_VI_gate.csv", index=False)

        pdf = _plot_VI_bundle(out, m_3, hierarchy_ns)
        pdf_path_val = str(pdf)
        if verbose:
            print(f"Artifacts → {out} ({pdf.name})")

    return {
        "output_dir": str(out) if out else None,
        "summary_pass": summary_pass,
        "M_3_MeV": m_3,
        "M_5_MeV": m_5,
        "figures_pdf": pdf_path_val,
        "summary_gate": df_summary,
        "chi_geom": geom,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper VI supplementary — reproducibility harness.")
    parser.add_argument("--no-write", action="store_true", help="Skip CSV/PDF output.")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    res = run_paper_vi(write_outputs=not args.no_write, verbose=not args.quiet)
    ok = bool(res["summary_gate"]["overall_ok"].iloc[0])
    if ok and not args.quiet:
        print(SEP)
        print("All Paper VI supplementary checks passed.")
        print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
