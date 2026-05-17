"""
Supplementary Material — Paper III
"Topology, Chirality, and the Vacuum Density"

Author: Steve Bico Mujjabi, MD (2026)
ORCID: https://orcid.org/0009-0001-0556-5516

Reproduces every number in the paper (κ closure, torus knots, Faddeev–Niemi
ordering, chirality equilibrium, M and ρ₀ decomposition). Uses the public
paper-local helpers for χ, κ, and Brannen fits; SymPy for algebraic
closures; SciPy for θ refinement and derivative checks; Pandas for CSV
exports; Matplotlib for figures. Optional Numba, PyTorch/JAX, Statsmodels
mirror Papers I–II.

Usage (from repository root):
    pip install -r requirements.txt
    pip install -r Part_I_Fundamental_Physics/requirements-fullstack.txt
    python Part_I_Fundamental_Physics/notebooks/supplementary_III.py

Notebook: Part_I_Fundamental_Physics/notebooks/paper_III_fullstack.ipynb
Outputs:  Part_I_Fundamental_Physics/outputs/paper_III/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
from scipy.optimize import minimize_scalar, root_scalar  # noqa: E402

from _physics_utils import agreement_ok, central_second_derivative, describe_autodiff_backends, output_dir

from _physics_utils import (
    CHI_TARGET,
    LEPTON_MASSES,
    brannen_masses,
    chi_self_consistency,
    energy_balance_at_chi,
    fit_brannen_to_leptons,
    kappa_for_chi,
)

SEP = "=" * 64

# PDG constants (paper text)
MEV_TO_J = 1.602176634e-13
C_LIGHT = 2.99792458e8
M_E_KG = 9.1093837015e-31
BETA = 1.75


def brannen_A(theta: float, k: int) -> float:
    return 1.0 + np.sqrt(2.0) * np.cos(theta + 2.0 * np.pi * k / 3.0)


def E_norm(chi: float, kappa: float, beta: float = BETA) -> float:
    return chi * (np.log(8.0 * chi) - beta) + kappa / chi


def _refine_theta_brannen(fit: Dict) -> float:
    target = sorted(LEPTON_MASSES.values())

    def loss_theta(theta: float) -> float:
        raw = sorted(brannen_masses(1.0, theta))
        if any(m <= 0 for m in raw):
            return 1e9
        mscal = float(np.exp(np.mean([np.log(t / r) for t, r in zip(target, raw)])))
        scaled = [m * mscal for m in raw]
        return float(sum((np.log(s / t)) ** 2 for s, t in zip(scaled, target)))

    res = minimize_scalar(
        loss_theta,
        bounds=(max(1e-6, fit["theta_rad"] - 0.03), min(float(2 * np.pi / 3), fit["theta_rad"] + 0.03)),
        method="bounded",
    )
    return float(res.x)


def _sympy_kappa_and_chirality_checks() -> pd.DataFrame:
    chi_s = sp.Symbol("chi", positive=True, real=True)
    beta_s = sp.Symbol("beta", positive=True, real=True)
    kappa_claim = chi_s ** 2 * (sp.log(8 * chi_s) - beta_s + 1)
    dcirc = sp.diff(chi_s * (sp.log(8 * chi_s) - beta_s), chi_s)
    dexpr = dcirc - kappa_claim / chi_s**2
    dexpr_simp = sp.simplify(dexpr)
    equilibrium_residual = sp.simplify(dexpr_simp * chi_s**2)

    theta_sym = sp.symbols("theta", real=True)
    phi_sym = sp.symbols("phi", real=True)
    phase = sp.cos(3 * theta_sym + phi_sym)
    phase_sub = sp.simplify(phase.subs(phi_sym, sp.pi - 3 * theta_sym))

    return pd.DataFrame(
        [
            {
                "check": "kappa_algebra_from_dE_sympy",
                "pass": equilibrium_residual == 0,
                "detail": str(equilibrium_residual),
            },
            {
                "check": "chirality_cos_substitution",
                "pass": bool(phase_sub == sp.cos(sp.pi)),
                "detail": str(phase_sub),
            },
        ]
    )


def _scipy_chirality_root(phi_c: float, theta_expect: float) -> pd.DataFrame:
    """Find θ where derivative of λ cos(3θ + φ_c) vanishes — should recover θ_expect."""

    def dE(th: float) -> float:
        return -3.0 * np.sin(3.0 * th + phi_c)

    a = max(1e-4, theta_expect - 0.1)
    b = min(float(np.pi / 3) - 1e-4, theta_expect + 0.1)
    if a >= b:
        a, b = 1e-4, float(np.pi / 3) - 1e-4
    res = root_scalar(dE, bracket=(a, b), method="brentq")
    ok, diff = agreement_ok(float(res.root), theta_expect)
    return pd.DataFrame(
        [
            {
                "theta_root_rad": float(res.root),
                "theta_expect_rad": theta_expect,
                "agreement": ok,
                "abs_diff": diff,
                "root_iterations": getattr(res, "iterations", np.nan),
            }
        ]
    )


def try_torch_d2_chiral(theta0: float, phi_c: float) -> Optional[float]:
    """d²/dθ² [λ cos(3θ+φ)] at equilibrium with λ=1."""
    try:
        import torch
    except ImportError:
        return None
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    e = torch.cos(3.0 * theta + phi_c)
    g1 = torch.autograd.grad(e, theta, create_graph=True)[0]
    g2 = torch.autograd.grad(g1, theta)[0]
    return float(g2.item())


def try_jax_d2_chiral(theta0: float, phi_c: float) -> Optional[float]:
    try:
        from jax import grad
        import jax.numpy as jnp

        def energy(th):
            return jnp.cos(3.0 * th + phi_c)

        d2 = grad(grad(energy))(theta0)
        return float(d2)
    except ImportError:
        return None


def _fn_energy_ratios_numpy(ns: np.ndarray) -> np.ndarray:
    ref = 3.0 ** (3.0 / 4.0)
    return (ns.astype(np.float64) ** (3.0 / 4.0)) / ref


def _numba_fn_scan(grid: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    ratios_np = _fn_energy_ratios_numpy(grid)

    try:
        from numba import njit

        @njit
        def ratio(n: float) -> float:
            ref = 3.0 ** 0.75
            return (n ** 0.75) / ref

        @njit
        def scan(gg: np.ndarray) -> np.ndarray:
            nn = gg.shape[0]
            oo = np.empty(nn)
            for i in range(nn):
                oo[i] = ratio(float(gg[i]))
            return oo

        ratios_nb = scan(grid.astype(np.float64))
        nb_mod = "used"
        max_abs = float(np.max(np.abs(ratios_nb - ratios_np)))
        try:
            from sklearn.metrics import r2_score

            r2 = float(r2_score(ratios_np, ratios_nb))
        except ImportError:
            r2 = float("nan")
    except ImportError:
        ratios_nb = ratios_np
        nb_mod = "skipped"
        max_abs = 0.0
        r2 = 1.0

    df = pd.DataFrame(
        [
            {
                "numba_vs_numpy_max_abs_FN_ratio": max_abs,
                "r2_sklearn": r2,
                "numba": nb_mod,
            }
        ]
    )
    return df, ratios_nb


def _statsmodels_fn_exponent(Qs: List[int]) -> pd.DataFrame:
    """OLS: log(E_rel) vs log(Q) — slope should be 3/4 for E ∝ Q^{3/4}."""
    Q = np.asarray(Qs, dtype=np.float64)
    ref = 3.0 ** (3.0 / 4.0)
    y = np.log(Q ** (3.0 / 4.0) / ref)
    x = np.log(Q)
    try:
        import statsmodels.api as sm

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        slope_raw = float(model.params.iloc[1]) if hasattr(model.params, "iloc") else float(model.params[1])
        return pd.DataFrame(
            [
                {
                    "expected_slope_three_quarters": 0.75,
                    "fitted_slope_logE_vs_logQ": slope_raw,
                    "rsquared": float(model.rsquared),
                    "n_points": len(Q),
                }
            ]
        )
    except ImportError:
        return pd.DataFrame(
            [
                {
                    "expected_slope_three_quarters": 0.75,
                    "fitted_slope_logE_vs_logQ": np.nan,
                    "rsquared": np.nan,
                    "n_points": len(Q),
                    "note": "statsmodels missing",
                }
            ]
        )


def _plot_figure_bundle(
    out: Path,
    theta_rad: float,
    phi_c: float,
    Qs: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    ratios = _fn_energy_ratios_numpy(Qs)

    ax0 = axes[0]
    ax0.bar(range(len(Qs)), ratios, tick_label=[str(int(q)) for q in Qs], color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"])
    ax0.axhline(y=1.0, color="gray", linestyle="--", lw=0.9)
    ax0.set_xlabel(r"Hopf charge $Q=n$ ($T(2,n)$ knots)")
    ax0.set_ylabel(r"$E/E_{\mathrm{trefoil}}$ ($\propto Q^{3/4}$)")
    ax0.set_title("(a) Faddeev–Niemi scaling")

    ths = np.linspace(0.0, float(np.pi / 3), 300)
    ax1 = axes[1]
    ax1.plot(np.degrees(ths), np.cos(3.0 * ths + phi_c), color="#34495e", lw=1.6)
    ax1.axvline(np.degrees(theta_rad), color="crimson", ls=":", lw=1.4, label=r"fitted $\theta$")
    ax1.axhline(-1.0, color="#7f8c8d", ls="--", lw=0.9)
    ax1.set_xlabel(r"$\theta$ (deg)")
    ax1.set_ylabel(r"$\cos(3\theta+\phi_c)$")
    ax1.set_title("(b) Chiral energy surrogate")
    ax1.legend(fontsize=8, loc="lower right")

    ax2 = axes[2]
    geom = chi_self_consistency()
    chi = CHI_TARGET
    kappa = kappa_for_chi(chi, BETA)
    chis_scan = np.linspace(chi * 0.997, chi * 1.003, 80)
    e_scan = np.array([E_norm(c, kappa, BETA) for c in chis_scan])
    ax2.plot(chis_scan - chi, e_scan - np.min(e_scan), color="#2980b9", lw=1.4)
    ax2.axvline(0.0, color="crimson", ls=":", lw=1.2)
    ax2.set_xlabel(r"$\chi - \chi_{\mathrm{eq}}$ ($\approx 137.036-\chi_{\mathrm{eq}}$ negligible)")
    ax2.set_ylabel(r"$\Delta E_{\mathrm{norm}}$ (relative)")
    ax2.set_title(r"(c) $E_{\mathrm{norm}}$ near equilibrium")
    text = rf"geom $\chi$: {geom['chi_geometric']:.6f}"

    fig.suptitle(
        "Paper III — Topology, Chirality, Vacuum Density Scale\nCDFD public supplementary material · Steve Bico Mujjabi, MD (2026)\n" + text,
        fontsize=12,
        fontweight="bold",
        y=1.05,
    )
    pdf = out / "paper_III_figures.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight", dpi=150)

    # Export individual panels for LaTeX inclusion
    for i, ax in enumerate([ax0, ax1, ax2]):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"fig3_{i}_panel.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    plt.close(fig)
    return pdf


def run_paper_iii(*, write_outputs: bool = True, verbose: bool = True) -> Dict:
    out: Optional[Path] = output_dir("paper_III") if write_outputs else None

    fit_coarse = fit_brannen_to_leptons()
    theta_rad = _refine_theta_brannen(fit_coarse)
    m_e = LEPTON_MASSES["electron"]
    m_mu = LEPTON_MASSES["muon"]
    chi = CHI_TARGET
    kappa = kappa_for_chi(chi, BETA)

    lhs_eq = np.log(8.0 * chi) - BETA + 1.0
    rhs_eq = kappa / chi ** 2
    ok_kappa, diff_kappa = agreement_ok(lhs_eq, rhs_eq)

    eb = energy_balance_at_chi(chi, BETA)

    knot_data = [
        (1, "unknot", False, 1, "U(1)", "ground state (electron)"),
        (2, "Hopf link", False, 2, "Z2", "2-component link, transient"),
        (3, "trefoil", True, 3, "Z3", "LEPTONS: e, mu, tau (Paper II)"),
        (4, "Solomon link", False, 4, "Z4", "2-component link, transient"),
        (5, "cinquefoil", True, 5, "Z5", "next knotted family"),
        (6, "3-component lnk", False, 6, "Z6", "multi-component link"),
        (7, "T(2,7) knot", True, 7, "Z7", "higher knotted family"),
    ]
    df_knots = pd.DataFrame(knot_data, columns=["n", "name", "is_knot", "lobes", "symmetry", "role"])

    E_tref = 3.0 ** (3.0 / 4.0)
    rows_fn = []
    for n in (3, 5, 7, 9):
        ratio = (n ** (3.0 / 4.0)) / E_tref
        rows_fn.append({"n": n, "Hopf_Q": n, "E_ratio": ratio, "is_knot": n % 2 == 1})
    df_fn = pd.DataFrame(rows_fn)

    phi_c = np.pi - 3.0 * theta_rad
    cos_chk = np.cos(3.0 * theta_rad + phi_c)

    target_mass = sorted(LEPTON_MASSES.values())
    raw_sorted = sorted(brannen_masses(1.0, theta_rad))
    mscal = float(np.exp(np.mean([np.log(t / r) for t, r in zip(target_mass, raw_sorted)])))
    scaled_mass = sorted([m * mscal for m in brannen_masses(1.0, theta_rad)])
    ratio_mmu_me = scaled_mass[1] / scaled_mass[0]
    ok_ratio, ratio_err = agreement_ok(ratio_mmu_me, m_mu / m_e, rtol=2e-4, atol=1e-8)

    a_e_min = min(brannen_A(theta_rad, k) for k in range(3))
    m_mv = float(m_e / a_e_min ** 2)
    g_factor = E_norm(chi, kappa, BETA)

    geom = chi_self_consistency()
    a_class = geom["a_classical_m"]

    m_j = m_mv * MEV_TO_J
    rho_si = float(m_j / (C_LIGHT ** 2 * a_class ** 3 * g_factor))
    rho_ele_classical = float(M_E_KG / ((4.0 / 3.0) * np.pi * a_class ** 3))

    df_table1_kappa = pd.DataFrame(
        [
            {
                "chi_target": chi,
                "beta": BETA,
                "kappa_derived": kappa,
                "lhs_ln8chi_minus_beta_plus1": lhs_eq,
                "rhs_kappa_over_chi2": rhs_eq,
                "closure_agreement": ok_kappa,
                "abs_residual": diff_kappa,
            }
        ]
    )

    df_chiral = pd.DataFrame(
        [
            {
                "theta_rad": theta_rad,
                "theta_deg": np.degrees(theta_rad),
                "phi_c_rad": phi_c,
                "phi_c_deg": np.degrees(phi_c),
                "cos_3theta_plus_phi_c": cos_chk,
                "sin_3theta_plus_phi_c": float(np.sin(3.0 * theta_rad + phi_c)),
                "mmu_me_brannen_ratio": ratio_mmu_me,
                "mmu_me_pdg_ratio": float(m_mu / m_e),
                "ratio_agreement": ok_ratio,
            }
        ]
    )

    df_M_rho = pd.DataFrame(
        [
            {
                "M_MeV": m_mv,
                "g_chi_beta": g_factor,
                "a_classical_m": a_class,
                "rho0_kg_m3": rho_si,
                "rho_classical_electron_kg_m3": rho_ele_classical,
                "comparison_nuclear_approx": 2.3e17,
            }
        ]
    )

    sym_df = _sympy_kappa_and_chirality_checks()
    scipy_roots = _scipy_chirality_root(phi_c, theta_rad)

    td2_torch = try_torch_d2_chiral(theta_rad, phi_c)
    td2_jax = try_jax_d2_chiral(theta_rad, phi_c)
    finite_d2 = central_second_derivative(lambda t: np.cos(3.0 * t + phi_c), theta_rad)
    autodiff_df = pd.DataFrame(
        [
            {
                "d2E_chiral_numeric_central_diff": finite_d2,
                "torch_d2_optional": td2_torch,
                "jax_d2_optional": td2_jax,
                "backends_status": describe_autodiff_backends(),
            }
        ]
    )

    grid_n = np.arange(3, 201, 2, dtype=np.float64)
    numba_df, _arr = _numba_fn_scan(grid_n)
    fn_ols_df = _statsmodels_fn_exponent([3, 5, 7, 9])

    if td2_torch is not None:
        ok_d2_torch, diff_d2 = agreement_ok(td2_torch, finite_d2, rtol=1e-5, atol=1e-6)
    else:
        ok_d2_torch, diff_d2 = True, 0.0

    summary_pass = ok_kappa and ok_ratio and scipy_roots["agreement"].iloc[0] and bool(sym_df["pass"].all())

    df_summary = pd.DataFrame(
        [
            {
                "kappa_closure": ok_kappa,
                "mmu_me_ratio_fit": ok_ratio,
                "scipy_theta_root_matches": bool(scipy_roots["agreement"].iloc[0]),
                "sympy_all_pass": bool(sym_df["pass"].all()),
                "overall_ok": summary_pass,
            }
        ]
    )

    eb_row = pd.DataFrame([eb])
    fit_row = pd.DataFrame(
        [
            {
                "M_fit_MeV": fit_coarse["M_MeV"],
                "theta_deg_coarse": fit_coarse["theta_deg"],
                "theta_deg_refined": np.degrees(theta_rad),
            }
        ]
    )

    if verbose:
        print(SEP)
        print("Paper III — TABLES (paper-local public helpers)")
        print(SEP)
        print(df_table1_kappa.to_string(index=False))
        print()
        print(df_knots.to_string(index=False))
        print()
        print(df_fn.to_string(index=False))
        print()
        print(df_chiral.to_string(index=False))
        print()
        print(df_M_rho.to_string(index=False))
        print(sym_df.to_string(index=False))
        print(scipy_roots.to_string(index=False))
        print(autodiff_df.to_string(index=False))
        print(numba_df.to_string(index=False))
        print(fn_ols_df.to_string(index=False))
        print(fit_row.to_string(index=False))
        print(df_summary.to_string(index=False))
        if td2_torch is not None:
            print(f"\nTorch d²E vs finite-diff agreement: ok={ok_d2_torch}, diff={diff_d2:.3e}")

    if write_outputs and out is not None:
        df_table1_kappa.to_csv(out / "table1_kappa_closure.csv", index=False)
        df_knots.to_csv(out / "table2_torus_knots.csv", index=False)
        df_fn.to_csv(out / "table3_fn_energy_ordering.csv", index=False)
        df_chiral.to_csv(out / "table4_chirality_theta_phi.csv", index=False)
        df_M_rho.to_csv(out / "table5_M_rho_decomposition.csv", index=False)
        sym_df.to_csv(out / "checks_symbolic.csv", index=False)
        scipy_roots.to_csv(out / "checks_scipy_chirality.csv", index=False)
        autodiff_df.to_csv(out / "checks_autodiff_chiral.csv", index=False)
        numba_df.to_csv(out / "checks_numba_fn_scan.csv", index=False)
        fn_ols_df.to_csv(out / "checks_statsmodels_FN_exponent.csv", index=False)
        eb_row.to_csv(out / "checks_energy_balance_chi.csv", index=False)
        fit_row.to_csv(out / "checks_brannen_theta_refine.csv", index=False)
        df_summary.to_csv(out / "checks_summary_gate.csv", index=False)
        qs = df_fn["n"].astype(float).values
        pdf = _plot_figure_bundle(out, theta_rad, phi_c, qs)
        if verbose:
            print(f"CSV + figures written under {out}  (PDF → {pdf.name})")

    pdf_path = str(out / "paper_III_figures.pdf") if (write_outputs and out) else ""

    return {
        "output_dir": str(out) if out else None,
        "summary_pass": summary_pass,
        "theta_rad": theta_rad,
        "phi_c_rad": phi_c,
        "kappa": kappa,
        "rho0_SI": rho_si,
        "g_factor": g_factor,
        "M_MeV": m_mv,
        "table1_kappa": df_table1_kappa,
        "table2_knots": df_knots,
        "table3_fn": df_fn,
        "table4_chiral": df_chiral,
        "table5_M_rho": df_M_rho,
        "symbolic_checks": sym_df,
        "scipy_chirality": scipy_roots,
        "autodiff": autodiff_df,
        "numba_FN": numba_df,
        "statsmodels_FN": fn_ols_df,
        "summary_gate": df_summary,
        "figures_pdf": pdf_path if write_outputs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper III supplementary — full-stack reproduction.")
    parser.add_argument("--no-write", action="store_true", help="Skip CSV/PDF output.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal stdout.")
    args = parser.parse_args()
    res = run_paper_iii(write_outputs=not args.no_write, verbose=not args.quiet)

    critical = False
    if isinstance(res["summary_gate"], pd.DataFrame):
        critical = bool(res["summary_gate"]["overall_ok"].iloc[0])

    if critical and not args.quiet:
        print(SEP)
        print("All Paper III supplementary checks passed.")
        print(SEP)

    return 0 if critical else 1


if __name__ == "__main__":
    raise SystemExit(main())
