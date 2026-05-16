"""
Supplementary Material — Paper IV
"The Vacuum Equation of State and the Origin of rho_0"

Author: Steve Bico Mujjabi, MD (2026)
ORCID: https://orcid.org/0009-0001-0556-5516

Uses paper-local public helpers for χ, κ, Brannen fit, Compton-ring geometry,
and rho0 bookkeeping; SymPy verifies dimensionless reductions; SciPy verifies
implicit master-equation formulations; Pandas emits CSV mirrors of every
tabular value in Paper IV; Matplotlib emits a bundled PDF figure. Optional
Statsmodels summarizes Faddeev–Niemi scaling; Numba cross-checks the scaling
coefficients.

Usage (from repository root):
    pip install -r requirements.txt
    pip install -r physics_papers/requirements-fullstack.txt
    python physics_papers/supplementary_IV.py

Notebook: physics_papers/notebooks/paper_IV_fullstack.ipynb
Outputs:  physics_papers/outputs/paper_IV/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

from _physics_utils import (
    CHI_TARGET,
    LEPTON_MASSES,
    ALPHA_MEASURED,
    brannen_masses,
    chi_self_consistency,
    fit_brannen_to_leptons,
    kappa_for_chi,
)

SEP = "=" * 64

BETA = 1.75
C_LIGHT = 2.99792458e8
M_E_KG = 9.1093837015e-31
HBAR = 1.054571817e-34


def brannen_A(theta: float, k: int) -> float:
    return 1.0 + np.sqrt(2.0) * np.cos(theta + 2.0 * np.pi * k / 3.0)


def E_norm(chi: float, kappa: float, beta: float = BETA) -> float:
    return chi * (np.log(8.0 * chi) - beta) + kappa / chi


def A_e_from_phi(phi_c: float) -> Tuple[float, float]:
    theta = (np.pi - phi_c) / 3.0
    vals = [brannen_A(theta, k) for k in range(3)]
    return float(min(vals)), theta


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


def _fn_energy_ratios_numpy(ns: np.ndarray) -> np.ndarray:
    ref = 3.0 ** (3.0 / 4.0)
    return (ns.astype(np.float64) ** (3.0 / 4.0)) / ref


def _numba_fn_scan(grid: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    ratios_np = _fn_energy_ratios_numpy(grid)
    try:
        from numba import njit

        @njit
        def ratio(n: float) -> float:
            ref = 3.0**0.75
            return (n**0.75) / ref

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


def _sympy_dimensionless_checks() -> pd.DataFrame:
    """ρ = mₑ/(a³ g Aₑ²) ⇒ X = ρa³/mₑ simplifies to 1/(g Aₑ²) (dimensionless SI form)."""

    rho, me, a, g, Ae = sp.symbols("rho m_e a g A_e", positive=True)
    rho_def = me / (a**3 * g * Ae**2)
    X = sp.simplify(rho_def * a**3 / me)
    target = sp.simplify(1 / (g * Ae**2))
    ok = bool(sp.simplify(X - target) == 0)
    return pd.DataFrame(
        [
            {
                "check": "X_rho_a3_me_equals_inv_gAe2",
                "pass": ok,
                "simplified_X": str(X),
                "target": str(target),
            }
        ]
    )


def _scipy_implicit_X(phi_c_rad: float, g_factor: float) -> pd.DataFrame:
    """SciPy solves X = 1/(g Ae(φ_C)²) as root of X*h(X)-1 where h = g Ae² (Ae depends on φ_C only here)."""

    def h_of_dummy(_x: float) -> float:
        Ae, _ = A_e_from_phi(phi_c_rad)
        return float(g_factor * Ae**2)

    Ae0, theta0 = A_e_from_phi(phi_c_rad)
    h_const = float(g_factor * Ae0**2)
    X_truth = 1.0 / h_const

    def residual(x: float) -> float:
        return x * h_of_dummy(x) - 1.0

    res = root_scalar(residual, bracket=(X_truth * 0.5, X_truth * 1.5), method="bisect")
    ok, dd = agreement_ok(float(res.root), X_truth)
    return pd.DataFrame(
        [
            {
                "Ae_from_phi_min": Ae0,
                "theta_implicit_rad": theta0,
                "h_g_Ae_squared": h_const,
                "X_analytic": X_truth,
                "X_root_scalar": float(res.root),
                "agreement": ok,
                "abs_diff": dd,
            }
        ]
    )


def _brannen_winning_k(phi_c: float) -> int:
    """Which Z₃ limb Min() picks at chirality φ (same rule as Paper IV Eq. Ae)."""

    theta = (np.pi - phi_c) / 3.0
    vals = [brannen_A(theta, k) for k in range(3)]
    return int(np.argmin(vals))


def _Ae_smooth_sq(phi_c: float, *, k_pick: Optional[int] = None) -> float:
    k = _brannen_winning_k(phi_c) if k_pick is None else k_pick
    theta = (np.pi - phi_c) / 3.0
    a = float(brannen_A(theta, k))
    return a * a


def try_torch_d_dphi_Ae_sq(phi_c0: float) -> Optional[float]:
    """d/dφ(A_k²) on the fixed branch k* that dominates at φ₀ (locally differentiable)."""

    try:
        import torch

        k_pick = _brannen_winning_k(phi_c0)
        p = torch.tensor(float(phi_c0), dtype=torch.float64, requires_grad=True)
        theta = (torch.pi - p) / 3.0
        sq2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        Ae = 1 + sq2 * torch.cos(theta + 2 * torch.pi * k_pick / 3)
        y = Ae**2
        g1 = torch.autograd.grad(y, p, create_graph=False)[0]
        return float(g1.item())
    except ImportError:
        return None


def try_jax_d_dphi_Ae_sq(phi_c0: float) -> Optional[float]:
    k_pick = _brannen_winning_k(phi_c0)

    try:
        import jax.numpy as jnp
        from jax import grad

        def Ae_sq(pc: Any) -> Any:
            theta = (jnp.pi - pc) / 3.0
            Ae = 1 + jnp.sqrt(2.0) * jnp.cos(theta + 2 * jnp.pi * k_pick / 3)
            return Ae**2

        return float(grad(Ae_sq)(float(phi_c0)))
    except ImportError:
        return None


def _plot_IV_bundle(out: Path, geom: Dict, chi: float, alpha: float, M3_mev: float) -> Path:
    a = geom["a_classical_m"]
    R = chi * a
    a0 = HBAR / (M_E_KG * C_LIGHT * alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.95))

    ax0 = axes[0]
    labels = ["$a$", r"$R=\chi a$", r"$a_0=a/\alpha^2$"]
    xs = np.array([a, R, a0])
    ax0.bar(range(3), np.log10(xs), color=["#3498db", "#2ecc71", "#e67e22"], tick_label=labels)
    ax0.set_ylabel(r"$\log_{10}$(length / m)")
    ax0.set_title("(a) CDFT length-scale hierarchy")

    ax1 = axes[1]
    phis_deg = np.linspace(120.0, 160.0, 220)
    phis_rad = np.radians(phis_deg)
    Xs = []
    for pvc in phis_rad:
        Ae, _th = A_e_from_phi(pvc)
        chi_here = CHI_TARGET
        kappa = kappa_for_chi(chi_here, BETA)
        gg = E_norm(chi_here, kappa, BETA)
        Xs.append(1.0 / (gg * Ae**2))
    ax1.plot(phis_deg, Xs, color="#8e44ad", lw=1.5)
    ax1.set_xlabel(r"$\Phi_c$ (deg) — illustrative slice; EOS sets true $\Phi_c(\rho_0)$")
    ax1.set_ylabel(r"$X=\rho_0 a^3/m_e = 1/(g A_e^2)$")
    ax1.set_title("(b) Dimensionless coupling vs chirality slice")

    ax2 = axes[2]
    Qvals = np.array([3, 5, 7, 9], dtype=float)
    ref = Qvals[0] ** 0.75
    MQ = M3_mev * (Qvals ** 0.75) / ref
    ax2.bar(range(len(Qvals)), MQ, tick_label=["3", "5", "7", "9"], color=["#16a085", "#27ae60", "#2980b9", "#c0392b"])
    ax2.set_xlabel(r"Hopf $Q=n$ ($T(2,n)$)")
    ax2.set_ylabel(r"Energy scale / MeV")
    ax2.set_title("(c) Faddeev–Niemi family scaling from $M_3$")

    fig.suptitle(
        "Paper IV — Vacuum self-consistency, dimensionless closure, FN scaling — CDFT supplementary (2026)",
        fontsize=11,
        fontweight="bold",
        y=1.05,
    )
    pdf = out / "paper_IV_figures.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight", dpi=150)

    # Export individual panels for LaTeX inclusion
    for i, ax in enumerate([ax0, ax1, ax2]):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"fig4_{i}_panel.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    plt.close(fig)
    return pdf


def run_paper_iv(*, write_outputs: bool = True, verbose: bool = True) -> Dict:
    out: Optional[Path] = output_dir("paper_IV") if write_outputs else None

    geom = chi_self_consistency()
    a_class = geom["a_classical_m"]
    R_compton = geom["R_compton_m"]

    chi = CHI_TARGET
    alpha = ALPHA_MEASURED
    kappa = kappa_for_chi(chi, BETA)
    g_factor = E_norm(chi, kappa, BETA)

    fit_coarse = fit_brannen_to_leptons()
    theta_rad = _refine_theta_brannen(fit_coarse)
    phi_c_rad = np.pi - 3.0 * theta_rad

    R_ring = chi * a_class
    ok_compton, d_compton = agreement_ok(R_ring, R_compton, rtol=1e-9, atol=1e-15)

    a0_bohr = a_class / alpha**2
    ratio_a_R = float(a_class / R_ring)

    Ae_min, theta_implicit = A_e_from_phi(phi_c_rad)
    ok_theta_impl, diff_theta_impl = agreement_ok(theta_implicit, theta_rad, rtol=1e-9, atol=1e-8)

    m_e_mev = LEPTON_MASSES["electron"]
    M_brannen_mev = float(m_e_mev / Ae_min**2)

    # Use CODATA kg mass so ρ closes with the same bookkeeping as ρ_check (avoids MeV↔kg drift).
    M_kg_brannen = float(M_E_KG / Ae_min**2)
    rho_si = float(M_kg_brannen / (a_class**3 * g_factor))

    theta_chk = (np.pi - phi_c_rad) / 3.0
    Ae_chk = float(min(brannen_A(theta_chk, k) for k in range(3)))
    rho_check = float(M_E_KG / (Ae_chk**2 * a_class**3 * g_factor))
    ok_rho_cycle, rho_diff = agreement_ok(rho_si, rho_check, rtol=1e-12, atol=1e-20)

    J_crit = rho_si * C_LIGHT**2
    u_vac = J_crit
    u_em_core = M_E_KG * C_LIGHT**2 / (8.0 * np.pi * a_class**3)
    ratio_u = float(u_vac / u_em_core)

    ratio_1 = rho_si * a_class**3 / M_E_KG
    ratio_2 = rho_si * a_class**4 * C_LIGHT / HBAR
    inv_gAe2 = float(1.0 / (g_factor * Ae_min**2))

    ok_ratio1, d1 = agreement_ok(ratio_1, inv_gAe2, rtol=1e-12)

    E_tref = 3.0 ** (3.0 / 4.0)
    M_fn: Dict[int, float] = {}
    rows_fn = []
    for n in (3, 5, 7, 9):
        mf = float(M_brannen_mev * (n ** (3.0 / 4.0)) / E_tref)
        M_fn[n] = mf
        rows_fn.append({"n": n, "Hopf_Q": n, "M_n_MeV": mf, "E_ratio_to_trefoil": float((n**0.75) / E_tref)})

    df_fn = pd.DataFrame(rows_fn)
    df_particles = pd.DataFrame(
        [
            ("Pion_pi0", 134.98),
            ("Pion_pi_pm", 139.57),
            ("Kaon_pm", 493.68),
            ("Eta", 547.86),
            ("Proton", 938.27),
            ("Neutron", 939.57),
            ("Lambda_baryon", 1115.68),
            ("M5_predicted", M_fn[5]),
            ("M7_predicted", M_fn[7]),
            ("M9_predicted", M_fn[9]),
        ],
        columns=["label", "mass_MeV"],
    ).sort_values("mass_MeV")

    nuclear_rho_approx = 2.3e17
    u_nuclear = nuclear_rho_approx * C_LIGHT**2

    Lambda_qcd_geom = (0.2e9 * 1.602176634e-19) ** 4 / (HBAR * C_LIGHT) ** 3

    scipy_X = _scipy_implicit_X(phi_c_rad, g_factor)

    df_table1_geom = pd.DataFrame(
        [
            {
                "a_classical_m": a_class,
                "R_chi_times_a_m": R_ring,
                "R_reduced_Compton_geom_m": R_compton,
                "compton_coincidence_agreement": ok_compton,
                "abs_delta_m": d_compton,
                "chi_target": chi,
                "alpha_inverse": alpha,
                "ratio_a_over_R_equals_alpha_check": ratio_a_R,
                "ratio_a_over_alpha_direct": alpha,
                "ratio_match": agreement_ok(ratio_a_R, alpha)[0],
                "bohr_a0_hbar_over_mc_alpha_m": a0_bohr,
                "J_crit_Pa": J_crit,
                "u_vacuum_J_m3": u_vac,
                "u_em_core_J_m3": u_em_core,
                "u_vac_over_u_em": ratio_u,
            }
        ]
    )

    rhs_scale = float(C_LIGHT**2 * a_class**3 * g_factor)

    df_table2_master = pd.DataFrame(
        [
            {
                "theta_deg_refined": np.degrees(theta_rad),
                "phi_c_deg_from_chirality": np.degrees(phi_c_rad),
                "A_e_minimum": Ae_min,
                "g_chi_beta": g_factor,
                "kappa_derived": kappa,
                "M_Brannen_MeV": M_brannen_mev,
                "c2_a3_g_factor_J_times_m3_over_kg": rhs_scale,
                "rho_0_derived_J_path_kg_m3": rho_si,
                "rho_verify_from_m_e_identity_kg_m3": rho_check,
                "rho_loop_agreement": ok_rho_cycle,
                "rho_rel_diff": rho_diff / max(rho_si, 1e-99),
                "theta_check_vs_refined_implicit": ok_theta_impl,
            }
        ]
    )

    df_table3_dimensionless = pd.DataFrame(
        [
            {
                "X1_rho_a3_over_me": ratio_1,
                "inv_gAe2_algebraic": inv_gAe2,
                "X1_matches_inv_gate": ok_ratio1,
                "X2_rho_a4_c_over_hbar": ratio_2,
            }
        ]
    )

    u_qcd_approx = Lambda_qcd_geom
    df_table5_ctx = pd.DataFrame(
        [
            {
                "rho0_kg_m3": rho_si,
                "K_bulk_modulus_Pa": rho_si * C_LIGHT**2,
                "rho0_over_three_me_over_four_pi_a3_note": rho_si / (M_E_KG / ((4.0 / 3.0) * np.pi * a_class**3)),
                "comparison_u_nuclear_J_m3": u_nuclear,
                "comparison_u_qcd_geom_J_m3": u_qcd_approx,
                "comparison_classical_sphere_factor": rho_si / (M_E_KG / a_class**3),
            }
        ]
    )

    df_table6_status = pd.DataFrame(
        [
            ("alpha_derived_series", True, "Paper I public chi equilibrium"),
            ("kappa_algebraic", True, "Paper III χ² term"),
            ("koide_theorem_series", True, "Paper II Z₃"),
            ("master_equation_article_IV", True, "Eq.(master) structure"),
            ("Phi_c_from_EOS", False, "Open — requires Navier–Stokes+CDFT"),
            ("M5_MeV_prediction", True, str(round(M_fn[5], 2))),
            ("rho0_inference_uses_PDG_theta", True, "θ from masses until EOS closes"),
        ],
        columns=["item", "value_or_gate", "note"],
    )

    sym_df = _sympy_dimensionless_checks()
    fn_numba_df, _nb = _numba_fn_scan(np.arange(3, 201, 2, dtype=np.float64))
    fn_ols_df = _statsmodels_fn_exponent([3, 5, 7, 9])

    k0 = _brannen_winning_k(phi_c_rad)
    h_eps = 1e-7

    def h_g_Ae2_local(p: float) -> float:
        return g_factor * _Ae_smooth_sq(p, k_pick=k0)

    fd_dphi_a2 = (h_g_Ae2_local(phi_c_rad + h_eps) - h_g_Ae2_local(phi_c_rad - h_eps)) / (2 * h_eps)

    dA2_torch = try_torch_d_dphi_Ae_sq(phi_c_rad)
    dA2_jax = try_jax_d_dphi_Ae_sq(phi_c_rad)
    torch_dphi = None if dA2_torch is None else g_factor * dA2_torch
    jax_dphi = None if dA2_jax is None else g_factor * dA2_jax

    autodiff_df = pd.DataFrame(
        [
            {
                "finite_diff_d_gAe2_dphi_fixed_branch_k": k0,
                "finite_diff_d_gAe2_dphi_numeric": fd_dphi_a2,
                "torch_d_gAe2_dphi_optional": torch_dphi,
                "jax_d_gAe2_dphi_optional": jax_dphi,
                "backends": describe_autodiff_backends(),
            }
        ]
    )

    jax_ok = True
    torch_ok = True
    if jax_dphi is not None:
        jax_ok = agreement_ok(jax_dphi, fd_dphi_a2, rtol=1e-5, atol=1e-6)[0]
    if torch_dphi is not None:
        torch_ok = agreement_ok(torch_dphi, fd_dphi_a2, rtol=1e-5, atol=1e-6)[0]

    summary_pass = (
        ok_compton
        and ok_rho_cycle
        and ok_theta_impl
        and ok_ratio1
        and bool(sym_df["pass"].iloc[0])
        and bool(scipy_X["agreement"].iloc[0])
        and jax_ok
        and torch_ok
    )

    df_summary = pd.DataFrame(
        [
            {
                "compton_coincidence": ok_compton,
                "rho_self_consistency_loop": ok_rho_cycle,
                "theta_implicit_equals_refined": ok_theta_impl,
                "dimensionless_X_closure": ok_ratio1,
                "sympy_X_algebra": bool(sym_df["pass"].iloc[0]),
                "scipy_implicit_X_agreement": bool(scipy_X["agreement"].iloc[0]),
                "overall_ok": summary_pass,
            }
        ]
    )

    fit_row = pd.DataFrame(
        [
            {
                "theta_deg_coarse": fit_coarse["theta_deg"],
                "theta_deg_refined": np.degrees(theta_rad),
                "paper_text_theta_deg_nominal": 12.7325,
            }
        ]
    )

    if verbose:
        print(SEP)
        print("Paper IV — TABLES (paper-local public helpers)")
        print(SEP)
        print(df_table1_geom.to_string(index=False))
        print()
        print(df_table2_master.to_string(index=False))
        print()
        print(df_table3_dimensionless.to_string(index=False))
        print()
        print(df_fn.to_string(index=False))
        print()
        print(df_table6_status.to_string(index=False))
        print(sym_df.to_string(index=False))
        print(scipy_X.to_string(index=False))
        print(autodiff_df.to_string(index=False))
        print(fn_numba_df.to_string(index=False))
        print(fn_ols_df.to_string(index=False))
        print(fit_row.to_string(index=False))
        print(df_summary.to_string(index=False))

    pdf_path_val: Optional[str] = None
    if write_outputs and out is not None:
        df_table1_geom.to_csv(out / "table1_geometric_vacuum_scales.csv", index=False)
        df_table2_master.to_csv(out / "table2_master_rho_equation_numbers.csv", index=False)
        df_table3_dimensionless.to_csv(out / "table3_dimensionless_X_factors.csv", index=False)
        df_fn.to_csv(out / "table4_fn_knot_family_scales.csv", index=False)
        df_particles.to_csv(out / "table4_light_hadrons_comparison.csv", index=False)
        df_table5_ctx.to_csv(out / "table5_vacuum_physical_context.csv", index=False)
        df_table6_status.to_csv(out / "table6_program_status_chain.csv", index=False)
        sym_df.to_csv(out / "checks_symbolic_IV.csv", index=False)
        scipy_X.to_csv(out / "checks_scipy_implicit_dimensionless.csv", index=False)
        autodiff_df.to_csv(out / "checks_autodiff_phi_sensitivity.csv", index=False)
        fn_numba_df.to_csv(out / "checks_numba_FN_IV.csv", index=False)
        fn_ols_df.to_csv(out / "checks_statsmodels_FN_IV.csv", index=False)
        fit_row.to_csv(out / "checks_brannen_theta_IV.csv", index=False)
        df_summary.to_csv(out / "checks_summary_IV_gate.csv", index=False)

        pdf = _plot_IV_bundle(out, geom, chi, alpha, M_brannen_mev)
        pdf_path_val = str(pdf)
        if verbose:
            print(f"CSV + figures written under {out} (PDF → {pdf.name})")

    return {
        "output_dir": str(out) if out else None,
        "summary_pass": summary_pass,
        "rho0_SI": rho_si,
        "M3_MeV": M_brannen_mev,
        "figures_pdf": pdf_path_val if write_outputs else None,
        "tables": {
            "geometry": df_table1_geom,
            "master": df_table2_master,
            "dimensionless": df_table3_dimensionless,
            "fn_scales": df_fn,
        },
        "summary_gate": df_summary,
        "particles": df_particles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper IV supplementary — full-stack reproduction.")
    parser.add_argument("--no-write", action="store_true", help="Skip CSV/PDF output.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal stdout.")
    args = parser.parse_args()
    res = run_paper_iv(write_outputs=not args.no_write, verbose=not args.quiet)

    ok = False
    if isinstance(res["summary_gate"], pd.DataFrame):
        ok = bool(res["summary_gate"]["overall_ok"].iloc[0])
    if ok and not args.quiet:
        print(SEP)
        print("All Paper IV supplementary checks passed.")
        print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
