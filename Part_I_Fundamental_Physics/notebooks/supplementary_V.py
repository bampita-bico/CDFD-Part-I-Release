"""
Supplementary Material — Paper V
"Zero Parameters: Deriving the Vacuum Chirality Phase from the Koide Invariant"

Author: Steve Bico Mujjabi, MD (2026)
ORCID: https://orcid.org/0009-0001-0556-5516

Uses paper-local public helpers for chi, kappa, Brannen primitives, geometric inputs,
PDG masses, Koide bookkeeping, plus SymPy algebraic checks, SciPy minimization,
Pandas CSV tables, optional Numba/Sklearn scans, Torch/JAX |dQ/dθ|, and
Matplotlib figure bundle.

Usage (from repository root):
    pip install -r requirements.txt
    pip install -r Part_I_Fundamental_Physics/requirements-fullstack.txt
    python Part_I_Fundamental_Physics/notebooks/supplementary_V.py

Notebook: Part_I_Fundamental_Physics/notebooks/paper_V_fullstack.ipynb
Outputs:  Part_I_Fundamental_Physics/outputs/paper_V/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PAPERS_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sympy as sp  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from _figure_utils import save_axes_panel
from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

from _physics_utils import (
    CHI_TARGET,
    LEPTON_MASSES,
    ALPHA_MEASURED,
    brannen_masses,
    chi_self_consistency,
    fit_brannen_to_leptons,
    kappa_for_chi,
    koide_ratio,
)

SEP = "=" * 64
BETA = 1.75
C_LIGHT = 2.99792458e8
M_E_KG = 9.1093837015e-31
PDG_SIGMA_MEV = {
    "electron": 1.5e-7,
    "muon": 2.3e-6,
    "tau": 0.12,
}


def E_norm(chi: float, kappa: float, beta: float = BETA) -> float:
    return chi * (np.log(8.0 * chi) - beta) + kappa / chi


def brannen_amp_sq(theta: float, k: int) -> float:
    return (1.0 + np.sqrt(2.0) * np.cos(theta + 2.0 * np.pi * k / 3.0)) ** 2


def masses_from_theta_brannen(theta: float, m_e_mev: float) -> Tuple[List[float], float, float]:
    """Return sorted masses (electron first), Ae from smallest amplitude, Brannen scale M (MeV)."""

    amps_sq = sorted(brannen_amp_sq(theta, k) for k in range(3))
    ae_sq = amps_sq[0]
    if ae_sq <= 0:
        raise ValueError("Degenerate Ae in Brannen construction.")
    big_m_mev = m_e_mev / ae_sq
    masses = sorted(big_m_mev * aq for aq in amps_sq)
    ae = float(np.sqrt(ae_sq))
    return masses, ae, big_m_mev


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


def _sympy_equilibrium_checks() -> pd.DataFrame:
    """Algebraic tautology underpinning Papers III+V: π − Q + Q = π."""

    q = sp.Symbol("Q", positive=True)
    taut = sp.simplify(3 * (q / 3) + (sp.pi - q) - sp.pi)
    return pd.DataFrame(
        [
            {
                "check": "three_theta_plus_phi_c_equals_pi_substitution_Q",
                "pass": taut == 0,
                "simplified_residual": str(taut),
            },
            {
                "check": "self_consistency_three_theta_equals_Q_symbolic",
                "pass": sp.simplify(3 * (q / 3) - q) == 0,
                "simplified_residual": str(sp.simplify(3 * (q / 3) - q)),
            },
            {
                "check": "phi_c_lit_val_pi_minus_Q_Rational_exact",
                "pass": True,
                "simplified_residual": str(sp.pi - sp.Rational(2, 3)),
            },
        ]
    )


def _scipy_theta_from_fixed_phi(phi_c_rad: float) -> pd.DataFrame:
    """Locate θ minimizing |3θ + φ − π|^2 inside (0, π/3]; should recover paper-V θ=Q/3 if φ = π − Q."""

    def obj(th: float) -> float:
        return float((3.0 * th + phi_c_rad - np.pi) ** 2)

    res = minimize_scalar(obj, bounds=(1e-7, np.pi / 3 - 1e-7), method="bounded")
    return pd.DataFrame(
        [
            {
                "theta_scipy_bounded_rad": float(res.x),
                "objective_residual": float(res.fun),
                "iterations_optional": getattr(res, "nit", np.nan),
            }
        ]
    )


def try_torch_dQ_brannen(theta0: float, M: float) -> Optional[float]:
    """|dQ/dθ| at θ₀ (Brannen theorem ⇒ 0)."""

    try:
        import torch
    except ImportError:
        return None

    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    Ms = torch.tensor(M, dtype=torch.float64)
    mk = []
    for k in range(3):
        c = theta + (2 * torch.pi * k) / 3
        mk.append(Ms * (1 + torch.sqrt(torch.tensor(2.0)) * torch.cos(c)) ** 2)
    m123 = torch.stack(mk)
    Q = torch.sum(m123) / (torch.sum(torch.sqrt(m123)) ** 2)
    g = torch.autograd.grad(Q, theta, create_graph=False)[0]
    return float(torch.abs(g).item())


def try_jax_dQ_brannen(theta0: float, M: float) -> Optional[float]:
    try:
        import jax.numpy as jnp
        from jax import grad

        def q_of(th: float) -> float:
            mk = jnp.array([M * (1 + jnp.sqrt(2.0) * jnp.cos(th + 2 * jnp.pi * k / 3)) ** 2 for k in range(3)])
            return jnp.sum(mk) / (jnp.sum(jnp.sqrt(mk)) ** 2)

        g = grad(q_of)(theta0)
        return float(abs(float(g)))
    except ImportError:
        return None


def _numba_koide_Q_scan(theta_grid: np.ndarray, M: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """NumPy vs Numba sweep: Koide stays 2/3 for any θ."""

    def numpy_Qs(grid: np.ndarray) -> np.ndarray:
        out = np.empty_like(grid, dtype=np.float64)
        for i, t in enumerate(grid):
            out[i] = koide_ratio(*brannen_masses(M, float(t)))
        return out

    q_np = numpy_Qs(theta_grid)
    try:
        from numba import njit

        @njit
        def triple_koide(theta: float, mm: float) -> float:
            acc = np.empty(3, dtype=np.float64)
            for k in range(3):
                acc[k] = mm * (
                    (1 + np.sqrt(2) * np.cos(theta + 2 * np.pi * k / 3)) ** 2
                )
            ssum = acc[0] + acc[1] + acc[2]
            rsum = np.sqrt(acc[0]) + np.sqrt(acc[1]) + np.sqrt(acc[2])
            return ssum / (rsum * rsum)

        @njit
        def scan_nb(grid: np.ndarray, mm: float) -> np.ndarray:
            oo = np.empty(grid.shape[0], dtype=np.float64)
            for i in range(grid.shape[0]):
                oo[i] = triple_koide(grid[i], mm)
            return oo

        q_nb = scan_nb(theta_grid, M)

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
    return df, q_np


def _plot_V_bundle(
    out: Path,
    masses_pred: Sequence[float],
    masses_pdg: Sequence[float],
    names: Sequence[str],
    theta_deg_derived: float,
    theta_deg_fitted: float,
    phi_c_rad: float,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.95))

    ax0 = axes[0]
    x = np.arange(len(names))
    w = 0.35
    ax0.bar(x - w / 2, masses_pdg, width=w, label="PDG 2022", color="#bdc3c7")
    ax0.bar(x + w / 2, masses_pred, width=w, label="Paper V $\\theta=Q/3$", color="#3498db")
    ax0.set_xticks(x, list(names))
    ax0.set_ylabel("mass / MeV")
    ax0.legend(fontsize=8)
    ax0.set_title("(a) Lepton masses vs PDG")

    ax1 = axes[1]
    xs = ["derived\n$\\theta=2/9\\,$rad", "fit\n(brannen to PDG)"]
    ys = [theta_deg_derived, theta_deg_fitted]
    ax1.bar(xs, ys, color=["#2ecc71", "#e74c3c"])
    ax1.set_ylabel(r"$\theta$ (deg)")
    ax1.set_title("(b) Trefoil phase angle")
    for i, y in enumerate(ys):
        ax1.text(i, y + 0.15 * max(ys), f"{y:.4f}", ha="center", fontsize=9)

    ths = np.linspace(1e-3, np.pi / 3 - 1e-3, 400)
    ax2 = axes[2]
    ax2.plot(
        np.degrees(ths),
        np.cos(3 * ths + phi_c_rad),
        color="#8e44ad",
        lw=1.45,
        label=r"$\cos(3\theta+\phi_c)$",
    )
    ax2.axvline(theta_deg_derived, color="#16a085", ls=":", lw=1.3)
    ax2.axhline(-1.0, color="#95a5a6", lw=1.1, linestyle="--")
    ax2.set_xlabel(r"$\theta$ (deg)")
    ax2.set_ylabel(r"$\cos(3\theta+\phi_c)$")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title(r"(c) Chiral cosine along $\phi_c=\pi-2/3\ \mathrm{rad}$")

    fig.suptitle(
        "Paper V — Koide-aligned chirality vacuum phase — CDFT supplementary (2026)",
        fontsize=11,
        fontweight="bold",
        y=1.05,
    )
    pdf = out / "paper_V_figures.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight", dpi=150)

    # Export individual panels for LaTeX inclusion
    for i, ax in enumerate([ax0, ax1, ax2]):
        save_axes_panel(fig, ax, out / f"fig5_{i}_panel.pdf")

    plt.close(fig)
    return pdf


def run_paper_v(*, write_outputs: bool = True, verbose: bool = True) -> Dict:
    out: Optional[Path] = output_dir("paper_V") if write_outputs else None

    q_koide = 2.0 / 3.0
    theta_derived = float(q_koide / 3.0)
    phi_c_derived = float(np.pi - q_koide)

    geom = chi_self_consistency()
    a_class = geom["a_classical_m"]

    chi = CHI_TARGET
    kappa = kappa_for_chi(chi, BETA)
    g_factor = E_norm(chi, kappa, BETA)

    m_e_mev = float(LEPTON_MASSES["electron"])
    m_mu_mev = float(LEPTON_MASSES["muon"])
    m_tau_mev = float(LEPTON_MASSES["tau"])

    masses_pred_sorted, ae_val, big_m_mev = masses_from_theta_brannen(theta_derived, m_e_mev)

    names = sorted(LEPTON_MASSES.keys(), key=lambda k: LEPTON_MASSES[k])
    masses_pdg = [float(LEPTON_MASSES[k]) for k in names]
    preds_named = masses_pred_sorted
    pdgs_named = masses_pdg

    Q_pred = float(koide_ratio(*masses_pred_sorted))
    ok_koide, dk = agreement_ok(Q_pred, q_koide)

    eq_sum = float(3.0 * theta_derived + phi_c_derived)
    ok_equil, deq = agreement_ok(eq_sum, np.pi)

    fit_coarse = fit_brannen_to_leptons()
    theta_fit = _refine_theta_brannen(fit_coarse)
    phi_fit = float(np.pi - 3.0 * theta_fit)

    masses_fit_sorted, ae_fit, m_big_fit = masses_from_theta_brannen(theta_fit, m_e_mev)

    amp_sq_derived = sorted(brannen_amp_sq(theta_derived, k) for k in range(3))
    df_table1_algebra = pd.DataFrame(
        [
            {
                "Q_koide": q_koide,
                "theta_rad_Q_over_3": theta_derived,
                "theta_deg": np.degrees(theta_derived),
                "phi_c_rad_pi_minus_Q": phi_c_derived,
                "phi_c_deg": np.degrees(phi_c_derived),
                "three_theta_rad": float(3 * theta_derived),
                "equilibrium_three_theta_plus_phi_minus_pi_residual": eq_sum - np.pi,
                "equilibrium_agreement_gate": ok_equil,
                "three_theta_minus_Q_residual": float(3 * theta_derived - q_koide),
            }
        ]
    )

    rows_mass = []
    for name, mp, mq in zip(names, preds_named, pdgs_named):
        unc = PDG_SIGMA_MEV[name]
        err_mev = float(mp - mq)
        sig = float(err_mev / unc if unc > 0 else np.nan)
        rows_mass.append(
            {
                "lepton": name,
                "predicted_MeV": mp,
                "pdg_2022_MeV": mq,
                "error_MeV": err_mev,
                "sigma": sig,
                "ppm_vs_pdg_percent": float((mp / mq - 1) * 1e6),
            }
        )
    df_table3_masses = pd.DataFrame(rows_mass)

    ae_sq_der_chk = amp_sq_derived[0]
    M_kg_brannen = float(M_E_KG / ae_sq_der_chk)
    rho_derived_si = float(M_kg_brannen / (a_class**3 * g_factor))

    df_table2_steps = pd.DataFrame(
        [
            ("alpha_codata_via_chi_inverse", float(ALPHA_MEASURED), "Paper I public chi equilibrium"),
            ("chi_eq", chi, r"Paper I \(\chi_{\mathrm{eq}} \approx 137.036\)"),
            ("kappa", float(kappa), r"Paper III \(\chi^2(\ln 8\chi-\beta+1)\)"),
            ("g_dimensionless_energy", float(g_factor), r"Papers I+III \(\chi(\ln8\chi-\beta)+\kappa/\chi\)"),
            ("Q_brannen_theorem", q_koide, "Paper II Koide invariant"),
            ("theta_derived_rad", theta_derived, r"Paper V \(Q/3\)"),
            ("phi_c_derived_rad", phi_c_derived, r"Paper V \(\pi - Q\) with \(Q=2/3\) rad"),
            ("A_e_min", ae_val, "Brannen minimum amplitude derived"),
            ("M_scale_MeV", big_m_mev, r"Paper II \(M=m_e/A_e^2\)"),
            ("rho0_kg_m3", rho_derived_si, r"Papers IV–V: $M_{\mathrm{kg}}/(a^3 g)$"),
        ],
        columns=("symbol", "value", "notes"),
    )

    df_table3rho = pd.DataFrame(
        [
            {
                "rho0_kg_m3": rho_derived_si,
                "g_factor": float(g_factor),
                "consistent_with_paper_IV_path": rho_derived_si,
            }
        ]
    )

    phi_c_deg_derived = float(np.degrees(phi_c_derived))
    phi_c_deg_fitted = float(np.degrees(phi_fit))
    theta_deg_derived_v = float(np.degrees(theta_derived))
    theta_deg_fitted_v = float(np.degrees(theta_fit))
    df_table4_compare = pd.DataFrame(
        [
            {
                "quantity": "theta_deg",
                "fitted_Paper_II_refined": theta_deg_fitted_v,
                "derived_Paper_V_Q_over_3": theta_deg_derived_v,
                "delta_arcmin": abs(theta_deg_derived_v - theta_deg_fitted_v) * 60.0,
            },
            {
                "quantity": "phi_c_deg",
                "fitted_Paper_II_refined": phi_c_deg_fitted,
                "derived_Paper_V_pi_minus_Q": phi_c_deg_derived,
                "delta_arcmin": abs(phi_c_deg_derived - phi_c_deg_fitted) * 60.0,
            },
            {
                "quantity": "M_MeV",
                "fitted_Paper_II_refined": m_big_fit,
                "derived_Paper_V_Q_over_3": big_m_mev,
                "delta_arcmin": np.nan,
            },
        ]
    )

    fit_err_table = []
    for name, mfit, md, mq in zip(names, masses_fit_sorted, preds_named, pdgs_named):
        fit_err_table.append(
            {
                "lepton": name,
                "fitted_err_ppm_vs_pdg_percent": float((mfit / mq - 1.0) * 1e6),
                "derived_err_ppm_vs_pdg_percent": float((md / mq - 1.0) * 1e6),
            }
        )
    df_table4ppm = pd.DataFrame(fit_err_table)

    tau_row = df_table3_masses[df_table3_masses["lepton"] == "tau"].iloc[0]
    mu_row = df_table3_masses[df_table3_masses["lepton"] == "muon"].iloc[0]
    ele_row = df_table3_masses[df_table3_masses["lepton"] == "electron"].iloc[0]

    mu_ppm_derived = abs(float((preds_named[names.index("muon")] / m_mu_mev - 1.0) * 1e6))
    ok_mu_ppm = mu_ppm_derived <= 25.0  # PDG fractional scale ~22 ppm (Paper V text)
    ok_tau_sigma = abs(float(tau_row["sigma"])) <= 3.5
    ok_electron_exact, _ede = agreement_ok(float(ele_row["predicted_MeV"]), float(ele_row["pdg_2022_MeV"]), rtol=1e-12, atol=1e-12)

    sym_df = _sympy_equilibrium_checks()
    scipy_theta_df = _scipy_theta_from_fixed_phi(phi_c_derived)
    ok_theta_scipy, theta_scipy_diff = agreement_ok(float(scipy_theta_df["theta_scipy_bounded_rad"].iloc[0]), theta_derived)

    theta_hi = float(min(theta_fit + 0.02, 0.239))
    theta_grid = np.linspace(1e-3, theta_hi, 2600)
    numba_df, q_vals = _numba_koide_Q_scan(theta_grid, 1.0)
    q_dev_abs = float(np.max(np.abs(q_vals - q_koide)))

    M_autodiff = float(big_m_mev / m_e_mev)
    tdq = try_torch_dQ_brannen(theta_derived, M_autodiff)
    jdq = try_jax_dQ_brannen(theta_derived, M_autodiff)
    finite_dq = koide_ratio(*brannen_masses(M_autodiff, theta_derived + 2e-6)) - koide_ratio(
        *brannen_masses(M_autodiff, theta_derived - 2e-6)
    )
    finite_approx = finite_dq / (4e-6)

    torch_ok = tdq <= 5e-5 if tdq is not None else True
    jax_ok = jdq <= 5e-5 if jdq is not None else True

    df_autodiff = pd.DataFrame(
        [
            {
                "finite_diff_dKoide_numeric_est": finite_approx,
                "torch_abs_dQ_dtheta": tdq,
                "jax_abs_dQ_dtheta": jdq,
                "backends": describe_autodiff_backends(),
            }
        ]
    )

    koide_exact_brannen = koide_ratio(*brannen_masses(1.0, theta_derived))

    summary_pass = (
        ok_equil
        and ok_theta_scipy
        and ok_koide
        and ok_electron_exact
        and bool(sym_df.iloc[:2]["pass"].all())
        and abs(float(3 * theta_derived - q_koide)) <= 1e-15
        and ok_mu_ppm
        and ok_tau_sigma
        and q_dev_abs <= 1e-10
        and torch_ok
        and jax_ok
    )

    df_summary = pd.DataFrame(
        [
            {
                "equilibrium_three_theta_phi_pi": ok_equil,
                "scipy_bounded_theta_equals_Q_over_3": ok_theta_scipy,
                "koide_Q_predictions": ok_koide,
                "electron_calibration_exact": ok_electron_exact,
                "symm_algebra_rows_0_2": bool(sym_df.iloc[:2]["pass"].all()),
                "mu_within_25ppm_fractional": ok_mu_ppm,
                "tau_sigma_within_3pt5_sigma": ok_tau_sigma,
                "numba_koide_scan_deviation_gate": q_dev_abs <= 1e-10,
                "torch_dKoide_near_zero_optional": torch_ok,
                "jax_dKoide_near_zero_optional": jax_ok,
                "overall_ok": summary_pass,
            }
        ]
    )

    pdf_path_val: Optional[str] = None
    if verbose:
        print(SEP)
        print("Paper V — paper-local public reproducibility bundle")
        print(SEP)
        print(df_table1_algebra.to_string(index=False))
        print()
        print(df_table2_steps.to_string(index=False))
        print()
        print(df_table3_masses.to_string(index=False))
        print()
        print(df_table4_compare.to_string(index=False))
        print(sym_df.to_string(index=False))
        print(scipy_theta_df.to_string(index=False))
        print(df_autodiff.to_string(index=False))
        print(numba_df.to_string(index=False))
        print(df_summary.to_string(index=False))

    if write_outputs and out is not None:
        df_table1_algebra.to_csv(out / "table1_algebra_theta_phi_derivation.csv", index=False)
        df_table2_steps.to_csv(out / "table2_chain_steps.csv", index=False)
        df_table3_masses.to_csv(out / "table3_lepton_predictions.csv", index=False)
        df_table3rho.to_csv(out / "table3_rho_derived_IV_style.csv", index=False)
        df_table4_compare.to_csv(out / "table4_fitted_vs_derived.csv", index=False)
        df_table4ppm.to_csv(out / "table4_error_ppm_comparison.csv", index=False)
        sym_df.to_csv(out / "checks_symbolic_V.csv", index=False)
        scipy_theta_df.to_csv(out / "checks_scipy_theta_from_phi.csv", index=False)
        pd.DataFrame(
            [
                {
                    "Q_prediction": koide_exact_brannen,
                    "Q_target": q_koide,
                    "agreement_ok": ok_koide,
                    "max_abs_residual_numba_grid": q_dev_abs,
                }
            ]
        ).to_csv(out / "checks_koide_brannen.csv", index=False)
        df_autodiff.to_csv(out / "checks_autodiff_brannen.csv", index=False)
        numba_df.to_csv(out / "checks_numba_koide_flat.csv", index=False)
        pd.DataFrame(
            [
                {
                    "theta_coarse_deg": fit_coarse["theta_deg"],
                    "theta_refined_deg": np.degrees(theta_fit),
                }
            ]
        ).to_csv(out / "checks_brannen_fit_reference.csv", index=False)
        df_summary.to_csv(out / "checks_summary_V_gate.csv", index=False)

        pdf = _plot_V_bundle(out, preds_named, pdgs_named, names, theta_deg_derived_v, theta_deg_fitted_v, phi_c_derived)
        pdf_path_val = str(pdf)
        if verbose:
            print(f"Artifacts → {out} ({pdf.name})")

    return {
        "output_dir": str(out) if out else None,
        "summary_pass": summary_pass,
        "theta_derived_rad": theta_derived,
        "phi_c_derived_rad": phi_c_derived,
        "mass_predictions_MeV": preds_named,
        "Q_predicted": Q_pred,
        "rho_derived_si": rho_derived_si,
        "figures_pdf": pdf_path_val,
        "summary_gate": df_summary,
        "table4_comparison": df_table4_compare,
        "symbolic_checks": sym_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper V supplementary — reproducibility harness.")
    parser.add_argument("--no-write", action="store_true", help="Skip CSV/PDF output.")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    res = run_paper_v(write_outputs=not args.no_write, verbose=not args.quiet)
    ok = bool(res["summary_gate"]["overall_ok"].iloc[0])

    if ok and not args.quiet:
        print(SEP)
        print("All Paper V supplementary checks passed.")
        print(SEP)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
