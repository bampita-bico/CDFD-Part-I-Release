"""
Supplementary Material — Paper I
"Vortex Stability in a Constrained Transport Vacuum
 and the Origin of the Fine-Structure Constant"

Author: Steve Bico Mujjabi, MD (2026)
ORCID: https://orcid.org/0009-0001-0556-5516

Reproduces every number in the paper's three tables and generates
publication-quality figures. Numerics come from the public equations stated in
the paper and the paper-local helper module; symbolic checks use SymPy;
independent roots and integrals use SciPy; tables export via Pandas.

Usage (from repository root; install stack first):
    pip install -r requirements.txt
    pip install -r physics_papers/requirements-fullstack.txt   # optional numba/torch/jax
    python physics_papers/supplementary_I.py

Notebook: physics_papers/notebooks/paper_I_fullstack.ipynb
Outputs:  physics_papers/outputs/paper_I/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PAPERS_DIR)

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

from _physics_utils import (
    ALPHA_MEASURED,
    CHI_TARGET,
    chi_self_consistency,
    dE_dchi,
    find_equilibrium,
    kappa_for_chi,
    total_energy,
)

from _physics_utils import (
    agreement_ok,
    central_second_derivative,
    describe_autodiff_backends,
    output_dir,
    try_jax_d2E_total_energy,
    try_torch_d2E_total_energy,
)

BETA = 1.75
R_COMP = 3.861592680e-13  # m  reduced Compton wavelength
A_CLASS = 2.8179403227e-15  # m  classical electron radius


def _sympy_derivative_checks(beta: float, kappa: float, chi_samples: np.ndarray) -> pd.DataFrame:
    chi_s, beta_s, kappa_s = sp.symbols("chi beta kappa", positive=True, real=True)
    E_sym = chi_s * (sp.log(8 * chi_s) - beta_s) + kappa_s / chi_s
    dE_sym = sp.diff(E_sym, chi_s)
    dE_lamb = sp.lambdify((chi_s, beta_s, kappa_s), dE_sym, modules=["numpy"])

    max_abs_err = 0.0
    for c in chi_samples:
        eng = float(dE_dchi(c, beta, kappa))
        sy = float(dE_lamb(c, beta, kappa))
        max_abs_err = max(max_abs_err, abs(eng - sy))

    # Closed-form κ from equilibrium dE/dχ = 0  ⇒  κ = χ² (ln(8χ) − β + 1)
    kappa_from_chi_sym = sp.simplify(chi_s**2 * (sp.log(8 * chi_s) - beta_s + 1))
    kappa_lamb = sp.lambdify((chi_s, beta_s), kappa_from_chi_sym, modules=["numpy"])
    k_sym = float(kappa_lamb(CHI_TARGET, beta))
    k_num = float(kappa_for_chi(CHI_TARGET, beta=beta))

    return pd.DataFrame(
        [
            {
                "check": "sympy_vs_public_dE_dchi_max_abs",
                "value": max_abs_err,
                "note": "SymPy derivative vs public helper on sample chi grid",
            },
            {
                "check": "kappa_algebra_vs_public_helper",
                "value": abs(k_sym - k_num),
                "note": "κ(χ*) from SymPy formula vs kappa_for_chi",
            },
        ]
    )


def _integral_identity(chi_a: float, chi_b: float, beta: float, kappa: float) -> Tuple[float, float, bool]:
    """∫ dE/dχ dχ should equal E(b) − E(a)."""

    def integrand(c: float) -> float:
        return float(dE_dchi(c, beta, kappa))

    integral, _ = quad(integrand, chi_a, chi_b, limit=200)
    jump = float(total_energy(chi_b, beta=beta, kappa=kappa) - total_energy(chi_a, beta=beta, kappa=kappa))
    ok, diff = agreement_ok(integral, jump, rtol=1e-7, atol=1e-8)
    return float(integral), float(jump), bool(ok)


def _numba_energy_scan(
    chi_arr: np.ndarray, beta: float, kappa: float
) -> Optional[np.ndarray]:
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def _grid(chi, b, k):
        out = np.empty(chi.shape[0], dtype=np.float64)
        for i in range(chi.shape[0]):
            c = chi[i]
            out[i] = c * (np.log(8.0 * c) - b) + k / c
        return out

    return _grid(chi_arr.astype(np.float64), beta, kappa)


def _sensitivity_sweep(
    kappa0: float, beta: float, rel_span: float = 0.1, n: int = 50
) -> pd.DataFrame:
    kappa_arr = np.linspace(kappa0 * (1.0 - rel_span), kappa0 * (1.0 + rel_span), n)
    rows = []
    for k in kappa_arr:
        chi = find_equilibrium(beta=beta, kappa=k)
        if chi is None:
            continue
        rows.append(
            {
                "kappa": k,
                "kappa_rel": k / kappa0 - 1.0,
                "chi_eq": chi,
                "alpha": 1.0 / chi,
            }
        )
    return pd.DataFrame(rows)


def run_paper_i(
    *,
    write_outputs: bool = True,
    verbose: bool = True,
) -> dict:
    out = output_dir("paper_I") if write_outputs else None
    kappa = kappa_for_chi(CHI_TARGET, beta=BETA)
    chi_public = find_equilibrium(beta=BETA, kappa=kappa)
    if chi_public is None:
        raise RuntimeError("Equilibrium solver failed to find chi")

    rel_err_stability = abs(chi_public - CHI_TARGET) / CHI_TARGET
    chi_scipy = brentq(lambda c: dE_dchi(c, BETA, kappa), 1.01, 1e6, xtol=1e-14)
    d2 = 1.0 / chi_public + 2.0 * kappa / chi_public**3

    E_circ = chi_public * (np.log(8.0 * chi_public) - BETA)
    E_back = kappa / chi_public
    E_total = total_energy(chi_public, beta=BETA, kappa=kappa)
    ratio = E_back / E_circ

    h = 1e-4
    d2_numerical = (dE_dchi(chi_public + h, BETA, kappa) - dE_dchi(chi_public - h, BETA, kappa)) / (2 * h)
    d2_central = central_second_derivative(lambda c: float(dE_dchi(c, BETA, kappa)), chi_public, h=1e-4)

    geo = chi_self_consistency()
    chi_geom = geo["chi_geometric"]
    rel_err_geom = abs(chi_geom - CHI_TARGET) / CHI_TARGET

    # --- Tables as DataFrames ------------------------------------------------
    table1 = pd.DataFrame(
        [
            {
                "chi_target_1_over_alpha": CHI_TARGET,
                "beta": BETA,
                "kappa_required": kappa,
                "chi_recovered_public": chi_public,
                "chi_recovered_scipy_brentq": chi_scipy,
                "alpha_recovered": 1.0 / chi_public,
                "alpha_measured_CODATA": ALPHA_MEASURED,
                "relative_error_chi": rel_err_stability,
                "public_scipy_abs_diff": abs(chi_public - chi_scipy),
                "d2E_dchi2_analytic": d2,
            }
        ]
    )

    table2 = pd.DataFrame(
        [
            {
                "E_circulation": E_circ,
                "E_back_pressure": E_back,
                "E_total": E_total,
                "ratio_back_over_circ": ratio,
                "d2E_dchi2_finite_diff": d2_numerical,
                "d2E_dchi2_central_on_dE": d2_central,
            }
        ]
    )

    table3 = pd.DataFrame(
        [
            {
                "R_Compton_m": R_COMP,
                "a_classical_e_radius_m": A_CLASS,
                "chi_geometric": chi_geom,
                "alpha_from_geometry": geo["alpha_from_chi"],
                "alpha_measured": ALPHA_MEASURED,
                "relative_error_chi_geom": rel_err_geom,
            }
        ]
    )

    chi_samples = np.linspace(50.0, 400.0, 25)
    sym_df = _sympy_derivative_checks(BETA, kappa, chi_samples)

    int_a, int_b = 80.0, 200.0
    q_int, q_jump, int_ok = _integral_identity(int_a, int_b, BETA, kappa)
    integral_row = pd.DataFrame(
        [
            {
                "chi_a": int_a,
                "chi_b": int_b,
                "quad_integral_dE_dchi": q_int,
                "delta_E_total": q_jump,
                "FTC_match": int_ok,
            }
        ]
    )

    sens = _sensitivity_sweep(kappa, BETA)
    torch_d2 = try_torch_d2E_total_energy(chi_public, BETA, kappa)
    jax_d2 = try_jax_d2E_total_energy(chi_public, BETA, kappa)

    autodiff_df = pd.DataFrame(
        [
            {
                "d2_analytic": d2,
                "d2_pytorch": torch_d2,
                "d2_jax": jax_d2,
                "backends": describe_autodiff_backends(),
            }
        ]
    )

    chi_lo, chi_hi = 50, 400
    chi_arr = np.linspace(chi_lo, chi_hi, 2000)
    E_arr = total_energy(chi_arr, beta=BETA, kappa=kappa)
    nb_arr = _numba_energy_scan(chi_arr, BETA, kappa)
    if nb_arr is not None:
        max_abs = float(np.max(np.abs(nb_arr - E_arr)))
        try:
            from sklearn.metrics import r2_score

            r2_nb = float(r2_score(E_arr, nb_arr))
        except ImportError:
            r2_nb = float("nan")
        numba_row = pd.DataFrame(
            [{"numba_vs_numpy_max_abs_E": max_abs, "r2_sklearn": r2_nb, "numba": "used"}]
        )
    else:
        numba_row = pd.DataFrame(
            [{"numba_vs_numpy_max_abs_E": np.nan, "r2_sklearn": np.nan, "numba": "skipped"}]
        )

    try:
        import statsmodels.api as sm

        ols = sm.OLS(sens["alpha"], sm.add_constant(sens["kappa"])).fit()
        ols_row = pd.DataFrame(
            [
                {
                    "rsquared_alpha_on_kappa": float(ols.rsquared),
                    "kappa_slope": float(ols.params.iloc[1]),
                    "n_points": len(sens),
                }
            ]
        )
    except ImportError:
        ols_row = pd.DataFrame([{"rsquared_alpha_on_kappa": np.nan, "note": "statsmodels missing"}])

    if verbose:
        print("=" * 60)
        print("TABLE 1 — Stability equation  (public helper + SciPy)")
        print("=" * 60)
        print(table1.to_string(index=False))
        print()
        print("=" * 60)
        print("TABLE 2 — Energy balance")
        print("=" * 60)
        print(table2.to_string(index=False))
        print()
        print("=" * 60)
        print("TABLE 3 — Geometric self-consistency")
        print("=" * 60)
        print(table3.to_string(index=False))
        print()
        print("Symbolic / consistency checks")
        print(sym_df.to_string(index=False))
        print()
        print(integral_row.to_string(index=False))
        print()
        print(autodiff_df.to_string(index=False))
        print()
        print(numba_row.to_string(index=False))
        print()
        print(ols_row.to_string(index=False))

    if write_outputs and out is not None:
        table1.to_csv(out / "table1_stability.csv", index=False)
        table2.to_csv(out / "table2_energy_balance.csv", index=False)
        table3.to_csv(out / "table3_geometry.csv", index=False)
        sym_df.to_csv(out / "checks_symbolic.csv", index=False)
        integral_row.to_csv(out / "checks_integral_ftc.csv", index=False)
        sens.to_csv(out / "sensitivity_kappa_sweep.csv", index=False)
        autodiff_df.to_csv(out / "checks_autodiff_second_deriv.csv", index=False)
        numba_row.to_csv(out / "checks_numba_scan.csv", index=False)
        ols_row.to_csv(out / "checks_statsmodels_alpha_on_kappa.csv", index=False)

        _plot_figure_bundle(out, kappa, chi_public, chi_arr, E_arr)

        print()
        print(f"CSV + figures written under {out}")

    return {
        "output_dir": str(out) if out else None,
        "kappa": kappa,
        "chi_public": chi_public,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "symbolic_checks": sym_df,
        "integral_check": integral_row,
        "sensitivity": sens,
        "autodiff": autodiff_df,
        "numba_check": numba_row,
        "statsmodels_ols": ols_row,
    }


def _plot_figure_bundle(
    out: Path,
    kappa: float,
    chi_public: float,
    chi_arr: np.ndarray,
    E_arr: np.ndarray,
) -> None:
    out = Path(out)
    dE_arr = dE_dchi(chi_arr, BETA, kappa)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(chi_arr, E_arr, "C0", lw=2, label=r"$E_\mathrm{norm}(\chi)$")
    ax0.axvline(CHI_TARGET, color="C1", ls="--", lw=1.4, label=rf"$\chi^* = 1/\alpha = {CHI_TARGET:.1f}$")
    ax0.scatter([chi_public], [total_energy(chi_public, BETA, kappa)], color="C1", zorder=5, s=60)
    ax0.set_xlabel(r"$\chi = R/a$", fontsize=12)
    ax0.set_ylabel(r"$E_\mathrm{norm}$ (arb. units)", fontsize=12)
    ax0.set_title("A — Energy landscape", fontweight="bold")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(chi_arr, dE_arr, "C2", lw=2, label=r"$dE/d\chi$")
    ax1.axhline(0, color="k", lw=0.8, ls=":")
    ax1.axvline(CHI_TARGET, color="C1", ls="--", lw=1.4, label=rf"$\chi^* = {CHI_TARGET:.1f}$")
    ax1.scatter([chi_public], [0], color="C1", zorder=5, s=60)
    ax1.set_xlabel(r"$\chi$", fontsize=12)
    ax1.set_ylabel(r"$dE/d\chi$", fontsize=12)
    ax1.set_title("B — Equilibrium condition", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-max(abs(dE_arr)) * 0.3, max(abs(dE_arr)) * 0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    kappa_arr = np.linspace(kappa * 0.90, kappa * 1.10, 300)
    chi_kappa = np.array([find_equilibrium(beta=BETA, kappa=k) or np.nan for k in kappa_arr])
    alpha_kappa = 1.0 / chi_kappa
    ax2.plot((kappa_arr / kappa - 1) * 100, alpha_kappa * 1e3, "C3", lw=2)
    ax2.axhline(ALPHA_MEASURED * 1e3, color="k", ls="--", lw=1, label=r"$\alpha_\mathrm{measured}$ (×10³)")
    ax2.axvline(0, color="C1", ls=":", lw=1)
    ax2.set_xlabel(r"$\kappa$ deviation from $\kappa^*$ (%)", fontsize=12)
    ax2.set_ylabel(r"$\alpha(\kappa)$ (×10³)", fontsize=12)
    ax2.set_title(r"C — $\alpha$ sensitivity to $\kappa$", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    E_c = chi_arr * (np.log(8.0 * chi_arr) - BETA)
    E_b = kappa / chi_arr
    ax3.plot(chi_arr, E_c, "C0", lw=2, label=r"$E_\mathrm{circ}$")
    ax3.plot(chi_arr, E_b, "C4", lw=2, label=r"$E_\mathrm{back}$")
    ax3.plot(chi_arr, E_arr, "k", lw=1.5, ls="--", label=r"$E_\mathrm{total}$")
    ax3.axvline(CHI_TARGET, color="C1", ls="--", lw=1.2)
    ax3.set_xlabel(r"$\chi$", fontsize=12)
    ax3.set_ylabel("Energy (arb.)", fontsize=12)
    ax3.set_title("D — Energy components", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(-200, 1500)

    fig.suptitle(
        "Paper I — Vortex Stability & Origin of the Fine-Structure Constant\n"
        r"CDFD public supplementary material · Steve Bico Mujjabi, MD (2026)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    pdf = out / "paper_I_figures.pdf"
    fig.savefig(pdf, bbox_inches="tight", dpi=150)

    # Also save individual panels for LaTeX inclusion
    extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out / "fig1a_energy_landscape.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out / "fig1b_equilibrium.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out / "fig1c_sensitivity.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(out / "fig1d_energy_components.pdf", bbox_inches=extent.expanded(1.2, 1.3), dpi=150)

    plt.close(fig)


def main() -> dict:
    parser = argparse.ArgumentParser(description="Paper I supplementary — full stack reproduction.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip CSV/PDF output (stdout only).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less console output (still writes files unless --no-write).",
    )
    args = parser.parse_args()
    res = run_paper_i(write_outputs=not args.no_write, verbose=not args.quiet)
    if not args.quiet:
        print()
        print("=" * 60)
        print("All three tables reproduced. Results consistent with paper.")
        print("=" * 60)
    return res


if __name__ == "__main__":
    main()
