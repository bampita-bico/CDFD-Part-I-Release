"""
Supplementary Material - Paper XI
Dark matter, dark energy, black holes, and the Blancken-layer synthesis.

This script replaces the old orphaned/static Paper XI figures with a
reproducible conceptual-output trail. The figures are not observational
evidence; they are deterministic schematics and falsification maps for the
cosmological bridge claims in Paper XI.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PAPERS_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.stats import linregress

from _physics_utils import describe_autodiff_backends, output_dir

SEP = "=" * 64


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def bridge_map(out: Path) -> pd.DataFrame:
    rows = [
        {
            "bridge": "dark_matter",
            "cdfd_kernel": "history-dependent vacuum constraint gradients",
            "observable": "lensing residual after baryon and standard-DM fits",
            "required_discriminant": "residual correlation with merger/shear/plasma history",
            "failure_condition": "no history term improves the residual field",
        },
        {
            "bridge": "dark_energy",
            "cdfd_kernel": "large-scale residual of incomplete vacuum self-consistency",
            "observable": "epoch or scale dependence in effective pressure",
            "required_discriminant": "unit-correct residual map rho_Lambda,eff from Phi-C",
            "failure_condition": "wrong units, unconstrained tuning, or incompatible w(z)",
        },
        {
            "bridge": "black_holes",
            "cdfd_kernel": "transport-capacity saturation boundary",
            "observable": "ringdown, accretion hysteresis, entropy correction",
            "required_discriminant": "recover GR plus one controlled deviation",
            "failure_condition": "no distinction from a relabeling of GR",
        },
        {
            "bridge": "blancken_layer",
            "cdfd_kernel": "pre-geometric constraint substrate",
            "observable": "invariant, selector, or boundary condition",
            "required_discriminant": "derivation of a fixed-point selector used upstream",
            "failure_condition": "no calculable invariant beyond metaphor",
        },
        {
            "bridge": "mass_topology",
            "cdfd_kernel": "stable masses as topological constraint states",
            "observable": "fixed mass ratios and family selectors",
            "required_discriminant": "precision comparison and independent review",
            "failure_condition": "selectors must be fitted independently per family",
        },
    ]
    df = pd.DataFrame(rows)
    _save_csv(df, out / "table1_cosmology_bridge_map.csv")

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.axis("off")
    ax.text(
        0.50,
        0.91,
        r"Paper XI bridge: constrained transport $\Psi_{\rm vac}=\Phi_{\rm vac}/C_{\rm vac}$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    center = (0.50, 0.50)
    ax.add_patch(Circle(center, 0.105, facecolor="#243B53", edgecolor="#17233A", lw=1.5))
    ax.text(0.50, 0.515, r"$\Psi=\Phi/C$", color="white", ha="center", va="center", fontsize=16)
    ax.text(0.50, 0.472, "constraint operator", color="white", ha="center", va="center", fontsize=8.5)

    nodes = [
        ("dark\nmatter", 0.17, 0.70, "#476A6F"),
        ("dark\nenergy", 0.83, 0.70, "#7B5E7B"),
        ("black\nholes", 0.17, 0.29, "#B85C38"),
        ("Blancken\nlayer", 0.83, 0.29, "#6A994E"),
        ("mass\ntopology", 0.50, 0.14, "#386FA4"),
    ]
    for label, x, y, color in nodes:
        ax.add_patch(Circle((x, y), 0.082, facecolor=color, edgecolor="#222", lw=1.0, alpha=0.96))
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.add_patch(
            FancyArrowPatch(
                posA=center,
                posB=(x, y),
                arrowstyle="->",
                mutation_scale=12,
                lw=1.4,
                color="#555",
                shrinkA=66,
                shrinkB=58,
            )
        )

    ax.add_patch(Rectangle((0.08, 0.02), 0.84, 0.055, facecolor="#F4F1DE", edgecolor="#B7B7A4"))
    ax.text(
        0.50,
        0.048,
        "Scientific status: hypotheses must produce discriminants beyond qualitative analogy",
        ha="center",
        va="center",
        fontsize=9.3,
    )
    fig.tight_layout()
    fig.savefig(out / "fig8_cosmology_bridge.pdf")
    plt.close(fig)
    return df


def dark_matter_lensing(out: Path) -> Dict[str, pd.DataFrame]:
    n = 220
    x = np.linspace(-3.0, 3.0, n)
    y = np.linspace(-3.0, 3.0, n)
    xx, yy = np.meshgrid(x, y)
    r1 = np.sqrt((xx + 0.75) ** 2 + (yy - 0.15) ** 2)
    r2 = np.sqrt((xx - 0.65) ** 2 + (yy + 0.25) ** 2)
    baryon = 1.4 * np.exp(-(r1 / 0.55) ** 2) + 1.1 * np.exp(-(r2 / 0.65) ** 2)
    shear_history = np.exp(-((yy - 0.45 * np.sin(1.2 * xx)) / 0.68) ** 2) * np.exp(-(xx / 2.45) ** 2)
    merger_memory = 0.72 * np.exp(-((xx + 1.15) ** 2 + (yy + 1.05) ** 2) / 1.05)
    constraint = 0.48 * baryon + 0.76 * shear_history + 0.46 * merger_memory
    standard_fit = 0.60 * baryon + 0.28 * np.exp(-(xx**2 + yy**2) / 3.6)
    residual = constraint - standard_fit

    flat = pd.DataFrame(
        {
            "baryon": baryon.ravel(),
            "history": (shear_history + merger_memory).ravel(),
            "residual": residual.ravel(),
        }
    )
    model_b = linregress(flat["baryon"], flat["residual"])
    residual_after_b = flat["residual"] - (model_b.intercept + model_b.slope * flat["baryon"])
    model_h = linregress(flat["history"], residual_after_b)
    checks = pd.DataFrame(
        [
            {
                "check": "history_correlation_after_baryons",
                "value": float(model_h.rvalue),
                "ok": abs(float(model_h.rvalue)) > 0.35,
            },
            {
                "check": "residual_has_positive_and_negative_structure",
                "value": float(residual.max() - residual.min()),
                "ok": float(residual.max() - residual.min()) > 0.8,
            },
        ]
    )
    _save_csv(checks, out / "checks_dark_matter_lensing_gate.csv")

    sample = flat.sample(800, random_state=110526)
    _save_csv(sample, out / "table2_dark_matter_residual_sample.csv")

    fig, ax = plt.subplots(1, 3, figsize=(12.8, 4.2))
    im0 = ax[0].imshow(baryon, extent=[-3, 3, -3, 3], origin="lower", cmap="magma")
    ax[0].contour(xx, yy, baryon, levels=5, colors="white", alpha=0.5, linewidths=0.6)
    ax[0].set_title("Baryonic source map")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)

    im1 = ax[1].imshow(constraint, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis")
    ax[1].contour(xx, yy, shear_history + merger_memory, levels=5, colors="white", alpha=0.55, linewidths=0.7)
    ax[1].set_title("History-weighted constraint field")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.03)

    lim = float(np.max(np.abs(residual)))
    im2 = ax[2].imshow(residual, extent=[-3, 3, -3, 3], origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[2].contour(xx, yy, residual, levels=[-0.35, 0.0, 0.35], colors="black", alpha=0.55, linewidths=0.7)
    ax[2].set_title("Residual after standard smooth fit")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03)
    for axis in ax:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.suptitle("Toy lensing residual: the falsifiable quantity is the remaining history term", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "paper_4_dark_matter_lensing.png", dpi=220)
    plt.close(fig)
    return {"sample": sample, "checks": checks}


def dark_energy_residual(out: Path) -> Dict[str, pd.DataFrame]:
    z = np.linspace(0.0, 3.0, 260)
    a = 1.0 / (1.0 + z)
    lambda_cdm_w = -np.ones_like(z)
    residual = 0.018 * np.tanh((z - 1.25) / 0.7) - 0.006 * np.exp(-((z - 0.35) / 0.28) ** 2)
    cdfd_w = -1.0 + residual
    delta_vac = 0.045 * np.exp(-z / 1.8) * np.cos(1.1 * z) - 0.012
    rho_eff = 1.0 + np.cumsum(delta_vac[::-1])[::-1] * (z[1] - z[0]) * 0.04

    df = pd.DataFrame(
        {
            "redshift_z": z,
            "scale_factor_a": a,
            "lambda_cdm_w": lambda_cdm_w,
            "cdfd_residual_w": cdfd_w,
            "delta_vac": delta_vac,
            "rho_lambda_eff_proxy": rho_eff,
        }
    )
    _save_csv(df, out / "table3_dark_energy_residual_curve.csv")

    checks = pd.DataFrame(
        [
            {
                "check": "near_lambda_cdm_limit",
                "value": float(np.max(np.abs(cdfd_w + 1.0))),
                "ok": float(np.max(np.abs(cdfd_w + 1.0))) < 0.05,
            },
            {
                "check": "nonzero_epoch_dependence",
                "value": float(np.ptp(cdfd_w)),
                "ok": float(np.ptp(cdfd_w)) > 0.02,
            },
        ]
    )
    _save_csv(checks, out / "checks_dark_energy_gate.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax[0].plot(z, lambda_cdm_w, "--", color="#444", label=r"$\Lambda$CDM reference")
    ax[0].plot(z, cdfd_w, color="#7B5E7B", lw=2.0, label="CDFD residual hypothesis")
    ax[0].set_xlabel("redshift z")
    ax[0].set_ylabel("effective w(z)")
    ax[0].set_title("Dark-energy discriminant")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(z, delta_vac, color="#386FA4", lw=1.8, label=r"$\delta_{\rm vac}$")
    ax[1].plot(z, rho_eff - 1.0, color="#B85C38", lw=1.8, label=r"$\rho_{\Lambda,\rm eff}-1$")
    ax[1].axhline(0.0, color="#444", ls="--", lw=1.0)
    ax[1].set_xlabel("redshift z")
    ax[1].set_title("Residual self-consistency proxy")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "paper_4_dark_energy_expansion.png", dpi=220)
    plt.close(fig)
    return {"curve": df, "checks": checks}


def black_hole_capacity(out: Path) -> Dict[str, pd.DataFrame]:
    r = np.linspace(1.02, 8.0, 320)
    horizon = 2.0
    j_over_jcrit = np.clip((horizon / r) ** 2.25 + 0.10 * np.exp(-((r - 2.4) / 0.55) ** 2), 0, 1.12)
    redshift_proxy = 1.0 / np.sqrt(np.maximum(1.0 - horizon / r, 1e-4))
    entropy_correction = 0.012 * np.log(1.0 + (horizon / r) ** 2)
    df = pd.DataFrame(
        {
            "radius_over_M": r,
            "J_over_Jcrit": j_over_jcrit,
            "redshift_proxy": redshift_proxy,
            "entropy_correction_proxy": entropy_correction,
        }
    )
    _save_csv(df, out / "table4_black_hole_capacity_boundary.csv")

    near = df.iloc[np.argmin(np.abs(df["J_over_Jcrit"] - 1.0))]
    checks = pd.DataFrame(
        [
            {
                "check": "capacity_boundary_near_horizon",
                "value": float(near["radius_over_M"]),
                "ok": 1.5 < float(near["radius_over_M"]) < 2.6,
            },
            {
                "check": "outer_region_unsaturated",
                "value": float(df.tail(20)["J_over_Jcrit"].mean()),
                "ok": float(df.tail(20)["J_over_Jcrit"].mean()) < 0.12,
            },
        ]
    )
    _save_csv(checks, out / "checks_black_hole_gate.csv")

    grid = np.linspace(-3.4, 3.4, 340)
    x, y = np.meshgrid(grid, grid)
    rgrid = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    field = np.exp(-((rgrid - 1.0) / 0.22) ** 2) + 0.35 * np.exp(-((rgrid - 1.8) / 0.55) ** 2)
    field *= 1.0 + 0.14 * np.sin(5 * theta)
    field = np.ma.masked_where(rgrid < 0.36, field)

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax[0].imshow(field, extent=[-3.4, 3.4, -3.4, 3.4], origin="lower", cmap="inferno")
    ax[0].add_patch(Circle((0, 0), 1.0, facecolor="black", edgecolor="white", lw=0.9))
    ax[0].add_patch(Circle((0, 0), 1.35, fill=False, edgecolor="#F4D35E", lw=1.4, ls="--"))
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    ax[0].set_title("Capacity-saturation surface")

    ax[1].plot(r, j_over_jcrit, color="#B85C38", lw=2.0, label=r"$J/J_{\rm crit}$")
    ax[1].axhline(1.0, color="#222", ls="--", lw=1.0)
    ax[1].axvline(horizon, color="#555", ls=":", lw=1.2, label="GR horizon scale")
    ax[1].set_xlabel(r"radius proxy $r/M$")
    ax[1].set_ylabel("transport saturation")
    ax[1].set_ylim(0, 1.16)
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.25)
    ax[1].set_title("Bridge must recover GR first")
    fig.tight_layout()
    fig.savefig(out / "paper_4_black_hole_rupture.png", dpi=220)
    plt.close(fig)
    return {"curve": df, "checks": checks}


def blancken_layer(out: Path) -> Dict[str, pd.DataFrame]:
    x = np.linspace(-4, 4, 420)
    y = np.linspace(-2.6, 2.6, 260)
    xx, yy = np.meshgrid(x, y)
    substrate = (
        np.sin(2.6 * xx + 0.8 * np.sin(yy))
        + 0.55 * np.cos(3.4 * yy - 0.7 * np.cos(xx))
        + 0.32 * np.sin(1.8 * (xx + yy))
    )
    envelope = np.exp(-0.045 * (xx**2 + yy**2))
    field = substrate * envelope

    rows = [
        {
            "requirement": "dimensionless invariant",
            "status": "open",
            "why_it_matters": "prevents Blancken layer from being only a name for unknown substrate",
        },
        {
            "requirement": "fixed-point selector boundary",
            "status": "highest_priority",
            "why_it_matters": "could derive the selectors used in Papers VI-X",
        },
        {
            "requirement": "scale law to observed constant",
            "status": "open",
            "why_it_matters": "would connect sub-geometric constraint to measured physics",
        },
        {
            "requirement": "no-go theorem for emergent geometries",
            "status": "open",
            "why_it_matters": "would make the substrate mathematically restrictive",
        },
    ]
    df = pd.DataFrame(rows)
    _save_csv(df, out / "table5_blancken_layer_requirements.csv")
    checks = pd.DataFrame(
        [
            {"check": "has_priority_selector_requirement", "value": 1, "ok": "highest_priority" in set(df["status"])},
            {"check": "minimum_requirements_count", "value": len(df), "ok": len(df) >= 4},
        ]
    )
    _save_csv(checks, out / "checks_blancken_layer_gate.csv")

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.imshow(field, extent=[-4, 4, -2.6, 2.6], origin="lower", cmap="cividis", alpha=0.95)
    for offset, color, label in [
        (1.15, "#FFFFFF", "emergent metric sheet"),
        (0.15, "#F4D35E", "constraint selector"),
        (-0.85, "#E76F51", "pre-geometric substrate"),
    ]:
        xs = np.linspace(-3.6, 3.6, 260)
        ys = offset + 0.22 * np.sin(1.55 * xs) + 0.08 * np.sin(4.1 * xs)
        ax.plot(xs, ys, color=color, lw=2.0, label=label)
    ax.set_title("Blancken-layer schematic: useful only if it yields an invariant or selector")
    ax.set_xlabel("pre-geometric coordinate proxy")
    ax.set_ylabel("constraint phase proxy")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.6, 2.6)
    fig.tight_layout()
    fig.savefig(out / "paper_4_blancken_layer.png", dpi=220)
    plt.close(fig)
    return {"requirements": df, "checks": checks}


def symbolic_checks(out: Path) -> pd.DataFrame:
    phi, c, alpha, beta, gamma = sp.symbols("Phi_vac C_vac alpha beta gamma", positive=True)
    psi = phi / c
    feedback = alpha * phi - beta * c + gamma * sp.Symbol("laplacian_C")
    rows = [
        {"check": "dPsi_dPhi_positive", "expression": str(sp.diff(psi, phi)), "ok": str(sp.diff(psi, phi)) == "1/C_vac"},
        {
            "check": "dPsi_dC_negative",
            "expression": str(sp.diff(psi, c)),
            "ok": str(sp.diff(psi, c)) == "-Phi_vac/C_vac**2",
        },
        {
            "check": "feedback_contains_flux_term",
            "expression": str(feedback),
            "ok": alpha * phi in sp.Add.make_args(feedback),
        },
    ]
    df = pd.DataFrame(rows)
    _save_csv(df, out / "checks_symbolic_XI.csv")
    return df


def bundle_pdf(out: Path) -> None:
    files = [
        "fig8_cosmology_bridge.pdf",
        "paper_4_dark_matter_lensing.png",
        "paper_4_dark_energy_expansion.png",
        "paper_4_black_hole_rupture.png",
        "paper_4_blancken_layer.png",
    ]
    with PdfPages(out / "paper_XI_figures.pdf") as pdf:
        for file_name in files:
            path = out / file_name
            fig = plt.figure(figsize=(8.5, 11.0))
            fig.text(0.5, 0.53, file_name, ha="center", va="center", fontsize=14)
            fig.text(0.5, 0.49, f"Standalone reproducible figure: {path.name}", ha="center", va="center", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)


def run_paper_xi(*, write_outputs: bool = True, verbose: bool = True) -> Dict[str, object]:
    out: Optional[Path] = output_dir("paper_XI") if write_outputs else None
    if out is None:
        raise RuntimeError("Paper XI outputs require write_outputs=True")

    bridge = bridge_map(out)
    dark_matter = dark_matter_lensing(out)
    dark_energy = dark_energy_residual(out)
    black_holes = black_hole_capacity(out)
    blancken = blancken_layer(out)
    symbolic = symbolic_checks(out)
    bundle_pdf(out)

    gate_rows = []
    for name, block in (
        ("dark_matter", dark_matter["checks"]),
        ("dark_energy", dark_energy["checks"]),
        ("black_holes", black_holes["checks"]),
        ("blancken_layer", blancken["checks"]),
        ("symbolic", symbolic),
    ):
        gate_rows.append({"component": name, "ok": bool(block["ok"].all())})
    gate_rows.append({"component": "bridge_rows", "ok": len(bridge) == 5})
    gate_rows.append({"component": "autodiff_backends", "ok": describe_autodiff_backends()})
    gate = pd.DataFrame(gate_rows)
    _save_csv(gate, out / "checks_summary_XI_gate.csv")

    if verbose:
        print(SEP)
        print("Paper XI outputs written")
        print(f"Output directory: {out}")
        print(f"Bridge classes: {len(bridge)}")
        print(f"Dark matter history correlation: {float(dark_matter['checks'].loc[0, 'value']):.3f}")
        print(SEP)
    return {
        "output_dir": str(out),
        "bridge": bridge,
        "dark_matter": dark_matter,
        "dark_energy": dark_energy,
        "black_holes": black_holes,
        "blancken": blancken,
        "symbolic": symbolic,
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper XI supplementary - reproducible cosmological bridge outputs.")
    parser.add_argument("--no-write", action="store_true", help="Reserved for interface consistency; Paper XI writes by default.")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    if args.no_write:
        raise SystemExit("Paper XI output generator is intended to write files; omit --no-write.")
    result = run_paper_xi(write_outputs=True, verbose=not args.quiet)
    ok = all(bool(v) if not isinstance(v, str) else bool(v) for v in result["gate"]["ok"].tolist())
    if ok and not args.quiet:
        print("All Paper XI supplementary checks passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
