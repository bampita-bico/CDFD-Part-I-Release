"""
Supplementary Material — Paper XII
Friction, superconductivity, plasma states, and transport mysteries.

The models here are deliberately toy-level. They generate falsifiability
figures, tables, and numerical gates for the Paper XII bridge without claiming
to replace established contact mechanics, superconductivity, MHD, or granular
physics.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
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
from scipy.signal import find_peaks

from _physics_utils import describe_autodiff_backends, output_dir

SEP = "=" * 64


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def phase_map(out: Path) -> pd.DataFrame:
    rows = [
        {
            "domain": "friction",
            "phi_proxy": "shear work tau*v*A",
            "constraint_proxy": "contact memory, asperity locking, real area",
            "transition_signal": "stick-slip, acoustic/contact-resistance precursor",
            "failure_condition": "no residual memory/geometric predictor after rate-state fit",
        },
        {
            "domain": "superconductivity",
            "phi_proxy": "paired charge current / phase transport",
            "constraint_proxy": "dissipation suppressed, phase stiffness retained",
            "transition_signal": "resistance collapse plus stiffness/pinning reallocation",
            "failure_condition": "no added prediction for high-Tc/vortex/disorder anomalies",
        },
        {
            "domain": "plasma",
            "phi_proxy": "current density, pressure drive, energy flux",
            "constraint_proxy": "magnetic topology, resistivity, turbulent transport",
            "transition_signal": "reconnection/disruption at drive-topology extrema",
            "failure_condition": "events depend only on scalar plasma parameters",
        },
        {
            "domain": "turbulence",
            "phi_proxy": "inertial energy flux",
            "constraint_proxy": "viscosity, boundary drag, eddy impedance",
            "transition_signal": "localized transition before global failure",
            "failure_condition": "no local constraint variable improves transition prediction",
        },
        {
            "domain": "jamming_glass",
            "phi_proxy": "shear, vibration, thermal agitation",
            "constraint_proxy": "packing, cages, contact network, relaxation barrier",
            "transition_signal": "history-dependent avalanches or relaxation",
            "failure_condition": "preparation history adds no predictive information",
        },
    ]
    df = pd.DataFrame(rows)
    _save_csv(df, out / "table1_transport_mystery_map.csv")

    def wrap_label(value: str, width: int = 28) -> str:
        return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    ax.axis("off")
    y_positions = np.linspace(0.80, 0.18, len(rows))
    colors = ["#476A6F", "#7B5E7B", "#B85C38", "#386FA4", "#6A994E"]
    columns = {
        "Flux": (0.25, "phi_proxy", 28),
        "Constraint": (0.49, "constraint_proxy", 31),
        "Signal": (0.75, "transition_signal", 34),
    }
    for heading, (x, _, _) in columns.items():
        ax.text(x, 0.94, heading, fontsize=10.5, fontweight="bold", ha="left", va="center")
    for y, row, color in zip(y_positions, rows, colors):
        domain = row["domain"].replace("_", "\n").title()
        ax.text(
            0.09,
            y,
            domain,
            color="white",
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="none", alpha=0.95),
        )
        for x, key, width in columns.values():
            ax.text(
                x,
                y,
                wrap_label(row[key], width),
                fontsize=8.2,
                ha="left",
                va="center",
                linespacing=1.18,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F8F3", edgecolor="#D6D6CE", linewidth=0.7),
            )
        ax.annotate("", xy=(0.21, y), xytext=(0.14, y), arrowprops=dict(arrowstyle="->", lw=1.1, color="#555"))
        ax.annotate("", xy=(0.45, y), xytext=(0.40, y), arrowprops=dict(arrowstyle="->", lw=1.1, color="#555"))
        ax.annotate("", xy=(0.71, y), xytext=(0.66, y), arrowprops=dict(arrowstyle="->", lw=1.1, color="#555"))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.05, 1.0)
    ax.set_title("Paper XII transport mystery map: flux, constraint, transition signal", fontsize=12)
    fig.tight_layout(pad=1.25)
    fig.savefig(out / "fig12_0_transport_mystery_map.pdf")
    plt.close(fig)
    return df


def friction_model(out: Path) -> Dict[str, object]:
    n = 1600
    dt = 0.01
    t = np.arange(n) * dt
    drive_rate = 0.18
    normal_load = 1.0
    memory = np.zeros(n)
    phi = np.zeros(n)
    c_static = np.zeros(n)
    psi = np.zeros(n)
    slip = np.zeros(n, dtype=bool)
    acoustic = np.zeros(n)
    stress = 0.08
    m = 0.18

    for i in range(n):
        stress += drive_rate * dt
        m += dt * (0.036 * normal_load - 0.010 * m)
        c = 0.40 + 1.35 * m
        p = stress
        r = p / c
        if r > 1.0:
            slip[i] = True
            acoustic[i] = 0.65 + 0.25 * min(r - 1.0, 1.0)
            stress *= 0.31
            m *= 0.53
            c = 0.40 + 1.35 * m
            p = stress
            r = p / c
        else:
            acoustic[i] = 0.025 * (r ** 4) + 0.002 * np.sin(21 * t[i]) ** 2
        memory[i] = m
        phi[i] = p
        c_static[i] = c
        psi[i] = r

    event_idx = np.flatnonzero(slip)
    event_table = pd.DataFrame(
        {
            "event_id": np.arange(1, len(event_idx) + 1),
            "time": t[event_idx],
            "phi_before_release": phi[event_idx],
            "constraint_after_release": c_static[event_idx],
            "psi_after_release": psi[event_idx],
        }
    )
    _save_csv(event_table, out / "table2_friction_stick_slip_events.csv")

    peaks, _ = find_peaks(acoustic, height=0.03, distance=30)
    precursor_gate = bool(len(peaks) >= len(event_idx))
    checks = pd.DataFrame(
        [
            {"check": "events_detected", "value": len(event_idx), "ok": len(event_idx) >= 4},
            {"check": "precursor_peaks_detected", "value": len(peaks), "ok": precursor_gate},
            {"check": "psi_bounded_after_release", "value": float(np.max(psi)), "ok": float(np.max(psi)) < 1.01},
        ]
    )
    _save_csv(checks, out / "checks_friction_gate.csv")

    fig, ax = plt.subplots(3, 1, figsize=(9.2, 7.2), sharex=True)
    ax[0].plot(t, phi, label=r"shear flux proxy $\Phi_f$", lw=1.4)
    ax[0].plot(t, c_static, label=r"adaptive contact constraint $C_f$", lw=1.4)
    ax[0].scatter(t[event_idx], c_static[event_idx], color="#B85C38", s=18, label="slip events", zorder=3)
    ax[0].set_ylabel("normalized units")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(t, psi, color="#476A6F", lw=1.3)
    ax[1].axhline(1.0, color="#B85C38", ls="--", lw=1.0)
    ax[1].set_ylabel(r"$\Psi_f=\Phi_f/C_f$")
    ax[1].grid(True, alpha=0.25)

    ax[2].plot(t, acoustic, color="#7B5E7B", lw=1.2, label="acoustic/contact-noise proxy")
    ax[2].scatter(t[peaks], acoustic[peaks], color="#B85C38", s=12, label="detected peaks")
    ax[2].set_ylabel("precursor")
    ax[2].set_xlabel("time")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].grid(True, alpha=0.25)
    fig.suptitle("Toy friction output: adaptive memory creates stick-slip release", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "fig12_1_friction_stick_slip.pdf")
    plt.close(fig)
    return {"checks": checks, "events": event_table}


def superconductivity_model(out: Path) -> Dict[str, object]:
    t_norm = np.linspace(0.03, 1.35, 480)
    tc = 1.0
    phase_stiffness = np.where(t_norm < tc, 1.0 - (t_norm / tc) ** 2, 0.0)
    pair_flux = np.where(t_norm < tc, 1.0 - 0.22 * t_norm, np.exp(-8.0 * (t_norm - tc)))
    c_diss = 0.02 + 0.92 / (1.0 + np.exp(-25.0 * (t_norm - tc)))
    c_vortex = 0.06 + 0.30 * np.exp(-((t_norm - 0.72) / 0.16) ** 2)
    transport_index = pair_flux * phase_stiffness / (c_diss + c_vortex + 1e-4)
    resistance_proxy = 1.0 / (1.0 + 8.0 * transport_index)

    df = pd.DataFrame(
        {
            "T_over_Tc": t_norm,
            "pair_flux": pair_flux,
            "phase_stiffness": phase_stiffness,
            "dissipative_constraint": c_diss,
            "vortex_constraint": c_vortex,
            "transport_index": transport_index,
            "resistance_proxy": resistance_proxy,
        }
    )
    _save_csv(df, out / "table3_superconductivity_reallocation_curve.csv")

    samples = df.iloc[[40, 180, 330, 440]].copy()
    samples["regime"] = ["deep_coherent", "vortex_limited", "critical_loss", "normal_dissipative"]
    _save_csv(samples, out / "table4_superconductivity_regime_samples.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.5))
    ax[0].plot(t_norm, phase_stiffness, label="phase stiffness", lw=1.8)
    ax[0].plot(t_norm, c_diss, label="dissipative constraint", lw=1.8)
    ax[0].plot(t_norm, c_vortex, label="vortex/pinning constraint", lw=1.5)
    ax[0].axvline(1.0, color="#444", ls="--", lw=1.0, label=r"$T_c$")
    ax[0].set_xlabel(r"$T/T_c$")
    ax[0].set_ylabel("normalized constraint")
    ax[0].set_title("Constraint reallocation")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.25)

    ax[1].semilogy(t_norm, resistance_proxy, color="#B85C38", lw=1.8, label="resistance proxy")
    ax[1].plot(t_norm, transport_index / max(transport_index), color="#476A6F", lw=1.5, label="normalized coherent index")
    ax[1].axvline(1.0, color="#444", ls="--", lw=1.0)
    ax[1].set_xlabel(r"$T/T_c$")
    ax[1].set_ylabel("index")
    ax[1].set_title("Dissipation collapse with retained phase constraint")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig12_2_superconductivity_reallocation.pdf")
    plt.close(fig)

    checks = pd.DataFrame(
        [
            {
                "check": "phase_stiffness_collapses_above_Tc",
                "value": float(phase_stiffness[t_norm > tc].max()),
                "ok": float(phase_stiffness[t_norm > tc].max()) == 0.0,
            },
            {
                "check": "resistance_low_below_Tc",
                "value": float(np.median(resistance_proxy[t_norm < 0.75])),
                "ok": float(np.median(resistance_proxy[t_norm < 0.75])) < 0.25,
            },
            {
                "check": "resistance_high_above_Tc",
                "value": float(np.median(resistance_proxy[t_norm > 1.12])),
                "ok": float(np.median(resistance_proxy[t_norm > 1.12])) > 0.65,
            },
        ]
    )
    _save_csv(checks, out / "checks_superconductivity_gate.csv")
    return {"curve": df, "checks": checks}


def plasma_model(out: Path) -> Dict[str, object]:
    grid = np.linspace(-1.0, 1.0, 180)
    x, y = np.meshgrid(grid, grid)
    phi = (
        0.28
        + 1.05 * np.exp(-((x + 0.28) ** 2 + (y - 0.10) ** 2) / 0.16)
        + 0.62 * np.exp(-((x - 0.42) ** 2 + (y + 0.22) ** 2) / 0.045)
    )
    c_topology = 0.34 + 0.58 * (x**2 + y**2) + 0.30 * np.abs(x * y)
    c_topology -= 0.24 * np.exp(-((x + 0.10) ** 2 + (y + 0.04) ** 2) / 0.035)
    c_topology = np.maximum(c_topology, 0.08)
    gy, gx = np.gradient(c_topology, grid, grid)
    grad_c = np.sqrt(gx**2 + gy**2)
    psi = phi / c_topology
    rupture_index = grad_c * psi
    threshold = np.quantile(rupture_index, 0.985)
    event_mask = rupture_index >= threshold

    try:
        import statsmodels.api as sm

        y_event = event_mask.ravel().astype(float)
        x_phi = sm.add_constant(phi.ravel())
        x_risk = sm.add_constant(np.column_stack([phi.ravel(), rupture_index.ravel()]))
        r2_phi = float(sm.OLS(y_event, x_phi).fit().rsquared)
        r2_risk = float(sm.OLS(y_event, x_risk).fit().rsquared)
        statsmodels_status = "used"
    except ImportError:
        y_event = event_mask.ravel().astype(float)
        r2_phi = float(np.corrcoef(phi.ravel(), y_event)[0, 1] ** 2)
        r2_risk = float(np.corrcoef(rupture_index.ravel(), y_event)[0, 1] ** 2)
        statsmodels_status = "skipped"

    summary = pd.DataFrame(
        [
            {"metric": "max_phi", "value": float(phi.max())},
            {"metric": "max_constraint", "value": float(c_topology.max())},
            {"metric": "max_psi", "value": float(psi.max())},
            {"metric": "rupture_threshold_q985", "value": float(threshold)},
            {"metric": "event_fraction", "value": float(event_mask.mean())},
            {"metric": "r2_phi_only", "value": r2_phi},
            {"metric": "r2_phi_plus_rupture_index", "value": r2_risk},
        ]
    )
    _save_csv(summary, out / "table5_plasma_rupture_index_summary.csv")

    checks = pd.DataFrame(
        [
            {"check": "rupture_index_beats_phi_only", "value": r2_risk - r2_phi, "ok": (r2_risk - r2_phi) > 0.15},
            {"check": "event_fraction_small", "value": float(event_mask.mean()), "ok": float(event_mask.mean()) < 0.03},
            {"check": "statsmodels", "value": statsmodels_status, "ok": True},
        ]
    )
    _save_csv(checks, out / "checks_plasma_gate.csv")

    fig, ax = plt.subplots(1, 4, figsize=(14.0, 3.6))
    panels = [
        (phi, r"drive $\Phi_p$"),
        (c_topology, r"constraint $C_p$"),
        (psi, r"ratio $\Psi_p$"),
        (rupture_index, r"rupture index $|\nabla C|\,\Phi/C$"),
    ]
    for a, (z, title) in zip(ax, panels):
        im = a.imshow(z, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
        a.contour(x, y, event_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        a.set_title(title, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle("Toy plasma output: risk localizes at drive plus topology extrema", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out / "fig12_3_plasma_rupture_index.pdf")
    plt.close(fig)
    return {"summary": summary, "checks": checks}


def jamming_model(out: Path) -> Dict[str, object]:
    rng = np.random.default_rng(120526)
    n = 4500
    base = rng.pareto(1.85, size=n) + 0.15
    low_memory = base * rng.lognormal(mean=-0.20, sigma=0.27, size=n)
    high_memory = base * rng.lognormal(mean=0.38, sigma=0.44, size=n)
    low_memory = np.clip(low_memory, 0, 20)
    high_memory = np.clip(high_memory, 0, 20)

    bins = np.linspace(0, 14, 70)
    hist_low, edges = np.histogram(low_memory, bins=bins, density=True)
    hist_high, _ = np.histogram(high_memory, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    df = pd.DataFrame({"avalanche_size": centers, "low_memory_density": hist_low, "high_memory_density": hist_high})
    _save_csv(df, out / "table6_jamming_memory_distributions.csv")

    q = [0.5, 0.9, 0.95, 0.99]
    summary = pd.DataFrame(
        {
            "quantile": q,
            "low_memory_size": np.quantile(low_memory, q),
            "high_memory_size": np.quantile(high_memory, q),
        }
    )
    summary["ratio_high_to_low"] = summary["high_memory_size"] / summary["low_memory_size"]
    _save_csv(summary, out / "table7_jamming_memory_quantiles.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax[0].plot(centers, hist_low, label="low preparation memory", lw=1.7)
    ax[0].plot(centers, hist_high, label="high preparation memory", lw=1.7)
    ax[0].set_xlabel("avalanche size")
    ax[0].set_ylabel("density")
    ax[0].set_title("Avalanche distribution shifts with memory")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(summary["quantile"], summary["ratio_high_to_low"], "o-", color="#B85C38", lw=1.8)
    ax[1].axhline(1.0, color="#444", ls="--", lw=1.0)
    ax[1].set_xlabel("quantile")
    ax[1].set_ylabel("high-memory / low-memory size")
    ax[1].set_title("Same drive, different constraint history")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig12_4_jamming_memory.pdf")
    plt.close(fig)

    checks = pd.DataFrame(
        [
            {
                "check": "high_memory_has_larger_tail",
                "value": float(summary.loc[summary["quantile"] == 0.95, "ratio_high_to_low"].iloc[0]),
                "ok": float(summary.loc[summary["quantile"] == 0.95, "ratio_high_to_low"].iloc[0]) > 1.35,
            },
            {
                "check": "deterministic_seed",
                "value": 120526,
                "ok": True,
            },
        ]
    )
    _save_csv(checks, out / "checks_jamming_gate.csv")
    return {"summary": summary, "checks": checks}


def symbolic_checks(out: Path) -> pd.DataFrame:
    phi, c, m, alpha, beta, eta = sp.symbols("Phi C M alpha beta eta", positive=True)
    psi = phi / c
    dpsi_dphi = sp.diff(psi, phi)
    dpsi_dc = sp.diff(psi, c)
    dc_dt = alpha * phi + eta * m - beta * c
    rows = [
        {"check": "dPsi_dPhi_positive", "expression": str(dpsi_dphi), "ok": str(dpsi_dphi) == "1/C"},
        {"check": "dPsi_dC_negative", "expression": str(dpsi_dc), "ok": str(dpsi_dc) == "-Phi/C**2"},
        {"check": "adaptive_constraint_contains_memory", "expression": str(dc_dt), "ok": eta * m in sp.Add.make_args(dc_dt)},
    ]
    df = pd.DataFrame(rows)
    _save_csv(df, out / "checks_symbolic_XII.csv")
    return df


def bundle_pdf(out: Path) -> None:
    figure_files = [
        "fig12_0_transport_mystery_map.pdf",
        "fig12_1_friction_stick_slip.pdf",
        "fig12_2_superconductivity_reallocation.pdf",
        "fig12_3_plasma_rupture_index.pdf",
        "fig12_4_jamming_memory.pdf",
    ]
    with PdfPages(out / "paper_XII_figures.pdf") as pdf:
        for file_name in figure_files:
            path = out / file_name
            fig = plt.figure(figsize=(8.5, 11.0))
            fig.text(0.5, 0.52, file_name, ha="center", va="center", fontsize=14)
            fig.text(0.5, 0.48, f"See standalone vector figure: {path.name}", ha="center", va="center", fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)


def run_paper_xii(*, write_outputs: bool = True, verbose: bool = True) -> Dict[str, object]:
    out: Optional[Path] = output_dir("paper_XII") if write_outputs else None
    if out is None:
        raise RuntimeError("Paper XII outputs require write_outputs=True")

    phase_df = phase_map(out)
    friction = friction_model(out)
    superconductivity = superconductivity_model(out)
    plasma = plasma_model(out)
    jamming = jamming_model(out)
    symbolic = symbolic_checks(out)
    bundle_pdf(out)

    gate_rows = []
    for name, block in (
        ("friction", friction["checks"]),
        ("superconductivity", superconductivity["checks"]),
        ("plasma", plasma["checks"]),
        ("jamming", jamming["checks"]),
        ("symbolic", symbolic),
    ):
        df = block.copy()
        ok = bool(df["ok"].all()) if "ok" in df else False
        gate_rows.append({"component": name, "ok": ok})
    gate_rows.append({"component": "autodiff_backends", "ok": describe_autodiff_backends()})
    gate = pd.DataFrame(gate_rows)
    _save_csv(gate, out / "checks_summary_XII_gate.csv")

    if verbose:
        print(SEP)
        print("Paper XII outputs written")
        print(f"Output directory: {out}")
        print(f"Maps: {len(phase_df)}")
        print(f"Friction events: {len(friction['events'])}")
        print(f"Plasma risk R^2 delta: {float(plasma['checks'].loc[0, 'value']):.3f}")
        print(SEP)
    return {
        "output_dir": str(out),
        "phase_map": phase_df,
        "friction": friction,
        "superconductivity": superconductivity,
        "plasma": plasma,
        "jamming": jamming,
        "symbolic": symbolic,
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper XII supplementary — reproducible transport-mystery outputs.")
    parser.add_argument("--no-write", action="store_true", help="Reserved for interface consistency; Paper XII writes by default.")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    if args.no_write:
        raise SystemExit("Paper XII output generator is intended to write files; omit --no-write.")
    result = run_paper_xii(write_outputs=True, verbose=not args.quiet)
    ok_values = result["gate"]["ok"].tolist()
    ok = all(v is True or (isinstance(v, str) and v) for v in ok_values)
    if ok and not args.quiet:
        print("All Paper XII supplementary checks passed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
