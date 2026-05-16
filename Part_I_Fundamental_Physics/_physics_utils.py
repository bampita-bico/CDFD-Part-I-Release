"""
Shared public helpers for Part I Fundamental Physics supplementary scripts.
Provides output layout, numerical tolerances, and public physics functions
(vortex energy, chi equilibrium, Koide/Brannen lepton parametrisation).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent

RTOL_DEFAULT = 1e-9
ATOL_DEFAULT = 1e-12

ALPHA_MEASURED = 1.0 / 137.035999084
CHI_TARGET = 1.0 / ALPHA_MEASURED
LEPTON_MASSES = {
    "electron": 0.51099895,
    "muon": 105.6583755,
    "tau": 1776.86,
}


def total_energy(chi: float | np.ndarray, beta: float = 1.75, kappa: float = 1.0) -> float | np.ndarray:
    """Normalized public vortex energy used in Paper I."""
    return chi * (np.log(8.0 * chi) - beta) + kappa / chi


def dE_dchi(chi: float | np.ndarray, beta: float, kappa: float) -> float | np.ndarray:
    """Derivative of the normalized Paper I energy."""
    return (np.log(8.0 * chi) - beta + 1.0) - kappa / chi**2


def find_equilibrium(
    beta: float = 1.75,
    kappa: float = 1.0,
    bracket: tuple[float, float] = (1.01, 1e6),
) -> Optional[float]:
    """Bisection solver for dE/dchi = 0."""
    a, b = bracket
    fa, fb = float(dE_dchi(a, beta, kappa)), float(dE_dchi(b, beta, kappa))
    if fa * fb > 0:
        return None
    for _ in range(200):
        mid = (a + b) / 2.0
        fm = float(dE_dchi(mid, beta, kappa))
        if abs(fm) < 1e-12 or (b - a) / max(abs(mid), 1e-12) < 1e-10:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2.0


def kappa_for_chi(target_chi: float, beta: float = 1.75) -> float:
    """Return the kappa value that places equilibrium at target_chi."""
    return float(target_chi**2 * (np.log(8.0 * target_chi) - beta + 1.0))


def chi_self_consistency() -> dict[str, float]:
    """Geometric alpha check from the reduced Compton and classical electron radii."""
    r_compton = 3.861592680e-13
    a_classical = 2.8179403227e-15
    chi_geom = r_compton / a_classical
    return {
        "R_compton_m": r_compton,
        "a_classical_m": a_classical,
        "chi_geometric": chi_geom,
        "chi_target": CHI_TARGET,
        "alpha_from_chi": 1.0 / chi_geom,
        "alpha_measured": ALPHA_MEASURED,
        "relative_error": abs(chi_geom - CHI_TARGET) / CHI_TARGET,
    }


def energy_balance_at_chi(chi: float, beta: float = 1.75) -> dict[str, float | str]:
    """Return the two normalized energy terms at a chosen chi."""
    kappa = kappa_for_chi(chi, beta)
    e_circ = float(chi * (np.log(8.0 * chi) - beta))
    e_back = float(kappa / chi)
    return {
        "chi": chi,
        "kappa": kappa,
        "E_circulation": e_circ,
        "E_back_pressure": e_back,
        "ratio_back_to_circ": e_back / e_circ,
        "note": "ratio near 1 means terms compete naturally at this equilibrium",
    }


def koide_ratio(m1: float, m2: float, m3: float) -> float:
    """Koide ratio for three positive masses."""
    return float((m1 + m2 + m3) / (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2)


def verify_koide_real_masses() -> dict[str, float | bool]:
    me, mmu, mtau = (LEPTON_MASSES[k] for k in ["electron", "muon", "tau"])
    q = koide_ratio(me, mmu, mtau)
    return {
        "Q": q,
        "target": 2.0 / 3.0,
        "error_absolute": abs(q - 2.0 / 3.0),
        "satisfied": abs(q - 2.0 / 3.0) < 1e-3,
    }


def test_power_law_modes(powers: Optional[np.ndarray] = None) -> dict[str, float | str]:
    """Scan m_n proportional to n^p against Koide Q."""
    if powers is None:
        powers = np.linspace(0.5, 15.0, 5000)
    best_p, best_q, best_err = None, None, np.inf
    for p in powers:
        q = koide_ratio(1.0, 2.0**p, 3.0**p)
        err = abs(q - 2.0 / 3.0)
        if err < best_err:
            best_err, best_p, best_q = err, p, q
    return {
        "best_power": float(best_p),
        "best_Q": float(best_q),
        "error": float(best_err),
        "n2_Q": koide_ratio(1.0, 4.0, 9.0),
        "conclusion": "No simple power law m proportional to n^p gives exact Koide closure",
    }


def brannen_masses(M: float, theta: float) -> list[float]:
    """Brannen-Koide parametrization for a Z3 triplet."""
    return [
        float(M * (1.0 + np.sqrt(2.0) * np.cos(theta + 2.0 * np.pi * k / 3.0)) ** 2)
        for k in range(3)
    ]


def fit_brannen_to_leptons() -> dict[str, Any]:
    """Fit the Brannen scale and phase to the three charged leptons."""
    target = sorted(LEPTON_MASSES.values())
    best: dict[str, Any] = {"err": np.inf}
    for theta in np.linspace(0, 2 * np.pi / 3, 10000):
        raw = sorted(brannen_masses(1.0, float(theta)))
        if any(m <= 0 for m in raw):
            continue
        scale = float(np.exp(np.mean([np.log(t / r) for t, r in zip(target, raw)])))
        scaled = [m * scale for m in raw]
        err = float(sum((np.log(s / t)) ** 2 for s, t in zip(scaled, target)))
        if err < best["err"]:
            best = {"err": err, "M": scale, "theta": float(theta), "masses": scaled}
    masses = best["masses"]
    return {
        "M_MeV": best["M"],
        "theta_rad": best["theta"],
        "theta_deg": float(np.degrees(best["theta"])),
        "fitted_masses": masses,
        "actual_masses": target,
        "koide_Q": koide_ratio(*masses),
        "max_error_pct": max(abs(f / a - 1) * 100 for f, a in zip(masses, target)),
        "prediction": "Lepton masses are represented by a Z3-symmetric Brannen-Koide triplet.",
    }


def output_dir(paper_id: str) -> Path:
    """Return `physics_papers/outputs/<paper_id>/`, creating it if needed."""
    d = ROOT / "outputs" / paper_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def agreement_ok(
    a: float,
    b: float,
    *,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
) -> Tuple[bool, float]:
    """Return (matches, abs_diff)."""
    diff = abs(float(a) - float(b))
    scale = max(abs(float(a)), abs(float(b)), 1.0)
    ok = diff <= atol + rtol * scale
    return ok, diff


def central_second_derivative(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """Second derivative of scalar f at x via central differences on f'."""
    def fp(t: float) -> float:
        return (f(t + h) - f(t - h)) / (2 * h)

    return (fp(x + h) - fp(x - h)) / (2 * h)


def try_torch_d2E_total_energy(
    chi_eq: float,
    beta: float,
    kappa: float,
) -> Optional[float]:
    """d²E/dχ² at χ_eq using PyTorch autograd; None if torch unavailable."""
    try:
        import torch
    except ImportError:
        return None

    chi = torch.tensor(chi_eq, dtype=torch.float64, requires_grad=True)
    E = chi * (torch.log(8.0 * chi) - beta) + kappa / chi
    g1 = torch.autograd.grad(E, chi, create_graph=True)[0]
    g2 = torch.autograd.grad(g1, chi)[0]
    return float(g2.item())


def try_jax_d2E_total_energy(
    chi_eq: float,
    beta: float,
    kappa: float,
) -> Optional[float]:
    """d²E/dχ² at χ_eq using JAX; None if jax unavailable."""
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad
    except ImportError:
        return None

    def E(chi: Any) -> Any:
        return chi * (jnp.log(8.0 * chi) - beta) + kappa / chi

    d2 = grad(grad(E))(chi_eq)
    return float(d2)


def describe_autodiff_backends() -> str:
    """Short status line for console / logs."""
    parts = []
    try:
        import torch  # noqa: F401

        parts.append("torch=ok")
    except ImportError:
        parts.append("torch=missing")
    try:
        import jax  # noqa: F401

        parts.append("jax=ok")
    except ImportError:
        parts.append("jax=missing")
    try:
        import numba  # noqa: F401

        parts.append("numba=ok")
    except ImportError:
        parts.append("numba=missing")
    return ", ".join(parts)
