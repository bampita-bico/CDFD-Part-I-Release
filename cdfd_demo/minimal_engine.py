"""
Minimal CDFD equation model — Φ / C / Ψ + Life Number Λ
NumPy only. No platform API.

Three coupled fields on an NX×NY grid:
  Φ (phi)  — flow / energy intensity
  C        — constraint / resistance
  Ψ_s      — system equilibrium = (Φ / C) * S * M_s

Evolution (explicit Euler, periodic boundaries):
  ∂Φ/∂t = ∇·(1/C · ∇Φ) + S - D
  ∂C/∂t = α|Φ| - βC + γ∇²C
  Ψ_s   = (Φ / C) * S * M_s
"""

import numpy as np

NX, NY = 64, 64
DT = 0.01
ALPHA = 0.1   # constraint growth rate
BETA = 0.05   # constraint relaxation rate
GAMMA = 0.1   # constraint diffusion rate


class State:
    def __init__(self, nx=NX, ny=NY, seed=42):
        rng = np.random.default_rng(seed)
        self.nx, self.ny = nx, ny
        self.phi = np.ones((nx, ny)) + rng.normal(0, 0.05, (nx, ny))
        self.C = np.ones((nx, ny)) + rng.normal(0, 0.01, (nx, ny))
        self.S = 1.0  # Surface responsiveness
        self.Ms = 1.0  # Structural memory
        self.psi_s = (self.phi / self.C) * self.S * self.Ms
        self.t = 0.0
        self.meta: dict = {}


def _laplacian(f):
    """5-point Laplacian with periodic boundaries."""
    return (
        np.roll(f, 1, 0) + np.roll(f, -1, 0) +
        np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f
    )


def _divergence_flux(phi, C):
    """∇·(1/C · ∇Φ) via central differences, periodic."""
    inv_C = 1.0 / np.maximum(C, 1e-9)
    gx = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2.0
    gy = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2.0
    fx = inv_C * gx
    fy = inv_C * gy
    div = (np.roll(fx, -1, 1) - np.roll(fx, 1, 1)) / 2.0 + \
          (np.roll(fy, -1, 0) - np.roll(fy, 1, 0)) / 2.0
    return div


def step(state: State, dt=DT,
         alpha=ALPHA, beta=BETA, gamma=GAMMA,
         source=0.01, dissipation=0.005) -> State:
    phi, C = state.phi, state.C

    dphi = _divergence_flux(phi, C) + source - dissipation * phi
    dC = alpha * np.abs(phi) - beta * C + gamma * _laplacian(C)

    state.phi = np.maximum(phi + dt * dphi, 1e-9)
    state.C = np.maximum(C + dt * dC, 1e-9)
    state.psi_s = (state.phi / state.C) * state.S * state.Ms
    state.t += dt
    return state


def compute_life_number(state: State) -> float:
    """
    Tri-Regime Bioenergetics (Chlorophyll, Magnetite, Water, Melanin):
    C_input     ~ Φ (energy absorption)
    C_electron  ~ σ_e (redox/electron mobility)
    C_proton    ~ σ_p (water/proton coherence)
    C_stability ~ 1/S (melanin-like buffering)
    
    J_max = min(C_input, C_electron, C_proton, C_stability)
    Λ = (C_input · C_electron · C_proton · τ_relax) / (S · E_maintenance)
    Λ < 1 → non-living  |  Λ ≈ 1 → proto-biological  |  Λ > 1 → sustained life

    The public demo reports normalized capacity terms so the value is stable
    under simple unit rescaling and does not diverge for nearly uniform fields.
    """
    phi = state.phi
    C = state.C
    
    phi_mean = float(np.mean(np.abs(phi))) + 1e-9
    c_mean = float(np.mean(np.abs(C))) + 1e-9
    phi_cv = float(np.std(phi)) / phi_mean
    c_cv = float(np.std(C)) / c_mean
    
    # 1. Energy input: saturating normalized absorption capacity.
    C_input = phi_mean / (1.0 + phi_mean)
    
    # 2. Electron transport: lower effective constraint means higher mobility.
    C_electron = 1.0 / (1.0 + c_mean)
    
    # 3. Proton coherence: smoother fields have higher normalized coherence.
    C_proton = 1.0 / (1.0 + phi_cv)
    
    # 4. Stabilization load: more heterogeneous constraint increases load.
    S = 1.0 + c_cv
    C_stability = 1.0 / S
    
    # Weakest-link throughput capacity
    J_max = min(C_input, C_electron, C_proton, C_stability)
    
    # Dimensionless Life Number with a fixed maintenance cost for this demo.
    E_maintenance = 0.25
    tau_relax = 1.0
    lam = (C_input * C_electron * C_proton * tau_relax) / (S * E_maintenance)
    
    state.meta["life_number"] = lam
    state.meta["J_max"] = J_max
    state.meta["C_input"] = C_input
    state.meta["C_electron"] = C_electron
    state.meta["C_proton"] = C_proton
    state.meta["C_stability"] = C_stability
    
    return lam


def run(steps=500, nx=NX, ny=NY, **kwargs) -> tuple[State, list]:
    state = State(nx=nx, ny=ny)
    history = []
    for _ in range(steps):
        state = step(state, **kwargs)
        lam = compute_life_number(state)
        history.append({
            "t": state.t,
            "mean_phi": float(np.mean(state.phi)),
            "mean_C": float(np.mean(state.C)),
            "mean_psi": float(np.mean(state.psi_s)),
            "lambda": lam,
        })
    return state, history


if __name__ == "__main__":
    state, history = run(steps=200)
    last = history[-1]
    print(f"t={last['t']:.2f}  Ψ_s={last['mean_psi']:.4f}  Λ={last['lambda']:.4f}")
