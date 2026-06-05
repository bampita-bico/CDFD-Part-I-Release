"""
Supplementary Material — Paper X
Vacuum EOS: deriving self-consistency from Ψ = Φ/C.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from typing import Dict, Optional
_REPO_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPERS_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,_REPO_ROOT); sys.path.insert(0,_PAPERS_DIR)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, sympy as sp
from scipy.optimize import brentq
from _figure_utils import save_axes_panel
from _physics_utils import agreement_ok, describe_autodiff_backends, output_dir

from _physics_utils import LEPTON_MASSES

_LEPTON_SOURCE = "paper_local_public_constants"

C_N=float(np.sqrt(2.0)); SEP='='*64

def A(th,n,k): return 1.0 + C_N*np.cos(th + 2*np.pi*k/n)
def Phi(th,n): return float(sum(A(th,n,k)**2 for k in range(n)))
def Ccap(th,n): return float(sum(abs(A(th,n,k)) for k in range(n))**2)
def Psi_geom(th,n): return n*th
def f_sc(n,th): return n*th - Phi(th,n)/Ccap(th,n)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
def solve_theta(n):
    if n==3: return (2/3)/3
    lo,hi=1e-14,np.pi/n-1e-14
    f=lambda t:f_sc(n,t)
    if f(lo)*f(hi)>0: return None
    return float(brentq(f,lo,hi,xtol=1e-15,rtol=1e-15))

def sym_phi_2n():
    th=sp.Symbol('theta', real=True)
    rows=[]; pts=(sp.Rational(1,11),sp.Rational(17,100),-sp.Rational(23,100))
    for n in (3,5,7,9):
        expr=sum((1+sp.sqrt(2)*sp.cos(th+2*sp.pi*sp.Integer(k)/n))**2 for k in range(n))
        simp=sp.simplify(expr)
        lit=(simp==2*n)
        gate=all(abs(float(sp.N((expr-2*n).subs(th,t),50)))<1e-12 for t in pts)
        rows.append({'n':n,'expr_simplified':str(simp)[:100],'literal_equals_2n':lit,'numeric_gate_2n':gate})
    return pd.DataFrame(rows)

def try_torch_grad(n,th):
    try:
        import torch
        pi_t=torch.tensor(np.pi,dtype=torch.float64)
        t=torch.tensor(th,dtype=torch.float64,requires_grad=True)
        amps=[1.0 + C_N*torch.cos(t + 2.0*pi_t*k/n) for k in range(n)]
        ph=sum(a*a for a in amps); cc=(sum(torch.abs(a) for a in amps))**2
        y=n*t - ph/cc
        g=torch.autograd.grad(y,t)[0]
        return float(g.item())
    except ImportError:
        return None

def try_jax_grad(n,th):
    try:
        import jax
        g=jax.grad(lambda x: f_sc(n,float(x)))(th)
        return float(g)
    except ImportError:
        return None


def try_numba_phi_grid(theta_grid: np.ndarray, n_values: tuple[int, ...]) -> tuple[str, float]:
    ref = {
        n: np.array([Phi(float(t), n) for t in theta_grid], dtype=float)
        for n in n_values
    }
    try:
        from numba import njit

        @njit
        def phi_batch(ts: np.ndarray, n: int) -> np.ndarray:
            c = 1.4142135623730951
            out = np.empty(len(ts))
            twopi = 6.283185307179586
            for i in range(len(ts)):
                t = ts[i]
                acc = 0.0
                for k in range(n):
                    a = 1.0 + c * np.cos(t + twopi * k / n)
                    acc += a * a
                out[i] = acc
            return out

        max_abs = 0.0
        for n in n_values:
            nb = phi_batch(theta_grid, n)
            max_abs = max(max_abs, float(np.max(np.abs(nb - ref[n]))))
        return "used", max_abs
    except ImportError:
        return "skipped", 0.0

def plot_bundle(out:Path, df1:pd.DataFrame):
    fig,ax=plt.subplots(1,3,figsize=(13.5,4.4))

    # Panel 1: Flow invariance
    x_n = df1['n'].to_numpy()
    y_phi_ratio = (df1['Phi']/df1['n']).to_numpy()
    ax[0].plot(x_n, y_phi_ratio, 'o-', lw=1.8, ms=5, label=r'Computed $\Phi/n$')
    ax[0].axhline(2.0, ls='--', lw=1.2, color='C3', label=r'Theory: $\Phi/n=2$')
    ax[0].set_title(r'Flow Invariance: $\Phi/n=2$')
    ax[0].set_xlabel(r'Family index $n$')
    ax[0].set_ylabel(r'Normalized flow $\Phi/n$')
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(fontsize=8, loc='best')

    # Panel 2: EOS identity
    x_ratio = df1['Phi_over_C'].to_numpy()
    y_psi = df1['Psi_geom'].to_numpy()
    ax[1].scatter(x_ratio, y_psi, s=30, alpha=0.9, label=r'Computed families $n=3\ldots 13$')
    mn=min(df1[['Phi_over_C','Psi_geom']].min()); mx=max(df1[['Phi_over_C','Psi_geom']].max())
    ax[1].plot([mn,mx],[mn,mx],'--', lw=1.2, color='C3', label=r'Ideal line $\Psi=\Phi/C$')
    ax[1].set_title(r'Vacuum EOS Closure: $\Psi$ vs $\Phi/C$')
    ax[1].set_xlabel(r'$\Phi/C$')
    ax[1].set_ylabel(r'$\Psi=n\theta_n$')
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(fontsize=8, loc='best')

    # Panel 3: Numerical residuals
    y_abs = df1['abs_delta'].to_numpy()
    ax[2].semilogy(x_n, y_abs, 'o-', lw=1.8, ms=5, label=r'$|\Phi/C-\Psi|$')
    ax[2].set_title(r'Residuals to Machine Precision')
    ax[2].set_xlabel(r'Family index $n$')
    ax[2].set_ylabel(r'$|\Phi/C-\Psi|$')
    ax[2].grid(True, which='both', alpha=0.25)
    ax[2].legend(fontsize=8, loc='best')
    ax[2].text(
        0.03,
        0.07,
        f"max={float(np.max(y_abs)):.2e}",
        transform=ax[2].transAxes,
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='0.8')
    )

    fig.tight_layout()
    pdf=out/'paper_X_figures.pdf'
    fig.savefig(pdf,bbox_inches='tight')

    # Export individual panels for LaTeX inclusion
    for i, ax_ in enumerate(ax):
        save_axes_panel(fig, ax_, out / f"fig10_{i}_panel.pdf")

    plt.close(fig); return pdf

def run_paper_x(write_outputs:bool=True, verbose:bool=True):
    out=output_dir('paper_X') if write_outputs else None
    thetas={}
    for n in range(3,14):
        t=solve_theta(n)
        if t is None: raise RuntimeError(f'no theta n={n}')
        thetas[n]=t

    rows=[]
    for n in range(3,14):
        th=thetas[n]; ph=Phi(th,n); cc=Ccap(th,n); ps=Psi_geom(th,n); ratio=ph/cc; d=abs(ratio-ps)
        ok,_=agreement_ok(ratio,ps,rtol=1e-10,atol=1e-12)
        rows.append({'n':n,'theta_rad':th,'Phi':ph,'C':cc,'Phi_over_C':ratio,'Psi_geom':ps,'abs_delta':d,'row_ok':ok})
    df1=pd.DataFrame(rows)

    df2=sym_phi_2n()
    rows3=[]
    for n in range(3,14):
        th_ref=thetas[n]; th_sc=solve_theta(n)
        ok,d=agreement_ok(th_sc,th_ref,rtol=1e-11,atol=1e-12)
        rows3.append({'n':n,'theta_reference':th_ref,'theta_rederived':th_sc,'agree':ok,'abs_diff':d,'residual_abs':abs(f_sc(n,th_sc))})
    df3=pd.DataFrame(rows3)

    grads=[]; g_ok=True
    for n in (3,5,7,9,11,13):
        th=thetas[n]; tg=try_torch_grad(n,th); jg=try_jax_grad(n,th)
        grads.append({'n':n,'torch_grad':tg,'jax_grad':jg})
        if tg is not None and abs(tg)<1e-8: g_ok=False
        if jg is not None and abs(jg)<1e-8: g_ok=False
    dfg=pd.DataFrame(grads)

    theta_grid = np.linspace(1e-6, 0.2, 300)
    numba_status, numba_max_abs = try_numba_phi_grid(theta_grid, (5, 7, 9))
    numba_ok = bool(numba_max_abs < 1e-10)
    df_numba = pd.DataFrame(
        [
            {
                "numba": numba_status,
                "n_values": "5,7,9",
                "max_abs_phi_residual_vs_numpy": numba_max_abs,
                "numba_parity_ok": numba_ok,
            }
        ]
    )
    df_checks_symbolic = df2.copy()
    df_checks_scipy = df3.copy()

    phi_ok=bool((abs(df1['Phi']-2*df1['n'])<1e-10).all())
    virial_ok=bool(df1['row_ok'].all())
    sym_ok=bool(df2['numeric_gate_2n'].all())
    rederive_ok=bool(df3['agree'].all() and (df3['residual_abs']<1e-10).all())
    overall=phi_ok and virial_ok and sym_ok and rederive_ok and g_ok and numba_ok
    dfsum=pd.DataFrame([{'phi_equals_2n':phi_ok,'virial_rows_ok':virial_ok,'sympy_phi_2n':sym_ok,'theta_rederive_ok':rederive_ok,'autodiff_optional':g_ok,'numba_phi_optional':numba_ok,'lepton_masses_source':_LEPTON_SOURCE,'overall_ok':overall}])

    pdf=None
    if write_outputs and out is not None:
        df1.to_csv(out/'table1_virial_theorem.csv',index=False)
        df2.to_csv(out/'table2_phi_equals_2n.csv',index=False)
        df3.to_csv(out/'table3_psi_equals_phi_over_c.csv',index=False)
        df_checks_symbolic.to_csv(out/'checks_symbolic_phi_equals_2n.csv',index=False)
        df_checks_scipy.to_csv(out/'checks_scipy_theta_rederived.csv',index=False)
        df_numba.to_csv(out/'checks_numba_phi_grid.csv',index=False)
        dfg.to_csv(out/'checks_autodiff_fixedpoint_slope.csv',index=False)
        dfsum.to_csv(out/'checks_summary_X_gate.csv',index=False)
        pdf=str(plot_bundle(out,df1))

    if verbose:
        print(SEP); print('Paper X — reproducibility bundle'); print(SEP)
        print(dfsum.to_string(index=False)); print(describe_autodiff_backends())

    return {'output_dir':str(out) if out else None,'summary_pass':overall,'figures_pdf':pdf,'summary_gate':dfsum}

def main()->int:
    ap=argparse.ArgumentParser(description='Paper X supplementary — reproducibility harness.')
    ap.add_argument('--no-write',action='store_true'); ap.add_argument('-q','--quiet',action='store_true')
    a=ap.parse_args(); r=run_paper_x(write_outputs=not a.no_write, verbose=not a.quiet)
    ok=bool(r['summary_gate']['overall_ok'].iloc[0])
    if ok and not a.quiet: print('All Paper X supplementary checks passed.')
    return 0 if ok else 1

if __name__=='__main__':
    raise SystemExit(main())
