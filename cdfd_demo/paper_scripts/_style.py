"""Shared Matplotlib styling for the public CDFD demo figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#2f6f9f",
    "red": "#b9413d",
    "green": "#3f8f5f",
    "gold": "#b98b2f",
    "gray": "#5b6268",
    "light_red": "#f5d9d6",
    "light_green": "#d9eadf",
    "light_gold": "#f4ead1",
}


def apply_style() -> None:
    """Apply a compact, publication-oriented style without extra dependencies."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2f3437",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#d8dee4",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 220,
        }
    )
