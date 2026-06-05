"""Shared helpers for publication-ready supplementary figures."""
from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def save_axes_panel(
    fig: Figure,
    target_ax: Axes,
    path: Path,
    *,
    dpi: int = 150,
    pad_inches: float = 0.08,
) -> None:
    """Save one axes with complete labels and without neighboring-panel bleed."""
    fig.canvas.draw()
    bbox = target_ax.get_tightbbox(fig.canvas.get_renderer())
    if bbox is None:
        raise RuntimeError("Unable to determine panel bounds")
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(pad_inches)

    other_axes = [(ax, ax.get_visible()) for ax in fig.axes if ax is not target_ax]
    try:
        for ax, _visible in other_axes:
            ax.set_visible(False)
        fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
    finally:
        for ax, visible in other_axes:
            ax.set_visible(visible)
