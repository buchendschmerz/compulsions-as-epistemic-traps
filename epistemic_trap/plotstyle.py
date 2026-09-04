"""Shared publication-quality matplotlib style and colour vocabulary."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# A single, consistent colour vocabulary used across every figure.
C = dict(
    healthy="#2E6F9E",       # variable-k / identifiable
    compulsive="#C1443C",    # fixed-k / non-identifiable
    mild="#E0913B",
    erp="#3E8E7E",           # response prevention (k = 0)
    graded="#6A4C93",        # graded / variable-k ERP
    reassurance="#9A8C98",
    neutral="#4A4A4A",
    ridge="#1B1B1B",
    accent="#D9A441",
)

CONDITION_COLORS = {
    "free_choice": C["compulsive"],
    "reassurance": C["reassurance"],
    "erp_k0": C["erp"],
    "graded_erp": C["graded"],
    "variable_k": C["healthy"],
}

CONDITION_LABELS = {
    "free_choice": "Free choice (compulsive)",
    "reassurance": "Reassurance (check after acting)",
    "erp_k0": "Response prevention ($k=0$ only)",
    "graded_erp": "Graded ERP (widening $k$ support)",
    "variable_k": "Variable-$k$ policy",
}


def use_style():
    """Apply the project-wide rcParams. Call once at the top of each figure script."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 3.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "mathtext.fontset": "dejavusans",
        "figure.constrained_layout.use": True,
    })


def panel_label(ax, letter, dx=-0.16, dy=1.06, size=11):
    """Put a bold panel letter (A, B, C ...) in axis-fraction coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left")


def save(fig, name, figdir=None):
    """Save a figure as both PDF (vector, for LaTeX) and PNG (for quick viewing)."""
    from .config import FIGURES
    figdir = FIGURES if figdir is None else figdir
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"{name}.{ext}")
    plt.close(fig)
    return figdir / f"{name}.pdf"
