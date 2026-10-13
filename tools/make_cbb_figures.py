#!/usr/bin/env python3
"""Render the CBB figures from corrected data without touching the PLOS artwork."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import make_submission_figures as main_figures
import make_figures as supplements

OUT = ROOT / "paper" / "figures_cbb"
OUT.mkdir(exist_ok=True)
main_figures.RESULTS = ROOT / "results" / "cbb_revision_20261013"
supplements.RESULTS = main_figures.RESULTS


def save(fig, stem):
    if fig._suptitle is not None:
        fig._suptitle.remove()
    for text in fig.findobj(match=plt.Text):
        value = text.get_text()
        if value in "ABCDEF" and len(value) == 1:
            text.set_text(value.lower())
        text.set_text(text.get_text().replace("Preregistered", "Proposed"))
        if text.get_text() == "Uncertainty after matched exposure":
            text.set_text("Final efficacy uncertainty")
    if stem in {"figure_4_interventions", "figure_5_predictions_boundary"}:
        for ax in fig.axes:
            for text in ax.texts:
                if text.get_text() in {"a", "b", "c", "d"}:
                    text.set_position((-0.08, 1.18))
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return OUT / f"{stem}.pdf"


main_figures.finish = save
supplements.save = save
main_figures.main()
supplements.fig4_phase_diagram()
supplements.fig5_hysteresis()
supplements.figS2_uncertainty_diagram()
print(f"CBB figure files: {OUT}")
