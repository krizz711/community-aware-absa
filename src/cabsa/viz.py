"""Shared plotting style so every figure in the report looks the same."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
GREY, INK, MUTED = "#9a9993", "#0b0b0b", "#52514e"
SERIES = [BLUE, ORANGE, AQUA]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#efeeea", BLUE])
FIG_DIR = "results/figures"


def setup():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 11, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def save(name: str):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", bbox_inches="tight")
