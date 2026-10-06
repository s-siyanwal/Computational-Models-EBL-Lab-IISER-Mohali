"""Shared figure style: validated categorical order, one-hue sequential ramp,
blue-gray-red diverging map, recessive axes. Colors follow the entity (a
(k1,k2) pair or a model variant keeps its color in every figure)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV = LinearSegmentedColormap.from_list("div_br", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a02a2a"])
STABLE_CMAP = ListedColormap(["#0d366b", "#eda100"])  # unstable (resolution) / stable (IaSC persists)
DET = INK  # deterministic oracle is always drawn in primary ink

# one colour per (k1,k2) pair -- fixed, so a pair is recognisable across figures
K_COLORS = {
    (0.2, 0.2): CAT[0], (0.8, 0.8): CAT[1], (0.5, 0.5): CAT[2], (0.2, 0.8): CAT[3],
    (0.8, 0.2): CAT[4], (0.5, 1.0): CAT[5], (0.0, 0.5): CAT[6],
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.labelsize": 10, "axes.titleweight": "semibold",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
        "legend.fontsize": 8.5, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def klabel(k):
    return f"k1={k[0]:g}, k2={k[1]:g}"


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


def err(lower_arm, upper_arm):
    """Error-bar arms clipped at 0 (Wilson/quantile bounds can sit a hair on
    the wrong side of the point estimate through rounding or skew)."""
    import numpy as np
    return np.clip(np.vstack([np.asarray(lower_arm, float), np.asarray(upper_arm, float)]), 0, None)
