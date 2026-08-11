"""Spatial niche maps and embedding visualizations (publication quality).

Style conventions for publication figures:
  - No embedded titles (titles belong in the caption, not on the figure).
  - 300 dpi PNG + editable SVG (text kept as text, not outlined).
  - Okabe-Ito colorblind-safe palette (separates green/vermillion for red-green CVD).
  - Liberation Sans (metric-equivalent to Arial), consistent sizes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- publication style ---
matplotlib.rcParams.update({
    "font.family": ["Liberation Sans", "Arimo", "DejaVu Sans"],
    "svg.fonttype": "none",          # editable text in SVG
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette
NICHE_COLORS = {
    "immune_cold": "#D55E00",   # vermillion
    "immune_hot": "#0072B2",    # blue
    "normal": "#009E73",        # bluish green
    "necrotic": "#CC79A7",      # reddish purple
    "exclude": "#BBBBBB",       # light grey
}
NICHE_CLASSES = ["immune_cold", "immune_hot", "normal", "necrotic"]


def _save(fig, out_path: str):
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    fig.savefig(out_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_niche_map(
    coords: np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str,
    out_path: str,
    spot_size: float = 14.0,
):
    """Scatter spots in array coordinates colored by niche.

    `title` is accepted for backward compatibility but not drawn (publication
    figures carry no embedded title).
    """
    labels = pd.Series(np.asarray(labels)).astype(str)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    # draw exclude first (background), then niches on top
    for niche in ["exclude", *NICHE_CLASSES]:
        m = (labels == niche).values
        if m.sum() == 0:
            continue
        ax.scatter(
            coords[m, 1], -coords[m, 0],
            s=spot_size, c=NICHE_COLORS[niche], label=niche.replace("_", " "),
            linewidths=0, alpha=0.9 if niche != "exclude" else 0.3,
        )
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
        markerscale=1.5, handletextpad=0.3, borderaxespad=0.0,
    )
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


def plot_mes_map(coords: np.ndarray, mes_prob: np.ndarray, title: str, out_path: str, spot_size: float = 14.0):
    """MES probability spatial map. `title` accepted but not drawn."""
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    sc = ax.scatter(
        coords[:, 1], -coords[:, 0], c=mes_prob, s=spot_size,
        cmap="viridis", vmin=0, vmax=1, linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("MES probability")
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.8)
    fig.tight_layout()
    _save(fig, out_path)


def plot_embedding(
    emb: np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str,
    out_path: str,
):
    """2D embedding (UMAP) colored by niche. `title` accepted but not drawn."""
    labels = pd.Series(np.asarray(labels)).astype(str)
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    for niche in ["exclude", *NICHE_CLASSES]:
        m = (labels == niche).values
        if m.sum() == 0:
            continue
        ax.scatter(
            emb[m, 0], emb[m, 1], s=5, c=NICHE_COLORS[niche],
            label=niche.replace("_", " "), linewidths=0,
            alpha=0.65 if niche != "exclude" else 0.25,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
        markerscale=2.2, handletextpad=0.3, borderaxespad=0.0,
    )
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
