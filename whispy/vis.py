"""Visualization module — matplotlib plots for results and live data.

Usage:
    from whispy.vis import plot_confusion_matrix, plot_accuracy_comparison
    plot_confusion_matrix(metrics.confusion_matrix, labels=["empty","one","two","five"])
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib required — pip install whispy[vis]")


# =========================================================================
# Confusion matrix
# =========================================================================
def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    show: bool = True,
) -> Any:
    """Plot a confusion matrix heatmap."""
    plt = _ensure_matplotlib()
    cm = np.array(cm)
    n = cm.shape[0]
    labels = labels or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(4, n * 0.8), max(3.5, n * 0.7)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontweight="bold", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =========================================================================
# Accuracy comparison bar chart
# =========================================================================
def plot_accuracy_comparison(
    results: dict[str, float],
    title: str = "Model Accuracy Comparison",
    save_path: str | None = None,
    show: bool = True,
) -> Any:
    """Bar chart of model accuracies.

    Parameters
    ----------
    results : dict mapping model name → accuracy.
    """
    plt = _ensure_matplotlib()

    names = list(results.keys())
    accs = list(results.values())

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.2), 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, accs, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_ylim(0, min(1.15, max(accs) + 0.1))
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =========================================================================
# PCA scatter
# =========================================================================
def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str] | None = None,
    title: str = "PCA 2D Projection",
    save_path: str | None = None,
    show: bool = True,
) -> Any:
    """2D PCA scatter plot colored by class."""
    plt = _ensure_matplotlib()
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    classes = np.unique(y)
    labels = labels or [str(c) for c in classes]
    cmap = plt.cm.get_cmap("Set1", max(len(classes), 3))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(classes):
        mask = y == c
        idx = np.where(mask)[0]
        if len(idx) > 1500:
            idx = np.random.choice(idx, 1500, replace=False)
        lbl = labels[i] if i < len(labels) else str(c)
        ax.scatter(X2[idx, 0], X2[idx, 1], s=10, alpha=0.6,
                   c=[cmap(i)], label=lbl, edgecolors="none")

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="best", markerscale=2)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =========================================================================
# Live CSI stream plot (blocking — runs until Ctrl+C)
# =========================================================================
def plot_live_stream(
    port: str,
    baud: int = 115200,
    n_subcarriers: int = 52,
    window: int = 300,
) -> None:
    """Real-time matplotlib plot of CSI amplitude from serial.

    Parameters
    ----------
    port : serial port.
    baud : baud rate.
    n_subcarriers : number of subcarriers to plot (after mask).
    window : number of time samples shown in the rolling window.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    from whispy.core import CSI_SUBCARRIER_MASK, parse_csi_line

    try:
        import serial
    except ImportError:
        raise ImportError("pyserial required — pip install whispy[collect]")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Live CSI Amplitude", fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Amplitude (mean across subcarriers)")

    data = np.zeros(window)
    line_obj, = ax.plot(data, color="steelblue", linewidth=0.8)
    ax.set_ylim(0, 50)

    plt.ion()
    plt.show()

    ser = serial.Serial(port, baudrate=baud, timeout=0.05)
    _ = ser.read(ser.in_waiting or 1)

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="ignore").strip()
            parsed = parse_csi_line(text)
            if parsed is None:
                continue
            mag = np.sqrt(parsed["real"]**2 + parsed["imag"]**2)
            mag_filtered = mag[CSI_SUBCARRIER_MASK]
            val = mag_filtered.mean()

            data = np.roll(data, -1)
            data[-1] = val
            line_obj.set_ydata(data)
            ax.set_ylim(0, max(data.max() * 1.2, 1))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        plt.ioff()
        plt.close()
        print("[vis] Live stream stopped.")
