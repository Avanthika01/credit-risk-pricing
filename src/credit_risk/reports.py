"""Charts (PNG) and markdown tables for the reports/ directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Validated categorical slots 1-2 (CVD-safe pair), recessive neutrals for axes/grid.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e0"


def _style(ax, title: str, xlabel: str, ylabel: str, subtitle: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig():
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def roc_chart(y, preds: dict[str, np.ndarray], aucs: dict[str, float], path: Path, subtitle: str) -> None:
    fig, ax = _fig()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Chance")
    for i, (name, p) in enumerate(preds.items()):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=SERIES[i], linewidth=2, label=f"{name} (AUC {aucs[name]:.3f})")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    _style(ax, "ROC curve", "False positive rate (goods declined)", "True positive rate (bads caught)", subtitle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)


def calibration_chart(tables: dict[str, pd.DataFrame], path: Path, subtitle: str) -> None:
    fig, ax = _fig()
    hi = max(max(t["mean_pd"].max(), t["observed"].max()) for t in tables.values()) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    markers = ["o", "s"]
    for i, (name, t) in enumerate(tables.items()):
        ax.plot(t["mean_pd"], t["observed"], color=SERIES[i], linewidth=2, marker=markers[i], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    _style(ax, "Calibration by predicted-PD decile", "Mean predicted PD", "Observed default rate", subtitle)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    _save(fig, path)


def cutoff_chart(table: pd.DataFrame, chosen: float, path: Path, subtitle: str) -> None:
    fig, ax = _fig()
    ax.plot(table["approval_rate"], table["bad_rate_approved"], color=SERIES[0], linewidth=2, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    for _, r in table.iterrows():
        if any(np.isclose(r["pd_cutoff"], v) for v in (0.10, 0.30, 0.40)):
            ax.annotate(f"PD \u2264 {r['pd_cutoff']:.2f}", (r["approval_rate"], r["bad_rate_approved"]),
                        textcoords="offset points", xytext=(-6, 8), ha="right", fontsize=8, color=INK_2)
    c = table.loc[np.isclose(table["pd_cutoff"], chosen)]
    if not c.empty:
        r = c.iloc[0]
        ax.plot(r["approval_rate"], r["bad_rate_approved"], marker="o", markersize=10, color=SERIES[1],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.annotate(f"policy cutoff PD \u2264 {chosen:.2f}\napprove {r['approval_rate']:.1%}, bad rate "
                    f"{r['bad_rate_approved']:.1%}", (r["approval_rate"], r["bad_rate_approved"]),
                    textcoords="offset points", xytext=(10, -28), fontsize=8, color=INK)
    _style(ax, "Approval rate vs bad rate among approved", "Approval rate", "Bad rate among approved", subtitle)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, path)


def pricing_chart(bench: pd.DataFrame, path: Path, subtitle: str) -> None:
    fig, ax = _fig()
    x = np.arange(len(bench))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, bench["lc_mean_rate"], width=w, color=SERIES[0], label="Lending Club actual rate")
    ax.bar(x + w / 2 + 0.01, bench["model_mean_rate"], width=w, color=SERIES[1], label="Model risk-based rate")
    ax.set_xticks(x, bench["grade"])
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    _style(ax, "Mean interest rate by Lending Club grade", "Lending Club grade", "Mean annual rate", subtitle)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def to_markdown(df: pd.DataFrame, floatfmt: dict | None = None, default_fmt: str = "{:.4f}") -> str:
    """Minimal markdown table writer (avoids a tabulate dependency)."""
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.get(c, default_fmt).format(v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
