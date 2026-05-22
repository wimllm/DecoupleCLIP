import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ADAPTER_ABLATION = {
    "Adapter dim": [128, 256, 384, 768],
    "Image AUROC": [90.5, 92.5, 92.7, 93.2],
    "Image AP": [95.9, 96.4, 96.6, 96.9],
    "Pixel AUROC": [91.5, 91.7, 92.5, 92.6],
    "Pixel AP": [37.7, 39.3, 40.3, 41.1],
}

SEG_LOSS_ABLATION = {
    "Seg loss weight": [0.25, 0.5, 1.0, 2.0],
    "Image AUROC": [91.6, 92.9, 93.2, 92.3],
    "Image AP": [93.5, 96.7, 96.9, 96.3],
    "Pixel AUROC": [87.9, 90.7, 92.6, 92.2],
    "Pixel AP": [35.9, 38.3, 41.1, 40.4],
}

PRIMARY_METRICS = ["Image AUROC", "Image AP", "Pixel AUROC"]
RIGHT_AXIS_METRIC = "Pixel AP"
STYLES = {
    "Image AUROC": {"color": "#1F77B4", "marker": "o", "hatch": ""},
    "Image AP": {"color": "#FF7F0E", "marker": "s", "hatch": "//"},
    "Pixel AUROC": {"color": "#2CA02C", "marker": "^", "hatch": "\\\\"},
    "Pixel AP": {"color": "#D62728", "marker": "D", "hatch": ".."},
}


def _set_sci_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 1.6,
    })


def _metric_ylim(values, pad_ratio=0.2):
    values = np.asarray(values, dtype=float)
    span = max(values.max() - values.min(), 1.0)
    lower = max(0.0, values.min() - span * pad_ratio)
    upper = min(100.0, values.max() + span * pad_ratio)
    return lower, upper


def _style_axes(ax, ax2=None):
    ax.grid(True, axis="y", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    if ax2 is not None:
        ax2.tick_params(axis="y", which="major", width=1.5, length=6)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.6)


def _combined_legend(fig, ax, ax2, y=0.99):
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=4,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
    )


def plot_adapter_combined_bar(output_path, dpi):
    _set_sci_style()
    dims = ADAPTER_ABLATION["Adapter dim"]
    x = np.arange(len(dims))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    ax2 = ax.twinx()

    offsets = [-1.5 * width, -0.5 * width, 0.5 * width]
    for metric, offset in zip(PRIMARY_METRICS, offsets):
        values = ADAPTER_ABLATION[metric]
        style = STYLES[metric]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=metric,
            color=style["color"],
            edgecolor="black",
            linewidth=1.0,
            hatch=style["hatch"],
            alpha=0.92,
        )

    pixel_ap = ADAPTER_ABLATION[RIGHT_AXIS_METRIC]
    style = STYLES[RIGHT_AXIS_METRIC]
    ax2.bar(
        x + 1.5 * width,
        pixel_ap,
        width=width,
        label=RIGHT_AXIS_METRIC,
        color=style["color"],
        edgecolor="black",
        linewidth=1.0,
        hatch=style["hatch"],
        alpha=0.92,
    )

    primary_values = np.concatenate([
        np.asarray(ADAPTER_ABLATION[metric], dtype=float) for metric in PRIMARY_METRICS
    ])
    ax.set_ylim(*_metric_ylim(primary_values, pad_ratio=0.18))
    ax2.set_ylim(*_metric_ylim(pixel_ap, pad_ratio=0.22))
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in dims])
    ax.set_xlabel("Adapter Dimension")
    ax.set_ylabel("AUROC / AP (%) (Image & Pixel AUROC)")
    ax2.set_ylabel("Pixel AP (%)")
    ax.set_title("Ablation Study on Visual Adapter Capacity", pad=30)

    _style_axes(ax, ax2)
    _combined_legend(fig, ax, ax2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_seg_loss_combined_line(output_path, dpi):
    _set_sci_style()
    weights = SEG_LOSS_ABLATION["Seg loss weight"]
    x = np.asarray(weights, dtype=float)

    fig, ax = plt.subplots(figsize=(11.2, 7.0))
    ax2 = ax.twinx()

    for metric in PRIMARY_METRICS:
        values = SEG_LOSS_ABLATION[metric]
        style = STYLES[metric]
        ax.plot(
            x,
            values,
            label=metric,
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            markeredgecolor=style["color"],
            markeredgewidth=1.2,
            linewidth=3.0,
        )

    pixel_ap = SEG_LOSS_ABLATION[RIGHT_AXIS_METRIC]
    style = STYLES[RIGHT_AXIS_METRIC]
    ax2.plot(
        x,
        pixel_ap,
        label=RIGHT_AXIS_METRIC,
        color=style["color"],
        marker=style["marker"],
        markersize=8,
        markeredgecolor=style["color"],
        markeredgewidth=1.2,
        linewidth=3.0,
    )

    primary_values = np.concatenate([
        np.asarray(SEG_LOSS_ABLATION[metric], dtype=float) for metric in PRIMARY_METRICS
    ])
    ax.set_ylim(*_metric_ylim(primary_values, pad_ratio=0.18))
    ax2.set_ylim(*_metric_ylim(pixel_ap, pad_ratio=0.22))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in weights])
    ax.set_xlabel("Seg Loss Weight")
    ax.set_ylabel("AUROC / AP (%) (Image & Pixel AUROC)")
    ax2.set_ylabel("Pixel AP (%)")
    ax.set_title("Ablation Study on Segmentation Loss Weight", pad=30)

    _style_axes(ax, ax2)
    _combined_legend(fig, ax, ax2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ablation figures for adapter capacity and segmentation loss weight."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="workspaces/ablation_figures",
        help="Directory to save the generated figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster formats.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = save_dir / f"table1_adapter_capacity_combined_bar.{args.format}"
    seg_loss_path = save_dir / f"table2_seg_loss_weight_combined_line.{args.format}"

    plot_adapter_combined_bar(adapter_path, args.dpi)
    plot_seg_loss_combined_line(seg_loss_path, args.dpi)

    print(f"Saved adapter capacity bar chart to: {adapter_path}")
    print(f"Saved segmentation loss weight line chart to: {seg_loss_path}")


if __name__ == "__main__":
    main()
