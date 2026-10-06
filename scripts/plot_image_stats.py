"""Plot image dimensions and PNG file sizes from report_image_stats.py output.

Run from the project root after installing the optional plotting dependency:
    python scripts/report_image_stats.py
    python scripts/plot_image_stats.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path


DATASETS = ("DIV2K_train", "DIV2K_valid", "CBSD68", "Kodak24")
LABELS = ("DIV2K train", "DIV2K validation", "CBSD68", "Kodak24")
COLORS = ("#2563eb", "#7c3aed", "#d97706", "#059669")


def read_rows(path: Path) -> dict[str, list[dict[str, float]]]:
    groups: dict[str, list[dict[str, float]]] = {name: [] for name in DATASETS}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset = row["dataset"]
            if dataset not in groups:
                continue
            groups[dataset].append(
                {
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                    "megapixels": float(row["megapixels"]),
                    "size_mb": float(row["size_mb"]),
                }
            )
    if any(not rows for rows in groups.values()):
        raise ValueError("The CSV must contain images from all four canonical datasets")
    return groups


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=root / "outputs" / "image_stats" / "images.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "image_stats" / "dataset_sizes.png")
    args = parser.parse_args()
    groups = read_rows(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / ".mplconfig"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit('Matplotlib is required. Install it with: python -m pip install -e ".[viz]"') from exc

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1), height_ratios=(1, 0.8))
    ax_dimensions = fig.add_subplot(grid[:, 0])
    ax_scatter = fig.add_subplot(grid[0, 1])
    ax_boxes = fig.add_subplot(grid[1, 1])

    for name, label, color in zip(DATASETS, LABELS, COLORS):
        rows = groups[name]
        dimension_counts = Counter((row["width"], row["height"]) for row in rows)
        ax_dimensions.scatter(
            [width for width, _ in dimension_counts],
            [height for _, height in dimension_counts],
            s=[24 + 24 * math.sqrt(count) for count in dimension_counts.values()],
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.7,
            label=f"{label} ({len(rows)})",
        )
        ax_scatter.scatter(
            [row["megapixels"] for row in rows],
            [row["size_mb"] for row in rows],
            s=14,
            color=color,
            alpha=0.45,
            edgecolor="none",
            label=label,
        )

    ax_dimensions.set(title="Image dimensions", xlabel="Width (pixels)", ylabel="Height (pixels)")
    ax_dimensions.set_xlim(0, 2200)
    ax_dimensions.set_ylim(0, 2200)
    ax_dimensions.set_aspect("equal", adjustable="box")
    ax_dimensions.legend(loc="upper left", frameon=False, fontsize=9)
    ax_dimensions.text(0.98, 0.03, "Circle area reflects image count", transform=ax_dimensions.transAxes,
                       ha="right", va="bottom", fontsize=8, color="#475569")

    ax_scatter.set(title="File size versus pixel count", xlabel="Megapixels", ylabel="PNG file size (MB)")
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim(bottom=0)

    box_data = [[row["size_mb"] for row in groups[name]] for name in DATASETS]
    boxes = ax_boxes.boxplot(box_data, orientation="horizontal", positions=range(1, 5), widths=0.55,
                             whis=(0, 100), showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
    for median in boxes["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.8)
    ax_boxes.set_yticks(range(1, 5), LABELS)
    ax_boxes.invert_yaxis()
    ax_boxes.set_xscale("log")
    ax_boxes.set(title="File size ranges and medians", xlabel="PNG file size (MB, log scale)")

    for ax in (ax_dimensions, ax_scatter, ax_boxes):
        ax.grid(True, alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    total_images = sum(len(rows) for rows in groups.values())
    total_gb = sum(row["size_mb"] for rows in groups.values() for row in rows) / 1000
    fig.suptitle(f"Image dataset sizes  ·  {total_images} images  ·  {total_gb:.2f} GB on disk", fontsize=16)
    fig.savefig(args.output, dpi=170)
    svg_path = args.output.with_suffix(".svg")
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved {args.output} and {svg_path}")


if __name__ == "__main__":
    main()
