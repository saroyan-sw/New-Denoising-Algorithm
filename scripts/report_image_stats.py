"""Summarize PNG dimensions and file sizes in the four canonical datasets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import struct
from collections import Counter
from pathlib import Path


DATASETS = ("DIV2K_train", "DIV2K_valid", "CBSD68", "Kodak24")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {0: "grayscale", 2: "RGB", 3: "indexed", 4: "grayscale+alpha", 6: "RGBA"}


def png_metadata(path: Path) -> tuple[int, int, int, str]:
    with path.open("rb") as image:
        header = image.read(33)
    if len(header) < 33 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    if width == 0 or height == 0 or color_type not in COLOR_TYPES:
        raise ValueError(f"Invalid PNG dimensions or color type: {path}")
    return width, height, bit_depth, COLOR_TYPES[color_type]


def summarize(rows: list[dict]) -> dict:
    widths = [row["width"] for row in rows]
    heights = [row["height"] for row in rows]
    pixels = [row["megapixels"] for row in rows]
    sizes = [row["size_mb"] for row in rows]
    return {
        "images": len(rows),
        "total_gb": round(sum(row["bytes"] for row in rows) / 1_000_000_000, 3),
        "distinct_dimensions": len(set((row["width"], row["height"]) for row in rows)),
        "width_px": {"min": min(widths), "median": statistics.median(widths), "max": max(widths)},
        "height_px": {"min": min(heights), "median": statistics.median(heights), "max": max(heights)},
        "megapixels": {"min": round(min(pixels), 3), "median": round(statistics.median(pixels), 3), "max": round(max(pixels), 3)},
        "size_mb": {"min": round(min(sizes), 3), "mean": round(statistics.mean(sizes), 3), "median": round(statistics.median(sizes), 3), "max": round(max(sizes), 3)},
        "orientations": dict(sorted(Counter(row["orientation"] for row in rows).items())),
        "formats": dict(sorted(Counter(f'{row["color_type"]} {row["bit_depth"]}-bit' for row in rows).items())),
        "common_dimensions": [
            {"dimensions": f"{width}x{height}", "images": count}
            for (width, height), count in Counter((row["width"], row["height"]) for row in rows).most_common(5)
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "image_stats")
    args = parser.parse_args()

    all_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for dataset in DATASETS:
        folder = args.data_dir / dataset
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise ValueError(f"No PNG images found in {folder}")
        rows = []
        for path in paths:
            width, height, bit_depth, color_type = png_metadata(path)
            size_bytes = path.stat().st_size
            row = {
                "dataset": dataset,
                "file": path.name,
                "width": width,
                "height": height,
                "megapixels": round(width * height / 1_000_000, 6),
                "bytes": size_bytes,
                "size_mb": round(size_bytes / 1_000_000, 6),
                "orientation": "landscape" if width > height else "portrait" if width < height else "square",
                "bit_depth": bit_depth,
                "color_type": color_type,
            }
            rows.append(row)
            all_rows.append(row)
        summary[dataset] = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "images.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary["all"] = summarize(all_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
