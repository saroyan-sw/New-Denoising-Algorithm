from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS))

from denoise_research import torch_optional
from denoise_research.experiment_utils import add_awgn, random_crop
from denoise_research.io import image_files, load_image
from denoise_research.metrics import full_reference_metrics
from run_archive_discriminator import optimize


DEFAULT_FRAGMENT = Path(
    r"C:\Users\user\.codex\visualizations\2026\09\30\01a0f28f-023d-7942-ad9f-1f49a1949a97\50-denoising-examples.html"
)


def save_rgb(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), "RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def thumbnail_data(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((112, 112), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=58, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_fragment(records: list[dict[str, object]], path: Path) -> None:
    for record in records:
        record["clean_src"] = thumbnail_data(Path(str(record.pop("clean_path"))))
        record["noisy_src"] = thumbnail_data(Path(str(record.pop("noisy_path"))))
        record["restored_src"] = thumbnail_data(Path(str(record.pop("restored_path"))))

    data = json.dumps(records, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    fragment = f'''<div id="denoise-examples-50">
  <h2>50 denoising examples</h2>
  <p class="text-muted">Select a restored preview to compare clean, noisy, and restored crops.</p>
  <div id="denoise-example-detail" aria-live="polite"></div>
  <div id="denoise-example-grid" class="viz-grid" aria-label="Restored previews from 50 examples"></div>
</div>
<script>
(() => {{
  const root = document.getElementById("denoise-examples-50");
  if (!root) return;
  const examples = {data};
  const grid = root.querySelector("#denoise-example-grid");
  const detail = root.querySelector("#denoise-example-detail");
  const state = window.openai && window.openai.widgetState;
  let selected = Number(state && state.modelContent && state.modelContent.selectedIndex);
  if (!Number.isInteger(selected) || selected < 0 || selected >= examples.length) selected = 0;

  function show(index, persist = true) {{
    selected = index;
    const item = examples[index];
    detail.innerHTML = `<p><strong>Example ${{index + 1}} / ${{examples.length}}</strong> · ${{item.dataset}} / ${{item.image}} · PSNR ${{item.noisy_psnr.toFixed(2)}} → ${{item.restored_psnr.toFixed(2)}} dB</p>
      <div class="viz-grid">
        <figure><img src="${{item.clean_src}}" width="112" height="112" alt="Clean reference crop"><figcaption>Clean</figcaption></figure>
        <figure><img src="${{item.noisy_src}}" width="112" height="112" alt="Noisy crop before denoising"><figcaption>Noisy</figcaption></figure>
        <figure><img src="${{item.restored_src}}" width="112" height="112" alt="Restored crop after coefficient fitting"><figcaption>Restored</figcaption></figure>
      </div>`;
    for (const button of grid.querySelectorAll("button")) {{
      const active = Number(button.dataset.index) === index;
      button.setAttribute("aria-pressed", String(active));
      button.classList.toggle("is-selected", active);
    }}
    if (persist && window.openai && window.openai.setWidgetState) {{
      window.openai.setWidgetState({{modelContent: {{selectedIndex: index}}}}).catch(() => {{}});
    }}
  }}

  grid.innerHTML = examples.map((item, index) => `<button type="button" class="btn viz-tile" data-index="${{index}}" aria-pressed="false" aria-label="Example ${{index + 1}}, ${{item.dataset}} ${{item.image}}">
    <img src="${{item.restored_src}}" width="72" height="72" alt=""><span>${{index + 1}}</span></button>`).join("");
  grid.addEventListener("click", (event) => {{
    const button = event.target.closest("button[data-index]");
    if (button) show(Number(button.dataset.index));
  }});
  show(selected, false);
  window.addEventListener("openai:set_globals", (event) => {{
    const next = event.detail && event.detail.globals && event.detail.globals.widgetState;
    const index = Number(next && next.modelContent && next.modelContent.selectedIndex);
    if (Number.isInteger(index) && index >= 0 && index < examples.length) show(index, false);
  }});
}})();
</script>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise and visualize 50 archive-faithful examples.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/netD_Clean_epoch_100.pth")
    parser.add_argument(
        "--output-dir", type=Path,
        default=EXPERIMENTS / "results/visual_50_archive_faithful",
    )
    parser.add_argument("--fragment-out", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Rebuild the visualization from already-denoised images without rerunning optimization.",
    )
    args = parser.parse_args()

    if args.reuse_existing:
        metrics_path = args.output_dir / "metrics.csv"
        image_dir = args.output_dir / "images"
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        records = []
        for row in rows:
            image_key = f"{int(row['example']):02d}_{row['dataset']}_{Path(row['image']).stem}"
            records.append(
                {
                    "dataset": row["dataset"],
                    "image": row["image"],
                    "low": float(row["low"]),
                    "mid": float(row["mid"]),
                    "high": float(row["high"]),
                    "noisy_psnr": float(row["noisy_psnr"]),
                    "restored_psnr": float(row["restored_psnr"]),
                    "clean_path": str(image_dir / f"{image_key}_clean.png"),
                    "noisy_path": str(image_dir / f"{image_key}_noisy.png"),
                    "restored_path": str(image_dir / f"{image_key}_restored.png"),
                }
            )
        write_fragment(records, args.fragment_out)
        print(f"Rebuilt visualization from {len(records)} saved examples: {args.fragment_out}")
        return

    torch_optional.require_torch()
    torch = torch_optional.torch
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    discriminator = torch_optional.load_clean_discriminator(args.checkpoint, args.device)
    image_dir = args.output_dir / "images"
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    selected: list[tuple[str, Path]] = []
    for dataset, count in (("CBSD68", 26), ("Kodak24", 24)):
        paths = image_files(ROOT / "data" / dataset)
        if len(paths) < count:
            raise RuntimeError(f"Expected at least {count} images in {dataset}; found {len(paths)}")
        selected.extend((dataset, path) for path in paths[:count])

    for index, (dataset, path) in enumerate(selected):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, index]))
        clean = random_crop(load_image(path), args.crop_size, rng)
        noisy = np.clip(add_awgn(clean, args.sigma, rng), 0.0, 1.0)
        noisy_tensor = torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))
        noisy_tensor = noisy_tensor.permute(2, 0, 1).unsqueeze(0).to(args.device)
        restored_tensor, fit = optimize(
            torch,
            noisy_tensor,
            discriminator,
            steps=args.steps,
            use_mean_constraint=True,
        )
        restored = restored_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_id = path.name
        image_key = f"{index + 1:02d}_{dataset}_{path.stem}"
        clean_path = image_dir / f"{image_key}_clean.png"
        noisy_path = image_dir / f"{image_key}_noisy.png"
        restored_path = image_dir / f"{image_key}_restored.png"
        save_rgb(clean, clean_path)
        save_rgb(noisy, noisy_path)
        save_rgb(restored, restored_path)
        noisy_metrics = full_reference_metrics(clean, noisy)
        restored_metrics = full_reference_metrics(clean, restored)
        weights = fit["coefficients_low_mid_high"]
        rows.append(
            {
                "example": index + 1,
                "dataset": dataset,
                "image": image_id,
                "low": weights[0],
                "mid": weights[1],
                "high": weights[2],
                "noisy_psnr": noisy_metrics["psnr"],
                "restored_psnr": restored_metrics["psnr"],
                "noisy_ssim": noisy_metrics["ssim"],
                "restored_ssim": restored_metrics["ssim"],
            }
        )
        records.append(
            {
                "dataset": dataset,
                "image": image_id,
                "low": weights[0],
                "mid": weights[1],
                "high": weights[2],
                "noisy_psnr": noisy_metrics["psnr"],
                "restored_psnr": restored_metrics["psnr"],
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "restored_path": str(restored_path),
            }
        )
        print(
            f"{index + 1}/50 {dataset}/{image_id}: "
            f"PSNR {noisy_metrics['psnr']:.2f} -> {restored_metrics['psnr']:.2f} dB; "
            f"weights {weights}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "algorithm": "archive Gaussian low + averaged Gabor mid + residual high",
        "coefficient_rule": "absolute coefficients normalized to sum to 1",
        "mean_constraint": True,
        "mean_constraint_weight": 5.0,
        "fidelity_weight": 10.0,
        "learning_rate": 0.01,
        "steps": args.steps,
        "noise": {"type": "AWGN then clipped to [0,1]", "sigma_8bit": args.sigma, "seed": args.seed},
        "crop_size": args.crop_size,
        "count": len(records),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_fragment(records, args.fragment_out)
    print(f"Saved 50 examples under {args.output_dir}")
    print(f"Wrote visualization fragment: {args.fragment_out}")


if __name__ == "__main__":
    main()
