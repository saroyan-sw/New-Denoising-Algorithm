from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from denoise_research import torch_optional
from denoise_research.io import image_files, load_image
from denoise_research.metrics import full_reference_metrics


DEFAULT_FRAGMENT = Path(
    r"C:\Users\user\.codex\visualizations\2026\09\30\01a0f28f-023d-7942-ad9f-1f49a1949a97\nima-100-crops.html"
)


def crop_and_noise(path: Path, size: int, sigma: float, seed_parts: list[int]) -> tuple[np.ndarray, np.ndarray, int, int]:
    image = load_image(path)
    height, width = image.shape[:2]
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    clean = image[top : top + size, left : left + size].copy()
    noise = rng.normal(0.0, sigma / 255.0, size=clean.shape)
    noisy = np.clip(clean + noise, 0.0, 1.0).astype(np.float32)
    return clean, noisy, top, left


def selected_crops(seed: int, size: int, sigma: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for dataset_index, (dataset, count) in enumerate((("CBSD68", 50), ("Kodak24", 50))):
        paths = image_files(ROOT / "data" / dataset)
        if not paths:
            raise RuntimeError(f"No images found under {ROOT / 'data' / dataset}")
        for local_index in range(count):
            source_index = local_index % len(paths)
            repeat_index = local_index // len(paths)
            path = paths[source_index]
            clean, noisy, top, left = crop_and_noise(
                path, size, sigma, [seed, dataset_index, source_index, repeat_index]
            )
            selected.append({
                "dataset": dataset,
                "path": path,
                "clean": clean,
                "noisy": noisy,
                "top": top,
                "left": left,
                "repeat": repeat_index,
            })
    return selected


def make_batch_fusion(torch: object, batch_size: int, channels: int = 3) -> object:
    nn = torch.nn
    functional = torch.nn.functional

    class BatchFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            coords = torch.arange(9, dtype=torch.float32) - 4
            profile = torch.exp(-(coords**2) / (2 * 2.0**2))
            profile /= profile.sum()
            gaussian = (profile[:, None] * profile[None, :]).view(1, 1, 9, 9)
            self.register_buffer("blur_kernel", gaussian.repeat(channels, 1, 1, 1))
            gabors = [
                cv2.getGaborKernel((5, 5), 3.0, (theta / 4) * np.pi, np.pi / 4, 0.5, 0, ktype=cv2.CV_32F)
                for theta in range(8)
            ]
            gabor = torch.from_numpy(np.mean(gabors, axis=0)).view(1, 1, 5, 5)
            self.register_buffer("gabor_kernel", gabor.repeat(channels, 1, 1, 1))
            self.raw_coefficients = nn.Parameter(torch.full((batch_size, 3), 0.33))

        def coefficients(self) -> object:
            values = self.raw_coefficients.abs()
            return values / (values.sum(dim=1, keepdim=True) + 1e-8)

        def forward(self, image: object) -> object:
            low = functional.conv2d(image, self.blur_kernel, padding=4, groups=channels)
            mid = functional.conv2d(image, self.gabor_kernel, padding=2, groups=channels)
            high = image - low - mid
            coeffs = self.coefficients()
            return (
                coeffs[:, 0, None, None, None] * low
                + coeffs[:, 1, None, None, None] * mid
                + coeffs[:, 2, None, None, None] * high
            )

    return BatchFusion()


def optimize_batch(torch: object, noisy: object, scorer: object, steps: int) -> tuple[object, list[dict[str, object]]]:
    model = make_batch_fusion(torch, noisy.shape[0], noisy.shape[1]).to(noisy.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    first_scores = None
    first_losses = None
    last_scores = None
    last_losses = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(noisy)
        scores = scorer(reconstruction)
        fidelity = (reconstruction - noisy).abs().mean(dim=(1, 2, 3))
        brightness = (reconstruction.mean(dim=(1, 2, 3)) - noisy.mean(dim=(1, 2, 3))).abs()
        losses = -scores / 10.0 + 10.0 * fidelity + 5.0 * brightness
        losses.mean().backward()
        optimizer.step()
        if first_scores is None:
            first_scores = scores.detach().cpu().tolist()
            first_losses = losses.detach().cpu().tolist()
        last_scores = scores.detach().cpu().tolist()
        last_losses = losses.detach().cpu().tolist()
    with torch.no_grad():
        result = model(noisy).clamp(0.0, 1.0)
        final_scores = scorer(result).detach().cpu().tolist()
        coefficients = model.coefficients().detach().cpu().tolist()
    records = [
        {
            "coefficients_low_mid_high": coefficients[i],
            "loss_first": first_losses[i],
            "loss_last": last_losses[i],
            "nima_score_first": first_scores[i],
            "nima_score_last_preclip": last_scores[i],
            "nima_score_final_clipped": final_scores[i],
        }
        for i in range(noisy.shape[0])
    ]
    return result, records


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255 + 0.5), "RGB").save(path)


def thumbnail_data(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((88, 88), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=48, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_fragment(records: list[dict[str, object]], path: Path) -> None:
    visual = []
    for row in records:
        visual.append({
            **{key: row[key] for key in ("example", "dataset", "image", "top", "left", "low", "mid", "high", "noisy_psnr", "restored_psnr", "noisy_ssim", "restored_ssim", "nima_before", "nima_after")},
            "clean_src": thumbnail_data(Path(str(row["clean_path"]))),
            "noisy_src": thumbnail_data(Path(str(row["noisy_path"]))),
            "restored_src": thumbnail_data(Path(str(row["restored_path"]))),
        })
    data = json.dumps(visual, separators=(",", ":")).replace("</", "<\\/")
    mean_noisy = float(np.mean([float(r["noisy_psnr"]) for r in records]))
    mean_restored = float(np.mean([float(r["restored_psnr"]) for r in records]))
    mean_ssim = float(np.mean([float(r["restored_ssim"]) for r in records]))
    fragment = f'''<div id="nima-crops-100">
  <h2>NIMA-guided denoising · 100 crops</h2>
  <div class="viz-grid">
    <div class="card viz-stat"><span class="text-muted">Noisy PSNR</span><span class="viz-stat-value tabular-nums">{mean_noisy:.2f} dB</span></div>
    <div class="card viz-stat"><span class="text-muted">Restored PSNR</span><span class="viz-stat-value tabular-nums">{mean_restored:.2f} dB</span></div>
    <div class="card viz-stat"><span class="text-muted">Restored SSIM</span><span class="viz-stat-value tabular-nums">{mean_ssim:.3f}</span></div>
  </div>
  <div id="nima-100-detail" aria-live="polite"></div>
  <div id="nima-100-grid" class="viz-grid" aria-label="One hundred restored crop previews"></div>
</div>
<script>
(() => {{
  const root = document.getElementById("nima-crops-100");
  if (!root) return;
  const examples = {data};
  const grid = root.querySelector("#nima-100-grid");
  const detail = root.querySelector("#nima-100-detail");
  const saved = window.openai && window.openai.widgetState;
  let selected = Number(saved && saved.modelContent && saved.modelContent.selectedIndex);
  if (!Number.isInteger(selected) || selected < 0 || selected >= examples.length) selected = 0;
  const f = value => Number(value).toFixed(3);
  function show(index, persist = true) {{
    selected = index;
    const item = examples[index];
    const gain = item.restored_psnr - item.noisy_psnr;
    detail.innerHTML = `<p><strong>Example ${{index + 1}} / 100</strong> · ${{item.dataset}}/${{item.image}} · crop (${{item.top}}, ${{item.left}}) · PSNR ${{item.noisy_psnr.toFixed(2)}} → ${{item.restored_psnr.toFixed(2)}} dB (${{gain >= 0 ? "+" : ""}}${{gain.toFixed(2)}})</p>
      <div class="viz-grid">
        <figure><img src="${{item.clean_src}}" width="88" height="88" alt="Clean reference crop"><figcaption>Clean</figcaption></figure>
        <figure><img src="${{item.noisy_src}}" width="88" height="88" alt="Crop after Gaussian noise"><figcaption>Noisy · SSIM ${{f(item.noisy_ssim)}}</figcaption></figure>
        <figure><img src="${{item.restored_src}}" width="88" height="88" alt="NIMA-guided restored crop"><figcaption>Restored · SSIM ${{f(item.restored_ssim)}}</figcaption></figure>
      </div>
      <p class="text-small text-muted">Weights L/M/H: ${{f(item.low)}} / ${{f(item.mid)}} / ${{f(item.high)}} · NIMA: ${{f(item.nima_before)}} → ${{f(item.nima_after)}}</p>`;
    for (const button of grid.querySelectorAll("button")) {{
      const active = Number(button.dataset.index) === index;
      button.setAttribute("aria-pressed", String(active));
      button.classList.toggle("is-selected", active);
    }}
    if (persist && window.openai && window.openai.setWidgetState) {{
      window.openai.setWidgetState({{modelContent: {{selectedIndex: index}}}}).catch(() => {{}});
    }}
  }}
  grid.innerHTML = examples.map((item, index) => `<button type="button" class="btn viz-tile" data-index="${{index}}" aria-pressed="false" aria-label="Example ${{index + 1}}, ${{item.dataset}} ${{item.image}}"><img src="${{item.restored_src}}" width="56" height="56" alt=""><span>${{index + 1}}</span></button>`).join("");
  grid.addEventListener("click", event => {{
    const button = event.target.closest("button[data-index]");
    if (button) show(Number(button.dataset.index));
  }});
  show(selected, false);
  window.addEventListener("openai:set_globals", event => {{
    const state = event.detail && event.detail.globals && event.detail.globals.widgetState;
    const index = Number(state && state.modelContent && state.modelContent.selectedIndex);
    if (Number.isInteger(index) && index >= 0 && index < examples.length) show(index, false);
  }});
}})();
</script>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and visualize NIMA technical fitting on 100 noisy crops.")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "results/nima_technical_100")
    parser.add_argument("--fragment-out", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch_optional.require_torch()
    torch = torch_optional.torch
    from deepinv.loss.metric import NIMA

    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    scorer = NIMA(variant="technical", device=args.device).eval()
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "nima_mobilenet_technical_weights.pt"
    examples = selected_crops(args.seed, args.crop_size, args.sigma)
    image_dir = args.output_dir / "images"
    rows: list[dict[str, object]] = []

    for start in range(0, len(examples), args.batch_size):
        group = examples[start : start + args.batch_size]
        noisy_batch = np.stack([item["noisy"] for item in group])
        tensor = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(args.device)
        restored_tensor, fits = optimize_batch(torch, tensor, scorer, args.steps)
        restored_batch = restored_tensor.permute(0, 2, 3, 1).cpu().numpy()
        for offset, (item, restored, fit) in enumerate(zip(group, restored_batch, fits)):
            index = start + offset + 1
            clean = item["clean"]
            noisy = item["noisy"]
            path = item["path"]
            noisy_metrics = full_reference_metrics(clean, noisy)
            restored_metrics = full_reference_metrics(clean, restored)
            key = f"{index:03d}_{item['dataset']}_{path.stem}_r{item['repeat']}"
            clean_path = image_dir / f"{key}_clean.png"
            noisy_path = image_dir / f"{key}_noisy.png"
            restored_path = image_dir / f"{key}_restored.png"
            save_rgb(clean, clean_path)
            save_rgb(noisy, noisy_path)
            save_rgb(restored, restored_path)
            weights = fit["coefficients_low_mid_high"]
            rows.append({
                "example": index,
                "dataset": item["dataset"],
                "image": path.name,
                "top": item["top"],
                "left": item["left"],
                "low": weights[0],
                "mid": weights[1],
                "high": weights[2],
                "noisy_psnr": noisy_metrics["psnr"],
                "restored_psnr": restored_metrics["psnr"],
                "noisy_ssim": noisy_metrics["ssim"],
                "restored_ssim": restored_metrics["ssim"],
                "nima_before": fit["nima_score_first"],
                "nima_after": fit["nima_score_final_clipped"],
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path),
                "restored_path": str(restored_path),
            })
        print(f"Completed {min(start + args.batch_size, len(examples))}/100 crops", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_fields = [key for key in rows[0] if not key.endswith("_path")]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in csv_fields} for row in rows])
    summary = {
        "quality_model": "NIMA technical MobileNet, TID2013",
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "count": 100,
        "datasets": {"CBSD68": 50, "Kodak24": 50},
        "crop_size": args.crop_size,
        "noise": {"type": "AWGN then clipped to [0,1]", "sigma_8bit": args.sigma, "seed": args.seed},
        "steps": args.steps,
        "batch_size": args.batch_size,
        "objective": "-NIMA/10 + 10*L1(reconstruction,noisy) + 5*mean-brightness difference",
        "mean_noisy_psnr": float(np.mean([r["noisy_psnr"] for r in rows])),
        "mean_restored_psnr": float(np.mean([r["restored_psnr"] for r in rows])),
        "mean_noisy_ssim": float(np.mean([r["noisy_ssim"] for r in rows])),
        "mean_restored_ssim": float(np.mean([r["restored_ssim"] for r in rows])),
        "mean_coefficients_low_mid_high": [float(np.mean([r[key] for r in rows])) for key in ("low", "mid", "high")],
        "improved_psnr_count": int(sum(r["restored_psnr"] > r["noisy_psnr"] for r in rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_fragment(rows, args.fragment_out)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote visualization: {args.fragment_out}", flush=True)


if __name__ == "__main__":
    main()
