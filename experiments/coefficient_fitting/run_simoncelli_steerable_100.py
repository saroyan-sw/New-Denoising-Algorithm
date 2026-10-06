from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from denoise_research.metrics import full_reference_metrics
from run_nima_100_visual import save_rgb, selected_crops


def make_masks(height: int, width: int, scales: int = 4, orientations: int = 4,
               device: str = "cpu") -> tuple[list[torch.Tensor], torch.Tensor]:
    """Simoncelli-style log-radial, steerable angular tight-frame masks.

    The order-3 cosine angular basis gives four orientations. Radial masks are
    split with complementary cosine/sine windows, and angular masks are
    normalized to unit squared sum. The returned real masks satisfy
    sum(mask**2) == 1 at every Fourier location, including the low residual.
    """
    fy = torch.fft.fftfreq(height, device=device).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device).view(1, width)
    radius = torch.sqrt(fx.square() + fy.square())
    angle = torch.atan2(fy, fx)

    angles = torch.arange(orientations, device=device, dtype=torch.float32) * (math.pi / orientations)
    raw_angles = torch.cos(angle.unsqueeze(0) - angles[:, None, None]).pow(3)
    angular = raw_angles / torch.sqrt(raw_angles.square().sum(dim=0, keepdim=True) + 1e-12)

    masks: list[torch.Tensor] = []
    residual = torch.ones_like(radius)
    for scale in range(scales):
        cutoff = 0.25 / (2**scale)
        # Smooth transition over one octave in log frequency.
        log_position = (torch.log2(radius.clamp_min(1e-8) / cutoff) + 1.0) / 1.0
        phase = (math.pi / 2.0) * log_position.clamp(0.0, 1.0)
        low = torch.cos(phase)
        high = torch.sin(phase)
        low = torch.where(radius <= cutoff / 2.0, torch.ones_like(low), low)
        high = torch.where(radius <= cutoff / 2.0, torch.zeros_like(high), high)
        low = torch.where(radius >= cutoff, torch.zeros_like(low), low)
        high = torch.where(radius >= cutoff, torch.ones_like(high), high)
        band_radial = residual * high
        for orientation in range(orientations):
            masks.append(band_radial * angular[orientation])
        residual = residual * low
    return masks, residual


def decompose(image: torch.Tensor, masks: list[torch.Tensor], low_mask: torch.Tensor
              ) -> tuple[list[torch.Tensor], torch.Tensor]:
    spectrum = torch.fft.fft2(image)
    bands = [torch.fft.ifft2(spectrum * mask) for mask in masks]
    low = torch.fft.ifft2(spectrum * low_mask)
    return bands, low


def reconstruct(bands: list[torch.Tensor], low: torch.Tensor, masks: list[torch.Tensor],
                low_mask: torch.Tensor) -> torch.Tensor:
    spectrum = torch.zeros_like(torch.fft.fft2(low))
    for coefficient, mask in zip(bands, masks):
        spectrum = spectrum + torch.fft.fft2(coefficient) * mask.conj()
    spectrum = spectrum + torch.fft.fft2(low) * low_mask
    return torch.fft.ifft2(spectrum).real


def fit_batch(torch_module: object, noisy: torch.Tensor, scorer: object,
              masks: list[torch.Tensor], low_mask: torch.Tensor,
              *, thresholding: bool, steps: int, sigma: float) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    bands, low = decompose(noisy, masks, low_mask)
    batch_size = noisy.shape[0]
    number_of_bands = len(bands)
    gain_raw = torch_module.nn.Parameter(torch.zeros(batch_size, number_of_bands, device=noisy.device))
    params = [gain_raw]

    band_noise = torch.tensor(
        [sigma * math.sqrt(float(mask.square().mean())) for mask in masks],
        dtype=noisy.dtype, device=noisy.device,
    ).view(1, number_of_bands, 1, 1, 1)
    if thresholding:
        # softplus(-0.43275) ~= 0.5: begin with a moderate, interpretable threshold.
        threshold_raw = torch_module.nn.Parameter(
            torch.full((batch_size, number_of_bands), -0.43275, device=noisy.device)
        )
        params.append(threshold_raw)

    optimizer = torch_module.optim.Adam(params, lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        gain = 2.0 * torch_module.sigmoid(gain_raw)
        updated: list[torch.Tensor] = []
        for band_index, coefficient in enumerate(bands):
            value = coefficient
            if thresholding:
                threshold = torch_module.nn.functional.softplus(threshold_raw[:, band_index])
                threshold = threshold[:, None, None, None] * band_noise[:, band_index]
                magnitude = value.abs()
                value = value * torch_module.relu(magnitude - threshold) / magnitude.clamp_min(1e-8)
            value = value * gain[:, band_index, None, None, None]
            updated.append(value)

        restored = reconstruct(updated, low, masks, low_mask)
        # Keep prior input handling identical to the earlier NIMA fit: score
        # the in-optimization reconstruction directly; clip only for metrics.
        scores = scorer(restored)
        fidelity = (restored - noisy).abs().mean(dim=(1, 2, 3))
        brightness = (restored.mean(dim=(1, 2, 3)) - noisy.mean(dim=(1, 2, 3))).abs()
        loss = (-scores / 10.0 + 10.0 * fidelity + 5.0 * brightness).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        gain = 2.0 * torch_module.sigmoid(gain_raw)
        final_bands: list[torch.Tensor] = []
        thresholds = torch.zeros_like(gain)
        for band_index, coefficient in enumerate(bands):
            value = coefficient
            if thresholding:
                multiplier = torch_module.nn.functional.softplus(threshold_raw[:, band_index])
                thresholds[:, band_index] = multiplier
                threshold = multiplier[:, None, None, None] * band_noise[:, band_index]
                magnitude = value.abs()
                value = value * torch_module.relu(magnitude - threshold) / magnitude.clamp_min(1e-8)
            final_bands.append(value * gain[:, band_index, None, None, None])
        restored = reconstruct(final_bands, low, masks, low_mask).clamp(0.0, 1.0)
    return restored, gain.detach().cpu().numpy(), thresholds.detach().cpu().numpy()


def write_html(path: Path, rows: list[dict[str, object]]) -> None:
    html_rows = []
    for row in rows:
        index = int(row["example"])
        dataset, image = str(row["dataset"]), str(row["image"])
        key = f"{index:03d}_{dataset}_{Path(image).stem}_r{row['repeat']}"
        shared = f"../nima_technical_100/images/{key}_"
        proposed = f"images/{key}_"
        html_rows.append({
            "index": index, "dataset": dataset, "image": image,
            "top": row["top"], "left": row["left"],
            "clean": shared + "clean.png", "noisy": shared + "noisy.png",
            "gain": proposed + "gain_only.png", "threshold": proposed + "threshold.png",
            "nima": shared + "restored.png", "disc": f"../discriminator_100/images/{key}_discriminator.png",
            "gain_psnr": row["gain_only_psnr"], "threshold_psnr": row["threshold_psnr"],
            "noisy_psnr": row["noisy_psnr"], "nima_psnr": row["nima_psnr"],
            "disc_psnr": row["disc_psnr"],
        })
    payload = json.dumps(html_rows).replace("</", "<\\/")
    document = f'''<!doctype html><meta charset="utf-8"><title>Steerable pyramid · 100 crops</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;color:#202124}}button,select{{font:inherit;padding:8px;margin:4px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0}}img{{width:100%;height:auto;image-rendering:auto}}.selected{{outline:3px solid #3478f6}}.thumbs{{display:grid;grid-template-columns:repeat(10,1fr);gap:4px;margin-top:20px}}.thumbs button{{padding:2px}}.thumbs img{{aspect-ratio:1;object-fit:cover}}</style>
<h1>Simoncelli style steerable pyramid · 100 matched crops</h1>
<p>Click a thumbnail to inspect at full crop resolution. Methods: noisy, gain only, gain plus soft threshold, prior NIMA and archive discriminator.</p>
<div id="summary"></div><label>Example <select id="choice"></select></label><div id="view" class="grid"></div><div id="thumbs" class="thumbs"></div>
<script>const data={payload};const choice=document.querySelector('#choice'),view=document.querySelector('#view'),summary=document.querySelector('#summary'),thumbs=document.querySelector('#thumbs');
for(const x of data){{const o=document.createElement('option');o.value=x.index-1;o.textContent=`${{x.index}} · ${{x.dataset}}/${{x.image}}`;choice.append(o)}}
for(const x of data){{const b=document.createElement('button');b.innerHTML=`<img src="${{x.threshold}}" alt=""><small>${{x.index}}</small>`;b.onclick=()=>show(x.index-1);thumbs.append(b)}}
function show(i){{choice.value=i;const x=data[i];summary.textContent=`${{x.index}}/100 · ${{x.dataset}}/${{x.image}} · crop (${{x.top}},${{x.left}}) · PSNR noisy ${{x.noisy_psnr.toFixed(2)}} | gain ${{x.gain_psnr.toFixed(2)}} | threshold ${{x.threshold_psnr.toFixed(2)}} | NIMA ${{x.nima_psnr.toFixed(2)}} | discriminator ${{x.disc_psnr.toFixed(2)}} dB`;
view.innerHTML=['clean','noisy','gain','threshold','nima','disc'].map((k,j)=>`<figure><img src="${{x[k]}}" alt="${{k}}"><figcaption>${{['Clean reference','Noisy','Pyramid gains','Pyramid + thresholds','Previous NIMA','Previous discriminator'][j]}}</figcaption></figure>`).join('')}}
choice.onchange=()=>show(Number(choice.value));show(0);</script>'''
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a multiscale steerable pyramid on the prior NIMA 100-crop AWGN benchmark.")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "results/simoncelli_steerable_100_matched150")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    from deepinv.loss.metric import NIMA
    scorer = NIMA(variant="technical", device=args.device).eval()
    scorer.requires_grad_(False)
    crops = selected_crops(args.seed, args.crop_size, args.sigma)
    old_nima_csv = EXPERIMENTS / "results/nima_technical_100/metrics.csv"
    old_disc_csv = EXPERIMENTS / "results/discriminator_100/comparison_metrics.csv"
    with old_nima_csv.open(newline="", encoding="utf-8") as f:
        old_nima = {int(r["example"]): r for r in csv.DictReader(f)}
    with old_disc_csv.open(newline="", encoding="utf-8") as f:
        old_disc = {int(r["example"]): r for r in csv.DictReader(f)}
    if len(crops) != 100 or len(old_nima) != 100 or len(old_disc) != 100:
        raise RuntimeError("Expected the complete, matching prior 100-crop results.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for start in range(0, 100, args.batch_size):
        group = crops[start:start + args.batch_size]
        clean_np = np.stack([item["clean"] for item in group])
        noisy_np = np.stack([item["noisy"] for item in group])
        noisy = torch.from_numpy(noisy_np).permute(0, 3, 1, 2).to(args.device)
        masks, low_mask = make_masks(args.crop_size, args.crop_size, device=args.device)
        with torch.no_grad():
            initial_bands, initial_low = decompose(noisy, masks, low_mask)
            identity = reconstruct(initial_bands, initial_low, masks, low_mask)
            identity_error = float((identity.real - noisy).abs().max().item())
        if identity_error > 2e-5:
            raise RuntimeError(f"Pyramid reconstruction check failed: max error={identity_error}")

        gain_out, gains, _ = fit_batch(torch, noisy, scorer, masks, low_mask,
                                       thresholding=False, steps=args.steps, sigma=args.sigma / 255.0)
        threshold_out, threshold_gains, thresholds = fit_batch(
            torch, noisy, scorer, masks, low_mask,
            thresholding=True, steps=args.steps, sigma=args.sigma / 255.0,
        )
        outputs = {
            "gain_only": gain_out.permute(0, 2, 3, 1).cpu().numpy(),
            "threshold": threshold_out.permute(0, 2, 3, 1).cpu().numpy(),
        }
        for offset, item in enumerate(group):
            number = start + offset + 1
            source: Path = item["path"]
            key = f"{number:03d}_{item['dataset']}_{source.stem}_r{item['repeat']}"
            clean = clean_np[offset]
            noisy_image = noisy_np[offset]
            row: dict[str, object] = {
                "example": number, "dataset": item["dataset"], "image": source.name,
                "top": item["top"], "left": item["left"], "repeat": item["repeat"],
                "identity_max_abs_error": identity_error,
            }
            row["noisy_psnr"] = full_reference_metrics(clean, noisy_image)["psnr"]
            row["noisy_ssim"] = full_reference_metrics(clean, noisy_image)["ssim"]
            for method, batch in outputs.items():
                restored = batch[offset]
                metrics = full_reference_metrics(clean, restored)
                row[f"{method}_psnr"] = metrics["psnr"]
                row[f"{method}_ssim"] = metrics["ssim"]
                path = image_dir / f"{key}_{method}.png"
                save_rgb(restored, path)
                row[f"{method}_image"] = str(path)
            row["nima_psnr"] = float(old_nima[number]["restored_psnr"])
            row["nima_ssim"] = float(old_nima[number]["restored_ssim"])
            row["disc_psnr"] = float(old_disc[number]["disc_psnr"])
            row["disc_ssim"] = float(old_disc[number]["disc_ssim"])
            row["gains_without_threshold"] = json.dumps(gains[offset].tolist())
            row["gains_with_threshold"] = json.dumps(threshold_gains[offset].tolist())
            row["threshold_sigma_multipliers"] = json.dumps(thresholds[offset].tolist())
            rows.append(row)
        print(f"Completed {min(start + len(group), 100)}/100 crops", flush=True)

    # Verify exact crop/noise alignment against the prior NIMA run before summarizing.
    for row in rows:
        previous = old_nima[int(row["example"])]
        if (row["dataset"] != previous["dataset"] or row["image"] != previous["image"]
                or int(row["top"]) != int(previous["top"]) or int(row["left"]) != int(previous["left"])):
            raise RuntimeError(f"Crop mismatch at example {row['example']}.")
        if abs(float(row["noisy_psnr"]) - float(previous["noisy_psnr"])) > 1e-6:
            raise RuntimeError(f"Noise mismatch at example {row['example']}.")

    fields = list(rows[0])
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    methods = ["noisy", "gain_only", "threshold", "nima", "disc"]
    summary: dict[str, object] = {
        "method": "frequency-domain order-3 steerable pyramid with 4 log-radial scales and 4 orientations",
        "quality_prior": "NIMA technical MobileNet, same loss and checkpoint as prior NIMA experiment",
        "dataset": "same 100 matched crops as NIMA/discriminator comparison: 50 CBSD68 + 50 Kodak24",
        "noise": {"type": "AWGN then clipped to [0,1]", "sigma_8bit": args.sigma, "seed": args.seed},
        "crop_size": args.crop_size, "optimization_steps": args.steps, "batch_size": args.batch_size,
        "objective": "-NIMA/10 + 10*L1(reconstruction,noisy) + 5*absolute mean-brightness difference",
        "gain_bounds": [0.0, 2.0], "thresholding": "complex soft threshold preserving coefficient phase; threshold = softplus(parameter) * sigma * filter RMS",
        "synthesis": "paired analysis/synthesis tight frame; maximum identity reconstruction error across batches recorded per crop",
        "mean_metrics": {}, "wins_over_noisy": {},
        "paired_psnr_comparisons": {},
        "max_identity_reconstruction_error": max(float(r["identity_max_abs_error"]) for r in rows),
        "mean_gain_only_coefficients": np.mean(np.stack([json.loads(str(r["gains_without_threshold"])) for r in rows]), axis=0).tolist(),
        "mean_thresholded_coefficients": np.mean(np.stack([json.loads(str(r["gains_with_threshold"])) for r in rows]), axis=0).tolist(),
        "mean_threshold_sigma_multipliers": np.mean(np.stack([json.loads(str(r["threshold_sigma_multipliers"])) for r in rows]), axis=0).tolist(),
        "crop_and_noise_match_verified": True,
        "nima_weights_sha256": hashlib.sha256((ROOT / ".cache/torch/hub/checkpoints/nima_mobilenet_technical_weights.pt").read_bytes()).hexdigest(),
    }
    for method in methods:
        for metric in ("psnr", "ssim"):
            values = [float(row[f"{method}_{metric}"]) for row in rows]
            summary["mean_metrics"][f"{method}_{metric}"] = float(np.mean(values))
        if method != "noisy":
            summary["wins_over_noisy"][method] = sum(float(r[f"{method}_psnr"]) > float(r["noisy_psnr"]) for r in rows)
    for method, reference in (("gain_only", "nima"), ("threshold", "nima"),
                              ("gain_only", "disc"), ("threshold", "disc")):
        deltas = [float(row[f"{method}_psnr"]) - float(row[f"{reference}_psnr"]) for row in rows]
        summary["paired_psnr_comparisons"][f"{method}_vs_{reference}"] = {
            "mean_delta_db": float(np.mean(deltas)),
            "wins": int(sum(delta > 0.0 for delta in deltas)),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_html(args.output_dir / "comparison.html", rows)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results saved under: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
