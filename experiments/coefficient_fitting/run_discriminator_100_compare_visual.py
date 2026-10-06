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
from denoise_research.metrics import full_reference_metrics
from run_nima_100_visual import make_batch_fusion, save_rgb, selected_crops


DEFAULT_FRAGMENT = Path(
    r"C:\Users\user\.codex\visualizations\2026\09\30\01a0f28f-023d-7942-ad9f-1f49a1949a97\nima-vs-discriminator-100.html"
)


def optimize_batch(torch: object, noisy: object, discriminator: object, steps: int) -> tuple[object, list[list[float]]]:
    model = make_batch_fusion(torch, noisy.shape[0], noisy.shape[1]).to(noisy.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(noisy)
        predictions, _ = discriminator(2.0 * reconstruction - 1.0)
        adversarial = sum(
            ((prediction - 1.0) ** 2).mean(dim=tuple(range(1, prediction.ndim)))
            for prediction in predictions
        )
        fidelity = (reconstruction - noisy).abs().mean(dim=(1, 2, 3))
        brightness = (reconstruction.mean(dim=(1, 2, 3)) - noisy.mean(dim=(1, 2, 3))).abs()
        losses = adversarial + 10.0 * fidelity + 5.0 * brightness
        losses.mean().backward()
        optimizer.step()
    with torch.no_grad():
        restored = model(noisy).clamp(0.0, 1.0)
        coefficients = model.coefficients().detach().cpu().tolist()
    return restored, coefficients


def image_data(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((112, 112), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=32, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_fragment(records: list[dict[str, object]], path: Path, summary: dict[str, object]) -> None:
    visual: list[dict[str, object]] = []
    for row in records:
        visual.append({
            **{key: row[key] for key in (
                "example", "dataset", "image", "top", "left",
                "noisy_psnr", "nima_psnr", "disc_psnr",
                "noisy_ssim", "nima_ssim", "disc_ssim",
            )},
            "clean_src": image_data(Path(str(row["clean_path"]))),
            "noisy_src": image_data(Path(str(row["noisy_path"]))),
            "nima_src": image_data(Path(str(row["nima_path"]))),
            "disc_src": image_data(Path(str(row["disc_path"]))),
        })
    data = json.dumps(visual, separators=(",", ":")).replace("</", "<\\/")
    fragment = f'''<div id="quality-model-comparison-100">
  <style>
    #quality-model-comparison-100 .comparison-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; align-items:start; }}
    #quality-model-comparison-100 .comparison-grid button {{ min-width:0; }}
    #quality-model-comparison-100 .comparison-grid img {{ display:block; width:100%; height:auto; image-rendering:auto; }}
    #quality-model-comparison-100 .focus-image {{ display:block; width:min(100%,640px); height:auto; margin:0 auto; image-rendering:auto; }}
    #quality-model-comparison-100 .example-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(52px,1fr)); gap:6px; }}
    #quality-model-comparison-100 .example-grid img {{ display:block; width:44px; height:44px; object-fit:cover; }}
    @media (max-width:600px) {{ #quality-model-comparison-100 .comparison-grid {{ grid-template-columns:1fr; }} }}
  </style>
  <h2>NIMA vs archive discriminator · 100 noisy crops</h2>
  <div class="viz-grid">
    <div class="card viz-stat"><span class="text-muted">Noisy PSNR</span><span class="viz-stat-value tabular-nums">{summary['mean_noisy_psnr']:.2f} dB</span></div>
    <div class="card viz-stat"><span class="text-muted">NIMA PSNR</span><span class="viz-stat-value tabular-nums">{summary['mean_nima_psnr']:.2f} dB</span></div>
    <div class="card viz-stat"><span class="text-muted">Discriminator PSNR</span><span class="viz-stat-value tabular-nums">{summary['mean_disc_psnr']:.2f} dB</span></div>
  </div>
  <div id="comparison-summary" aria-live="polite"></div>
  <div class="viz-controls" aria-label="Image display mode">
    <button type="button" class="btn btn-primary" id="show-compare" aria-pressed="true">Compare four</button>
    <button type="button" class="btn" id="show-focus" aria-pressed="false">Large single view</button>
  </div>
  <div id="comparison-view"></div>
  <hr>
  <div id="comparison-example-grid" class="example-grid" aria-label="Select one of 100 crop comparisons"></div>
</div>
<script>
(() => {{
  const root = document.getElementById("quality-model-comparison-100");
  if (!root) return;
  const examples = {data};
  const summary = root.querySelector("#comparison-summary");
  const view = root.querySelector("#comparison-view");
  const grid = root.querySelector("#comparison-example-grid");
  const compareButton = root.querySelector("#show-compare");
  const focusButton = root.querySelector("#show-focus");
  const labels = {{clean:"Clean reference", noisy:"Noisy input", nima:"NIMA restoration", disc:"Archive discriminator restoration"}};
  const saved = window.openai && window.openai.widgetState;
  let selected = Number(saved && saved.modelContent && saved.modelContent.selectedIndex);
  let mode = saved && saved.privateContent && saved.privateContent.mode;
  let focus = saved && saved.privateContent && saved.privateContent.focus;
  if (!Number.isInteger(selected) || selected < 0 || selected >= examples.length) selected = 0;
  if (!['compare','focus'].includes(mode)) mode = 'compare';
  if (!['clean','noisy','nima','disc'].includes(focus)) focus = 'nima';
  const f = value => Number(value).toFixed(3);
  function persist() {{
    if (window.openai && window.openai.setWidgetState) window.openai.setWidgetState({{
      modelContent: {{selectedIndex:selected}}, privateContent: {{mode,focus}}
    }}).catch(() => {{}});
  }}
  function panel(item, key, metric) {{
    return `<button type="button" class="btn" data-focus="${{key}}" aria-label="Open ${{labels[key]}} in large view"><img src="${{item[key + '_src']}}" alt="${{labels[key]}}"><span>${{labels[key]}}${{metric ? ' · ' + metric : ''}}</span></button>`;
  }}
  function render(persistState = true) {{
    const item = examples[selected];
    const delta = item.nima_psnr - item.disc_psnr;
    summary.innerHTML = `<p><strong>Example ${{selected + 1}} / 100</strong> · ${{item.dataset}}/${{item.image}} · crop (${{item.top}}, ${{item.left}}) · NIMA minus discriminator: ${{delta >= 0 ? '+' : ''}}${{delta.toFixed(2)}} dB</p>`;
    compareButton.setAttribute('aria-pressed', String(mode === 'compare'));
    focusButton.setAttribute('aria-pressed', String(mode === 'focus'));
    compareButton.classList.toggle('btn-primary', mode === 'compare');
    focusButton.classList.toggle('btn-primary', mode === 'focus');
    if (mode === 'compare') {{
      view.innerHTML = `<div class="comparison-grid">
        ${{panel(item,'clean','')}}
        ${{panel(item,'noisy',item.noisy_psnr.toFixed(2)+' dB · SSIM '+f(item.noisy_ssim))}}
        ${{panel(item,'nima',item.nima_psnr.toFixed(2)+' dB · SSIM '+f(item.nima_ssim))}}
        ${{panel(item,'disc',item.disc_psnr.toFixed(2)+' dB · SSIM '+f(item.disc_ssim))}}
      </div>`;
    }} else {{
      const metric = focus === 'clean' ? '' : focus === 'noisy'
        ? item.noisy_psnr.toFixed(2)+' dB · SSIM '+f(item.noisy_ssim)
        : focus === 'nima' ? item.nima_psnr.toFixed(2)+' dB · SSIM '+f(item.nima_ssim)
        : item.disc_psnr.toFixed(2)+' dB · SSIM '+f(item.disc_ssim);
      view.innerHTML = `<p><strong>${{labels[focus]}}</strong>${{metric ? ' · '+metric : ''}}</p><img class="focus-image" src="${{item[focus + '_src']}}" alt="${{labels[focus]}} enlarged">`;
    }}
    for (const button of grid.querySelectorAll('button')) {{
      const active = Number(button.dataset.index) === selected;
      button.setAttribute('aria-pressed', String(active));
      button.classList.toggle('is-selected', active);
    }}
    if (persistState) persist();
  }}
  grid.innerHTML = examples.map((item,index) => `<button type="button" class="btn viz-tile" data-index="${{index}}" aria-pressed="false" aria-label="Example ${{index+1}}, ${{item.dataset}} ${{item.image}}"><img src="${{item.disc_src}}" alt=""><span>${{index+1}}</span></button>`).join('');
  grid.addEventListener('click', event => {{ const button=event.target.closest('button[data-index]'); if(button){{selected=Number(button.dataset.index);render();}} }});
  view.addEventListener('click', event => {{ const button=event.target.closest('button[data-focus]'); if(button){{focus=button.dataset.focus;mode='focus';render();}} }});
  compareButton.addEventListener('click', () => {{mode='compare';render();}});
  focusButton.addEventListener('click', () => {{mode='focus';render();}});
  render(false);
  window.addEventListener('openai:set_globals', event => {{
    const state=event.detail && event.detail.globals && event.detail.globals.widgetState;
    const index=Number(state && state.modelContent && state.modelContent.selectedIndex);
    if(Number.isInteger(index) && index>=0 && index<examples.length){{selected=index;render(false);}}
  }});
}})();
</script>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NIMA and archive-discriminator fitting on the same 100 crops.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/netD_Clean_epoch_100.pth")
    parser.add_argument("--nima-dir", type=Path, default=EXPERIMENTS / "results/nima_technical_100")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "results/discriminator_100")
    parser.add_argument("--fragment-out", type=Path, default=DEFAULT_FRAGMENT)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-existing", action="store_true", help="Rebuild only the visualization from saved metrics and images.")
    args = parser.parse_args()

    if args.reuse_existing:
        examples = selected_crops(args.seed, args.crop_size, args.sigma)
        with (args.output_dir / "comparison_metrics.csv").open(newline="", encoding="utf-8") as handle:
            saved_rows = list(csv.DictReader(handle))
        summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
        records: list[dict[str, object]] = []
        for saved, item in zip(saved_rows, examples):
            index = int(saved["example"])
            source = item["path"]
            key = f"{index:03d}_{item['dataset']}_{source.stem}_r{item['repeat']}"
            records.append({
                **{key_name: (float(value) if key_name not in ("example", "dataset", "image") else value)
                   for key_name, value in saved.items()},
                "example": index,
                "clean_path": str(args.nima_dir / "images" / f"{key}_clean.png"),
                "noisy_path": str(args.nima_dir / "images" / f"{key}_noisy.png"),
                "nima_path": str(args.nima_dir / "images" / f"{key}_restored.png"),
                "disc_path": str(args.output_dir / "images" / f"{key}_discriminator.png"),
            })
        write_fragment(records, args.fragment_out, summary)
        print(f"Rebuilt visualization from {len(records)} saved comparisons: {args.fragment_out}")
        return

    torch_optional.require_torch()
    torch = torch_optional.torch
    discriminator = torch_optional.load_clean_discriminator(args.checkpoint, args.device)
    examples = selected_crops(args.seed, args.crop_size, args.sigma)
    with (args.nima_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        nima_rows = {int(row["example"]): row for row in csv.DictReader(handle)}
    image_dir = args.output_dir / "images"
    records: list[dict[str, object]] = []

    for start in range(0, 100, args.batch_size):
        group = examples[start : start + args.batch_size]
        noisy_batch = np.stack([item["noisy"] for item in group])
        tensor = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(args.device)
        restored_tensor, coefficients = optimize_batch(torch, tensor, discriminator, args.steps)
        restored_batch = restored_tensor.permute(0, 2, 3, 1).cpu().numpy()
        for offset, (item, restored, weights) in enumerate(zip(group, restored_batch, coefficients)):
            index = start + offset + 1
            clean = item["clean"]
            noisy = item["noisy"]
            source = item["path"]
            disc_metrics = full_reference_metrics(clean, restored)
            noisy_metrics = full_reference_metrics(clean, noisy)
            nima = nima_rows[index]
            key = f"{index:03d}_{item['dataset']}_{source.stem}_r{item['repeat']}"
            disc_path = image_dir / f"{key}_discriminator.png"
            save_rgb(restored, disc_path)
            nima_path = args.nima_dir / "images" / f"{key}_restored.png"
            clean_path = args.nima_dir / "images" / f"{key}_clean.png"
            noisy_path = args.nima_dir / "images" / f"{key}_noisy.png"
            if not all(path.is_file() for path in (nima_path, clean_path, noisy_path)):
                raise FileNotFoundError(f"Missing saved NIMA comparison images for {key}")
            records.append({
                "example": index,
                "dataset": item["dataset"],
                "image": source.name,
                "top": item["top"], "left": item["left"],
                "low": weights[0], "mid": weights[1], "high": weights[2],
                "noisy_psnr": noisy_metrics["psnr"], "noisy_ssim": noisy_metrics["ssim"],
                "nima_psnr": float(nima["restored_psnr"]), "nima_ssim": float(nima["restored_ssim"]),
                "disc_psnr": disc_metrics["psnr"], "disc_ssim": disc_metrics["ssim"],
                "clean_path": str(clean_path), "noisy_path": str(noisy_path),
                "nima_path": str(nima_path), "disc_path": str(disc_path),
            })
        print(f"Completed {min(start + args.batch_size, 100)}/100 crops", flush=True)

    summary = {
        "count": 100,
        "checkpoint": str(args.checkpoint),
        "noise": {"type": "AWGN then clipped to [0,1]", "sigma_8bit": args.sigma, "seed": args.seed},
        "steps": args.steps,
        "mean_noisy_psnr": float(np.mean([r["noisy_psnr"] for r in records])),
        "mean_nima_psnr": float(np.mean([r["nima_psnr"] for r in records])),
        "mean_disc_psnr": float(np.mean([r["disc_psnr"] for r in records])),
        "mean_noisy_ssim": float(np.mean([r["noisy_ssim"] for r in records])),
        "mean_nima_ssim": float(np.mean([r["nima_ssim"] for r in records])),
        "mean_disc_ssim": float(np.mean([r["disc_ssim"] for r in records])),
        "nima_psnr_wins": int(sum(r["nima_psnr"] > r["disc_psnr"] for r in records)),
        "discriminator_psnr_wins": int(sum(r["disc_psnr"] > r["nima_psnr"] for r in records)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [key for key in records[0] if not key.endswith("_path")]
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in records])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_fragment(records, args.fragment_out, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote visualization: {args.fragment_out}", flush=True)


if __name__ == "__main__":
    main()
