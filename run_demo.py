"""run_demo.py

Version: 0.1.0

Small demo runner for the proposal-aligned baseline.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from pipeline import build_config, run_pipeline

OUTPUT_DIR = Path("outputs/demo")
IMAGE_PATH = r"C:\Users\PF5MX\00_Devin\OneDrive\Documents\Part Time Work\FYP\FYPB\260324_pipeline_test\sample_images\img2.png"



def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float16, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(x)
    return x



def _save_img(path, img, title=None, cmap="gray"):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    results = run_pipeline(image_path=IMAGE_PATH, cfg=cfg)

    _save_img(OUTPUT_DIR / "01_gray.png", results["gray"], title="Gray input")
    _save_img(OUTPUT_DIR / "02_photoreceptor_frame0.png", results["photoreceptor_video"][0], title="Photoreceptor LPF")
    _save_img(OUTPUT_DIR / "03_dog_frame0.png", results["dog_video"][0], title="DoG", cmap="coolwarm")
    _save_img(OUTPUT_DIR / "04_norm_frame0.png", results["norm_video"][0], title="Normalized DoG", cmap="coolwarm")
    _save_img(OUTPUT_DIR / "05_on_frame0.png", results["on_video"][0], title="ON")
    _save_img(OUTPUT_DIR / "06_off_frame0.png", results["off_video"][0], title="OFF")
    _save_img(OUTPUT_DIR / "07_on_mosaic_frame0.png", results["on_mosaic"][0], title="ON mosaic")
    _save_img(OUTPUT_DIR / "08_off_mosaic_frame0.png", results["off_mosaic"][0], title="OFF mosaic")
    _save_img(OUTPUT_DIR / "09_reconstruction_frame0.png", results["reconstruction_frame0"], title="Reconstruction")
    _save_img(OUTPUT_DIR / "10_phosphene_frame0.png", results["phosphene_video"][0], title="Phosphene")

    summary = {
        "rate_summary": results["rate_summary"],
        "spike_summary": results.get("spike_summary", None),
        "reconstruction_mse_frame0": results["reconstruction_mse_frame0"],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_to_jsonable(summary), indent=2))



if __name__ == "__main__":
    main()
