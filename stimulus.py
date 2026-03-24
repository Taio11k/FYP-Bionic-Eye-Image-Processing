"""stimulus.py

Version: 0.1.0

Simple image/movie preparation utilities for the proposal-aligned baseline.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import shift
from skimage import color, data, io, img_as_float32
from skimage.transform import resize



def _coerce_to_float01(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32, copy=False)
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            raise ValueError("Input image contains no finite values.")
        if finite.min() >= 0.0 and finite.max() > 1.0 and finite.max() <= 255.0 + 1e-6:
            img = img / 255.0
        return img.astype(np.float32, copy=False)
    return img_as_float32(img)



def load_grayscale_image(image_path=None, target_hw=(128, 128)):
    if image_path is None:
        raw = data.camera()
    else:
        raw = io.imread(Path(image_path))

    raw = _coerce_to_float01(raw)

    if raw.ndim == 3:
        if raw.shape[2] == 4:
            raw = color.rgba2rgb(raw)
        gray = color.rgb2gray(raw)
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported image shape: {raw.shape}")

    gray = resize(
        gray,
        target_hw,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return np.clip(gray, 0.0, 1.0)



def robust_normalize(gray, p_low=1.0, p_high=99.0):
    gray = np.asarray(gray, dtype=np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        raise ValueError("Image contains no finite values.")
    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    scale = max(hi - lo, 1e-6)
    out = np.clip((gray - lo) / scale, 0.0, 1.0)
    return out.astype(np.float32)



def make_jittered_movie(gray, fps=10.0, duration_s=0.6, jitter_std_px=0.25, seed=123):
    gray = np.asarray(gray, dtype=np.float32)
    n_frames = max(1, int(round(float(fps) * float(duration_s))))
    rng = np.random.default_rng(seed)

    frames = []
    for _ in range(n_frames):
        dy = float(rng.normal(0.0, jitter_std_px))
        dx = float(rng.normal(0.0, jitter_std_px))
        frame = shift(gray, shift=(dy, dx), order=1, mode="nearest", prefilter=False)
        frames.append(np.asarray(frame, dtype=np.float32))
    movie = np.stack(frames, axis=0)
    t_s = np.arange(n_frames, dtype=np.float64) / float(fps)
    return movie.astype(np.float32), t_s



def prepare_stimulus(
    image_path=None,
    target_hw=(128, 128),
    p_low=1.0,
    p_high=99.0,
    fps=10.0,
    duration_s=0.6,
    jitter_std_px=0.25,
    seed=123,
):
    gray = load_grayscale_image(image_path=image_path, target_hw=target_hw)
    gray = robust_normalize(gray, p_low=p_low, p_high=p_high)
    movie, t_s = make_jittered_movie(
        gray,
        fps=fps,
        duration_s=duration_s,
        jitter_std_px=jitter_std_px,
        seed=seed,
    )
    return {
        "gray": gray,
        "movie": movie,
        "t_frames_s": t_s,
        "fps": float(fps),
    }
