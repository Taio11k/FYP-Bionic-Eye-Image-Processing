"""phosphene.py

Version: 0.1.0

Simple phosphene renderer for the proposal-aligned baseline.

This is a visualization layer, not a claim of implant biophysics.
It renders a scoreboard-like phosphene image from electrode amplitudes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter



def rates_to_electrode_currents(
    on_rates_hz,
    off_rates_hz,
    mode="on_off_sum",
    current_scale_uA_per_hz=0.20,
    current_clip_uA=(0.0, 24.0),
):
    on_rates_hz = np.asarray(on_rates_hz, dtype=np.float32)
    off_rates_hz = np.asarray(off_rates_hz, dtype=np.float32)
    if on_rates_hz.shape != off_rates_hz.shape:
        raise ValueError("ON and OFF rate maps must have the same shape.")

    if mode == "on_off_sum":
        env = on_rates_hz + off_rates_hz
    elif mode == "on_minus_off_abs":
        env = np.abs(on_rates_hz - off_rates_hz)
    else:
        raise ValueError(f"Unsupported electrode mode: {mode}")

    current = float(current_scale_uA_per_hz) * env
    lo, hi = float(current_clip_uA[0]), float(current_clip_uA[1])
    current = np.clip(current, lo, hi).astype(np.float32)
    return current



def render_phosphene_video(
    elec_currents_uA,
    mosaic_hw,
    out_hw,
    dot_sigma_px=3.0,
    dot_gain=1.0,
    dropout_fraction=0.0,
    gray_levels=8,
    seed=123,
):
    elec_currents_uA = np.asarray(elec_currents_uA, dtype=np.float32)
    if elec_currents_uA.ndim != 2:
        raise ValueError(f"`elec_currents_uA` must have shape [T,E], got {elec_currents_uA.shape}.")

    T, E = elec_currents_uA.shape
    mh, mw = int(mosaic_hw[0]), int(mosaic_hw[1])
    H, W = int(out_hw[0]), int(out_hw[1])
    if mh * mw != E:
        raise ValueError("Mosaic shape does not match number of electrodes.")

    rng = np.random.default_rng(seed)
    dropout_mask = np.ones(E, dtype=np.float32)
    n_drop = int(round(float(dropout_fraction) * E))
    if n_drop > 0:
        drop_idx = rng.choice(E, size=n_drop, replace=False)
        dropout_mask[drop_idx] = 0.0

    frames = []
    for t in range(T):
        grid = elec_currents_uA[t] * dropout_mask
        grid = grid.reshape(mh, mw)
        up = np.kron(grid, np.ones((max(1, H // mh), max(1, W // mw)), dtype=np.float32))
        up = up[:H, :W]
        frame = gaussian_filter(up, sigma=float(dot_sigma_px), mode="nearest")
        frame = float(dot_gain) * frame
        if np.max(frame) > 0:
            frame = frame / float(np.max(frame))
        if gray_levels is not None and int(gray_levels) > 1:
            g = int(gray_levels)
            frame = np.round(frame * (g - 1)) / float(g - 1)
        frames.append(frame.astype(np.float32))

    video = np.stack(frames, axis=0).astype(np.float32)
    return {
        "phosphene_video": video,
        "dropout_mask": dropout_mask.astype(np.float32),
    }
