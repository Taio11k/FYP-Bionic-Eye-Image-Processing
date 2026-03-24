"""retina_encoder.py

Version: 0.1.0

Proposal-aligned retinal encoder baseline.

Mathematical structure
----------------------
For each frame I(x, y):

1) Photoreceptor low-pass:
       P = G_sigma_p * I

2) Center-surround receptive field (DoG):
       C = G_sigma_c * P
       S = G_sigma_s * P
       D = C - k_s * S

3) Contrast gain control (divisive normalization):
       N = D / (eps + 1 + g * (G_sigma_n * |D|))

4) ON/OFF split:
       ON  = max(N, 0)
       OFF = max(-N, 0)

5) Spatial downsampling to a ganglion mosaic using stride s.

6) Rate coding:
       r_on  = r0 + a_on  * ON^gamma
       r_off = r0 + a_off * OFF^gamma

7) Optional Poisson spikes on a fine time grid.
"""

import numpy as np
from scipy.ndimage import gaussian_filter



def _check_video(video, name="video"):
    video = np.asarray(video, dtype=np.float32)
    if video.ndim != 3:
        raise ValueError(f"`{name}` must have shape [T,H,W], got {video.shape}.")
    return video



def _downsample_stride(frame, stride_px=8, mode="mean"):
    if stride_px <= 0:
        raise ValueError("`stride_px` must be positive.")
    h, w = frame.shape
    out_h = h // stride_px
    out_w = w // stride_px
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Stride too large for frame size.")

    cropped = frame[: out_h * stride_px, : out_w * stride_px]
    view = cropped.reshape(out_h, stride_px, out_w, stride_px)

    if mode == "mean":
        return view.mean(axis=(1, 3)).astype(np.float32)
    if mode == "max":
        return view.max(axis=(1, 3)).astype(np.float32)
    raise ValueError(f"Unsupported pooling mode: {mode}")



def _sample_xy_for_stride(hw, stride_px):
    h, w = int(hw[0]), int(hw[1])
    out_h = h // stride_px
    out_w = w // stride_px
    yy = (np.arange(out_h, dtype=np.float32) + 0.5) * float(stride_px)
    xx = (np.arange(out_w, dtype=np.float32) + 0.5) * float(stride_px)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return xy.astype(np.float32), (out_h, out_w)



def encode_movie(
    movie,
    photoreceptor_sigma_px=1.0,
    center_sigma_px=1.2,
    surround_sigma_px=2.8,
    surround_gain=0.85,
    cg_sigma_px=3.5,
    cg_strength=0.60,
    cg_epsilon=1e-3,
    stride_px=8,
    rf_pool="mean",
    on_rate_gain_hz=90.0,
    off_rate_gain_hz=90.0,
    baseline_rate_hz=2.0,
    rate_gamma=1.0,
):
    movie = _check_video(movie)
    T, H, W = movie.shape

    photoreceptor = np.empty_like(movie)
    dog = np.empty_like(movie)
    norm = np.empty_like(movie)
    on_video = np.empty_like(movie)
    off_video = np.empty_like(movie)

    for t in range(T):
        p = gaussian_filter(movie[t], sigma=float(photoreceptor_sigma_px), mode="nearest")
        c = gaussian_filter(p, sigma=float(center_sigma_px), mode="nearest")
        s = gaussian_filter(p, sigma=float(surround_sigma_px), mode="nearest")
        d = c - float(surround_gain) * s
        gain_pool = gaussian_filter(np.abs(d), sigma=float(cg_sigma_px), mode="nearest")
        n = d / (1.0 + float(cg_strength) * gain_pool + float(cg_epsilon))
        photoreceptor[t] = p
        dog[t] = d
        norm[t] = n
        on_video[t] = np.maximum(n, 0.0)
        off_video[t] = np.maximum(-n, 0.0)

    on_mosaic = np.stack(
        [_downsample_stride(on_video[t], stride_px=stride_px, mode=rf_pool) for t in range(T)],
        axis=0,
    )
    off_mosaic = np.stack(
        [_downsample_stride(off_video[t], stride_px=stride_px, mode=rf_pool) for t in range(T)],
        axis=0,
    )

    on_rates = float(baseline_rate_hz) + float(on_rate_gain_hz) * np.power(np.maximum(on_mosaic, 0.0), float(rate_gamma))
    off_rates = float(baseline_rate_hz) + float(off_rate_gain_hz) * np.power(np.maximum(off_mosaic, 0.0), float(rate_gamma))

    cell_xy, mosaic_hw = _sample_xy_for_stride((H, W), stride_px)
    n_sites = cell_xy.shape[0]

    on_rates_flat = on_rates.reshape(T, n_sites).astype(np.float32)
    off_rates_flat = off_rates.reshape(T, n_sites).astype(np.float32)
    rates = np.concatenate([on_rates_flat, off_rates_flat], axis=1)
    cell_types = np.array(["ON"] * n_sites + ["OFF"] * n_sites, dtype=object)
    cell_xy_all = np.concatenate([cell_xy, cell_xy], axis=0).astype(np.float32)

    return {
        "photoreceptor_video": photoreceptor.astype(np.float32),
        "dog_video": dog.astype(np.float32),
        "norm_video": norm.astype(np.float32),
        "on_video": on_video.astype(np.float32),
        "off_video": off_video.astype(np.float32),
        "on_mosaic": on_mosaic.astype(np.float32),
        "off_mosaic": off_mosaic.astype(np.float32),
        "on_rates_hz": on_rates_flat,
        "off_rates_hz": off_rates_flat,
        "rgc_rates_hz": rates.astype(np.float32),
        "cell_xy": cell_xy_all,
        "cell_types": cell_types,
        "mosaic_hw": mosaic_hw,
        "stride_px": int(stride_px),
    }



def generate_poisson_spikes(rgc_rates_hz, frame_dt_s, spike_dt_s=0.001, seed=123):
    rgc_rates_hz = np.asarray(rgc_rates_hz, dtype=np.float32)
    if rgc_rates_hz.ndim != 2:
        raise ValueError(f"`rgc_rates_hz` must have shape [T,N], got {rgc_rates_hz.shape}.")
    if frame_dt_s <= 0 or spike_dt_s <= 0:
        raise ValueError("Time steps must be positive.")

    T, N = rgc_rates_hz.shape
    reps = max(1, int(round(frame_dt_s / spike_dt_s)))
    fine_rates = np.repeat(rgc_rates_hz, reps, axis=0)
    fine_t = np.arange(fine_rates.shape[0], dtype=np.float64) * float(spike_dt_s)

    rng = np.random.default_rng(seed)
    spike_trains = []
    lam = fine_rates * float(spike_dt_s)
    for n in range(N):
        counts = rng.poisson(np.clip(lam[:, n], 0.0, None))
        times = []
        for idx, c in enumerate(counts):
            if c <= 0:
                continue
            if c == 1:
                times.append(fine_t[idx])
            else:
                jitter = rng.random(int(c)) * float(spike_dt_s)
                times.extend((fine_t[idx] + jitter).tolist())
        spike_trains.append(np.asarray(times, dtype=np.float64))

    return {
        "spike_trains_s": spike_trains,
        "spike_time_axis_s": fine_t,
        "spike_dt_s": float(spike_dt_s),
    }
