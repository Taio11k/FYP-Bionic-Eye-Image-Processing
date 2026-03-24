"""metrics.py

Version: 0.1.0

Metrics for the proposal-aligned baseline.
"""

import numpy as np



def summarize_rates(rgc_rates_hz):
    rgc_rates_hz = np.asarray(rgc_rates_hz, dtype=np.float32)
    active = rgc_rates_hz > 1e-6
    return {
        "shape": tuple(int(v) for v in rgc_rates_hz.shape),
        "global_mean_hz": float(np.mean(rgc_rates_hz)),
        "global_std_hz": float(np.std(rgc_rates_hz)),
        "global_max_hz": float(np.max(rgc_rates_hz)),
        "active_fraction": float(np.mean(active)),
    }



def summarize_spikes(spike_trains_s):
    counts = np.array([len(x) for x in spike_trains_s], dtype=np.int32)
    return {
        "n_channels": int(len(spike_trains_s)),
        "total_spikes": int(np.sum(counts)),
        "mean_spikes_per_channel": float(np.mean(counts)) if counts.size else 0.0,
        "max_spikes_per_channel": int(np.max(counts)) if counts.size else 0,
    }



def image_reconstruction_from_mosaic(on_mosaic, off_mosaic, stride_px=8, out_hw=None):
    on_mosaic = np.asarray(on_mosaic, dtype=np.float32)
    off_mosaic = np.asarray(off_mosaic, dtype=np.float32)
    if on_mosaic.shape != off_mosaic.shape:
        raise ValueError("ON and OFF mosaics must have the same shape.")

    recon = on_mosaic - off_mosaic
    recon = recon - np.min(recon)
    if np.max(recon) > 0:
        recon = recon / np.max(recon)

    if out_hw is None:
        return recon.astype(np.float32)

    H, W = int(out_hw[0]), int(out_hw[1])
    kron = np.kron(recon, np.ones((stride_px, stride_px), dtype=np.float32))
    return kron[:H, :W].astype(np.float32)



def mse(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Inputs must have the same shape.")
    return float(np.mean((a - b) ** 2))
