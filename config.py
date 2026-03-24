"""config.py

Version: 0.1.0

Proposal-aligned baseline configuration for the bionic-eye project.

Scope
-----
This baseline follows the *retinal-encoding-first* structure described in the
FYP A proposal:

    image/movie
    -> photoreceptor low-pass filtering
    -> center-surround filtering (DoG)
    -> contrast gain control (divisive normalization)
    -> ON/OFF split
    -> ganglion mosaic downsampling
    -> rate maps
    -> optional Poisson spikes
    -> electrode map
    -> simple phosphene rendering

This is intentionally simpler than a full Virtual Retina or pulse2percept model.
It is meant to be a transparent, inspectable baseline that preserves the logic of
proposal FYP A without claiming implant-grade biophysical realism.
"""

from copy import deepcopy


CFG = {
    "seed": 123,
    "stimulus": {
        "target_hw": (128, 128),
        "fps": 10.0,
        "duration_s": 0.60,
        "jitter_std_px": 0.25,
        "norm_percentile_low": 1.0,
        "norm_percentile_high": 99.0,
    },
    "retina": {
        # Photoreceptor front-end
        "photoreceptor_sigma_px": 1.0,
        # Center-surround receptive field (Difference of Gaussians)
        "center_sigma_px": 1.2,
        "surround_sigma_px": 2.8,
        "surround_gain": 0.85,
        # Contrast gain control (divisive normalization)
        "cg_sigma_px": 3.5,
        "cg_strength": 0.60,
        "cg_epsilon": 1e-3,
        # Spatial sampling / ganglion mosaic
        "stride_px": 8,
        "rf_pool": "mean",
        # Rate coding
        "on_rate_gain_hz": 90.0,
        "off_rate_gain_hz": 90.0,
        "baseline_rate_hz": 2.0,
        "rate_gamma": 1.0,
        # Spike generation
        "generate_spikes": True,
        "spike_dt_s": 0.001,
    },
    "electrode": {
        # Baseline assumption: one sampled ON site and one sampled OFF site both
        # contribute to stimulation. This can later be replaced by a geometric map.
        "mode": "on_off_sum",
        "current_scale_uA_per_hz": 0.20,
        "current_clip_uA": (0.0, 24.0),
    },
    "phosphene": {
        # Simple scoreboard-like renderer for baseline visualization.
        "dot_sigma_px": 3.0,
        "dot_gain": 1.0,
        "dropout_fraction": 0.0,
        "gray_levels": 8,
    },
}


def get_default_config():
    return deepcopy(CFG)
