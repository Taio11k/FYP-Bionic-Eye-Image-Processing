"""pipeline.py

Version: 0.1.0

End-to-end pipeline for the proposal-aligned baseline.
"""

from copy import deepcopy

import numpy as np

from config import get_default_config
from stimulus import prepare_stimulus
from retina_encoder import encode_movie, generate_poisson_spikes
from phosphene import rates_to_electrode_currents, render_phosphene_video
from metrics import summarize_rates, summarize_spikes, image_reconstruction_from_mosaic, mse



def _deep_update(base, updates):
    merged = deepcopy(base)
    for k, v in updates.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_update(merged[k], v)
        else:
            merged[k] = v
    return merged



def build_config(config_overrides=None):
    cfg = get_default_config()
    if config_overrides is not None:
        cfg = _deep_update(cfg, config_overrides)
    return cfg



def run_pipeline(image_path=None, cfg=None):
    if cfg is None:
        cfg = build_config()
    else:
        cfg = deepcopy(cfg)

    stim_cfg = cfg["stimulus"]
    ret_cfg = cfg["retina"]
    elec_cfg = cfg["electrode"]
    ph_cfg = cfg["phosphene"]

    stim = prepare_stimulus(
        image_path=image_path,
        target_hw=stim_cfg["target_hw"],
        p_low=stim_cfg["norm_percentile_low"],
        p_high=stim_cfg["norm_percentile_high"],
        fps=stim_cfg["fps"],
        duration_s=stim_cfg["duration_s"],
        jitter_std_px=stim_cfg["jitter_std_px"],
        seed=cfg["seed"],
    )

    enc = encode_movie(
        movie=stim["movie"],
        photoreceptor_sigma_px=ret_cfg["photoreceptor_sigma_px"],
        center_sigma_px=ret_cfg["center_sigma_px"],
        surround_sigma_px=ret_cfg["surround_sigma_px"],
        surround_gain=ret_cfg["surround_gain"],
        cg_sigma_px=ret_cfg["cg_sigma_px"],
        cg_strength=ret_cfg["cg_strength"],
        cg_epsilon=ret_cfg["cg_epsilon"],
        stride_px=ret_cfg["stride_px"],
        rf_pool=ret_cfg["rf_pool"],
        on_rate_gain_hz=ret_cfg["on_rate_gain_hz"],
        off_rate_gain_hz=ret_cfg["off_rate_gain_hz"],
        baseline_rate_hz=ret_cfg["baseline_rate_hz"],
        rate_gamma=ret_cfg["rate_gamma"],
    )

    spikes = None
    if ret_cfg.get("generate_spikes", True):
        spikes = generate_poisson_spikes(
            enc["rgc_rates_hz"],
            frame_dt_s=1.0 / float(stim_cfg["fps"]),
            spike_dt_s=ret_cfg["spike_dt_s"],
            seed=cfg["seed"],
        )

    elec = rates_to_electrode_currents(
        enc["on_rates_hz"],
        enc["off_rates_hz"],
        mode=elec_cfg["mode"],
        current_scale_uA_per_hz=elec_cfg["current_scale_uA_per_hz"],
        current_clip_uA=elec_cfg["current_clip_uA"],
    )

    ph = render_phosphene_video(
        elec_currents_uA=elec,
        mosaic_hw=enc["mosaic_hw"],
        out_hw=stim["gray"].shape,
        dot_sigma_px=ph_cfg["dot_sigma_px"],
        dot_gain=ph_cfg["dot_gain"],
        dropout_fraction=ph_cfg["dropout_fraction"],
        gray_levels=ph_cfg["gray_levels"],
        seed=cfg["seed"],
    )

    recon = image_reconstruction_from_mosaic(
        enc["on_mosaic"][0],
        enc["off_mosaic"][0],
        stride_px=ret_cfg["stride_px"],
        out_hw=stim["gray"].shape,
    )

    results = {
        "cfg": cfg,
        **stim,
        **enc,
        "elec_currents_uA": elec,
        **ph,
        "reconstruction_frame0": recon,
        "rate_summary": summarize_rates(enc["rgc_rates_hz"]),
        "reconstruction_mse_frame0": mse(stim["gray"], recon),
    }
    if spikes is not None:
        results.update(spikes)
        results["spike_summary"] = summarize_spikes(spikes["spike_trains_s"])
    return results
