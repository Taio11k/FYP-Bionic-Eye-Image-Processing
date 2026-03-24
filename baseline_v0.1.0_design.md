# Baseline v0.1.0 redesign plan

## Decision

The current phosphene-only pipeline is **not aligned** with the original FYP A proposal.
The proposal is explicitly a **retinal encoding** project first, with phosphene/electrode visualization as a downstream output layer.

## Correct baseline architecture

1. Input image or short movie
2. Photoreceptor low-pass filtering
3. Center-surround filtering using Difference of Gaussians (DoG)
4. Contrast gain control using divisive normalization
5. ON/OFF split
6. Spatial downsampling to a ganglion mosaic
7. Rate coding
8. Optional Poisson spike generation
9. Electrode activation map
10. Simple phosphene rendering for visualization

## Why this is the right compromise

This keeps the core proposal contribution:
- retinal-style preprocessing,
- spike-based encoding,
- task-dependent evaluation.

But it removes heavy model complexity:
- no full Virtual Retina XML stack,
- no pulse2percept dependency,
- no patient-specific axon-map fitting,
- no deep-learning encoder.

## Mathematical summary

For each frame `I(x,y)`:

- Photoreceptor smoothing:
  `P = G_sigma_p * I`
- Center-surround:
  `D = G_sigma_c * P - k_s (G_sigma_s * P)`
- Contrast gain control:
  `N = D / (1 + g (G_sigma_n * |D|) + eps)`
- ON/OFF split:
  `ON = max(N,0)`, `OFF = max(-N,0)`
- Downsampling with stride `s`
- Rates:
  `r_on = r0 + a_on * ON^gamma`
  `r_off = r0 + a_off * OFF^gamma`
- Poisson spikes:
  `k ~ Poisson(r * dt)`

## Meaning of DoG

Difference of Gaussians approximates a center-surround receptive field:
- a narrow Gaussian models local excitation,
- a broader Gaussian models inhibitory surround,
- subtraction highlights edges and local contrast.

So DoG is **not** arbitrary image processing here. It is the computational version of retinal center-surround antagonism.

## Meaning of stride

Stride is the spatial sampling step used to convert a dense image into a coarse ganglion mosaic.

Example:
- image size = `128 x 128`
- stride = `8`
- mosaic size = `16 x 16`

So stride controls the trade-off:
- smaller stride -> more cells, more detail, more computation
- larger stride -> fewer cells, lower detail, simpler baseline

In this baseline, stride is the cleanest transparent control for sampling density.

## Evaluation for this baseline

Primary:
- firing-rate statistics
- spike-count summary
- simple reconstruction error from ON/OFF mosaic

Secondary:
- object-recognition task performance later
- phosphene visualization only as a qualitative output

## Critical project boundary

If the goal is to match FYP A, the project should **not** be reduced to only image -> electrode -> phosphene.
That would become a simulated prosthetic vision / scene simplification project instead of a retinal encoding project.
