# Plan to the midterm report (due Fri Oct 23, 11:59 PM ET)

## What the midterm must show (course rubric)

- Baseline implemented and explained (mandatory), with our own outputs and metrics as evidence.
- At least two training runs or configurations, with training and validation curves.
- Model description with inputs and outputs, data preparation steps, metric definitions.
- Challenges so far, 2–3 concrete next steps, role breakdown, GitHub link. 4–5 pages.

## Timeline

| Week | Dates | Goal | Done when |
|---|---|---|---|
| 1 | Sep 29 – Oct 5 | Data pipeline and color labels | CelebAMask-HQ loaded; hair/cloth masks merged and dilated; color word per image and region; 26k/2k/2k split saved |
| 2 | Oct 6 – Oct 12 | Metrics, color classifier, B0 and B1 | Metric script runs on any folder of edits; B0 and B1 numbers on the test split |
| 3 | Oct 13 – Oct 18 | First LoRA runs | Latents cached; one 10k-step run with loss and validation curves; GPU benchmark done |
| 4 | Oct 19 – Oct 23 | Second configuration, midterm report | E1 or one E2 setting compared with B1; report submitted |

HW2 is due Oct 9, so week 2 is short; keep week 2 tasks small.

## Owners (tentative, not decided yet)

Starting point taken from the proposal; the team will confirm or change it.

| Person | Area |
|---|---|
| Will Sun | Training pipeline and LoRA (`src/train_lora.py`), latent cache, PSC allocation, GPU benchmark |
| Shawn Wang | Metrics: color classifier, ArcFace identity, LPIPS on unedited pixels, FID (`src/metrics/`) |
| Yubo Chen | Baselines B0 and B1 (`src/baselines/`), synthetic target-color pairs for E1 |
| Andy Zhang | E2 configs, figures, report and video |

Week 1 data work (loading, latent cache, color labels) is not assigned yet.

## Code layout

| Path | Contents |
|---|---|
| `src/data/` | CelebAMask-HQ loading, mask merge and dilation, color labels, splits |
| `src/baselines/` | B0 CIELAB recoloring, B1 zero-shot inpainting |
| `src/train_lora.py` | LoRA training on cached latents |
| `src/metrics/` | Color accuracy, identity, LPIPS, FID |
| `configs/` | One config per run; `compute_plan.json` is the full experiment plan |
| `scripts/benchmark_compute.py` | Measures FLOPs and throughput on the current GPU and prints device-hours for the plan |

Datasets, weights and checkpoints stay out of git.

## Compute

- Measured on the real model: 1.49 TFLOP per training sample, 44 TFLOP per edited image (30 steps with classifier-free guidance).
- With published V100 throughput, the planned runs need about 16–20 V100 GPU-hours. We request 120–150 on PSC to cover reruns, extra seeds and interactive development (PSC bills allocated GPU time, idle or not).
- The only unmeasured number is V100 training speed. First job on any GPU: `python scripts/benchmark_compute.py --batch-size 8` (about 10 minutes), then update the estimate.
- PSC request (Piazza @388): compute estimate + recommendation letter from a TA mentor; send the mentor a one-sentence project summary.
- Local Macs: batch 8 does not fit in 24 GB (attention matrices are stored in full on MPS); use batch 2 with gradient accumulation for debugging.

## Open decisions

- Share of synthetic target-color pairs in the main run (for example 50% of samples).
- Seeds: one per configuration for the midterm; add two more for the main run and E1 before the final report if the gap is small.
