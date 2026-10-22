# Method figure (T4)

Figure: `method_diagram.pdf/.png` (full text width, 7.0 x 3.64 in). Regenerate on Bridges-2 with
`PYTHONPATH=src python scripts/plot_method_diagram.py` (defaults point at `shared/data/CelebAMask-HQ` and the
first evaluation run). CPU only.

What it shows, and where in the code it comes from:

- (a) `cache.py` / `data.py`: the current color of the region is labeled by the median CIELAB color inside the
  mask, matched to the nearest palette color. A new color is drawn from `synthetic_colors`
  (red, blue, green, purple, pink, brown), never equal to the source. `recolor_lab` shifts the region's
  a*b* channels so their median matches the new color and keeps L* (lightness, so shading and texture stay).
  Example: training image 8613, black -> red.
- Mixing (`train.py`): each sample in a batch is the synthetic pair with probability `synthetic_fraction` = 0.5
  (target = recolored image, prompt names the new color), otherwise reconstruction (target = original, prompt
  names the source color). E1 sets the fraction to 0.
- (b) `train.py` / `evaluate.py`: the U-Net input is 9 channels: noisy target latent (4), mask resized to 64x64
  (1), VAE latent of the photo with the dilated mask zeroed (4). The prompt goes through the CLIP text encoder
  into cross-attention. The SD 1.5 inpainting U-Net is frozen; only LoRA (rank 16) on attention q, k, v, out
  is trained, with the noise-prediction MSE loss. Inference: 30 steps, guidance 7.5, raw output (no paste-back).
  Example: test image 4727, hair -> blue, Main seed 35 output (correct on both color metrics).

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/method_diagram.pdf}
  \caption{Method overview. (a) Synthetic recolor pairs: the current color of the hair or clothing region is
  labeled by its median CIELAB color, a different target color $c$ is sampled, and the region's a*b* channels
  are shifted to $c$ while lightness L* is kept, giving a target image $\tilde{x}$ and the prompt for $c$.
  During training each sample is this synthetic pair with probability 0.5 and otherwise a reconstruction of
  the original photo with its own color in the prompt (E1 uses reconstruction only). (b) The Stable
  Diffusion~1.5 inpainting U-Net takes the noisy latent, the mask and the latent of the masked photo
  (9 channels) and is conditioned on the prompt through cross-attention. The base model is frozen and only
  rank-16 LoRA weights on the attention projections are trained with the standard noise-prediction loss.
  At test time the edit is generated in 30 denoising steps and the raw output is evaluated. Photos from
  CelebAMask-HQ.}
  \label{fig:method}
\end{figure*}
```
