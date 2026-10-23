# Per-target-color edit success (T5)

Figure: `per_color_accuracy.pdf/.png` (rule-based color accuracy, main figure); `per_color_accuracy_clip.pdf/.png`
(CLIP color accuracy, companion). Numbers: `per_color_accuracy.csv` (both metrics, all methods incl. E2, per-seed values).
Regenerate with `python scripts/plot_per_color.py --runs /ocean/projects/cis260330p/shared/runs/wsun8`.

Data: eval runs `20261019T163222…_evaluate_2745b5d9` (B0, B1, E1, Main; seed 35), `…174034…_evaluate_38eac49d`
(E2: rank4, rank64, data10, data50; seed 35) and `…175622…_evaluate_210e0ca5` (Main and E1, training seeds 36/37).
Same 578 cases in all runs (292 hair, 286 clothing). Main and E1 are the mean over training seeds 35/36/37 with
standard error bars; B0 is deterministic and B1 has no adapter, so both are single runs. Accuracy is measured on the
raw inpainting output; the pipeline writes no separate composite images (the composite is only used for LPIPS
outside the mask).

## Findings (for the report)

Under the rule-based metric, Main (LoRA) beats plain inpainting (B1) on every target: 1.00 vs 0.12/0.01 for red/blue
hair and 0.98/0.92/0.86 vs 0.09/0.15/0.09 for red/blue/white clothing. Blond hair is the hardest target for every
method (at most 0.14), but CLIP rates the same blond edits at 0.95–0.98 for B1, E1 and Main, and even the direct Lab
recolor B0 scores only 0.08 on the rule. So the low blond number seems to come from the median-Lab rule rather than
from the edits. The synthetic recoloring pairs (Main vs E1) help most on red/blue hair (1.00 vs 0.18/0.02) and
red/blue clothing, but not on white clothing (0.86 vs 0.86); white and blond are not among the synthetic training
colors (red, blue, green, purple, pink, brown). Under CLIP, B1 is already strong on hair (0.79–0.95), so most of
Main's gain over B1 is on clothing (red 0.99 vs 0.29, white 0.89 vs 0.28).

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/per_color_accuracy.pdf}
  \caption{Color accuracy per target color for hair (left) and clothing (right). Color accuracy counts an edit
  as correct when the median CIELAB color of the edited region is closest to the target in a fixed palette.
  B0: direct CIELAB recolor inside the mask; B1: SD~1.5 inpainting without adapter; E1: LoRA trained on
  reconstruction only; Main: LoRA trained with reconstruction and synthetic recoloring pairs. Main and E1 show the
  mean over three training seeds with standard-error bars; B0 and B1 are single runs.
  $n = 88$--100 cases per bar. Blond and white are not among the synthetic training colors.}
  \label{fig:per-color}
\end{figure}
```

## Notes for Andy (not for the report text)

- The two color metrics disagree strongly. B1 hair is 0.01–0.12 on the rule but 0.79–0.95 on CLIP, and B0 hair blond
  is 0.08 on the rule vs 0.48 on CLIP. B0 white clothing is 0.00 on the rule. Worth one sentence in the report on
  the metric's limits; it ties to Dhiksha's point about noisy automatic color labels.
- "Painted-on" red/blue hair: these metrics cannot show it, so I left it out of the findings. Related number: Main's
  LPIPS outside the mask on red/blue hair is 0.042/0.045 vs 0.026 for B1 and E1 (blond: 0.027), so Main also
  changes more of the image outside the hair for those colors. Check the qualitative grid before claiming anything.
- Seed spread: Main hair blond per seed is 0.26 / 0.05 / 0.09 (rule), hence the large error bar.
- The first eval run was made from commit `7a1bd0d` with uncommitted changes; the other two from `92d013e`. B0/B1
  numbers are identical across all three runs.
- E2 (rank4/rank64/data10/data50, seed 35 only) is in the CSV but not plotted.
