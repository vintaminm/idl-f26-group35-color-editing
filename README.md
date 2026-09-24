# Text-Guided Hair and Clothing Color Editing

11-785 Introduction to Deep Learning, Fall 2026 — Project Group 35

Members: shawnwan, yuboc, andyzha2, wills

We adapt a frozen Stable Diffusion v1.5 inpainting model with LoRA, trained on CelebAMask-HQ hair and clothing masks paired with color captions, so that one adapter edits hair and clothing color from text. Baselines: non-learned CIELAB recoloring (B0) and zero-shot inpainting (B1).

## Layout

| Folder | Contents |
|---|---|
| `src/` | Data loading, color labels, LoRA training, editing, metrics |
| `configs/` | One config file per experiment run |
| `scripts/` | Entry points for training, editing and evaluation |
| `notebooks/` | Colab notebooks that call code in `src/` |
| `results/` | Small outputs only: metric tables, plots, sample grids |

Datasets, model weights and checkpoints are not stored in this repository (see `.gitignore`).
