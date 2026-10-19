# Text-Guided Hair and Clothing Color Editing

11-785 Introduction to Deep Learning, Fall 2026 — Group 35

Members: Will Sun, Shawn Wang, Yubo Chen, Andy Zhang

Our project uses LoRA with Stable Diffusion v1.5 inpainting to edit hair and
clothing colors from text. Baselines: CIELAB recoloring (B0) and zero-shot
inpainting (B1). See [docs/plan.md](docs/plan.md).

## Set up once on PSC

Each member needs access to the `cis260330p` project and the GitHub repo.
Run `newgrp` by itself first; it starts a project-group shell:

```bash
newgrp cis260330p
```

Then clone into the member's Ocean directory and create the Python 3.11.9
environment. Run the commands in order; stop if one fails:

```bash
cd "/ocean/projects/cis260330p/$USER"
git clone git@github.com:vintaminm/idl-f26-group35-color-editing.git
cd idl-f26-group35-color-editing
/usr/bin/python3.11 --version
/usr/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --index-url https://pypi.org/simple pip==26.2.1
python -m pip install --no-cache-dir --no-deps \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r requirements-psc.lock.txt
python -m pip check
```

The PSC lock includes PyTorch, project libraries, and notebook tools. Use
this lock on PSC. `requirements.txt` contains the direct dependencies,
including notebook tools, for machines with a different PyTorch build. The
`.venv` stays in each member's clone for later sessions.

## Work in a new terminal

`newgrp` applies to one terminal, so run it again when opening a new project
terminal:

```bash
newgrp cis260330p
```

Enter the clone:

```bash
cd "/ocean/projects/cis260330p/$USER/idl-f26-group35-color-editing"
```

For editing or checking installed packages on the login node, activate the environment:

```bash
source .venv/bin/activate
```

VS Code can select this `.venv` as a notebook kernel. Run computational
notebooks on an allocated node.

## Use a GPU

From the project directory on a login node, request a GPU under the project
account:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 02:00:00 -A cis260330p
```

After the compute-node prompt appears, enter the clone again and activate the
same environment:

```bash
cd "/ocean/projects/cis260330p/$USER/idl-f26-group35-color-editing"
source .venv/bin/activate
nvidia-smi
```

Run Python scripts here. To use VS Code notebooks, start Jupyter in this
compute-node terminal:

```bash
hostname -f
jupyter notebook --no-browser --ip=0.0.0.0
```

In VS Code choose **Select Kernel → Select Another Kernel → Existing Jupyter
Server**. Use the Jupyter URL, port, and token; replace `localhost` in the URL
with the hostname printed above. Keep Jupyter running while using the notebook.
For a small test, use [notebooks/gpu_mask_smoke_test.ipynb](notebooks/gpu_mask_smoke_test.ipynb).
Stop Jupyter with **Ctrl+C**, then `exit` to release the interactive job.

`interact` allows up to 8 hours. `GPU-shared` is limited to one node and
four GPUs; other GPU types include H100 and L40S. Multi-node GPU work uses the
`GPU` partition, and longer jobs use `sbatch`. See the
[PSC GPU guide](https://www.psc.edu/resources/bridges-2/user-guide/).

## Check jobs and resources

On a login node (the `squeue` command also works on a compute node):

```bash
projects
squeue -u "$USER" -o "%i %t %M %L %N"
sacct -X -u "$USER" -S today -o JobID,JobName,State,Elapsed,NodeList
```

These show project balances, current jobs (`TIME_LEFT` is time remaining),
and today's job history. On the allocated compute node:

```bash
nvidia-smi
ps -u "$USER" -o pid,etime,%cpu,%mem,cmd
```

These show GPU usage and the member's processes on that node.

## Ocean and Git

- `/jet/home/$USER`: personal SSH keys and settings.
- `/ocean/projects/cis260330p/$USER`: each member's clone and `.venv`.
- `/ocean/projects/cis260330p/shared`: team datasets and model weights.

`$USER` is the PSC username. Ocean is available on both login and compute
nodes. Project storage is shared across all member directories and `shared`;
`projects` shows the current quota. `newgrp` sets the file group for the
terminal; `-A cis260330p` charges GPU time to the project account.

Each member works on a Git branch and pushes code, notebooks, configs, and
dependency files. Clear notebook outputs before committing. Generated results
and logs go in `results/` and are ignored by Git; datasets, checkpoints, and
`.venv` are also ignored.

## Notebook workflow

Use `configs/pilot.json` for the shared root, region, masks, and run size.
`split_seed` fixes the data partition; `seed` controls training and generation.
Start with notebook 01, then try 02 using the compute-node Jupyter kernel above.
Run 03 after the data previews and base-model inference work.

| Notebook | Flow | Default size |
|---|---|---|
| [01_data_preview](notebooks/01_data_preview.ipynb) | Photos, raw/dilated masks, provisional color labels, fixed splits | 4 previews; CPU |
| [02_baseline_editing](notebooks/02_baseline_editing.ipynb) | B0 vs B1; optional saved LoRA | 2 validation images, 20 inference steps; GPU |
| [03_lora_pilot](notebooks/03_lora_pilot.ipynb) | Reconstruction LoRA, save adapter, reload and compare | 16 train images, batch 1, rank 4, 20 training steps; GPU |

Run cells in order. GPU functions check the kernel's project allocation and load
only the existing local model. Text embeddings and latents are computed inside
the training flow and cached in RAM for the selected samples. The initial color
palette/labels need visual review; the pilot is a workflow check, not evidence of
training quality. E1 synthetic supervision and formal evaluation come later.

Every run gets a unique directory under the member's clone:

```text
results/<timestamp>_<username>_<kind>_<id>/
  config.json, run.json       # settings, code/package versions, seed, job, sources
  samples.json               # exact image IDs, regions, labels/prompts
  splits.json, palette.json  # data-preview runs
  previews/                  # data-preview grids
  images/, images.jsonl      # comparison PNGs and per-output records
  loss.jsonl                 # training progress
  adapter/, checkpoints/     # final LoRA and intermediate adapter snapshots
```

Fields depend on the run type. Adapter snapshots do not include optimizer state
for exact training resume. Comparisons retain raw and composited images separately;
background copied from the original is postprocessing, not learned preservation.

Code, configs, and cleared notebooks go to GitHub. Outputs stay in Ocean and are
ignored by Git. To share a completed run, explicitly uncomment its final
`publish_run` cell; it copies to `shared/runs/<username>/<run-id>/` and refuses to
overwrite. Publish the training run to share its `adapter/`, and the comparison
run to share its images. Original shared data and base weights remain unchanged.

## Full training and evaluation

Batch jobs run the modules in `src/color_editing/` through
`scripts/psc/run.sbatch` (one L40S, project account, logs in `results/logs/`).
Submit from the clone root. `configs/main.json` trains on reconstruction plus
50% synthetic target-color pairs (B0 recolorings captioned with the new
color); `configs/e1.json` trains on reconstruction only.

Cache latents once (VAE and text encoder are frozen), then train:

```bash
C="/ocean/projects/cis260330p/$USER/cache"
R=scripts/psc/run.sbatch
sbatch -J cache-train0 -t 01:30:00 $R cache --config configs/main.json --split train --shard 0 --num-shards 2 --out $C/train-0.pt
sbatch -J cache-train1 -t 01:30:00 $R cache --config configs/main.json --split train --shard 1 --num-shards 2 --out $C/train-1.pt
sbatch -J cache-val -t 00:30:00 $R cache --config configs/main.json --split val --limit 500 --out $C/val.pt
sbatch -J train-main $R train --config configs/main.json --train-cache $C/train-0.pt $C/train-1.pt --val-cache $C/val.pt
```

Each training run writes `loss.jsonl` (training loss), `val.jsonl`
(validation loss with fixed noise and timesteps, on reconstruction and
synthetic pairs), checkpoints and `adapter/` under `results/`.

Evaluate B0, B1 and any adapters on the same test edits, then plot:

```bash
sbatch -J eval $R evaluate --config configs/main.json \
  --adapter main=results/<main run>/adapter e1=results/<e1 run>/adapter
PYTHONPATH=src .venv/bin/python -m color_editing.plots --out-dir results/figures \
  --runs main=results/<main run> e1=results/<e1 run> \
  --eval-run results/<eval run> --methods B0 B1 e1 main
```

The evaluation writes `cases.json`, generated images, per-edit
`metrics.jsonl` and `summary.csv` (color accuracy, CIELAB color error, and
LPIPS/MAE outside the edit mask).

## Shared data and model

Use the team's existing copy at `/ocean/projects/cis260330p/shared` from
any login or compute node. Members do not need to download or copy it into
their own clones.

| Path relative to `shared/` | Contents |
|---|---|
| `data/CelebAMask-HQ/CelebA-HQ-img/` | 30,000 photos, 1024 × 1024 |
| `data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/` | Binary masks, 512 × 512; 29,300 hair and 17,607 clothing masks |
| `data/CelebAMask-HQ/` | Original attribute, pose, and CelebA mapping files, plus the dataset README |
| `models/sd15-inpainting/` | SD 1.5 inpainting: fp16 weights, tokenizer, and pipeline configs |
| `metadata/assets.json`, `metadata/setup.log` | Sources, pinned model revision, checksums, validation results, and setup log |

Photo IDs run from 0 to 29999. For example, these paths select one photo
and its hair mask; use `"cloth"` for clothing:

```python
from pathlib import Path

data = Path("/ocean/projects/cis260330p/shared/data/CelebAMask-HQ")
image_id, region = 0, "hair"
image_path = data / "CelebA-HQ-img" / f"{image_id}.jpg"
mask_path = (
    data / "CelebAMask-HQ-mask-anno" / str(image_id // 2000)
    / f"{image_id:05d}_{region}.png"
)
```

Skip a region sample if its mask file is missing. Resize the photo to
512 × 512 to align with the mask; use nearest-neighbor interpolation if
resizing masks. White mask pixels mark the selected region to edit.
Splits, color labels, and synthetic target pairs are generated separately;
keep the original photos and annotations unchanged.

Load the shared model on an allocated GPU node using the project's environment:

```python
import torch
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "/ocean/projects/cis260330p/shared/models/sd15-inpainting",
    variant="fp16",
    dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
).to("cuda")
```

Only the fp16 safetensors variant is stored. `local_files_only=True` uses
the shared files without downloading another copy.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Data loading, color labels, LoRA training, editing, and metrics |
| `configs/` | Experiment configurations |
| `scripts/` | Training, editing, and evaluation entry points |
| `notebooks/` | Data preview, baseline comparison, LoRA pilot, and GPU checks |
| `results/` | Personal runs: configs, images, losses, and adapters (ignored by Git) |

Outside PSC, recreate a Python 3.11 environment, install a suitable PyTorch
build, then use `requirements.txt`. The PSC lock targets Linux with CUDA 11.8.
