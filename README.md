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
| `notebooks/` | Exploration and GPU checks |
| `results/` | Generated tables, plots, and sample grids (ignored by Git) |

Outside PSC, recreate a Python 3.11 environment, install a suitable PyTorch
build, then use `requirements.txt`. The PSC lock targets Linux with CUDA 11.8.
