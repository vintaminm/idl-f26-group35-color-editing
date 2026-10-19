#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:l40s-48:1
#SBATCH -A cis260330p
#SBATCH -t 03:00:00
#SBATCH -o results/logs/%x-%j.out
# run from the repo root:
# sbatch -J <name> scripts/psc/run.sbatch <module> [args]
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PWD/results/_hf_cache" TORCH_HOME="$PWD/results/_torch_cache"
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
module="$1"
shift
.venv/bin/python -m "color_editing.$module" "$@"
echo "finished $(date)"
