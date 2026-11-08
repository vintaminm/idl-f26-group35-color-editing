#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -A cis260330p
#SBATCH -t 02:30:00
#SBATCH -o results/logs/%x-%j.out
# run from the repo root:
# sbatch -J <name> [-d afterok:<eval job>] scripts/psc/metrics_final.sbatch
#   <eval run dir | eval job id>
# arcface identity and fid/kid for every method of one evaluation run
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PWD/results/_hf_cache" TORCH_HOME="$PWD/results/_torch_cache"
# onnxruntime 1.22 is a cuda 12 build, so arcface runs on the gpu
# with the cuda libs that came with torch
NV=$PWD/.venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$LD_LIBRARY_PATH
# without onnxruntime-gpu 1.22 in $ORT arcface runs on the cpu:
# pip install --target results/_ort onnxruntime-gpu==1.22.0
ORT="${ORT:-$PWD/results/_ort}"
export PYTHONPATH=$ORT:$PYTHONPATH
PY=.venv/bin/python
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
RUN="$1"
# a bare job id means the eval run written by that job
if [ ! -d "$RUN" ]; then
  RUN=$(grep -l "\"slurm_job_id\": \"$1\"" results/*_evaluate_*/run.json | head -1 | xargs dirname)
fi
if [ -z "$RUN" ]; then echo "no eval run for $1"; exit 1; fi
echo "eval run $RUN"
$PY -m color_editing.arcface_metrics --eval-run "$RUN"
$PY -m color_editing.fid_metrics --eval-run "$RUN"
echo "finished $(date)"
