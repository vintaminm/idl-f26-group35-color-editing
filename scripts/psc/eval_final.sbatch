#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -A cis260330p
#SBATCH -t 03:00:00
#SBATCH -o results/logs/%x-%j.out
# run from the repo root:
# sbatch -J <name> [--export=ALL,CFG=<config>] scripts/psc/eval_final.sbatch
#   [--skip-b1] Label=<train run name> [...]
# evaluates the newest adapter of each named run, then clip metrics and grid
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=src PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PWD/results/_hf_cache" TORCH_HOME="$PWD/results/_torch_cache"
PY=.venv/bin/python
CFG="${CFG:-configs/main.json}"
echo "job $SLURM_JOB_ID on $(hostname) $(date)"
extra=""
b1="B1"
if [ "$1" = "--skip-b1" ]; then extra="--skip-b1"; b1=""; shift; fi
adapters=""
names=""
for a in "$@"; do
  label="${a%%=*}"
  name="${a#*=}"
  dir=$(ls -d results/*_${USER}_${name}_*/adapter 2>/dev/null | tail -1)
  if [ -z "$dir" ]; then echo "no adapter for $name"; exit 1; fi
  adapters="$adapters $label=$dir"
  names="$names $label"
done
echo "adapters:$adapters"
$PY -m color_editing.evaluate --config "$CFG" $extra --adapter $adapters
RUN=$(grep -l "\"slurm_job_id\": \"$SLURM_JOB_ID\"" results/*_evaluate_*/run.json | head -1 | xargs dirname)
echo "eval run $RUN"
$PY -m color_editing.clip_metrics --eval-run "$RUN"
$PY -m color_editing.plots --out-dir "$RUN/figures" --eval-run "$RUN" --methods B0 $b1 $names
echo "finished $(date)"
