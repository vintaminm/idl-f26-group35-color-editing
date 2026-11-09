# Final follow-up experiments and color evaluation

Run these commands from the repository root. GPU commands require an allocated
compute node and this clone's activated `.venv`. Preparation, training and
inference are explicit commands; none of these scripts submits a Slurm job.

## Completed experiment scope

- Six rank-16, 10,000-step adapters: NoRed and P75, each with training seeds
  35/36/37. The three seeds are independent training repetitions.
- NoRed excludes red from both reconstruction and synthetic adapter-training
  prompts. The pretrained backbone already knows color words.
- P75 samples synthetic targets with probability 0.75 and reconstruction targets
  otherwise; the original Main configuration uses 0.5.
- The completed CNN used 14,842 training, 2,912 validation and 2,959 test region
  crops, with 15 epochs and 487,538 trainable parameters.
- The completed human audit has 184 items: 148 clear colors, 30 mixed and
  6 uncertain. It is a small exploratory audit, not a comprehensive human study.

| Follow-up | Hair CLIP accuracy | Clothing CLIP accuracy |
|---|---:|---:|
| NoRed, red targets only | 41.0 ± 16.1% | 57.9 ± 5.6% |
| P75, all evaluation targets | 98.2 ± 1.3% | 94.4 ± 1.3% |

Values are means ± sample standard deviations across the three training seeds.
The rows have different target scopes and are not a direct ranking. New adapters
share one rebuilt cache; comparisons with the original campaign have a cache
provenance difference. P75 has not established an improvement over Main.

The CNN's 93.9% test score measures agreement with automatic Lab weak labels,
not human color accuracy. On the 148 clear audit items, Lab/CLIP/CNN agree with
human labels on 63.5%/77.0%/62.8%. All three are diagnostic proxies.

## Required experiment artifacts

These are PSC campaign entry points. The configs select the shared data/model
root; evaluation also expects the original shared evaluation runs and the
published 578-case protocol. `evaluate_followup.py` checks its reference
`cases.json`; `prepare_color_data.py` builds a catalog of the original and new
runs. A fresh clone alone does not include those artifacts.

Keep the training/evaluation manifests, adapters, source run metrics, audit
responses and CNN selection metadata in the separate artifact backup. In
particular, the exact completed CNN subset is defined by
`data/bounded-selection.json` and `data/dataset-bounded.json`; code alone cannot
reconstruct the interrupted export journal's prefix. When moving systems,
restore those files and update the saved artifact paths/configs before rerunning.
Generated images, weights, crops, returned CSVs and annotation answer keys stay
under ignored `results/`, outside public Git.

## Train the additional seeds

`scripts/final/followup.py` runs one replicate at a time in an allocated GPU
terminal. NoRed checks transfer of a color word excluded from adapter training;
P75 checks the effect of using 75% synthetic pairs. Both use rank 16, batch 8,
10,000 steps, and training seeds 35/36/37.

The old latent cache is not shared. This workflow rebuilds one fixed seed-35 B0
cache and uses it for all six replicates. Repeating seed 35 checks the same
training protocol with the rebuilt cache; it does not reproduce the exact cached
VAE samples from the original run. Report the new three-seed results as a separate
reproduction; do not combine the old seed 35 with new seeds 36/37 as though the
cached VAE samples were identical.

From the repository root, after activating `.venv` on a GPU node:

```bash
python scripts/final/followup.py prepare
python scripts/final/followup.py train --experiment nored --seed 35
```

Run the training command separately for seeds 36 and 37. After NoRed, use
`--experiment p75` for the same three seeds if resources permit. Each command
starts only the requested replicate. Preparation reuses completed cache shards;
training skips a recorded, completed replicate. An interrupted training attempt
starts a new run from step zero; adapter checkpoints do not contain optimizer
state. The script checks the allocation has at least 125 minutes left before
starting a V100 training run and never requests another allocation itself.

Cache construction and training show dynamic terminal progress. Paths are
recorded in `results/final-followup/manifest.json`; cache files and console logs
are under `results/final-followup/{cache,logs}/`. Each training run retains its
normal `results/<run>/` directory with `adapter/`, checkpoints, `loss.jsonl`,
`val.jsonl`, and configuration. These paths are ignored by Git. Check saved paths:

```bash
python scripts/final/followup.py status
```

Evaluate the saved adapters using `configs/main.json` (seed 35), so test cases and
generation seeds match the published evaluation. Changing a training seed must
not change evaluation seeds. All results stay in the personal clone until an
explicit publication step; shared photos and base model weights are read only.

## Evaluate the additional seeds

Prepare the evaluation dependencies and public model weights once, on a login
node. Constraints retain the existing PyTorch CUDA 11.8 environment. ArcFace uses
CPU ONNX Runtime; CLIP, LPIPS and Inception use the allocated GPU during evaluation.

```bash
source .venv/bin/activate
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  -c requirements-psc.lock.txt -r requirements-evaluation.txt
python -u scripts/final/prepare_evaluation.py
```

Preparation downloads roughly 2.5 GB into this clone's `results/` caches and records
versions and checksums in `results/final-followup/evaluation-assets.json`. It does
not run inference or train a model. Wait for `Evaluation weights ready`.

On an allocated GPU terminal with `.venv` activated:

```bash
python -u scripts/final/evaluate_followup.py run
```

The command evaluates NoRed and then P75, three adapters each. It asserts that all
578 cases, prompts, masks and generation seeds match the published test protocol.
No B1 regeneration is needed; B0 is a deterministic CPU control. Method names
`NoRed-r35/r36/r37` and `P75-r35/r36/r37` identify the new reproduction campaign.
For each group it saves raw generated images, Lab/LPIPS summaries, CLIP scores,
ArcFace scores, full/regional FID and KID, training curves and a qualitative grid.
KID retains the original 100 subsets, subset size 100 and seed 35.

Progress is shown in the terminal and paths/stages are recorded in
`results/final-followup/evaluation.json`. Console logs are in the adjacent `logs/`.
Rerun the same command after a failure: valid generated batches and completed
stages are reused. Interrupted scoring is restarted without appending duplicate
rows; independent metric failures do not require regenerating images. Run a
single group with `run --group nored` or `run --group p75`, or inspect saved state
with `status`. All outputs remain personal until explicitly published.

## Train the color evaluator and run the human audit

On an allocated GPU node, activate `.venv` and run:

```bash
bash scripts/final/run_visible.sh final-color \
  python -u scripts/final/finish_color.py run
```

When switching from a partially completed full crop export, stop that command and
wait for its terminal prompt before running the bounded continuation instead:

```bash
bash scripts/final/run_visible.sh final-color-bounded \
  python -u scripts/final/finish_color_bounded.py
```

It preserves and reuses completed training crops (up to 20,000 regions), records
the actual selected subset in `data/dataset-bounded.json`, and still exports the
full original validation/test splits. The CNN receives that manifest explicitly;
the 13-run image-scoring catalog and 184-item human audit remain the same. Do not
describe this CNN as trained on all 37,971 LoRA training regions. The full export
recipe, journal, and all completed LoRA experiments are preserved separately.

This exports real region crops with the existing image-ID splits, builds a blinded
offline annotation package, trains a small color CNN, and scores every completed
evaluation in the exported catalog. It also saves three-seed NoRed/P75 report
tables. Install the base environment and `requirements-evaluation.txt` as described above.
The CNN uses those packages and requires no additional pretrained model download. Progress, ETA, and a persistent terminal log are displayed. A failed
command stops the workflow; rerunning it reuses validated exports, epoch
checkpoints, and completed metric runs.

The CNN learns automatic Lab **weak labels**, with `blond`/`yellow` merged into one
class. Its score is a diagnostic proxy. Report class support, held-out human
agreement, and uncertain cases alongside it. Human test labels never train or
select this evaluator. The audit contains 100 original regions and 84 paired edits
covering B0, B1, E1, Main, Light, NoRed, and P75; each method has only 12 audited
edits, so per-method human percentages are exploratory.

When `ANNOTATION PACKAGE READY` appears, download
`results/final-color/human-audit/annotation-pack.zip` using VS Code Explorer.
Unzip on a local computer and open `index.html`. The CNN can keep running on PSC
while a person labels the images locally.

Give an annotator only the ZIP and an anonymous ID such as `A01`. The page and
`INSTRUCTIONS.txt` explain the task: select the actual visible region color, allow
mixed/uncertain answers, classify clothing as plain/patterned, and download the
completed CSV. No account, installation, or project knowledge is required. Browser
storage is a convenience; the downloaded CSV is the returned submission. Keep only
the intended final CSV per annotator in the response directory.

Upload the returned `annotations_*.csv` files to
`results/final-color/human-audit/responses/`, then run after CNN scoring finishes:

```bash
python scripts/final/human_audit.py analyze
python scripts/final/finish_color.py verify
```

Analysis validates pack/item IDs, missing answers and conflicting submissions;
reports Lab/CLIP/CNN agreement with each person, plain/patterned clothing, human
target-color matching, ambiguous-case coverage, and disagreements if two people
participate. One completed annotator is supported and must be described as such.
Wait for `FINAL_AUDIT_COMPLETE` and `FINAL_COLOR_WORK_COMPLETE`.

| Output under `results/final-color/` | Contents |
|---|---|
| `data/` | 128-pixel real crops, weak labels, fixed split manifest, evaluation catalog |
| `cnn/` | Classifier checkpoint, training curves, real-test confusion matrix, per-run edit scores |
| `human-audit/annotation-pack.zip` | Blinded images, offline annotation page, instructions |
| `human-audit/key.json` | Private mapping to methods, targets and automatic labels; keep out of the annotator ZIP |
| `human-audit/responses/` | Returned annotation CSVs |
| `human-audit/analysis/` | Per-person statistics, joined labels, disagreements, readable report |
| `followup_summary.csv` | NoRed/P75 three-seed means and sample standard deviations |
| `status.json` | Compute and human-audit completion status |

These files are ignored by Git. Back them up with the original evaluation runs;
GitHub does not store the generated artifacts. The interactive allocation remains
open after the compute script finishes; `exit` releases it.

