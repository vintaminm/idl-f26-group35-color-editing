#!/bin/bash
# final-report runs: p sweep and withheld colors on the existing cache,
# lightness-aware targets on a rebuilt cache, then one eval job per group
set -eo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/logs
# latent cache from the readme; the lightness-aware one is built beside it
C="${CACHE:-/ocean/projects/cis260330p/$USER/cache}"
L="${LIGHT_CACHE:-$C-light}"
R=scripts/psc/run.sbatch
E=scripts/psc/eval_final.sbatch
GPU="${GPU:-v100-32}"
sub() { sbatch --parsable --gres=gpu:$GPU:1 "$@"; }
T="--train-cache $C/train-0.pt $C/train-1.pt --val-cache $C/val.pt"
TL="--train-cache $L/train-0.pt $L/train-1.pt --val-cache $L/val.pt"

# p sweep with the old b0 targets (p=0 is e1, p=0.5 is main)
dp=""
dp="$dp:$(sub -J train-p25 -t 03:00:00 $R train --config configs/final-p25.json $T)"
for s in 36 37; do
  dp="$dp:$(sub -J train-p25-s$s -t 03:00:00 $R train --config configs/final-p25.json --seed $s $T)"
done
dp="$dp:$(sub -J train-p75 -t 03:00:00 $R train --config configs/final-p75.json $T)"

# withheld color words
dw=""
dw="$dw:$(sub -J train-noblue -t 03:00:00 $R train --config configs/final-noblue.json $T)"
for s in 36 37; do
  dw="$dw:$(sub -J train-noblue-s$s -t 03:00:00 $R train --config configs/final-noblue.json --seed $s $T)"
done
dw="$dw:$(sub -J train-nored -t 03:00:00 $R train --config configs/final-nored.json $T)"

# lightness-aware targets: same shards and seeds as the old cache
mkdir -p $L
a=$(sub -J cache-light0 -t 01:30:00 $R cache --config configs/final-light.json --split train --shard 0 --num-shards 2 --out $L/train-0.pt)
b=$(sub -J cache-light1 -t 01:30:00 $R cache --config configs/final-light.json --split train --shard 1 --num-shards 2 --out $L/train-1.pt)
v=$(sub -J cache-lightval -t 00:30:00 $R cache --config configs/final-light.json --split val --limit 500 --out $L/val.pt)
D="afterok:$a:$b:$v"
dl=""
dlp=""
dl="$dl:$(sub -J train-light -t 03:00:00 -d $D $R train --config configs/final-light.json $TL)"
dlp="$dlp:$(sub -J train-light-p25 -t 03:00:00 -d $D $R train --config configs/final-light-p25.json $TL)"
for s in 36 37; do
  dl="$dl:$(sub -J train-light-s$s -t 03:00:00 -d $D $R train --config configs/final-light.json --seed $s $TL)"
  dlp="$dlp:$(sub -J train-light-p25-s$s -t 03:00:00 -d $D $R train --config configs/final-light-p25.json --seed $s $TL)"
done

# evals wait for their adapters; one keeps b1 as a reproducibility check
e1=$(sub -J eval-psweep -d afterok$dp $E P25=train-p25 P25-s36=train-p25-seed36 P25-s37=train-p25-seed37 P75=train-p75)
e2=$(sub -J eval-withheld -d afterok$dw $E --skip-b1 NoBlue=train-noblue NoBlue-s36=train-noblue-seed36 NoBlue-s37=train-noblue-seed37 NoRed=train-nored)
e3=$(sub -J eval-light -d afterok$dl $E --skip-b1 Light=train-light Light-s36=train-light-seed36 Light-s37=train-light-seed37)
e4=$(sub -J eval-lightp25 -d afterok$dlp $E --skip-b1 LightP25=train-light-p25 LightP25-s36=train-light-p25-seed36 LightP25-s37=train-light-p25-seed37)
echo "p sweep${dp//:/ } | withheld${dw//:/ } | cache $a $b $v | light${dl//:/ } | light p25${dlp//:/ } | eval $e1 $e2 $e3 $e4"
