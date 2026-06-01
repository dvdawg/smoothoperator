#!/bin/zsh
# Parallel CPU benchmark driver. Runs one process per dataset (limited
# concurrency) for the main std-vs-ca comparison and the extension subset,
# then merges the per-dataset CSVs.
set -e
cd "$(dirname "$0")"

# limit per-process threads so concurrent processes don't thrash
export OMP_NUM_THREADS=${THREADS:-2}
export MKL_NUM_THREADS=${THREADS:-2}
export VECLIB_MAXIMUM_THREADS=${THREADS:-2}

COMMON="--epochs 50 --grid_size 64 --n_train 800 --n_test 200 --modes 12 --batch_size 20 --device cpu"
MAIN_SEEDS=${MAIN_SEEDS:-20}
EXT_SEEDS=${EXT_SEEDS:-6}
CONC=${CONC:-4}
mkdir -p runs

run_limited() {  # throttle to $CONC background jobs
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n 2>/dev/null || sleep 1; done
}

echo "=== MAIN (std vs ca, $MAIN_SEEDS seeds) ==="
for ds in poisson heat advection darcy wave darcy_multi heat_sensor; do
  run_limited
  python3 -W ignore robust_benchmark.py --datasets $ds --models std ca \
    --trials $MAIN_SEEDS $COMMON --output_dir runs/main_$ds \
    > runs/main_$ds.log 2>&1 &
done
wait

echo "=== EXTENSIONS (all models, $EXT_SEEDS seeds) ==="
for ds in heat wave darcy_multi heat_sensor; do
  run_limited
  python3 -W ignore robust_benchmark.py --datasets $ds \
    --models std ca ll per_layer dynamic anisotropic \
    --trials $EXT_SEEDS $COMMON --output_dir runs/ext_$ds \
    > runs/ext_$ds.log 2>&1 &
done
wait

echo "=== MERGE ==="
python3 - <<'PY'
import pandas as pd, glob
m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('runs/main_*/benchmark_raw.csv'))], ignore_index=True)
m.to_csv('runs/main_raw.csv', index=False)
e = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('runs/ext_*/benchmark_raw.csv'))], ignore_index=True)
e.to_csv('runs/ext_raw.csv', index=False)
print('main rows', len(m), 'ext rows', len(e))
PY
echo ALL_DONE
