#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-gpu}
case "$mode" in cpu|gpu) ;; *) echo "Usage: $0 [cpu|gpu]" >&2; exit 2 ;; esac

export MPLCONFIGDIR=${MPLCONFIGDIR:-"$PWD/build/matplotlib"}
mkdir -p results "$MPLCONFIGDIR"
archive="proof-${mode}.zip"
rm -f "$archive" results/comparison.csv
rm -rf results/evidence
exec > >(tee results/execution_log.txt) 2>&1
echo "K-means execution mode: $mode"
date -u
uname -sm
g++ --version
if [[ $mode == gpu ]]; then
  nvidia-smi
  nvcc --version
  # Force a CUDA rebuild if the GPU architecture changes between runs.
  make -B cuda
  binary=build/kmeans
  backends=(cpu gpu)
else
  make cpu
  binary=build/kmeans_cpu
  backends=(cpu)
fi

python3 generate_dataset.py
echo 'mode,points,k,iterations,repeats,median_ms,inertia' > results/benchmark.csv
for size in 10000 50000 100000 500000 1000000; do
  for backend in "${backends[@]}"; do
    "$binary" "dataset/input_${size}.csv" 5 100 "$backend" "output/$backend/$size" 5 |
      tee -a results/benchmark.csv
  done
done
python3 report.py "$mode"

for backend in "${backends[@]}"; do
  mkdir -p "results/evidence/$backend"
  cp "output/$backend/10000/"*.csv "results/evidence/$backend/"
  cp "output/$backend/1000000/centroids.csv" "results/evidence/$backend/centroids_1000000.csv"
  wc -l "output/$backend/1000000/clusters.csv"
done
cp dataset/input_10000.csv results/evidence/input_10000.csv
sha256sum dataset/input_*.csv > results/evidence/input_sha256.txt
echo "Finished $mode run and generated evidence."
# Close the logging pipe and wait for tee to finish before archiving the log.
exec 1>&- 2>&-
wait "$!"
python3 -c 'import shutil, sys; shutil.make_archive(sys.argv[1], "zip", ".", "results")' "proof-${mode}"
