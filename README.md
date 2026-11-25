# CUDA K-Means Clustering

This capstone compares serial C++ and CUDA C++ implementations of Lloyd's K-means on five reproducible two-dimensional datasets, ranging from 10,000 to 1,000,000 points. Both use identical starting points, distance calculations, stopping rules, and inputs.

## Run on Google Colab

Open the supplied CUDA_KMeans_Clustering.ipynb notebook in Colab. Select **Runtime → Change runtime type → T4 GPU**, then choose **Run all**. The notebook writes the project files into the runtime, checks for NVIDIA hardware and nvcc, runs the benchmark, and offers a download of proof-gpu.zip. It uses a fresh project directory so earlier work is preserved.

## Run on Linux

Use Bash, GNU Make, a C++17 compiler supported by your CUDA toolkit, and Python 3.10–3.12. GPU execution needs an NVIDIA driver and CUDA Toolkit with nvcc on PATH. The default build targets the Tesla T4. Set CUDA_ARCH to your GPU's SM target when using a different card.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash run.sh gpu
```

The CPU-only workflow uses the same dependencies and runs with `bash run.sh cpu`. For a build only, use `make cpu` or `make cuda`.

## Command-line interface

```text
build/kmeans INPUT.csv K MAX_ITERATIONS {cpu|gpu} OUTPUT_DIR [REPEATS]
```

The CPU-only executable is `build/kmeans_cpu`. Input uses the header `x,y` and finite coordinates within ±1e100. K must be between 1 and 256 and no larger than the row count. Output includes `clusters.csv` and `centroids.csv`. Standard output reports mode, points, K, iterations, repeats, median milliseconds, and inertia.

## Algorithm and measurement

- The first input row seeds the first centroid. Successive seeds are the points farthest from their nearest existing seed.
- A CUDA thread assigns one point to the nearest centroid. Threads accumulate FP64 coordinate sums and integer counts in block-local shared memory. Each block merges its occupied bins into global totals with atomics.
- The host reads the compact totals, updates centroid means, and retains previous centers for empty clusters. The run stops when the largest centroid movement is at most 1e-6 or the iteration limit is reached. Both implementations recompute final labels from the returned centroids.
- Timings are wall-clock medians of five runs after one warm-up per mode and size. The measured CUDA call includes device allocation, transfers, kernels, host updates and final labels. Both modes exclude CSV I/O, shared seed initialization and inertia calculation. Speedup is CPU median divided by CUDA median. Atomic summation order can change low-order digits.
- For these seeded datasets the comparison requires identical labels, maximum centroid distance at most 1e-6, and relative inertia difference at most 1e-8.

## Evidence and submission

The runner creates benchmark.csv, before-and-after and runtime plots, an execution log, the complete 10,000-point input and output, checksums for all generated inputs, and the one-million-point centroids. CUDA mode also creates comparison.csv. Full intermediate CSVs remain under output/. The archive contains the results/ directory.

## Measured Tesla T4 results

These are the CPU and GPU medians from the same Google Colab session using CUDA Toolkit 13.0.

| Points | CPU (ms) | GPU (ms) | CPU / GPU |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.284 | 0.388 | 0.73× |
| 50,000 | 1.312 | 0.773 | 1.70× |
| 100,000 | 3.842 | 1.239 | 3.10× |
| 500,000 | 35.365 | 5.304 | 6.67× |
| 1,000,000 | 36.409 | 9.209 | 3.95× |

All five runs had zero label disagreements. The maximum centroid difference across the suite was 2.45e-13 and the reported inertia matched at every size. The T4 was slower at 10,000 points and faster at every larger tested size. These medians describe one Colab session and can vary with runtime load.

The accompanying proof-gpu.zip contains the locally packaged Colab evidence. The Colab export cell also offers the original runtime archive. The repository includes the source and benchmark evidence.
