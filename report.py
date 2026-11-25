import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_dataset import SIZES

mode = sys.argv[1]
with open("results/benchmark.csv") as stream:
    rows = {(r["mode"], int(r["points"])): r for r in csv.DictReader(stream)}

if mode == "gpu":
    with open("results/comparison.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["points", "cpu_over_gpu", "label_disagreements",
                         "max_centroid_distance", "relative_inertia_difference"])
        for size in SIZES:
            def read(backend, filename, **options):
                return np.loadtxt(Path("output") / backend / str(size) / filename,
                                  delimiter=",", skiprows=1, **options)

            cpu = rows["cpu", size]
            gpu = rows["gpu", size]
            cpu_labels = read("cpu", "clusters.csv", usecols=2, dtype=int)
            gpu_labels = read("gpu", "clusters.csv", usecols=2, dtype=int)
            if cpu_labels.shape != (size,) or gpu_labels.shape != (size,):
                raise ValueError(f"Incorrect row count for {size:,} points")
            disagreements = np.count_nonzero(cpu_labels != gpu_labels)
            distances = np.linalg.norm(
                read("cpu", "centroids.csv", usecols=(1, 2)) -
                read("gpu", "centroids.csv", usecols=(1, 2)), axis=1
            )
            inertia_error = abs(float(cpu["inertia"]) - float(gpu["inertia"]))
            inertia_error /= max(1.0, float(cpu["inertia"]))
            speedup = float(cpu["median_ms"]) / float(gpu["median_ms"])
            writer.writerow([size, speedup, disagreements, max(distances), inertia_error])
            if disagreements or max(distances) > 1e-6 or inertia_error > 1e-8:
                raise ValueError(f"CPU and CUDA disagree on the seeded dataset of {size:,} points")
    print("CPU and CUDA agree on labels, centroids and inertia at all five sizes")

points = np.loadtxt("output/" + mode + "/10000/clusters.csv", delimiter=",", skiprows=1)
centers = np.loadtxt("output/" + mode + "/10000/centroids.csv", delimiter=",", skiprows=1)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharex=True, sharey=True)
for ax, title in zip(axes, ("Input points", f"{mode.upper()} cluster assignments")):
    ax.set(title=title, xlabel="X", ylabel="Y", aspect="equal")
    ax.grid(alpha=0.2)
axes[0].scatter(points[:, 0], points[:, 1], s=2, alpha=0.35, color="#677886")
axes[1].scatter(points[:, 0], points[:, 1], s=2, alpha=0.5, c=points[:, 2], cmap="viridis")
axes[1].scatter(centers[:, 1], centers[:, 2], marker="X", c="black",
                edgecolors="white", s=100, label="Centroids")
axes[1].legend(loc="lower center")
fig.savefig("results/clusters.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
for backend in ("cpu", "gpu") if mode == "gpu" else ("cpu",):
    ax.plot(SIZES, [float(rows[backend, size]["median_ms"]) for size in SIZES],
            marker="o", label=backend.upper())
ax.set(xscale="log", yscale="log", xlabel="Number of points",
       ylabel="Median clustering time (ms)", title="K-means runtime by dataset size")
ax.grid(alpha=0.25, which="both")
ax.legend()
fig.savefig("results/performance.png", dpi=160)
