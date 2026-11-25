from pathlib import Path

import numpy as np

SIZES = (10_000, 50_000, 100_000, 500_000, 1_000_000)
CENTERS = np.array([[-15, -15], [-15, 15], [0, 0], [15, -15], [15, 15]])

if __name__ == "__main__":
    random = np.random.RandomState(42)
    Path("dataset").mkdir(exist_ok=True)
    for size in SIZES:
        labels = random.randint(0, len(CENTERS), size=size)
        points = CENTERS[labels] + random.normal(0, 1.5, size=(size, 2))
        path = Path("dataset") / f"input_{size}.csv"
        np.savetxt(path, points, delimiter=",", fmt="%.6f", header="x,y", comments="")
        print(f"Generated {path} with {size:,} points")
