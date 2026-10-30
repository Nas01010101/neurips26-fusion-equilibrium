"""npzcmp.py A.npz B.npz: same keys, and per-key max abs difference over all arrays."""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ka, kb = set(a.files), set(b.files)
print("keys only in A:", len(ka - kb), " only in B:", len(kb - ka), " shared:", len(ka & kb))
worst, nz = 0.0, 0
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.shape != y.shape or x.dtype != y.dtype:
        print("shape/dtype differs", k, x.shape, y.shape, x.dtype, y.dtype); nz += 1; continue
    d = float(np.nanmax(np.abs(x.astype(np.float64) - y))) if x.size else 0.0
    if not np.array_equal(x, y, equal_nan=True):
        nz += 1; worst = max(worst, d)
print("arrays differing:", nz, " max abs diff:", worst)
