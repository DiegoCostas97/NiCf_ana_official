#!/usr/bin/env python3
"""
Minimal end-to-end check of the vendored multilaterator.

Generates a handful of synthetic clusters from a known vertex, reconstructs
them, and reports the bias. Useful after moving the code to a new machine, and
as a demonstration that the reconstructed coordinates are in WCSim centimetres.

    python external/multilat_smoke_test.py --geo-file data/geofile_...txt
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import functions_bonsai, functions_multilateration

p = argparse.ArgumentParser()
p.add_argument("--geo-file", default=None)
p.add_argument("--n", type=int, default=20)
a = p.parse_args()

geo_df = functions_bonsai.get_geo_mapping(a.geo_file)
functions_bonsai.geo = functions_bonsai.build_lookup_table(geo_df)

rng = np.random.default_rng(0)
truth = np.array([0.0, 80.0, 0.0])          # WCSim cm, near the top endcap
c = 29.9792458 / 1.33                        # cm/ns

res = []
for _ in range(a.n):
    rows = geo_df.sample(12, random_state=int(rng.integers(1e6)))
    xyz = rows[["x", "y", "z"]].values
    t = np.linalg.norm(xyz - truth, axis=1) / c + rng.normal(0, 1.0, len(xyz))
    out = functions_multilateration.run_multilateration_candidate(
        t, rows.mpmtid.values, rows.spmtid.values - 1,
        sigma_t=1.0, early_window_ns=100.0, robust_loss="soft_l1")
    if out["success"]:
        res.append([out["x"], out["y"], out["z"], out["chi2_ndof"]])

res = np.array(res)
print(f"reconstructed {len(res)}/{a.n} clusters")
print("truth          [cm]:", truth)
print("mean vertex    [cm]:", np.round(res[:, :3].mean(axis=0), 2))
print("rms            [cm]:", np.round(res[:, :3].std(axis=0), 2))
print("median chi2/ndf    :", round(float(np.median(res[:, 3])), 3))
print("\nIf the mean is close to the truth in CENTIMETRES, the frame assumption")
print("used by run_angular_vertex.py (--vertex-frame wcsim-cm) is the right one.")
