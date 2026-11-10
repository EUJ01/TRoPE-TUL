"""Builds the GeoLife trips CSV of settings/geolife.json: the 5 columns data.load_trips reads,
taken from the 3.6 GB POI CSV in chunks (never loaded whole).

The POI vectors stay in the released row-aligned file. De-duplicating them gains nothing: the
embedded text holds each POI's distance and bearing from the point, so about 95% of rows are unique
(checked on the first 3.7M rows, 2026-10-08).

Usage: python scripts/build_geolife.py <dataset_dir>
"""
import os
import sys

import pandas as pd

d = sys.argv[1]
src_csv = os.path.join(d, 'geolife-Beijing-POIs-v2.csv')
out_csv = os.path.join(d, 'geolife-Beijing-TrajTRoPE.csv')
assert not os.path.exists(out_csv), f"{out_csv} exists; not overwriting"

cols = ['traj_id', 'user_id', 'lng', 'lat', 'timestamp']
n = 0
for i, chunk in enumerate(pd.read_csv(src_csv, usecols=cols, chunksize=500_000)):
    chunk[cols].to_csv(out_csv, mode='a', header=i == 0, index=False)
    n += len(chunk)
print(f"trips: {n} rows -> {out_csv}")
