"""Nested user subsets of the scale-up dataset (500 within 1000 within all), for the scaling curve.

Users are permuted with seed 42; each subset takes the first n. Per subset: a trips CSV with users re-indexed
0..C-1 in original order, the matching POI index slice, and the full split restricted to its trajectories
(so each subset's val/test lie inside the full split's val/test).

Usage: python scripts/make_tky_subsets.py <dataset_dir> 500 1000
Reads tky-scale-TrajTRoPE.csv, pois/gemma/tky-scale-osm-poi-index.npy, splits/tky_scale_seed42.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

d, sizes = sys.argv[1], [int(s) for s in sys.argv[2:]]
df = pd.read_csv(os.path.join(d, 'tky-scale-TrajTRoPE.csv'))
poi_index = np.load(os.path.join(d, 'pois/gemma/tky-scale-osm-poi-index.npy'))
with open(os.path.join(d, 'splits/tky_scale_seed42.json')) as fp:
    split = json.load(fp)
assert len(poi_index) == len(df)

order = np.random.default_rng(42).permutation(df.user_id.nunique())
prev = set()
for n in sorted(sizes):
    users = set(order[:n].tolist())
    assert prev <= users
    prev = users
    m = df.user_id.isin(users).to_numpy()
    sub = df[m].reset_index(drop=True)
    sub['user_id'] = sub.user_id.rank(method='dense').astype(int) - 1
    trajs = set(sub.traj_id)
    outs = (f'tky-scale-u{n}-TrajTRoPE.csv', f'pois/gemma/tky-scale-u{n}-osm-poi-index.npy', f'splits/tky_scale_u{n}_seed42.json')
    assert not any(os.path.exists(os.path.join(d, o)) for o in outs), f"subset {n} exists; not overwriting"
    sub.to_csv(os.path.join(d, outs[0]), index=False)
    np.save(os.path.join(d, outs[1]), poi_index[m])
    with open(os.path.join(d, outs[2]), 'w') as fp:
        json.dump({k: [t for t in v if t in trajs] for k, v in split.items()}, fp)
    print(f"u{n}: users {sub.user_id.nunique()}, trajectories {len(trajs)}, points {len(sub)}")
