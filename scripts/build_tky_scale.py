"""TKY-2049 trips: the full Foursquare TSMC2014 Tokyo file, split into local-day trajectories.

Exact duplicate rows are dropped (same-time rows at other venues are kept, as in the old data).
Day = calendar day at fixed UTC+9 (the loader's Asia/Tokyo; 19 rows with other offsets are treated as Tokyo).
Filters: >= 3 points per trajectory, then >= 10 trajectories per user (user decision, 2026-10-08).
Columns are those of 4square-TrajTRoPE.csv; users are re-indexed 0..C-1 in raw-id order.

Usage: python scripts/build_tky_scale.py <dataset_TSMC2014_TKY.txt> <out_csv>
"""
import os
import sys

import pandas as pd

MIN_POINTS, MIN_TRAJ = 3, 10

src, out_csv = sys.argv[1:3]
assert not os.path.exists(out_csv), f"{out_csv} exists; not overwriting"
raw = pd.read_csv(src, sep='\t', header=None, encoding='latin-1',
                  names=['user', 'venue_id', 'cat_id', 'venue_category_name', 'lat', 'lng', 'tz', 'utc'])
n_raw = len(raw)
raw = raw.drop_duplicates().reset_index(drop=True)
print(f"raw rows {n_raw}, exact duplicates dropped {n_raw - len(raw)}")

raw['t'] = pd.to_datetime(raw.utc, format='%a %b %d %H:%M:%S +0000 %Y', utc=True)
raw['day'] = (raw.t + pd.Timedelta(hours=9)).dt.date
raw = raw.sort_values(['user', 't'], kind='stable')

size = raw.groupby(['user', 'day'])['t'].transform('size')
raw = raw[size >= MIN_POINTS]
n_traj = raw.groupby('user')['day'].transform('nunique')
df = raw[n_traj >= MIN_TRAJ].copy()

df['user_id'] = df.user.rank(method='dense').astype(int) - 1
df['traj_id'] = df.groupby(['user', 'day'], sort=True).ngroup()
df['trip'] = df.traj_id - df.groupby('user_id').traj_id.transform('min')
df['seq_i'] = df.groupby('traj_id').cumcount()
df['timestamp'] = (df.t - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(seconds=1)  # unit-safe (pandas 3 uses us)
df['delta_t'] = df.timestamp - df.groupby('traj_id').timestamp.transform('first')
df['time'] = df.t.dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
cols = ['traj_id', 'user_id', 'trip', 'seq_i', 'time', 'lng', 'lat', 'timestamp', 'delta_t', 'venue_id',
        'venue_category_name']
df = df[cols].reset_index(drop=True)

per_user = df.groupby('user_id').traj_id.nunique()
assert df.notna().all().all() and per_user.min() >= MIN_TRAJ and df.groupby('traj_id').size().min() >= MIN_POINTS
assert (df.traj_id.diff().fillna(0) >= 0).all(), "rows must be grouped by trajectory"
df.to_csv(out_csv, index=False)
print(f"users {df.user_id.nunique()}, trajectories {df.traj_id.nunique()}, points {len(df)} -> {out_csv}")
