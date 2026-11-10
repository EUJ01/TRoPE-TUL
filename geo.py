"""Trips, coordinates and grid cells: numpy/pandas/pyproj only (no torch), so local tools can use them."""
import numpy as np
import pandas as pd
from pyproj import Transformer

TRAJ_ID_COL = 'traj_id'
USER_COL = 'user_id'
X_COL = 'lng'
Y_COL = 'lat'
T_COL = 'timestamp'


def load_trips(path):
    cols = [TRAJ_ID_COL, USER_COL, X_COL, Y_COL, T_COL]
    df = pd.read_hdf(path, key='trips')[cols] if path.endswith('.h5') else pd.read_csv(path, usecols=cols)
    # The row number is also the POI embedding row, so it must be 0..N-1.
    assert (df.index == np.arange(len(df))).all(), "trips index must be 0..N-1 (it is the POI row)"
    df[T_COL] = df[T_COL].astype(np.int64)
    return df


def project(lng, lat, utm_zone, bbox, proj=None):
    """UTM metres (northern zone), or metres in `proj` (a PROJ string) for data wider than a UTM zone.
    Points outside bbox [lng_min, lat_min, lng_max, lat_max]
    are clamped onto it; rows are kept so the split and POI rows stay unchanged."""
    lng_c, lat_c = np.clip(lng, bbox[0], bbox[2]), np.clip(lat, bbox[1], bbox[3])
    n = int(((lng_c != lng) | (lat_c != lat)).sum())
    print(f"Clamped {n} of {len(lng)} points onto the dataset box")
    tr = Transformer.from_crs('EPSG:4326', proj or f'EPSG:326{utm_zone:02d}', always_xy=True)
    xy = np.stack(tr.transform(lng_c, lat_c), -1)
    assert np.isfinite(xy).all(), "non-finite projected coordinates"
    return xy


def in_bbox(lng, lat, bbox):
    """True where a point is inside bbox [lng_min, lat_min, lng_max, lat_max] (bounds inclusive), i.e. not clamped."""
    return (lng >= bbox[0]) & (lng <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])


def cell_ids(xy_m, ok, train_rows, row_traj, levels, max_rows=65536, min_traj=2):
    """Square grid cells of `levels` metres on xy_m (metres). Per level the vocabulary is the cells of in-box (ok)
    training points visited by >= min_traj distinct training trajectories, capped to the max_rows - 1 most
    visited (by training points, ties to the smaller key). Returns ids (N, len(levels)) in 1..V, 0 = UNK
    (not ok, or not in the vocabulary), and the table rows V + 1 per level."""
    assert all(a < b for a, b in zip(levels, levels[1:])), f"levels must be strictly ascending: {levels}"
    tr = np.zeros(len(xy_m), bool)
    tr[train_rows] = True
    fit = tr & ok
    m = int(row_traj.max()) + 1
    cells, rows = np.zeros((len(xy_m), len(levels)), np.int64), []
    for k, s in enumerate(levels):
        c = np.floor(xy_m / s).astype(np.int64)
        assert np.abs(c).max() < 2 ** 30, f"cell index overflow at {s} m"
        key = c[:, 0] * 2 ** 32 + c[:, 1]
        u, inv, cnt = np.unique(key[fit], return_inverse=True, return_counts=True)
        n_traj = np.bincount(np.unique(inv.ravel() * m + row_traj[fit]) // m, minlength=len(u))  # distinct trajectories
        keep = np.flatnonzero(n_traj >= min_traj)
        keep = np.sort(keep[np.argsort(-cnt[keep], kind='stable')[:max_rows - 1]])
        vocab = u[keep]
        hit = ok & np.isin(key, vocab)
        cells[hit, k] = np.searchsorted(vocab, key[hit]) + 1
        rows.append(len(vocab) + 1)
        print(f"Cells {s} m: {rows[-1]} rows (min_traj {min_traj}), known train points {(cells[tr, k] > 0).mean() * 100:.2f}%"
              f", non-train {(cells[~tr, k] > 0).mean() * 100 if (~tr).any() else 0:.2f}%")
    return cells, rows
