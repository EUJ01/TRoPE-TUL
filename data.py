import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from geo import load_trips, project, TRAJ_ID_COL, USER_COL, X_COL, Y_COL, T_COL  # noqa: F401 (re-exported)

# Per-point input features: x, y (scaled UTM), day-of-week, hour, minute (local time),
# minutes since the trajectory's first point.
N_FEATS = 6


def make_split(traj_ids, users, seed):
    """Stratified 80/10/10 split; draws the same ids as the released utils.stratify_dataset."""
    idx = np.arange(len(traj_ids))
    train, val_test = train_test_split(idx, test_size=0.2, stratify=users, random_state=seed)
    # F14: users with one trajectory in val+test (GeoLife users 42, 64) crash the second split;
    # they share one stratum. No such users on Foursquare/Gowalla, so their split is unchanged.
    strata = users[val_test].copy()
    vals, counts = np.unique(strata, return_counts=True)
    rare = np.isin(strata, vals[counts < 2])
    strata[rare] = -1
    if rare.sum() == 1:
        strata[rare] = vals[counts.argmax()]
    val, test = train_test_split(val_test, test_size=0.5, stratify=strata, random_state=seed)
    return {k: traj_ids[v].tolist() for k, v in (('train', train), ('val', val), ('test', test))}


def time_features(ts, tz):
    """Day-of-week, hour and minute (fractional) in local time, from int64 Unix seconds."""
    t = pd.to_datetime(ts, unit='s', utc=True).tz_convert(tz)
    minute = np.asarray(t.minute + t.second / 60)
    hour = np.asarray(t.hour) + minute / 60
    dow = np.asarray(t.dayofweek) + hour / 24
    return np.stack([dow, hour, minute], -1)


class TrajDataset(Dataset):
    """One item per trajectory: (features (L, N_FEATS) float32, POI table rows (L,) or (L, 1 + levels) int64, user id)."""

    def __init__(self, point_feats, ts, users, rows_by_traj, traj_ids, poi_index):
        self.items = []
        for t in traj_ids:
            r = rows_by_traj[t]
            r = r[np.argsort(ts[r], kind='stable')]
            dt = ((ts[r] - ts[r[0]]) / 60).astype(np.float32)[:, None]
            feats = np.concatenate([point_feats[r], dt], 1).astype(np.float32)
            self.items.append((feats, poi_index[r].astype(np.int64), int(users[r[0]])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    """Pads to the longest trajectory. Returns feats, POI rows, pad mask (True = padding), users."""
    B, L = len(batch), max(len(f) for f, _, _ in batch)
    feats = np.zeros((B, L, N_FEATS), np.float32)
    poi = np.zeros((B, L, *batch[0][1].shape[1:]), np.int64)  # (B, L) or (B, L, 1 + levels); pad 0
    pad = np.ones((B, L), bool)
    for i, (f, p, _) in enumerate(batch):
        feats[i, :len(f)], poi[i, :len(f)], pad[i, :len(f)] = f, p, False
    users = np.array([u for _, _, u in batch], np.int64)
    return torch.from_numpy(feats), torch.from_numpy(poi), torch.from_numpy(pad), torch.from_numpy(users)


def cap_rows(rows_by_traj, ts, max_len):
    """C5: each trajectory's first max_len rows in time order (stable, as TrajDataset sorts)."""
    return {t: r[np.argsort(ts[r], kind='stable')][:max_len] for t, r in rows_by_traj.items()}


class MPerUserSampler:
    """C6 batch sampler: each user's trajectories are shuffled and cut into chunks of m (a user's last chunk may be
    smaller); the chunks are shuffled and every batch takes batch_size // m of them. No replacement, an epoch is
    exactly N items, at most batch_size // m users per batch; draws only from `generator`.
    ponytail: short chunks make some batches smaller than batch_size (and a few more batches per epoch)."""

    def __init__(self, users, m, batch_size, generator):
        assert batch_size % m == 0, f"m {m} must divide the batch size {batch_size}"
        users = np.asarray(users)
        order = np.argsort(users, kind='stable')
        self.by_user = np.split(order, np.cumsum(np.bincount(users))[:-1])
        self.m, self.k, self.g = m, batch_size // m, generator

    def __iter__(self):
        chunks = []
        for idx in self.by_user:
            idx = idx[torch.randperm(len(idx), generator=self.g).numpy()]
            chunks += [idx[i:i + self.m] for i in range(0, len(idx), self.m)]
        order = torch.randperm(len(chunks), generator=self.g).tolist()
        for s in range(0, len(order), self.k):
            yield np.concatenate([chunks[c] for c in order[s:s + self.k]]).tolist()

    def __len__(self):
        n_chunks = sum(-(-len(i) // self.m) for i in self.by_user)
        return -(-n_chunks // self.k)
