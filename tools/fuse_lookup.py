"""P1: fuse a run's saved logits with a train-only cell x user lookup (numpy/scipy/pandas, no torch).
  python tools/fuse_lookup.py <run>/settings.json <run>_preds.npz [--data_dir dataset] [--final]
  python tools/fuse_lookup.py check
score = log_softmax(logits) + w * L, L = per-trajectory log-normalised lookup (vote or NB) at one cell size or the sum
over sizes; w, method and size chosen on val ACC@1 (ties -> smaller w). Test metrics only with --final."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import log_softmax, logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geo import cell_ids, in_bbox, load_trips, project, TRAJ_ID_COL, USER_COL, X_COL, Y_COL  # noqa: E402

LEVELS = [10, 50, 200, 1000]
SCALES = LEVELS + ['all']
ALPHA = 0.5  # NB smoothing, as logs/diag/memo_stats.py
EPS = 1e-6  # vote floor per user, relative to the trajectory's total vote (its known points x levels)
WS = np.concatenate([[0.0], np.logspace(-2, 3, 41)])  # same 8 steps per decade as before; to 1000 so w is not clipped (base chose 10, the old top)


def metrics(S, y):
    """ACC@1, ACC@5, macro-F1 over users 0..C-1 with zero_division 0 (as pipeline.compute_metrics)."""
    n = S.shape[1]
    top5 = np.argsort(-S, 1, kind='stable')[:, :5]  # position 0 = argmax
    pr = top5[:, 0]
    tp = np.bincount(y[pr == y], minlength=n)
    d = np.bincount(pr, minlength=n) + np.bincount(y, minlength=n)
    pp, ap = np.bincount(pr, minlength=n), np.bincount(y, minlength=n)  # 2026-10-27: P / R for the phase 3 table (--five)
    return {'ACC@1': (pr == y).mean(), 'ACC@5': (top5 == y[:, None]).any(1).mean(),
            'Macro-P': np.where(pp > 0, tp / np.maximum(pp, 1), 0).mean(), 'Macro-R': np.where(ap > 0, tp / np.maximum(ap, 1), 0).mean(),
            'Macro-F1': np.where(d > 0, 2 * tp / np.maximum(d, 1), 0).mean()}


def tables(cells, rows, user, train_rows, n_users):
    """Per level: P(user | cell) (vote), NB log-count table and NB normaliser, from training points with a known cell."""
    out = []
    for k in range(cells.shape[1]):
        r = train_rows[cells[train_rows, k] > 0]
        C = sparse.csr_matrix((np.ones(len(r)), (cells[r, k], user[r])), shape=(rows[k], n_users))
        D = C.copy()
        D.data = np.log1p(D.data / ALPHA)  # multinomial NB, sparse form
        out.append((sparse.diags(1 / np.maximum(C.sum(1).A1, 1)) @ C, D, np.log(C.sum(0).A1 + ALPHA * (rows[k] - 1))))
    return out


def lookup(tabs, cells, T, scale, log_prior):
    """{'vote', 'nb'}: L (len(T), C) for trajectories T (row arrays); unknown-cell points add nothing, rows with
    no known cell are 0."""
    tid, r = np.repeat(np.arange(len(T)), [len(t) for t in T]), np.concatenate(T)
    v = ll = n = 0
    for k in range(len(tabs)) if scale == 'all' else [LEVELS.index(scale)]:
        P, D, norm = tabs[k]
        m = cells[r, k] > 0
        V = sparse.csr_matrix((np.ones(m.sum()), (tid[m], cells[r[m], k])), shape=(len(T), D.shape[0]))
        nk = V.sum(1).A1[:, None]
        v, ll, n = v + (V @ P).toarray(), ll + (V @ D).toarray() - nk * norm, n + nk
    known = n > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        e = v + EPS * v.sum(1, keepdims=True)
        Lv = np.log(e / e.sum(1, keepdims=True))
    nb = ll + log_prior
    return {'vote': np.where(known, Lv, 0), 'nb': np.where(known, nb - logsumexp(nb, 1, keepdims=True), 0)}


def select(ls, L, y, name, best):
    """Best (ACC@1, w, name) so far; ties -> smaller w (w = 0 is the start, so it wins every tie)."""
    for w in WS[1:]:
        a = (np.argmax(ls + w * L, 1) == y).mean()
        if a > best[0] or (a == best[0] and w < best[1]):
            best = (a, w, name)
    return best


def main(a):
    run = json.load(open(a.settings))
    ds = run['settings']['dataset']
    path = {k: os.path.join(a.data_dir, ds[k]) for k in ('trips', 'split')}
    for p in path.values():
        assert os.path.exists(p), f"missing {p} (fusion needs the run's own frozen split and trips)"
    pr = np.load(a.preds)
    split = json.load(open(path['split']))
    for k in ('val', 'test'):
        assert np.array_equal(pr[f'{k}_traj'], np.array(split[k])), f"{k}_traj in the npz != split['{k}']"

    df = load_trips(path['trips'])
    rows_by_traj = df.groupby(TRAJ_ID_COL, sort=False).indices
    user = df[USER_COL].to_numpy()
    n_users = pr['val_logits'].shape[1]
    assert set(np.unique(user)) == set(range(n_users)), "user ids must be 0..C-1"
    lng, lat = df[X_COL].to_numpy(), df[Y_COL].to_numpy()
    xy = project(lng, lat, ds['utm_zone'], ds['bbox'], ds.get('proj'))
    train_rows = np.concatenate([rows_by_traj[t] for t in split['train']])
    xy_m = xy - np.median(xy[train_rows], 0)  # metres, as main.py before / scale
    # ponytail: lookup vocabulary = every training cell (min_traj 1, no cap); the model's cell token uses min_traj/max_rows
    cells, rows = cell_ids(xy_m, in_bbox(lng, lat, ds['bbox']), train_rows, pd.factorize(df[TRAJ_ID_COL])[0],
                           LEVELS, max_rows=2 ** 62, min_traj=1)
    tabs = tables(cells, rows, user, train_rows, n_users)
    prior = np.bincount(user[[rows_by_traj[t][0] for t in split['train']]], minlength=n_users).astype(float)
    log_prior, tie = np.log(prior + 1), 1e-9 * prior / prior.sum()  # tie: no known cell -> most frequent user

    def inputs(k):
        T = [rows_by_traj[t] for t in split[k]]
        y = pr[f'{k}_labels'].astype(np.int64)
        assert np.array_equal(y, user[[t[0] for t in T]]), f"{k}_labels do not match the trips"
        return T, y, log_softmax(pr[f'{k}_logits'].astype(np.float64), 1)

    T, y, ls = inputs('val')
    assert np.array_equal(ls.argmax(1), pr['val_logits'].argmax(1))
    best, alone = ((ls.argmax(1) == y).mean(), 0.0, None), {}
    for s in SCALES:
        for meth, L in lookup(tabs, cells, T, s, log_prior).items():
            acc = ((L + tie).argmax(1) == y).mean()
            if acc > alone.get(meth, (-1,))[0]:
                alone[meth] = (acc, s)
            best = select(ls, L, y, (meth, s), best)
    assert best[2] is not None or best[0] == (pr['val_logits'].argmax(1) == y).mean()

    cols = ('ACC@1', 'ACC@5', 'Macro-P', 'Macro-R', 'Macro-F1') if a.five else ('ACC@1', 'ACC@5', 'Macro-F1')
    print(f"\n{'split':5s} {'row':34s} " + ' '.join(f'{c.replace("Macro-", ""):>6s}' for c in cols))
    for k in ('val', 'test') if a.final else ('val',):
        T, y, ls = inputs(k)
        out = [('model (w = 0)', ls)]
        out += [(f'lookup {m} @ {s}', lookup(tabs, cells, T, s, log_prior)[m] + tie) for m, (_, s) in alone.items()]
        if best[2] is not None:
            (m, s), w = best[2], best[1]
            out.append((f'fused {m} @ {s}, w = {w:.4g}', ls + w * lookup(tabs, cells, T, s, log_prior)[m]))
        else:
            out.append(('fused: w = 0 chosen', ls))
        for name, S in out:
            mt = metrics(S, y)
            print(f"{k:5s} {name:34s} " + ' '.join(f"{mt[c] * 100:6.2f}" for c in cols))


def check():
    # macro-F1: check_v2.check_metrics example; classes 0, 1, 2 F1 2/3, 1, 1; classes 3, 4, 5 count as 0 -> 4/9
    logits = np.array([[3, 1, 0, 0, 0, 0], [0, 3, 1, 0, 0, 0], [0, 0, 3, 1, 0, 0], [3, 0, 0, 0, 0, 1]], float)
    m = metrics(logits, np.array([0, 1, 2, 5]))
    assert m['ACC@1'] == 0.75 and m['ACC@5'] == 1.0 and abs(m['Macro-F1'] - 4 / 9) < 1e-12, m
    print('ok macro-F1 = 4/9 (sklearn f1_score(average=macro, labels=arange(6), zero_division=0))')

    # cell_ids at 10 m. Train: traj 0 has 4 points in cell A (rows 0-2, 6) and row 7 in A outside the box; traj 1 one
    # point in A and one in B; traj 2 one in C. Val traj 3: points in B, C and an unseen cell D. At 20 m A and B merge.
    xy = np.array([[5, 5], [6, 5], [7, 5], [5, 6], [15, 5], [25, 5], [8, 8], [5, 5], [15, 5], [25, 5], [95, 95]], float)
    row_traj = np.array([0, 0, 0, 1, 1, 2, 0, 0, 3, 3, 3])
    ok = np.ones(len(xy), bool)
    ok[7] = False
    tr = np.arange(8)
    c2, r2 = cell_ids(xy, ok, tr, row_traj, [10, 20], min_traj=2)
    assert np.array_equal(c2, cell_ids(xy, ok, tr, row_traj, [10, 20], min_traj=2)[0]), 'not deterministic'
    assert r2[0] == 2 and (c2[[0, 1, 2, 3, 6], 0] == 1).all(), c2  # only A has 2 trajectories at 10 m
    assert c2[7].sum() == 0 and c2[10].sum() == 0, 'not-ok or unseen cell must be 0'
    assert c2[4, 0] == 0 and c2[5, 0] == 0, 'cells of one trajectory must be UNK at min_traj 2'
    assert c2[4, 1] > 0 and c2[5, 1] == 0, '20 m: A+B (2 trajectories) known, C not'
    c1, r1 = cell_ids(xy, ok, tr, row_traj, [10], min_traj=1)
    assert r1 == [4] and (c1[[4, 5, 8, 9], 0] > 0).all() and c1[10, 0] == 0 and c1[7, 0] == 0
    cc, rc = cell_ids(xy, ok, tr, row_traj, [10], max_rows=2, min_traj=1)
    assert rc == [2] and (cc[[0, 1, 2, 3, 6], 0] == 1).all() and cc[[4, 5], 0].sum() == 0, 'cap keeps the most frequent'
    for lv in ([20, 10], [10, 10]):
        try:
            cell_ids(xy, ok, tr, row_traj, lv)
        except AssertionError:
            continue
        raise AssertionError(f'levels {lv} must raise')
    print('ok cell_ids')

    # lookup: an all-UNK trajectory gives L = 0; a known one is log-normalised
    users = np.array([0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 1])
    tabs = tables(c1, r1, users, tr, 3)
    for s in (10, 'all'):
        L = lookup(tabs, c1, [np.array([10]), np.array([7, 10]), np.array([8])], s, np.zeros(3))
        for meth in L:
            assert (L[meth][:2] == 0).all(), (meth, L[meth])
            assert np.allclose(np.exp(L[meth][2]).sum(), 1) and L[meth][2].argmax() == 1, (meth, L[meth])
    print('ok lookup: all-UNK row -> L = 0')

    # w = 0 returns argmax of logits: a lookup that is always wrong is never chosen
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((50, 4))
    y = lg.argmax(1)
    ls = log_softmax(lg, 1)
    best = select(ls, np.log(np.eye(4)[(y + 1) % 4] + 1e-6), y, 'bad', ((ls.argmax(1) == y).mean(), 0.0, None))
    assert best == (1.0, 0.0, None) and np.array_equal(ls.argmax(1), lg.argmax(1))
    print('ok w = 0 -> argmax of logits')


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        check()
    else:
        p = argparse.ArgumentParser()
        p.add_argument('settings', help="the run's settings.json ({'settings': config, 'args': ...})")
        p.add_argument('preds', help="the run's <run>_preds.npz")
        p.add_argument('--data_dir', default='dataset', help='dataset paths in the config are relative to this')
        p.add_argument('--final', action='store_true', help='also print test (final configs only, once)')
        p.add_argument('--five', action='store_true', help='print ACC@1, ACC@5, Macro-P, Macro-R, Macro-F1 (default: ACC@1, ACC@5, F1)')
        main(p.parse_args())
