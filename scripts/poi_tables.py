"""E6 (POI encoder cost vs benefit, 2026-10-27): POI tables beside the Qwen3 ones, and the counts the cost tables need.

trunc    <in.npy> <out.npy> <d>                          first d dims, L2 re-normalised (Matryoshka), input dtype kept; chunked mmap
cats     <pois_csv> <trips_csv> <split_json> <prefix>    counts of each POI type among a point's named POIs / 10; type vocabulary
                                                         = types covering 99 % of TRAIN slots + 'other' -> <prefix>-cats.npy + -cats-index.npy
w2v      <pois_csv> <trips_csv> <split_json> <prefix>    word2vec (sg, 256-d) over POI names + types, fitted on the unique TRAIN token
                                                         sets; point vector = mean of known tokens -> <prefix>-w2v256.npy + -w2v256-index.npy
scan     <pois_csv> <trips_csv> <split_json> <out_json>  one pass: unique texts (format_row_pois_as_json, hashed), cache-hit rate of
                                                         val / test texts, POI catalogue size, 2,000 sample texts with their rows
catalogue <pois_csv> <out_json>                          unique POIs (name, type, lat, lng) and their name + type texts (embed-once cost)
cachehit <index.npy> <trips_csv> <split_json>            share of val / test rows whose text index also occurs in train rows
--self-check                                             synthetic data; asserts the rules above

Every POI CSV is read in chunks with explicit dtypes (a plain chunked read of the GeoLife CSV crashes inside pandas) and is checked
row by row against the trips CSV (traj_id, lat, lng). POIs count only when they have a name, as in format_row_pois_as_json.
Outputs are never overwritten. Each command prints wall time and peak RSS. Run w2v with PYTHONHASHSEED=0 (gensim hashes with it).
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

K, CHUNK = 10, 200_000
NAME = [f'POI_{i}_name' for i in range(1, K + 1)]
TYPE = [f'POI_{i}_type' for i in range(1, K + 1)]
NUM = [f'POI_{i}_{c}' for i in range(1, K + 1) for c in ('lat', 'lng', 'dist(m)', 'dir(deg)')]
T0 = time.time()


def peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except ImportError:  # Windows
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e9
        except ImportError:
            return float('nan')


def done(msg):
    print(f"{msg}; wall {time.time() - T0:.1f} s, peak RSS {peak_rss_gb():.2f} GB", flush=True)


def fresh(*paths):
    for p in paths:
        assert not os.path.exists(p), f"{p} exists; not overwriting"


def read_pois(pois_csv, cols):
    dtype = {c: str for c in NAME + TYPE}
    dtype.update({c: 'float64' for c in NUM + ['lat', 'lng']})
    dtype['traj_id'] = 'int64'
    cols = ['traj_id', 'lat', 'lng'] + [c for c in cols if c not in ('traj_id', 'lat', 'lng')]
    return pd.read_csv(pois_csv, usecols=cols, dtype={c: dtype[c] for c in cols}, chunksize=CHUNK)


def aligned_chunks(pois_csv, trips_csv, cols):
    """POI-CSV chunks, each asserted equal to the trips CSV rows (traj_id, lat, lng)."""
    trips = pd.read_csv(trips_csv, usecols=['traj_id', 'lat', 'lng'], dtype={'traj_id': 'int64'}, chunksize=CHUNK)
    for p in read_pois(pois_csv, cols):
        t = next(trips)
        assert len(t) == len(p) and (t.traj_id.to_numpy() == p.traj_id.to_numpy()).all() and \
            np.allclose(t[['lat', 'lng']].to_numpy(), p[['lat', 'lng']].to_numpy(), rtol=0, atol=1e-9), \
            f"{pois_csv} rows do not match {trips_csv}"
        yield p
    assert next(trips, None) is None, f"{trips_csv} has more rows than {pois_csv}"


def split_of(split_json):
    s = json.load(open(split_json))
    return {t: k for k in ('train', 'val', 'test') for t in s[k]}


def named(p):
    """(rows, K) bool: slot i holds a named POI (the POIs format_row_pois_as_json keeps)."""
    n = p[NAME].to_numpy(dtype=object)
    return np.array([[isinstance(x, str) and x.strip() != '' for x in r] for r in n]) if len(n) else np.zeros((0, K), bool)


def trunc(src, out, d):
    d = int(d)
    fresh(out)
    a = np.load(src, mmap_mode='r')
    assert d < a.shape[1], f"{src} is {a.shape[1]}-d already"
    o = np.lib.format.open_memmap(out, mode='w+', dtype=a.dtype, shape=(a.shape[0], d))
    for s in range(0, len(a), CHUNK):
        x = np.asarray(a[s:s + CHUNK, :d], dtype=np.float32)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        o[s:s + CHUNK] = (x / np.where(n > 0, n, 1)).astype(a.dtype)
    o.flush()
    done(f"trunc {src} {a.shape} -> {out} {o.shape} {o.dtype}")


def cats(pois_csv, trips_csv, split_json, prefix):
    tab, idx, meta = f'{prefix}-cats.npy', f'{prefix}-cats-index.npy', f'{prefix}-cats.json'
    fresh(tab, idx, meta)
    part = split_of(split_json)
    counts = pd.Series(dtype='int64')
    for p in aligned_chunks(pois_csv, trips_csv, NAME + TYPE):  # pass 1: vocabulary from train rows only
        tr = p.traj_id.map(part).to_numpy() == 'train'
        m = named(p)[tr]
        t = p[TYPE].to_numpy(dtype=object)[tr][m]
        counts = counts.add(pd.Series(t).fillna('').value_counts(), fill_value=0)
    counts = counts.sort_values(ascending=False)
    vocab = list(counts.index[:int(((counts.cumsum() / counts.sum()) < 0.99).sum()) + 1])
    pos = {t: i for i, t in enumerate(vocab)}
    other = len(vocab)
    rows = []
    for p in aligned_chunks(pois_csv, trips_csv, NAME + TYPE):  # pass 2: counts per row (uint8, 0..10)
        m, t = named(p), p[TYPE].to_numpy(dtype=object)
        c = np.zeros((len(p), other + 1), np.uint8)
        for j in range(K):
            col = np.array([pos.get(x if isinstance(x, str) else '', other) for x in t[:, j]])
            np.add.at(c, (np.nonzero(m[:, j])[0], col[m[:, j]]), 1)
        rows.append(c)
    rows = np.concatenate(rows)
    uniq, inv = np.unique(rows, axis=0, return_inverse=True)
    np.save(tab, (uniq / K).astype(np.float32))
    np.save(idx, inv.reshape(-1).astype(np.int64))
    json.dump({'vocab': vocab, 'other_slot': other, 'train_slots': int(counts.sum()), 'types_in_train': len(counts),
               'rows': len(rows), 'unique_rows': len(uniq)}, open(meta, 'w', encoding='utf-8'), ensure_ascii=False)
    done(f"cats {pois_csv}: {len(counts)} train types, {len(vocab)} kept (99 %) + other -> {tab} {uniq.shape}, {len(rows)} rows")


def tokens(p):
    m, n, t = named(p), p[NAME].to_numpy(dtype=object), p[TYPE].to_numpy(dtype=object)
    return [tuple([n[r, j].strip() for j in range(K) if m[r, j]] +
                  ['T:' + str(t[r, j]).strip() for j in range(K) if m[r, j] and isinstance(t[r, j], str)]) for r in range(len(p))]


def w2v(pois_csv, trips_csv, split_json, prefix, dim=256):
    assert os.environ.get('PYTHONHASHSEED') == '0', "run with PYTHONHASHSEED=0 (gensim's token hashing)"
    from gensim.models import Word2Vec
    tab, idx, meta = f'{prefix}-w2v{dim}.npy', f'{prefix}-w2v{dim}-index.npy', f'{prefix}-w2v{dim}.json'
    fresh(tab, idx, meta)
    part = split_of(split_json)
    sets, index, train_sets = {}, [], set()
    for p in aligned_chunks(pois_csv, trips_csv, NAME + TYPE):
        tr = p.traj_id.map(part).to_numpy() == 'train'
        for r, tok in enumerate(tokens(p)):
            k = sets.setdefault(tok, len(sets))
            index.append(k)
            if tr[r]:
                train_sets.add(k)
    by_id = list(sets)
    sentences = [list(by_id[k]) for k in sorted(train_sets) if by_id[k]]
    t1 = time.time()
    model = Word2Vec(sentences, vector_size=dim, window=10, min_count=1, sg=1, epochs=5, workers=1, seed=42)
    fit = time.time() - t1
    table = np.zeros((len(by_id), dim), np.float32)
    for k, tok in enumerate(by_id):
        v = [model.wv[x] for x in tok if x in model.wv.key_to_index]
        if v:
            table[k] = np.mean(v, 0)
    np.save(tab, table)
    np.save(idx, np.asarray(index, np.int64))
    json.dump({'train_sentences': len(sentences), 'vocab': len(model.wv), 'rows': len(index), 'unique_sets': len(by_id),
               'fit_seconds': round(fit, 1), 'zero_rows': int((table == 0).all(1).sum())}, open(meta, 'w'))
    done(f"w2v {pois_csv}: {len(sentences)} unique train token sets, vocab {len(model.wv)}, fit {fit:.1f} s -> {tab} {table.shape}")


def scan(pois_csv, trips_csv, split_json, out_json, n_sample=2000):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attach_pois import format_row_pois_as_json
    stem = os.path.splitext(out_json)[0]
    fresh(out_json, stem + '-hashes.npy', stem + '-sample.json')
    part = split_of(split_json)
    hashes, kind, poi_h, cat_h, start = [], [], [], [], 0
    rng = np.random.default_rng(42)
    sample = {}
    for p in aligned_chunks(pois_csv, trips_csv, NAME + TYPE + NUM):
        txt = p.apply(format_row_pois_as_json, axis=1)
        hashes.append(pd.util.hash_pandas_object(txt, index=False).to_numpy())
        kind.append(p.traj_id.map(part).to_numpy())
        m = named(p)
        for j in range(K):  # catalogue: unique (name, type, lat, lng) and unique (name, type) texts
            q = p.loc[m[:, j], [NAME[j], TYPE[j], f'POI_{j + 1}_lat', f'POI_{j + 1}_lng']]
            q.columns = ['n', 't', 'a', 'o']
            poi_h.append(np.unique(pd.util.hash_pandas_object(q, index=False).to_numpy()))
            cat_h.append(np.unique(pd.util.hash_pandas_object(q[['n', 't']], index=False).to_numpy()))
        pick = rng.random(len(p)) < n_sample * 1.2 / 5_941_128  # about n_sample rows over the whole file
        for r in np.nonzero(pick)[0]:
            sample[int(start + r)] = txt.iloc[r]
        start += len(p)
        poi_h, cat_h = [np.unique(np.concatenate(poi_h))], [np.unique(np.concatenate(cat_h))]
    h, kd = np.concatenate(hashes), np.concatenate(kind)
    train = np.unique(h[kd == 'train'])
    hit = {k: float(np.isin(h[kd == k], train).mean()) for k in ('val', 'test')}
    np.save(stem + '-hashes.npy', h)
    keys = sorted(sample)[:n_sample]
    json.dump({'rows': keys, 'texts': [sample[k] for k in keys]}, open(stem + '-sample.json', 'w', encoding='utf-8'), ensure_ascii=False)
    res = {'rows': int(len(h)), 'unique_texts': int(len(np.unique(h))), 'cache_hit_val': hit['val'], 'cache_hit_test': hit['test'],
           'catalogue_pois': int(len(poi_h[0])), 'catalogue_texts': int(len(cat_h[0])), 'sample': len(keys)}
    json.dump(res, open(out_json, 'w'))
    done(f"scan {pois_csv}: {res}")


def catalogue(pois_csv, out_json):
    fresh(out_json)
    pois, texts = set(), {}
    for p in read_pois(pois_csv, NAME + TYPE + NUM):
        m = named(p)
        for j in range(K):
            q = p.loc[m[:, j], [NAME[j], TYPE[j], f'POI_{j + 1}_lat', f'POI_{j + 1}_lng']].itertuples(index=False)
            for n, t, a, o in q:
                pois.add((n.strip(), t if isinstance(t, str) else '', a, o))
                texts.setdefault((n.strip(), t if isinstance(t, str) else ''), None)
    tx = [json.dumps({'name': n, 'type': t}, ensure_ascii=False) for n, t in texts]
    json.dump({'pois': len(pois), 'texts': tx}, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False)
    done(f"catalogue {pois_csv}: {len(pois)} unique POIs, {len(tx)} unique name + type texts -> {out_json}")


def cachehit(index_npy, trips_csv, split_json):
    idx = np.load(index_npy)
    part = split_of(split_json)
    kd = pd.read_csv(trips_csv, usecols=['traj_id'], dtype={'traj_id': 'int64'}).traj_id.map(part).to_numpy()
    assert len(kd) == len(idx), "index does not match the trips"
    train = np.unique(idx[kd == 'train'])
    res = {k: round(float(np.isin(idx[kd == k], train).mean()), 4) for k in ('val', 'test')}
    done(f"cachehit {index_npy}: {res}")
    return res


def self_check():
    """Synthetic trips / POIs / split: vocabulary from train only, dedup round trip, alignment check, trunc norms."""
    os.environ.setdefault('PYTHONHASHSEED', '0')
    d = tempfile.mkdtemp(prefix='poi_tables_check_')
    trips = pd.DataFrame({'traj_id': [0, 0, 1, 1, 2, 2], 'lat': [1., 1.1, 2., 2., 3., 3.1], 'lng': [5., 5.1, 6., 6., 7., 7.1]})
    trips.to_csv(f'{d}/t.csv', index=False)
    p = trips.copy()
    for c in NAME + TYPE:
        p[c] = pd.Series([None] * len(p), dtype=object)
    for c in NUM:
        p[c] = np.nan
    p.loc[:, 'POI_1_name'] = ['a', 'a', 'b', 'b', 'only_val', '']  # row 5: empty name -> no POI
    p.loc[:, 'POI_1_type'] = ['cafe', 'cafe', 'shop', 'shop', 'val_type', 'shop']
    p.loc[:, 'POI_2_name'] = ['c', None, None, None, None, None]
    p.loc[:, 'POI_2_type'] = ['shop', None, None, None, None, None]
    p.to_csv(f'{d}/p.csv', index=False)
    json.dump({'train': [0, 1], 'val': [2], 'test': []}, open(f'{d}/s.json', 'w'))
    cats(f'{d}/p.csv', f'{d}/t.csv', f'{d}/s.json', f'{d}/x')
    meta, tab, idx = json.load(open(f'{d}/x-cats.json')), np.load(f'{d}/x-cats.npy'), np.load(f'{d}/x-cats-index.npy')
    assert 'val_type' not in meta['vocab'], "val-only type entered the vocabulary"
    assert len(idx) == 6 and idx[2] == idx[3] and idx[0] != idx[1], "cats dedup"
    assert np.isclose(tab[idx[4]].sum(), 0.1) and tab[idx[4], meta['other_slot']] == 0.1, "val type -> other slot"
    assert tab[idx[5]].sum() == 0, "an unnamed POI must not count"
    w2v(f'{d}/p.csv', f'{d}/t.csv', f'{d}/s.json', f'{d}/x')
    wt, wi = np.load(f'{d}/x-w2v256.npy'), np.load(f'{d}/x-w2v256-index.npy')
    assert wi[0] != wi[1] and wi[2] == wi[3], "w2v dedup"
    assert (wt[wi[4]] == 0).all(), "val-only tokens entered the word2vec vocabulary"
    assert (wt[wi[5]] == 0).all() and not (wt[wi[0]] == 0).all(), "row without POIs -> zero; train row -> non-zero"
    trips.iloc[[0, 2, 1, 3, 4, 5]].to_csv(f'{d}/t2.csv', index=False)  # rows of two trajectories swapped
    try:
        list(aligned_chunks(f'{d}/p.csv', f'{d}/t2.csv', NAME))
        raise SystemExit("alignment check did not fire")
    except AssertionError:
        pass
    np.save(f'{d}/e.npy', np.random.default_rng(0).standard_normal((50, 16)).astype(np.float16))
    trunc(f'{d}/e.npy', f'{d}/e8.npy', 8)
    e8 = np.load(f'{d}/e8.npy').astype(np.float32)
    assert e8.shape == (50, 8) and np.allclose(np.linalg.norm(e8, axis=1), 1, atol=2e-3), "trunc norms"
    print(f"self-check passed ({d})")


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        cmd, *a = sys.argv[1:]
        {'trunc': trunc, 'cats': cats, 'w2v': w2v, 'scan': scan, 'catalogue': catalogue, 'cachehit': cachehit}[cmd](*a)
