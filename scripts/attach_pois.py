"""POI context for each point, in the format of Foursquare-POIs.csv, and the deduplicated POI texts.

attach: the 10 nearest POIs within 250 m, or within 500 m if fewer than 3 lie within 250 m (the rule of the
        old Geoapify data); distance = WGS84 geodesic, dir = spherical initial bearing (both reproduce the old
        values; see --check). Selection runs on unique coordinates in UTM zone 54.
texts:  format_row_pois_as_json (verbatim from code/poi/POI_embedding_qwen.ipynb), deduplicated:
        <prefix>-poi-texts.json (unique texts) and <prefix>-poi-index.npy (row -> text).
check:  recomputes POI_i dist/dir of an existing POI CSV and reports the largest errors.

Usage: python scripts/attach_pois.py attach <trips_csv> <osm_pois_csv> <out_pois_csv>
       python scripts/attach_pois.py texts <pois_csv> <out_prefix>
       python scripts/attach_pois.py check <pois_csv>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree

K, R1, R2, MIN_AT_R1 = 10, 250, 500, 3
GEOD = Geod(ellps='WGS84')


def bearing(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng2 - lng1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng2 - lng1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def distance(lat1, lng1, lat2, lng2):
    return GEOD.inv(lng1, lat1, lng2, lat2)[2]


def attach(trips_csv, osm_csv, out_csv):
    assert not os.path.exists(out_csv), f"{out_csv} exists; not overwriting"
    trips = pd.read_csv(trips_csv)
    pois = pd.read_csv(osm_csv)
    to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32654', always_xy=True)
    coord_id, coords = pd.factorize(pd.MultiIndex.from_arrays([trips.lat, trips.lng]))
    clat, clng = coords.get_level_values(0).to_numpy(), coords.get_level_values(1).to_numpy()
    tree = cKDTree(np.column_stack(to_utm.transform(pois.lng.to_numpy(), pois.lat.to_numpy())))
    d, j = tree.query(np.column_stack(to_utm.transform(clng, clat)), k=K, distance_upper_bound=R2)
    use_r2 = (d <= R1).sum(1) < MIN_AT_R1
    keep = d <= np.where(use_r2, R2, R1)[:, None]

    out = {'radius_used(m)': np.where(use_r2, R2, R1)}
    names, dists = {}, {}
    for i in range(K):
        m = keep[:, i]
        p = pois.iloc[np.where(m, j[:, i], 0)]
        plat, plng = p.lat.to_numpy(), p.lng.to_numpy()
        for c, v in (('name', p['name'].to_numpy()), ('type', p['type'].to_numpy()), ('lat', plat), ('lng', plng)):
            names[f'POI_{i + 1}_{c}'] = np.where(m, v, None)
        dists[f'POI_{i + 1}_dist(m)'] = np.where(m, distance(clat, clng, plat, plng), np.nan)
        dists[f'POI_{i + 1}_dir(deg)'] = np.where(m, bearing(clat, clng, plat, plng), np.nan)
    # Nearest first by UTM distance; geodesic order agrees within 0.04% scale, so no re-sort.
    per_coord = pd.DataFrame({**out, **names, **dists})
    res = pd.concat([trips, per_coord.iloc[coord_id].reset_index(drop=True)], axis=1)
    res.to_csv(out_csv, index=False)
    n = keep.sum(1)[coord_id]
    print(f"{len(res)} points, {len(coords)} unique coordinates -> {out_csv}")
    print(f"500 m radius used: {use_r2[coord_id].mean():.1%} of points; no POI: {(n == 0).mean():.2%}; "
          f"mean POIs per point {n.mean():.2f}")
    types = pd.Series(per_coord[[f'POI_{i + 1}_type' for i in range(K)]].to_numpy().ravel()).dropna()
    print('POI slots by key:', types.str.split(':').str[0].value_counts(normalize=True).round(3).head(12).to_dict())


def format_row_pois_as_json(row, max_pois=10):
    """
    Format POIs for a single row as JSON string.
    Returns JSON string with format: {POI_1: {name: ..., type: ..., lat: ..., lng: ...}, ...}
    For rows with no POIs, returns "No POIs found within the given radius."
    """
    poi_dict = {}
    has_any_poi = False

    for i in range(1, max_pois + 1):
        name_col = f"POI_{i}_name"
        type_col = f"POI_{i}_type"
        lat_col = f"POI_{i}_lat"
        lng_col = f"POI_{i}_lng"

        if name_col in row.index and pd.notna(row[name_col]) and str(row[name_col]).strip() != '':
            poi_info = {}

            if name_col in row.index and pd.notna(row[name_col]):
                poi_info['name'] = str(row[name_col]).strip()
            if type_col in row.index and pd.notna(row[type_col]):
                poi_info['type'] = str(row[type_col]).strip()
            if lat_col in row.index and pd.notna(row[lat_col]):
                poi_info['lat'] = float(row[lat_col])
            if lng_col in row.index and pd.notna(row[lng_col]):
                poi_info['lng'] = float(row[lng_col])

            dist_col = f"POI_{i}_dist(m)"
            if dist_col in row.index and pd.notna(row[dist_col]):
                poi_info['dist(m)'] = float(row[dist_col])

            dir_col = f"POI_{i}_dir(deg)"
            if dir_col in row.index and pd.notna(row[dir_col]):
                poi_info['dir(deg)'] = float(row[dir_col])

            if poi_info:
                poi_dict[f"POI_{i}"] = poi_info
                has_any_poi = True

    if not has_any_poi:
        return "No POIs found within the given radius."

    return json.dumps(poi_dict, ensure_ascii=False)


def texts(pois_csv, prefix):
    out_json, out_idx = f'{prefix}-poi-texts.json', f'{prefix}-poi-index.npy'
    assert not os.path.exists(out_json) and not os.path.exists(out_idx), f"{prefix} outputs exist; not overwriting"
    poi_cols = [f'POI_{i}_{c}' for i in range(1, K + 1) for c in ('name', 'type', 'lat', 'lng', 'dist(m)', 'dir(deg)')]
    df = pd.read_csv(pois_csv, usecols=lambda c: c in poi_cols, low_memory=False)
    t = df.apply(format_row_pois_as_json, axis=1)
    index, uniq = pd.factorize(t)
    assert len(index) == len(df) and index.min() == 0
    with open(out_json, 'w', encoding='utf-8') as fp:
        json.dump(list(uniq), fp, ensure_ascii=False)
    np.save(out_idx, index.astype(np.int64))
    print(f"{len(df)} rows, {len(uniq)} unique texts -> {out_json}, {out_idx}")


def check(pois_csv):
    df = pd.read_csv(pois_csv, usecols=lambda c: c in ('lat', 'lng') or c.startswith('POI_'), low_memory=False)
    ed = eb = 0.0
    for i in range(1, K + 1):
        m = df[f'POI_{i}_lat'].notna() & df[f'POI_{i}_dist(m)'].notna()
        r = df[m]
        a = (r.lat, r.lng, r[f'POI_{i}_lat'], r[f'POI_{i}_lng'])
        ed = max(ed, np.abs(distance(*a) - r[f'POI_{i}_dist(m)']).max())
        eb = max(eb, np.abs((bearing(*a) - r[f'POI_{i}_dir(deg)'] + 180) % 360 - 180).max())
    print(f"max error: distance {ed:.2e} m, bearing {eb:.2e} deg")
    assert ed < 0.01 and eb < 1e-6


if __name__ == '__main__':
    {'attach': attach, 'texts': texts, 'check': check}[sys.argv[1]](*sys.argv[2:])
