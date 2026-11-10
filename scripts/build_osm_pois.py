"""Named OpenStreetMap POIs inside the Foursquare Tokyo box, from the Geofabrik Kanto extract.

Nodes, plus areas (closed ways and multipolygons) placed at the mean of their outer-ring nodes.
type = the first matching "key:value" in KEYS (raw OSM tags, not mapped onto Geoapify's categories).
Private homes (building=residential/apartments/house/detached) are excluded.

Usage: python scripts/build_osm_pois.py <kanto.osm.pbf> <out_csv>
"""
import sys
from collections import Counter

import osmium
import pandas as pd

BBOX = (139.0, 35.0, 140.5, 36.3)  # lng_min, lat_min, lng_max, lat_max (settings/foursquare*.json)
KEYS = ['amenity', 'shop', 'tourism', 'leisure', 'office', 'public_transport', 'railway', 'historic',
        'healthcare', 'craft', 'sport', 'building']
HOMES = {'residential', 'apartments', 'house', 'detached'}


def poi_type(tags):
    for k in KEYS:
        v = tags.get(k)
        if v and not (k == 'building' and v in HOMES):
            return f'{k}:{v}'
    return None


class Handler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def add(self, tags, lat, lng):
        name = tags.get('name')
        if not name or not (BBOX[0] <= lng <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
            return
        t = poi_type(tags)
        if t:
            self.rows.append((name.strip(), t, lat, lng))

    def node(self, n):
        if 'name' in n.tags:
            self.add(n.tags, n.location.lat, n.location.lon)

    def area(self, a):
        if 'name' not in a.tags:
            return
        pts = [(nd.lat, nd.lon) for ring in a.outer_rings() for nd in ring if nd.location.valid()]
        if pts:
            self.add(a.tags, sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


if __name__ == '__main__':
    pbf, out_csv = sys.argv[1:3]
    h = Handler()
    h.apply_file(pbf, locations=True, idx='flex_mem')
    df = pd.DataFrame(h.rows, columns=['name', 'type', 'lat', 'lng'])
    n = len(df)
    df = df[~df.assign(la=df.lat.round(4), ln=df.lng.round(4)).duplicated(['name', 'type', 'la', 'ln'])]
    df.to_csv(out_csv, index=False)
    keys = Counter(t.split(':')[0] for t in df.type)
    print(f'{len(df)} POIs ({n - len(df)} duplicates dropped) -> {out_csv}')
    print('by key:', dict(keys.most_common()))
    print(f"building share: {keys['building'] / len(df):.1%}")
