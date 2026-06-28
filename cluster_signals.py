"""
Cluster the raw OSM signal-tagged points into intersection-level signals,
then check how many cluster centroids lie near the route.

Why
---
NL OSM editors place multiple `highway=traffic_signals` nodes per real
intersection (one per approach lane, plus signal-controlled crossings).
Raw count is therefore ~10x the real signalized-intersection count.
Clustering at ~30 m radius collapses these to one point per intersection.

We then check how many cluster centroids lie within a tolerance of any
route node. That's the correct signal count for routing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    lat1r, lon1r, lat2r, lon2r = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def cluster_points(points, radius_m=30.0):
    """Greedy single-linkage clustering: a point joins an existing cluster
    if it's within `radius_m` of ANY member; otherwise it starts a new one.

    Returns list of clusters; each cluster is a list of point dicts.
    """
    clusters = []
    for p in points:
        joined = False
        for cluster in clusters:
            for member in cluster:
                if haversine_m(p['lat'], p['lon'],
                               member['lat'], member['lon']) <= radius_m:
                    cluster.append(p)
                    joined = True
                    break
            if joined:
                break
        if not joined:
            clusters.append([p])
    return clusters


def cluster_centroid(cluster):
    """Mean lat/lon of a cluster."""
    n = len(cluster)
    return (sum(p['lat'] for p in cluster) / n,
            sum(p['lon'] for p in cluster) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start',  required=True)
    ap.add_argument('--end',    required=True)
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--cluster-radius-m', type=float, default=30.0,
                    help='Points within this distance form one intersection')
    ap.add_argument('--route-tol-m', type=float, default=30.0,
                    help='A cluster centroid within this distance of any route node counts as on-route')
    ap.add_argument('--out-dir', default='./signal_diag')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_lat, s_lon = [float(x) for x in args.start.split(',')]
    e_lat, e_lon = [float(x) for x in args.end.split(',')]
    south = min(s_lat, e_lat) - args.margin
    north = max(s_lat, e_lat) + args.margin
    west  = min(s_lon, e_lon) - args.margin
    east  = max(s_lon, e_lon) + args.margin

    import osmnx as ox
    print(f"Bbox: [{south:.4f}, {west:.4f}] -> [{north:.4f}, {east:.4f}]")

    # ----- Fetch every signal-tagged node ---------------------------------
    print("\nFetching all OSM signal points (truth set)...")

    # highway=traffic_signals — direct
    g1 = ox.features_from_bbox(
        bbox=(west, south, east, north),
        tags={'highway': 'traffic_signals'},
    )
    print(f"  highway=traffic_signals: {len(g1)}")

    # highway=crossing — filter to signal-controlled only
    g2 = ox.features_from_bbox(
        bbox=(west, south, east, north),
        tags={'highway': 'crossing'},
    )
    # Keep rows where crossing in signal-values OR crossing:signals=yes
    signal_cross_mask = pd.Series(False, index=g2.index)
    if 'crossing' in g2.columns:
        signal_cross_mask |= g2['crossing'].isin(
            ['traffic_signals', 'pelican', 'puffin', 'toucan']
        )
    if 'crossing:signals' in g2.columns:
        signal_cross_mask |= g2['crossing:signals'].astype(str).str.lower().isin(
            ['yes', 'true']
        )
    g2_sig = g2[signal_cross_mask]
    print(f"  highway=crossing (signal-controlled only): {len(g2_sig)}")

    # Union — convert both to a flat list of dicts, dedup by osm id
    points = []
    seen = set()
    for gdf, src in [(g1, 'traffic_signals'), (g2_sig, 'crossing_signal')]:
        for idx, row in gdf.iterrows():
            osm_id = idx[-1] if isinstance(idx, tuple) else idx
            if osm_id in seen:
                continue
            seen.add(osm_id)
            points.append({
                'osm_id': osm_id,
                'lat':    row.geometry.y,
                'lon':    row.geometry.x,
                'source': src,
            })
    print(f"  Union (deduped by OSM id): {len(points)} points")

    # ----- Cluster --------------------------------------------------------
    print(f"\nClustering at radius={args.cluster_radius_m:.0f} m...")
    clusters = cluster_points(points, radius_m=args.cluster_radius_m)
    print(f"  Clusters (= estimated real intersections): {len(clusters)}")
    sizes = sorted([len(c) for c in clusters], reverse=True)
    print(f"  Cluster size distribution (top 10): {sizes[:10]}")
    print(f"  Mean nodes per cluster: {sum(sizes)/len(sizes):.1f}")

    # Save cluster centroids
    centroids = []
    for i, c in enumerate(clusters):
        lat, lon = cluster_centroid(c)
        centroids.append({
            'cluster_id': i,
            'lat': lat,
            'lon': lon,
            'n_nodes': len(c),
            'sources': ','.join(sorted({p['source'] for p in c})),
        })
    pd.DataFrame(centroids).to_csv(out_dir / 'signal_clusters.csv', index=False)

    # ----- Build same drive graph + route ---------------------------------
    print(f"\nBuilding drive graph + route for comparison...")
    from signal_locations import load_network, find_route
    G, src_n, dst_n = load_network(s_lat, s_lon, e_lat, e_lon,
                                   margin_deg=args.margin)
    path = find_route(G, src_n, dst_n, 'fastest')
    print(f"  Route: {len(path)} nodes")

    # For each cluster centroid, find the nearest route node and its distance
    on_route = []
    for c in centroids:
        best_d = float('inf')
        best_node = None
        for n in path:
            nd = G.nodes[n]
            d = haversine_m(c['lat'], c['lon'], nd['y'], nd['x'])
            if d < best_d:
                best_d = d
                best_node = n
        if best_d <= args.route_tol_m:
            on_route.append({
                **c,
                'nearest_route_node': best_node,
                'distance_m': round(best_d, 1),
            })

    on_route_df = pd.DataFrame(on_route).sort_values('distance_m') if on_route else pd.DataFrame()
    on_route_df.to_csv(out_dir / 'route_signals_clustered.csv', index=False)

    # ----- Summary --------------------------------------------------------
    print(f"\n=== RESULTS ===")
    print(f"OSM signal points (raw):         {len(points)}")
    print(f"Signalized intersections (clusters at {args.cluster_radius_m:.0f}m): {len(clusters)}")
    print(f"On route (centroid within {args.route_tol_m:.0f}m of any route node): {len(on_route)}")
    print(f"\nCurrent code reports 0 signals on this route.")
    print(f"After clustering + edge-proximity, it should report: {len(on_route)}.")
    print(f"\nFiles:")
    print(f"  signal_clusters.csv         all {len(clusters)} clusters")
    print(f"  route_signals_clustered.csv {len(on_route)} clusters on this route")


if __name__ == '__main__':
    main()
