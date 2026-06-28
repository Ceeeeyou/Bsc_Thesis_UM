"""
Signal inventory pipeline for vehicle-aware eco-routing.

WHY THIS MODULE EXISTS
======================
Detecting signal-controlled intersections by checking `highway=traffic_signals`
on route nodes (the original approach in signal_costs.has_traffic_signal)
under-reports signals in NL OSM data, badly. The reason isn't the tag
patterns; it's how OSMnx builds the routing graph.

A single real signalized intersection in NL OSM is typically tagged with
multiple `highway=traffic_signals` nodes (one per approach lane) plus
several `highway=crossing` nodes (one per pedestrian crosswalk). Diagnostic
data for one Eindhoven bbox: 1136 raw OSM signal-tagged points clustering
to ~150 real intersections (mean 7.6 nodes per intersection).

When OSMnx builds the drive graph, it keeps only ONE representative node
per intersection — and that representative is usually NOT one of the
signal-tagged nodes. So checking route-node tags finds none, even though
the route passes through real signalized intersections.

THE TWO-STAGE PIPELINE
======================
Stage 1: BUILD INVENTORY (independent of the routing graph)
    1.1 Fetch every signal-tagged feature in the bbox via
        features_from_bbox (returns nodes regardless of network filter).
    1.2 Cluster spatially (~30 m) so each real intersection collapses
        to one centroid. Eliminates OSM's per-lane multi-tagging.
    1.3 Disk-cache per bbox (SHA1 hash of rounded coords). The OSM
        fetch is ~30-60 s; we don't want to repeat it for the same OD.

Stage 2: MATCH INVENTORY TO ROUTE
    2.1 For each cluster centroid, find the nearest route node by
        point-to-segment distance against the route polyline.
    2.2 If within tolerance (~30 m), that route node is signal-controlled.
    2.3 Deduplicate by route node: multiple clusters mapping to the same
        node count as ONE signal cost. (You cannot physically stop
        twice at the same point on your route.)

FAILURE MODE
============
If features_from_bbox raises (no network, OSMnx version mismatch, etc.)
build_signal_inventory returns None and the caller falls back to the
original tag-based path. Offline work still runs.

For the thesis methodology section: this is a two-stage signal-detection
pipeline (OSM inventory → geometric route match) replacing the single-stage
node-tag check. The validation methodology in signal_locations.py applies
identically; only the detection algorithm changed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
import hashlib
import json
import warnings


# =============================================================================
# Defaults — tuned against Eindhoven-area validation (see thesis Section X.Y)
# =============================================================================
DEFAULT_CLUSTER_RADIUS_M  = 30.0   # OSM nodes within this radius = same intersection
DEFAULT_ROUTE_MATCH_TOL_M = 30.0   # Cluster centroid within this distance of any route edge = on route
DEFAULT_CACHE_DIR         = Path('./cache_signals')


# =============================================================================
# Data types
# =============================================================================
@dataclass
class SignalPoint:
    """One signal-tagged OSM node before clustering."""
    osm_id:  int
    lat:     float
    lon:     float
    source:  str    # 'traffic_signals' or 'crossing_signal'


@dataclass
class SignalCluster:
    """A spatially-clustered group of OSM signal nodes.
    Represents one real signalized intersection."""
    cluster_id: int
    lat:        float        # centroid
    lon:        float        # centroid
    members:    list[SignalPoint] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.members)

    @property
    def sources(self) -> str:
        return ','.join(sorted({m.source for m in self.members}))


@dataclass
class RouteSignal:
    """A signal-controlled route node, after matching clusters to the path."""
    route_node:        int           # graph node id on the route
    cluster_centroid:  tuple[float, float]   # (lat, lon)
    n_osm_nodes:       int           # how many OSM points the cluster contained
    sources:           str           # 'traffic_signals' | 'crossing_signal' | both
    distance_m:        float         # cluster centroid to nearest route edge


# =============================================================================
# Geometry helpers
# =============================================================================
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    lat1r, lon1r, lat2r, lon2r = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def _point_to_segment_m(plat, plon, alat, alon, blat, blon) -> float:
    """Point-to-segment distance in metres, locally projecting to planar XY
    around the segment midpoint. Accurate to <0.5% over a few km."""
    mlat = (alat + blat) / 2
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * cos(radians(mlat))
    def xy(lat, lon):
        return (lon - alon) * m_per_deg_lon, (lat - alat) * m_per_deg_lat
    px, py = xy(plat, plon)
    bx, by = xy(blat, blon)
    seg2 = bx * bx + by * by
    if seg2 < 1e-9:
        return sqrt(px * px + py * py)
    t = max(0.0, min(1.0, (px * bx + py * by) / seg2))
    cx, cy = t * bx, t * by
    return sqrt((px - cx) ** 2 + (py - cy) ** 2)


# =============================================================================
# Stage 1.1: Fetch OSM signal points (cached)
# =============================================================================
def _bbox_hash(south, north, west, east) -> str:
    key = f"{south:.5f}_{north:.5f}_{west:.5f}_{east:.5f}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def fetch_osm_signals(
    south: float, north: float, west: float, east: float,
    *, cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Optional[list[SignalPoint]]:
    """Fetch all signal-tagged points in bbox. Disk-cached per bbox.

    Returns None on failure (no network, OSMnx error, etc.) — caller is
    expected to fall back to tag-based detection.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"signals_{_bbox_hash(south, north, west, east)}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            points = [SignalPoint(**p) for p in data]
            print(f"  [signal cache] hit ({len(points)} points)")
            return points
        except Exception as e:
            print(f"  [signal cache] read failed ({e}); refetching")

    try:
        import osmnx as ox
    except ImportError:
        warnings.warn("osmnx not installed — cannot build signal inventory")
        return None

    try:
        # Single combined query for both highway=traffic_signals and
        # highway=crossing. One Overpass call instead of two — important
        # for batch runs against the rate-limited public Overpass server.
        gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags={'highway': ['traffic_signals', 'crossing']},
        )
    except Exception as e:
        warnings.warn(f"OSM signal fetch failed: {e}")
        return None

    points: list[SignalPoint] = []
    seen: set[int] = set()

    def _classify_row(row, columns) -> Optional[str]:
        """Return 'traffic_signals' | 'crossing_signal' | None for one row."""
        h = row.get('highway') if 'highway' in columns else None
        h_list = h if isinstance(h, list) else [h]

        if 'traffic_signals' in h_list:
            sub = row.get('traffic_signals') if 'traffic_signals' in columns else None
            if sub in ('blinker', 'emergency'):
                return None
            return 'traffic_signals'

        if 'crossing' in h_list:
            cval = row.get('crossing') if 'crossing' in columns else None
            csig = (row.get('crossing:signals')
                    if 'crossing:signals' in columns else None)
            is_signal = (
                cval in ('traffic_signals', 'pelican', 'puffin', 'toucan')
                or str(csig).lower() in ('yes', 'true')
            )
            return 'crossing_signal' if is_signal else None
        return None

    if gdf is not None and len(gdf):
        columns = gdf.columns
        for idx, row in gdf.iterrows():
            osm_id = idx[-1] if isinstance(idx, tuple) else idx
            if osm_id in seen:
                continue
            source = _classify_row(row, columns)
            if source is None:
                continue
            try:
                geom = row.geometry
                lat = float(geom.y); lon = float(geom.x)
            except Exception:
                continue   # non-Point geometry (way/relation) — skip
            seen.add(osm_id)
            points.append(SignalPoint(
                osm_id=int(osm_id),
                lat=lat, lon=lon,
                source=source,
            ))

    # Cache
    try:
        cache_file.write_text(json.dumps([p.__dict__ for p in points]))
        print(f"  [signal cache] wrote {len(points)} points -> {cache_file.name}")
    except Exception as e:
        print(f"  [signal cache] write failed ({e}); continuing without cache")

    return points


# =============================================================================
# Stage 1.2: Cluster
# =============================================================================
def cluster_points(
    points: list[SignalPoint],
    *, radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
) -> list[SignalCluster]:
    """Greedy single-linkage clustering by haversine distance.

    A point joins an existing cluster if it's within `radius_m` of ANY
    member; otherwise it starts a new cluster. Stable but order-sensitive
    at boundaries — fine for our use because we deduplicate by route node
    downstream regardless.
    """
    clusters: list[SignalCluster] = []
    for p in points:
        joined = False
        for c in clusters:
            for m in c.members:
                if haversine_m(p.lat, p.lon, m.lat, m.lon) <= radius_m:
                    c.members.append(p)
                    joined = True
                    break
            if joined:
                break
        if not joined:
            clusters.append(SignalCluster(
                cluster_id=len(clusters),
                lat=p.lat, lon=p.lon,
                members=[p],
            ))
    # Recompute centroids as the mean of all members
    for c in clusters:
        n = len(c.members)
        c.lat = sum(m.lat for m in c.members) / n
        c.lon = sum(m.lon for m in c.members) / n
    return clusters


# =============================================================================
# Stage 2: Match clusters to route, deduplicate by route node
# =============================================================================
def _nearest_route_node_by_edge_distance(G, path, lat, lon) -> tuple[int, float]:
    """Find the route node whose incident path edges come closest to (lat, lon).

    Returns (route_node_id, distance_m). The distance is measured against
    the *route polyline* near that node, not the node position itself —
    this matters because OSMnx route nodes can be hundreds of metres apart
    even when the polyline passes within 5 m of a signal centroid.
    """
    best_node = path[0]
    best_d = float('inf')

    # For each consecutive pair (u, v) in the path, compute polyline distance
    # to the cluster, and attribute the result to whichever endpoint is closer
    # to the polyline's closest point. This is a reasonable heuristic.
    for u, v in zip(path[:-1], path[1:]):
        edges = G[u][v]
        edge = next(iter(edges.values()))
        if 'geometry' in edge:
            coords = list(edge['geometry'].coords)   # (lon, lat) order
        else:
            coords = [
                (G.nodes[u]['x'], G.nodes[u]['y']),
                (G.nodes[v]['x'], G.nodes[v]['y']),
            ]
        for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
            d = _point_to_segment_m(lat, lon, lat1, lon1, lat2, lon2)
            if d < best_d:
                best_d = d
                # Attribute to the closer endpoint of (u, v)
                du = haversine_m(lat, lon, G.nodes[u]['y'], G.nodes[u]['x'])
                dv = haversine_m(lat, lon, G.nodes[v]['y'], G.nodes[v]['x'])
                best_node = u if du <= dv else v
    return best_node, best_d


def match_clusters_to_route(
    clusters: list[SignalCluster],
    G, path: list,
    *, tol_m: float = DEFAULT_ROUTE_MATCH_TOL_M,
) -> list[RouteSignal]:
    """For each cluster, find its nearest route node by edge-distance.
    Keep only those within `tol_m`. Deduplicate by route node so each
    physical node is charged at most one signal cost.

    If multiple clusters map to the same route node, the kept record uses
    the closest cluster (smallest distance_m); n_osm_nodes is summed
    across all merged clusters for diagnostic transparency.
    """
    # First pass: cluster -> (route_node, distance) if within tolerance
    pairs: list[tuple[SignalCluster, int, float]] = []
    for c in clusters:
        node, d = _nearest_route_node_by_edge_distance(G, path, c.lat, c.lon)
        if d <= tol_m:
            pairs.append((c, node, d))

    # Second pass: deduplicate by route node, keep nearest
    per_node: dict[int, RouteSignal] = {}
    n_osm_per_node: dict[int, int] = {}
    src_per_node: dict[int, set[str]] = {}
    for c, node, d in pairs:
        n_osm_per_node[node] = n_osm_per_node.get(node, 0) + c.n_nodes
        src_per_node.setdefault(node, set()).update(c.sources.split(','))
        if node not in per_node or d < per_node[node].distance_m:
            per_node[node] = RouteSignal(
                route_node=node,
                cluster_centroid=(c.lat, c.lon),
                n_osm_nodes=c.n_nodes,           # overwritten below
                sources=c.sources,               # overwritten below
                distance_m=d,
            )

    # Finalize: write the summed n_osm_nodes and unioned sources
    for node, rs in per_node.items():
        rs.n_osm_nodes = n_osm_per_node[node]
        rs.sources = ','.join(sorted(src_per_node[node]))

    return list(per_node.values())


# =============================================================================
# Top-level entry point
# =============================================================================
def build_signal_inventory(
    G, *,
    bbox: Optional[tuple[float, float, float, float]] = None,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Optional[list[SignalCluster]]:
    """End-to-end Stage 1: fetch + cluster.

    Returns None on failure (caller falls back to tag-based detection).

    bbox: optional (south, north, west, east). If given, used verbatim —
        pass the SAME request bbox used to build the routing graph so the
        on-disk signal cache key is stable across runs. If omitted, the
        bbox is inferred from G's node extent and rounded to 4 dp (~11 m)
        so minor run-to-run node-set differences don't bust the cache.
    """
    if bbox is not None:
        south, north, west, east = bbox
    else:
        lats = [d['y'] for _, d in G.nodes(data=True) if 'y' in d]
        lons = [d['x'] for _, d in G.nodes(data=True) if 'x' in d]
        if not lats or not lons:
            warnings.warn("Graph has no coords; cannot build signal inventory")
            return None
        south, north = round(min(lats), 4), round(max(lats), 4)
        west,  east  = round(min(lons), 4), round(max(lons), 4)

    points = fetch_osm_signals(south, north, west, east, cache_dir=cache_dir)
    if points is None:
        return None

    clusters = cluster_points(points, radius_m=cluster_radius_m)
    return clusters


def match_route_to_inventory(
    G, path: list, inventory: list[SignalCluster],
    *, tol_m: float = DEFAULT_ROUTE_MATCH_TOL_M,
) -> list[RouteSignal]:
    """Stage 2 wrapper. Returns the deduplicated, route-node-attached list."""
    return match_clusters_to_route(inventory, G, path, tol_m=tol_m)
