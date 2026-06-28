"""
Step 2 — Vehicle-aware eco-routing.

Given an OD pair (A, B) and a vehicle configuration, compute three routes:
  1. FASTEST route  : minimize travel time (no CO2 awareness)
  2. ECO route      : minimize CO2 subject to time ≤ (1 + slack) × T_fastest
  3. ECO route for a SECOND vehicle on the same OD pair, same slack

Then compare per-route CO2, time, distance, and whether the routes are
geometrically identical. This directly tests Hypothesis H2 from Step 1:

    H2 — regime-dependent vehicle parameter importance implies that eco-optimal
    routes diverge between vehicles, particularly for OD pairs with mixed-regime
    alternatives.

DESIGN OUTLINE
==============
Network:
    OSMnx → drivable subgraph for the bounding box of the OD pair, plus a
    margin. Each edge has `length` (m) and `maxspeed` (or a class-based default).

Edge cost components (computed once per (edge, vehicle)):
    time_s            edge length / free-flow speed for that edge's class
    co2_g             integrate the energy model over a synthetic speed profile
                      drawn from the WLTP phase that matches the edge's OSM
                      `highway` tag (consistent with road_aware_ece.py)

Eco objective for a given (vehicle, slack):
    Find the path minimizing total CO2, subject to total time ≤ T_budget,
    where T_budget = (1 + slack) × T_fastest.

    Implementation: penalty-based weighted Dijkstra. Define a combined edge
    weight w_α(e) = α · co2_g(e) + (1−α) · time_s(e), with both terms scaled
    to [0, 1] over the network. Sweep α from 0 (pure time) to 1 (pure CO2).
    For each α, run Dijkstra and record the path. Among paths satisfying
    the time budget, pick the one with the lowest CO2.

    This produces both the eco-optimal route and a Pareto frontier as a
    side-product. Robust, simple, reproducible — the standard approach in the
    eco-routing literature (Ahn & Rakha 2013, Boriboonsomsin 2012).

Reproducibility:
    All inputs are CLI args: --start, --end (lat,lon), --vehicle, --slack,
    --place (or --bbox), --out. No code edits to switch vehicles or OD pairs.

DEPENDENCIES (run on your machine):
    pip install osmnx networkx folium pandas numpy

If OSMnx is unavailable, the --synthetic flag runs the same algorithm on
a hand-built test graph (used here to validate the pipeline end-to-end).
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from iso23795_model import (
    Vehicle, VEHICLES, compute_work_per_second, aggregate_trip,
)
from road_aware_ece import ROAD_TYPE_TO_PHASE
from elevation import ElevationProvider, SyntheticElevationProvider, add_elevation_to_graph
from signal_costs import (
    compute_path_signal_cost,
    compute_path_signal_cost_from_inventory,
    DEFAULT_EXPECTED_IDLE_S,
    DEFAULT_STOP_PROBABILITY,
)
from signal_inventory import (
    build_signal_inventory,
    match_route_to_inventory,
)


# =============================================================================
# WLTP phase data — speed profiles per regime (loaded once)
# =============================================================================
def load_wltp_phases(wltp_csv: str) -> dict:
    """Load the four WLTP phases as dicts of (speed, distance, time) arrays."""
    df = pd.read_csv(wltp_csv)
    phases = {}
    for ph in ['LOW', 'MIDDLE', 'HIGH', 'EXTRA-HIGH']:
        sub = df[df['phase'] == ph].reset_index(drop=True)
        speed = sub['speed_ms'].values.astype(float)
        # Distance per row matches TEST_XLSX/LCMM convention (d = v · 1s)
        distance = speed.copy()
        # 1 Hz timestamps in ms
        time_ms = (np.arange(len(speed)) * 1000).astype(float)
        # Phase totals
        phase_distance_m = float(distance.sum())
        phase_duration_s = float(len(speed))   # 1 Hz so #rows = seconds
        phase_avg_speed_kmh = (
            (speed[speed >= 0.5].mean() * 3.6) if (speed >= 0.5).any() else 0
        )
        phases[ph] = {
            'speed_ms':   speed,
            'distance_m': distance,
            'time_ms':    time_ms,
            'total_distance_m':  phase_distance_m,
            'total_duration_s':  phase_duration_s,
            'avg_speed_kmh':     phase_avg_speed_kmh,
        }
    return phases


# Default free-flow speeds (km/h) per OSM road class — used when `maxspeed`
# tag is missing on an edge. EU/NL-calibrated.
DEFAULT_MAXSPEED_KMH = {
    'motorway':       100,    # NL motorway 100 km/h common
    'motorway_link':   80,
    'trunk':           80,
    'trunk_link':      60,
    'primary':         70,
    'primary_link':    50,
    'secondary':       50,
    'secondary_link':  50,
    'tertiary':        50,
    'tertiary_link':   30,
    'residential':     30,
    'living_street':   15,
    'service':         20,
    'unclassified':    50,
    'road':            50,
    'pedestrian':       5,
    'track':           20,
}


# =============================================================================
# Edge cost computation
# =============================================================================
def precompute_phase_costs(vehicle: Vehicle, wltp_phases: dict) -> dict:
    """Compute per-(vehicle, phase) per-km motion cost and moving-average
    cruise speed. Called ONCE per vehicle, then per-edge cost is a simple
    multiplication.

    This replaces the previous "slice WLTP trace per edge" approach with the
    cleaner "phase-average per-km" model. Three reasons:

    1. Internally consistent. Previous code used WLTP-derived motion energy
       (includes WLTP's internal stops) AND legal-speed-limit time (assumes
       no stops at all). The two views of WLTP were inconsistent, which
       caused model travel times to be 2-3x faster than Google Maps.

    2. No per-edge slicing noise. Previous code picked different WLTP rows
       for adjacent edges of the same road type, giving slightly different
       per-edge costs based on arbitrary slicing decisions. Phase-average
       gives every edge of the same length and road type the same cost.

    3. ~100x faster. Pre-compute once, multiply per edge.

    Returns
    -------
    Dict mapping phase name → dict with:
        motion_J_per_m       motion energy per meter (cruise-only)
        co2_g_per_m          CO2 per meter (cruise-only, from motion energy)
        avg_moving_ms        moving-average speed in m/s (stops EXCLUDED).
                             This is used for edge time. Stops are added
                             explicitly via signal_costs when an OSM-tagged
                             signal lies on the route.
    """
    costs = {}
    for name, phase in wltp_phases.items():
        speed = phase['speed_ms']
        distance = phase['distance_m']

        # Run the energy model on the FULL phase ONCE.
        # We use only the moving rows so motion cost is "cruise-only".
        moving = speed >= 0.5   # standstill threshold (per ISO 23795-1)
        per_sec = compute_work_per_second(
            speed_ms=speed,
            altitude_m=np.zeros_like(speed),
            distance_m=distance,
            time_ms=phase['time_ms'],
            vehicle=vehicle,
        )
        trip = aggregate_trip(per_sec, vehicle)

        # Moving-only distance — this is what motion energy was spent on
        moving_dist_m = float(distance[moving].sum())
        if moving_dist_m <= 0:
            moving_dist_m = 1.0  # avoid division by zero

        # Motion energy per moving meter
        motion_J_per_m = trip.total_motion_J / moving_dist_m
        # Convert to CO2 per meter
        fuel_l_per_m = motion_J_per_m / (vehicle.eta * vehicle.lhv_J_per_l)
        co2_g_per_m  = fuel_l_per_m * vehicle.co2_kg_per_l * 1000

        # Moving-average speed in m/s (stops excluded — matches WLTP standard
        # report values: 25.3, 44.5, 60.7, 94.0 km/h for LOW/MIDDLE/HIGH/EH)
        avg_moving_ms = float(speed[moving].mean()) if moving.any() else 0.0

        costs[name] = {
            'motion_J_per_m':  motion_J_per_m,
            'co2_g_per_m':     co2_g_per_m,
            'avg_moving_ms':   avg_moving_ms,
        }
    return costs


def compute_edge_cost_co2_g(
    length_m: float, road_type: str, vehicle: Vehicle, phase_costs: dict,
) -> tuple[float, float]:
    """Per-edge cost: linear lookup against pre-computed phase costs.

    Returns (co2_grams, time_seconds) where:
      co2_g  = phase per-km motion cost × length    (motion only; grade and
               signals added separately by caller)
      time_s = length / phase moving-average speed  (cruise-only; idle/stop
               time at OSM-tagged signals added separately by caller)
    """
    phase_name = ROAD_TYPE_TO_PHASE.get(road_type, 'LOW')
    pc = phase_costs[phase_name]

    co2_g  = pc['co2_g_per_m']    * length_m
    avg_v  = pc['avg_moving_ms']
    if avg_v <= 0:
        time_s = length_m / 5.0  # walking pace fallback
    else:
        time_s = length_m / avg_v
    return co2_g, time_s


def annotate_graph_costs(G, vehicle: Vehicle, wltp_phases: dict,
                          include_grade: bool = True) -> None:
    """Annotate every edge with vehicle-specific cost attributes.

    Per-edge attributes written:
        co2_g           total CO2 in grams (motion + grade)
        co2_motion_g    motion-only CO2 (diagnostic; from WLTP phase avg)
        co2_grade_g     grade-only CO2 (diagnostic; from m·g·dh_pos)
        time_s          edge travel time in seconds (cruise-only;
                        signal idle time is added route-level, not per-edge)
        time_freeflow_s pure free-flow time using OSM maxspeed (diagnostic)
        maxspeed_kmh    parsed maxspeed value (diagnostic)

    METHODOLOGY NOTES (cite in thesis):

    Motion cost is computed per WLTP phase moving-average, NOT per-edge slice
    of the WLTP trace. This means every residential edge gets the same per-km
    cost, every motorway edge gets the same per-km cost, etc. Pre-computed
    once per (vehicle, phase) at the start of routing.

    Edge time uses the WLTP phase MOVING average (stops excluded) — 25.3,
    44.5, 60.7, 94.0 km/h for LOW/MIDDLE/HIGH/EXTRA-HIGH. This represents
    the speed when the vehicle is actually moving. Idle time at OSM-tagged
    signals is added separately via signal_costs.augment_route_with_signals().

    This is "Option B" in the discussion: cruise on every edge + explicit
    stops at OSM-tagged signal nodes. Limitation: OSM signal coverage is
    incomplete in NL, so this under-counts urban friction on routes where
    signals are mistagged or untagged. Document this in thesis.
    """
    G_GRAV = 9.81

    # Pre-compute per-phase costs ONCE for this vehicle
    phase_costs = precompute_phase_costs(vehicle, wltp_phases)

    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get('length', 1.0))
        road_type = data.get('highway', 'unclassified')
        if isinstance(road_type, list):
            road_type = road_type[0]

        # Diagnostic only: free-flow time using OSM maxspeed
        if 'time_freeflow_s' not in data:
            maxspeed = data.get('maxspeed')
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if maxspeed is None:
                v_kmh = DEFAULT_MAXSPEED_KMH.get(road_type, 50)
            else:
                try:
                    v_kmh = float(str(maxspeed).split()[0])
                except Exception:
                    v_kmh = DEFAULT_MAXSPEED_KMH.get(road_type, 50)
            data['time_freeflow_s'] = length_m / max(1.0, v_kmh / 3.6)
            data['maxspeed_kmh'] = v_kmh

        # Per-edge motion cost and cruise time, from phase aggregates
        co2_g_motion, edge_time_s = compute_edge_cost_co2_g(
            length_m, road_type, vehicle, phase_costs
        )

        # Grade work added linearly (ISO clip: only positive grade work counts)
        # Prefer the multi-point integrated positive-gain dh_pos_m if present;
        # fall back to endpoint dh_m for backward compatibility.
        if include_grade:
            if 'dh_pos_m' in data:
                dh_pos = float(data['dh_pos_m'])
            else:
                dh_pos = max(0.0, float(data.get('dh_m', 0.0)))
            grade_work_J = vehicle.mass_kg * G_GRAV * dh_pos
            grade_fuel_l = grade_work_J / (vehicle.eta * vehicle.lhv_J_per_l)
            grade_co2_g = grade_fuel_l * vehicle.co2_kg_per_l * 1000
        else:
            grade_co2_g = 0.0

        data['co2_g']        = co2_g_motion + grade_co2_g
        data['co2_motion_g'] = co2_g_motion   # diagnostic
        data['co2_grade_g']  = grade_co2_g    # diagnostic
        data['time_s']       = edge_time_s    # cruise time from WLTP moving avg
                                                # (signal idle added at route level)


# =============================================================================
# Routing
# =============================================================================
@dataclass
class Route:
    name: str
    nodes: list
    co2_g: float
    time_s: float
    distance_m: float
    edge_count: int


def shortest_path_by(G, source, target, weight: str) -> Route:
    """Run Dijkstra on G with the given edge attribute as weight."""
    nodes = nx.shortest_path(G, source, target, weight=weight)
    co2 = 0.0; time = 0.0; dist = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        # MultiDiGraph: pick the parallel edge minimizing the active weight
        edges = G[u][v]
        best = min(edges.values(), key=lambda d: d.get(weight, float('inf')))
        co2  += best['co2_g']
        time += best['time_s']
        dist += best['length']
    return Route(name='?', nodes=nodes, co2_g=co2, time_s=time,
                 distance_m=dist, edge_count=len(nodes) - 1)


def find_eco_route(G, source, target, time_budget_s: float,
                   k: int = 20) -> tuple[Route, list]:
    """Find the route minimizing CO2 subject to time ≤ time_budget_s.

    Method: enumerate the top-k time-shortest simple paths via Yen's algorithm
    (NetworkX shortest_simple_paths), evaluate each by CO2 and time, pick
    the lowest-CO2 path that satisfies the time budget.

    This handles cases the weighted-sum (alpha-sweep) approach misses —
    routes that are Pareto-dominated in the linear sense but become optimal
    once a non-linear constraint (the time budget) is applied. Standard
    technique in constrained shortest path problems.

    Returns the chosen route and the full list of evaluated alternatives.
    """
    # NetworkX requires single-edge graphs for shortest_simple_paths;
    # collapse parallel edges by taking the minimum-time edge per (u,v).
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    for u, v, data in G.edges(data=True):
        if H.has_edge(u, v):
            existing = H[u][v]
            if data['time_s'] < existing['time_s']:
                for k_, val in data.items():
                    H[u][v][k_] = val
        else:
            H.add_edge(u, v, **data)

    candidates = []
    try:
        gen = nx.shortest_simple_paths(H, source, target, weight='time_s')
        for i, path in enumerate(gen):
            if i >= k:
                break
            co2 = 0.0; time = 0.0; dist = 0.0
            for u, v in zip(path[:-1], path[1:]):
                e = H[u][v]
                co2 += e['co2_g']; time += e['time_s']; dist += e['length']
            r = Route(name=f'k={i+1}', nodes=path, co2_g=co2, time_s=time,
                      distance_m=dist, edge_count=len(path) - 1)
            candidates.append(r)
    except nx.NetworkXNoPath:
        return None, []

    # Among candidates within budget, pick lowest CO2
    feasible = [r for r in candidates if r.time_s <= time_budget_s]
    if not feasible:
        # No feasible path found — return the fastest of the candidates
        best = min(candidates, key=lambda r: r.time_s)
        best.name = 'eco_fallback_no_feasible'
        return best, candidates

    best = min(feasible, key=lambda r: r.co2_g)
    best.name = f'eco_k={candidates.index(best)+1}'
    return best, candidates


# =============================================================================
# Network loading — OSMnx and synthetic
# =============================================================================
def load_osmnx_network(start_lat: float, start_lon: float,
                        end_lat: float,   end_lon: float,
                        margin_deg: float = 0.01):
    """Download a drivable OSM subgraph spanning A and B with a margin."""
    try:
        import osmnx as ox
    except ImportError:
        raise RuntimeError("Run: pip install osmnx")

    # Ensure signal-related sub-tags are retained on graph nodes. OSMnx default
    # is ['highway', 'junction', 'railway', 'ref']. We add:
    #   - 'crossing'           — legacy "highway=crossing + crossing=traffic_signals"
    #                            tagging (still common in NL).
    #   - 'crossing:signals'   — modern boolean "crossing:signals=yes" tagging
    #                            (spreading since ~2020, increasingly common
    #                            in recent Dutch edits).
    #   - 'traffic_signals'    — sub-tag on highway=traffic_signals nodes used
    #                            to distinguish normal signals from blinkers,
    #                            emergency-only signals, and ramp meters. Lets
    #                            us EXCLUDE false-positive sub-types in
    #                            signal_costs.has_traffic_signal().
    # Without these, the corresponding detection branches in
    # signal_costs.classify_signal_node() can never fire.
    desired = ['highway', 'junction', 'railway', 'ref',
               'crossing', 'crossing:signals', 'traffic_signals']
    ox.settings.useful_tags_node = list(set(ox.settings.useful_tags_node + desired))

    # Disable OSMnx caching. The cache stores graphs as built with the
    # useful_tags_node settings *at first download*. If we later add tags
    # (like 'crossing'), a cached graph silently lacks them — a silent
    # correctness bug. Forcing fresh downloads avoids this. Cost: ~10-30s
    # of extra download time per OD pair, which is negligible for thesis
    # work involving a small handful of routes.
    ox.settings.use_cache = False

    south = min(start_lat, end_lat) - margin_deg
    north = max(start_lat, end_lat) + margin_deg
    west  = min(start_lon, end_lon) - margin_deg
    east  = max(start_lon, end_lon) + margin_deg
    print(f"  Downloading OSM bbox [{south:.4f}, {west:.4f}] → "
          f"[{north:.4f}, {east:.4f}]...")

    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
    print(f"  Network: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Stash the REQUEST bbox so the signal inventory uses the identical box
    # (and therefore a stable on-disk cache key) rather than re-deriving it
    # from graph node extent, which wobbles run-to-run.
    G.graph['request_bbox'] = (south, north, west, east)

    src = ox.distance.nearest_nodes(G, start_lon, start_lat)
    dst = ox.distance.nearest_nodes(G, end_lon, end_lat)
    return G, src, dst


def build_synthetic_network():
    """A test graph that mimics a city with THREE route alternatives:
        FAST_PATH    — short hop to motorway, ~7.7 km, motorway-dominated
        URBAN_PATH   — through residential streets, ~6 km, low speed throughout
        MIXED_PATH   — primary/secondary mix, ~6.8 km, intermediate speeds
    Used when OSMnx is unavailable. Designed so different vehicles may
    legitimately prefer different routes — H2 test scenario.
    """
    G = nx.MultiDiGraph()
    G.graph['crs'] = 'EPSG:4326'

    nodes = {
        'A':  (5.4789, 51.4430),    # Eindhoven Centraal (start)
        'B':  (5.3778, 51.4503),    # Eindhoven Airport  (end)
        # Motorway path nodes (north loop)
        'M1': (5.4500, 51.4600),
        'M2': (5.4000, 51.4700),
        'M3': (5.3850, 51.4550),
        # Urban path nodes (direct through residential)
        'U1': (5.4500, 51.4400),
        'U2': (5.4200, 51.4400),
        'U3': (5.4000, 51.4450),
        # Mixed path nodes (south loop, primary/secondary)
        'P1': (5.4500, 51.4300),
        'P2': (5.4000, 51.4350),
        'P3': (5.3850, 51.4400),
    }
    for n, (x, y) in nodes.items():
        G.add_node(n, x=x, y=y)

    # Tag urban-route intermediate nodes as traffic_signals (3 signals on the
    # urban path, 1 on the mixed path entrance, 0 on motorway).
    # This mimics realistic OSM tagging: motorways have no at-grade signals,
    # urban arterials have several, primary roads have some.
    G.nodes['U1']['highway'] = 'traffic_signals'
    G.nodes['U2']['highway'] = 'traffic_signals'
    G.nodes['U3']['highway'] = 'traffic_signals'
    G.nodes['P1']['highway'] = 'traffic_signals'

    def add_edge(u, v, length_m, highway, maxspeed_kmh):
        G.add_edge(u, v, key=0, length=length_m, highway=highway,
                   maxspeed=str(maxspeed_kmh))
        G.add_edge(v, u, key=0, length=length_m, highway=highway,
                   maxspeed=str(maxspeed_kmh))

    # Fast path: A → M1 → M2 → M3 → B  (motorway-dominated, fastest, high aero)
    add_edge('A',  'M1', 1500,  'primary',     70)
    add_edge('M1', 'M2', 3500,  'motorway',   100)
    add_edge('M2', 'M3', 1500,  'motorway',   100)
    add_edge('M3', 'B',  1200,  'primary',     70)

    # Urban path: A → U1 → U2 → U3 → B  (residential streets, low aero, lots of stop-go)
    # Residential streets here represent a slow but short route. Stop-and-go
    # implied by the LOW WLTP phase the cost function uses.
    add_edge('A',  'U1', 1500,  'secondary',   50)
    add_edge('U1', 'U2', 1500,  'residential', 30)
    add_edge('U2', 'U3', 1500,  'residential', 30)
    add_edge('U3', 'B',  1500,  'secondary',   50)

    # Mixed path: A → P1 → P2 → P3 → B  (primary/secondary mix, ~60-70 km/h)
    # Should be the sweet spot for a heavy vehicle: enough speed to reduce
    # idle/per-km overhead but not motorway-aero-penalty territory.
    add_edge('A',  'P1', 1700,  'secondary',   50)
    add_edge('P1', 'P2', 3300,  'primary',     70)
    add_edge('P2', 'P3', 1300,  'primary',     70)
    add_edge('P3', 'B',  1300,  'secondary',   50)

    return G, 'A', 'B'


# =============================================================================
# Main
# =============================================================================
# Module-level inventory cache: keyed by id(G) so we don't rebuild the
# OSM signal inventory for r1 and r2 on the same graph. Cleared on each
# run because the in-memory graph is rebuilt per OD pair in main().
_INVENTORY_CACHE: dict[int, object] = {}


def _get_or_build_inventory(G):
    """Build the OSM signal inventory for graph G, cached per process run.
    Returns None on failure (caller falls back to tag-based detection)."""
    key = id(G)
    if key in _INVENTORY_CACHE:
        return _INVENTORY_CACHE[key]
    print(f"  Building OSM signal inventory...")
    inv = build_signal_inventory(G, bbox=G.graph.get('request_bbox'))
    if inv is not None:
        print(f"    {len(inv)} signalized intersections in bbox")
    else:
        print(f"    inventory unavailable — falling back to node-tag detection")
    _INVENTORY_CACHE[key] = inv
    return inv


def augment_route_with_signals(G, route, vehicle: Vehicle,
                                  deterministic: bool = False) -> None:
    """Add signal_count, signal_co2_g, signal_time_s to a Route and fold
    them into route's total co2_g / time_s. Modifies the route in place.

    Detection strategy (two-stage pipeline; see signal_inventory module):
      1. Build OSM signal inventory for the graph's bbox (cached).
      2. Match inventory clusters to the route geometrically, dedup by
         route node. One signal cost per matched route node.

    Falls back to the legacy node-tag detection if the inventory is
    unavailable (no network, OSMnx error). Both paths return the same
    dict shape, so downstream code is identical.
    """
    inv = _get_or_build_inventory(G)
    if inv is not None:
        route_signals = match_route_to_inventory(G, route.nodes, inv)
        sig = compute_path_signal_cost_from_inventory(
            G, route.nodes, vehicle, route_signals,
            deterministic=deterministic,
        )
    else:
        # Legacy fallback
        sig = compute_path_signal_cost(G, route.nodes, vehicle,
                                        deterministic=deterministic)

    route.signal_count   = sig['signal_count']
    route.signal_co2_g   = sig['co2_g']
    route.signal_time_s  = sig['time_s']
    # Save edge-level totals before folding signals in
    route.co2_motion_g   = route.co2_g
    route.co2_g          = route.co2_g + sig['co2_g']
    route.time_s         = route.time_s + sig['time_s']


def run_routing(G, source, target, vehicle: Vehicle,
                wltp_phases: dict, slack: float, label: str,
                include_grade: bool = True,
                include_signals: bool = True,
                deterministic_signals: bool = False) -> dict:
    """Annotate edges with this vehicle's costs, then find fastest + eco routes.

    include_grade: whether to add m·g·dh grade work to per-edge CO2.
    include_signals: whether to add per-signal idle/KE excess to route totals.
    deterministic_signals: if True, every signal-tagged node is a guaranteed
        stop (conservative upper bound). If False, expected-value (p_stop=0.5).
    """
    print(f"\n[{label}] Annotating edge costs for {vehicle.name} "
          f"(grade={'on' if include_grade else 'off'})...")
    annotate_graph_costs(G, vehicle, wltp_phases, include_grade=include_grade)

    # 1) Fastest route
    fastest = shortest_path_by(G, source, target, weight='time_s')
    fastest.name = 'fastest'
    if include_signals:
        augment_route_with_signals(G, fastest, vehicle,
                                    deterministic=deterministic_signals)
    print(f"  Fastest: {fastest.distance_m/1000:.2f} km, "
          f"{fastest.time_s/60:.1f} min, {fastest.co2_g:.0f} g CO2 "
          f"(signals: {getattr(fastest, 'signal_count', 0)}, "
          f"+{getattr(fastest, 'signal_co2_g', 0):.0f} g)")

    # 2) Eco route within time budget
    time_budget = (1 + slack) * fastest.time_s
    eco, pareto = find_eco_route(G, source, target, time_budget)
    if include_signals:
        # Re-augment all candidates with signal cost so the comparison is fair
        for r in pareto:
            augment_route_with_signals(G, r, vehicle,
                                        deterministic=deterministic_signals)
        # Re-pick eco among feasible after signal augmentation
        feasible = [r for r in pareto if r.time_s <= time_budget]
        if feasible:
            eco = min(feasible, key=lambda r: r.co2_g)
            eco.name = f'eco_post_signals'
    print(f"  Eco (slack={slack*100:.0f}%, budget={time_budget/60:.1f} min):")
    print(f"    {eco.distance_m/1000:.2f} km, {eco.time_s/60:.1f} min, "
          f"{eco.co2_g:.0f} g CO2 "
          f"(signals: {getattr(eco, 'signal_count', 0)}, "
          f"+{getattr(eco, 'signal_co2_g', 0):.0f} g)")

    return {
        'vehicle': vehicle.name,
        'fastest': fastest,
        'eco':     eco,
        'pareto':  pareto,
    }


def routes_are_identical(r1: Route, r2: Route) -> bool:
    return r1.nodes == r2.nodes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', help='Start lat,lon (e.g. 51.4720,5.5500)')
    ap.add_argument('--end',   help='End lat,lon')
    ap.add_argument('--vehicle1', default='polo',
                    choices=list(VEHICLES.keys()),
                    help='First vehicle (default: polo)')
    ap.add_argument('--vehicle2', default='volvo_fl',
                    choices=list(VEHICLES.keys()),
                    help='Second vehicle (default: volvo_fl truck)')
    ap.add_argument('--slack', type=float, default=0.30,
                    help='Time slack as fraction (default: 0.30 = 30%%)')
    ap.add_argument('--wltp', default='d:/Maastricht University/thesis/codes/data/wltp_class3_reference.csv',
                    help='WLTP reference CSV')
    ap.add_argument('--out', default='d:/Maastricht University/thesis/codes/outputs/step2_routes.json',
                    help='Output JSON summary path')
    ap.add_argument('--od-label', default='',
                    help='Human-readable name for this OD pair (e.g. "Nuenen-Aalst"), '
                         'used by summarize_results.py to label the corridor in graphs')
    ap.add_argument('--synthetic', action='store_true',
                    help='Use the built-in synthetic graph (no OSMnx required)')
    ap.add_argument('--dem', default='synthetic',
                    choices=['ahn3', 'srtm', 'flat', 'synthetic'],
                    help='DEM source for elevation (default: synthetic, '
                         'use ahn3 for real Dutch LIDAR)')
    ap.add_argument('--no-grade', action='store_true',
                    help='Disable grade work (ablation study)')
    ap.add_argument('--no-signals', action='store_true',
                    help='Disable traffic-signal cost (ablation study)')
    ap.add_argument('--deterministic-signals', action='store_true',
                    help='Treat every signal as a guaranteed stop')
    args = ap.parse_args()

    # Load WLTP phases
    print("Loading WLTP phases...")
    wltp_phases = load_wltp_phases(args.wltp)

    # Load the network
    if args.synthetic:
        print("\nUsing synthetic Eindhoven test graph...")
        G, source, target = build_synthetic_network()
    else:
        if not args.start or not args.end:
            ap.error("--start and --end required unless --synthetic")
        s_lat, s_lon = [float(x) for x in args.start.split(',')]
        e_lat, e_lon = [float(x) for x in args.end.split(',')]
        print(f"\nDownloading OSM network for {s_lat},{s_lon} → {e_lat},{e_lon}...")
        G, source, target = load_osmnx_network(s_lat, s_lon, e_lat, e_lon)

    # Annotate elevation
    include_grade = not args.no_grade
    if include_grade:
        print(f"\nAnnotating elevation (DEM source: {args.dem})...")
        if args.dem == 'synthetic':
            dem = SyntheticElevationProvider()
        else:
            dem = ElevationProvider(source=args.dem)
            # Build bbox from graph extents
            lats = [d['y'] for _, d in G.nodes(data=True)]
            lons = [d['x'] for _, d in G.nodes(data=True)]
            dem.prepare_bbox(min(lats), max(lats), min(lons), max(lons))
        add_elevation_to_graph(G, dem)
        elevs = [d.get('elevation_m', float('nan'))
                 for _, d in G.nodes(data=True)]
        valid = [e for e in elevs if e == e]   # non-NaN
        if valid:
            print(f"  Node elevations: {min(valid):.1f}–{max(valid):.1f} m "
                  f"(range {max(valid)-min(valid):.1f} m)")

    # Count signal-tagged nodes, broken down by which OSM tagging pattern
    # matched. This is the LEGACY (node-tag) count — kept for comparison.
    # The new two-stage pipeline (OSM inventory + route match) is what
    # routing actually uses; that count appears per-route below.
    #   traffic_signals  — canonical highway=traffic_signals
    #   crossing_value   — legacy highway=crossing + crossing=traffic_signals
    #   crossing_signals — modern highway=crossing + crossing:signals=yes
    from signal_costs import classify_signal_node
    from collections import Counter
    sig_breakdown = Counter()
    for _, d in G.nodes(data=True):
        kind = classify_signal_node(d)
        if kind is not None:
            sig_breakdown[kind] += 1
    n_sig_nodes = sum(sig_breakdown.values())
    print(f"  Signal nodes in drive graph (legacy detection): {n_sig_nodes}")
    if n_sig_nodes:
        parts = [f"{k}={v}" for k, v in sorted(sig_breakdown.items())]
        print(f"    by pattern: {', '.join(parts)}")

    v1 = VEHICLES[args.vehicle1]
    v2 = VEHICLES[args.vehicle2]

    include_signals = not args.no_signals
    r1 = run_routing(G, source, target, v1, wltp_phases, args.slack, v1.name,
                     include_grade=include_grade,
                     include_signals=include_signals,
                     deterministic_signals=args.deterministic_signals)
    r2 = run_routing(G, source, target, v2, wltp_phases, args.slack, v2.name,
                     include_grade=include_grade,
                     include_signals=include_signals,
                     deterministic_signals=args.deterministic_signals)

    # Compare
    print("\n" + "=" * 90)
    print(f"COMPARISON — slack={args.slack*100:.0f}%, grade={'on' if include_grade else 'off'}, "
          f"signals={'on' if include_signals else 'off'}")
    print("=" * 90)
    hdr = f"{'Route':<40s} {'Distance':>10s} {'Time':>8s} {'CO2 total':>11s} {'  motion':>10s} {'  signals':>10s}"
    print(hdr)
    print("-" * 90)
    for label, route in [
        (f'1. Fastest ({v1.name})',   r1['fastest']),
        (f'2. Eco for {v1.name}',     r1['eco']),
        (f'3. Eco for {v2.name}',     r2['eco']),
    ]:
        motion = getattr(route, 'co2_motion_g', route.co2_g)
        sig = getattr(route, 'signal_co2_g', 0)
        print(f"{label:<40s} {route.distance_m/1000:>9.2f} km {route.time_s/60:>7.1f} m "
              f"{route.co2_g:>10.0f} g {motion:>9.0f} g {sig:>9.0f} g")

    print(f"\nFastest({v1.name}) = Eco({v1.name})?  "
          f"{routes_are_identical(r1['fastest'], r1['eco'])}")
    print(f"Eco({v1.name})     = Eco({v2.name})?  "
          f"{routes_are_identical(r1['eco'], r2['eco'])}")

    # Save JSON summary
    def route_dict(r):
        co2_grade = sum(
            (G[u][v][min(G[u][v], key=lambda k: G[u][v][k].get('time_s', 1e9))]
             .get('co2_grade_g', 0))
            for u, v in zip(r.nodes[:-1], r.nodes[1:])
        ) if not args.synthetic or True else 0
        return {
            'route_nodes':  [str(n) for n in r.nodes],
            'distance_m':   r.distance_m,
            'time_s':       r.time_s,
            'co2_g':        r.co2_g,
            'co2_motion_g': getattr(r, 'co2_motion_g', r.co2_g),
            'co2_grade_g':  round(co2_grade, 1),
            'signal_count': getattr(r, 'signal_count', 0),
            'signal_co2_g': getattr(r, 'signal_co2_g', 0),
        }
    out_data = {
        'config': {
            'od_label':  args.od_label,
            'start':     args.start,
            'end':       args.end,
            'vehicle1': args.vehicle1, 'vehicle2': args.vehicle2,
            'slack': args.slack, 'synthetic': args.synthetic,
            'dem': args.dem, 'include_grade': include_grade,
            'include_signals': include_signals,
            'deterministic_signals': args.deterministic_signals,
        },
        'fastest_v1': route_dict(r1['fastest']),
        'eco_v1':     route_dict(r1['eco']),
        'eco_v2':     route_dict(r2['eco']),
        'identical': {
            'fastest_v1_eq_eco_v1': routes_are_identical(r1['fastest'], r1['eco']),
            'eco_v1_eq_eco_v2':     routes_are_identical(r1['eco'], r2['eco']),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"\nWrote summary to {args.out}")


if __name__ == '__main__':
    main()
