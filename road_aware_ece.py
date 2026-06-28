"""
Road-type-aware ECE percentages: classify each CSV row by the OSM road type
at its (lat, lon), map road type to a WLTP phase, and compute per-segment ECE
percentages against the matching reference.

This module implements the road-coloring methodology from ISO 23795-1's
fleet-management use case: compare real-trip work intensity against the WLTP
phase that matches each road type traversed.

DEPENDENCIES (run on your local machine):
    pip install osmnx pandas numpy

USAGE:
    python road_aware_ece.py \
        --csv 2026_04_18_polo_*.csv \
        --vehicle polo \
        --out per_row_ece.csv

If OSMnx is not available (e.g. no internet), fall back to speed-based
regime classification using the --no-osm flag.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from iso23795_model import (
    Vehicle, VEHICLES, compute_work_per_second, aggregate_trip,
    STANDSTILL_V_THRESH,
)
from wltp_ece import build_phase_references


# =============================================================================
# Road-type → WLTP phase mapping
# =============================================================================
# Mapping derived from typical free-flow speeds for each OSM highway tag in
# urban EU networks, calibrated against WLTP Class 3 phase moving averages
# (LOW 26, MIDDLE 45, HIGH 61, EXTRA-HIGH 94 km/h).
ROAD_TYPE_TO_PHASE = {
    # Highest-speed roads
    'motorway':        'EXTRA-HIGH',
    'motorway_link':   'EXTRA-HIGH',
    'trunk':           'EXTRA-HIGH',
    'trunk_link':      'EXTRA-HIGH',
    # Major arterials
    'primary':         'HIGH',
    'primary_link':    'HIGH',
    # Secondary/tertiary urban
    'secondary':       'MIDDLE',
    'secondary_link':  'MIDDLE',
    'tertiary':        'MIDDLE',
    'tertiary_link':   'MIDDLE',
    # Local urban
    'residential':     'LOW',
    'living_street':   'LOW',
    'service':         'LOW',
    'unclassified':    'LOW',
    'road':            'LOW',          # OSM placeholder when type unknown
    'pedestrian':      'LOW',
    # Track and other low-speed
    'track':           'LOW',
}


def classify_road_with_osmnx(lats: np.ndarray, lons: np.ndarray,
                              buffer_m: float = 200) -> list[str]:
    """Classify each (lat, lon) point by the highway tag of its nearest
    OSM road segment.

    Requires `osmnx` and internet access.

    Parameters
    ----------
    lats, lons : per-row coordinates
    buffer_m : padding around the trace bbox when downloading the network

    Returns
    -------
    List of OSM `highway` tag values (one per point), with 'unknown' for
    points where no road could be matched.
    """
    try:
        import osmnx as ox
    except ImportError:
        raise RuntimeError(
            "osmnx not installed. Run: pip install osmnx\n"
            "Or use --no-osm for the speed-based fallback."
        )

    # Build a bbox around the trace and download the drivable network
    north, south = float(np.max(lats)) + 0.005, float(np.min(lats)) - 0.005
    east,  west  = float(np.max(lons)) + 0.005, float(np.min(lons)) - 0.005
    print(f"  Downloading OSM network for bbox "
          f"[{south:.4f},{west:.4f} — {north:.4f},{east:.4f}]...")

    # OSMnx 2.x API: bbox is (left, bottom, right, top)
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
    print(f"  Network: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Find nearest edge for each (lat, lon) — vectorized
    # nearest_edges returns (u, v, key) tuples
    edges = ox.distance.nearest_edges(G, X=lons, Y=lats)

    # Look up the highway tag of each matched edge
    road_types = []
    for u, v, k in zip(*([e for e in zip(*edges)] if isinstance(edges, tuple)
                          else [edges])):
        try:
            tag = G.edges[u, v, k].get('highway', 'unknown')
            # Some edges have a list of tags; take the most specific one
            if isinstance(tag, list):
                tag = tag[0]
            road_types.append(tag)
        except KeyError:
            road_types.append('unknown')

    return road_types


def classify_road_by_speed(speeds_ms: np.ndarray,
                            window: int = 11) -> list[str]:
    """Speed-based fallback when OSMnx is unavailable.

    Uses a rolling median of speed over a window of rows to estimate the
    road type. Returns OSM-style highway tags so the rest of the pipeline
    treats both methods identically.

    The thresholds are derived from typical free-flow speeds:
        > 80 km/h  → motorway/trunk
        50-80 km/h → primary/secondary
        25-50 km/h → tertiary/residential
        < 25 km/h  → residential/local
    """
    speed_kmh = speeds_ms * 3.6
    rolling = pd.Series(speed_kmh).rolling(window, center=True, min_periods=1).median()

    types = []
    for v_kmh in rolling:
        if v_kmh > 80:
            types.append('motorway')
        elif v_kmh > 50:
            types.append('primary')
        elif v_kmh > 25:
            types.append('secondary')
        elif v_kmh > 5:
            types.append('residential')
        else:
            types.append('residential')   # standstill or near-standstill
    return types


# =============================================================================
# Per-row and per-segment ECE
# =============================================================================
def compute_per_row_ece(per_sec: pd.DataFrame, vehicle: Vehicle,
                         road_types: list[str], wltp_csv: str) -> pd.DataFrame:
    """For each row of a real trip, compute ECE percentages against the WLTP
    phase that matches the row's road type.

    ECE per row is computed against the matching phase's average per-km
    work intensity, so the result is dimensionally a percentage:
        AccECE_pct = (per_row_acc_pos_J / row_distance_m * 1000) /
                     (phase_acc_pos_J / phase_distance_km) * 100
    """
    # Map road types → WLTP phases
    phases = [ROAD_TYPE_TO_PHASE.get(t, 'LOW') for t in road_types]

    # Build per-phase reference work intensities (J per km, by component)
    refs = build_phase_references(vehicle, wltp_csv)
    # refs[phase] is a PhaseReference dataclass with aero/roll/acc per km

    df = per_sec.copy()
    df['road_type'] = road_types
    df['phase'] = phases

    # For each row compute per-km equivalents, then divide by phase reference
    # (only for moving rows; standstill rows get ECE = NaN)
    moving = df['speed_ms'] >= STANDSTILL_V_THRESH

    df['AccECE_pct']  = np.nan
    df['AeroECE_pct'] = np.nan
    df['RollECE_pct'] = np.nan
    df['WorkECE_pct'] = np.nan

    for idx in df[moving].index:
        ph = df.at[idx, 'phase']
        ref = refs[ph]
        d_m = df.at[idx, 'distance_m']
        if d_m <= 0:
            continue
        # Per-km row work
        acc_pos = max(0, df.at[idx, 'AccWork_J'])
        aero    = max(0, df.at[idx, 'AeroWork_J'])
        roll    = max(0, df.at[idx, 'RollWork_J'])
        total   = max(0, df.at[idx, 'TotalWork_J'])

        # divide by 1m to get per-meter, multiply by 1000 to get per-km
        # (equivalent to multiplying by 1000/d_m)
        acc_per_km  = acc_pos / d_m * 1000
        aero_per_km = aero    / d_m * 1000
        roll_per_km = roll    / d_m * 1000
        work_per_km = total   / d_m * 1000

        df.at[idx, 'AccECE_pct']  = acc_per_km  / ref.acc_pos_J_per_km  * 100 if ref.acc_pos_J_per_km else np.nan
        df.at[idx, 'AeroECE_pct'] = aero_per_km / ref.aero_J_per_km     * 100 if ref.aero_J_per_km    else np.nan
        df.at[idx, 'RollECE_pct'] = roll_per_km / ref.roll_J_per_km     * 100 if ref.roll_J_per_km    else np.nan
        df.at[idx, 'WorkECE_pct'] = work_per_km / ref.total_motion_J_per_km * 100 if ref.total_motion_J_per_km else np.nan

    return df


def aggregate_to_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive rows on the same road type into segments and compute
    per-segment ECE percentages and color buckets.

    Color buckets follow conventional eco-driving feedback thresholds:
        green   : WorkECE < 90%   (much more efficient than reference)
        yellow  : 90% ≤ WorkECE < 110%
        orange  : 110% ≤ WorkECE < 130%
        red     : WorkECE ≥ 130%  (much less efficient)
    These are NOT specified by the standard; they are a conventional
    visualization choice. Document this in the thesis methodology.
    """
    # Identify segment boundaries (where road_type changes)
    df = df.copy().reset_index(drop=True)
    df['segment_id'] = (df['road_type'] != df['road_type'].shift()).cumsum()

    segments = []
    for seg_id, sub in df.groupby('segment_id'):
        d_total = sub['distance_m'].sum()
        if d_total <= 0:
            continue
        acc_pos  = sub.loc[sub['AccWork_J']  > 0, 'AccWork_J'].sum()
        aero_tot = sub.loc[sub['AeroWork_J'] > 0, 'AeroWork_J'].sum()
        roll_tot = sub.loc[sub['RollWork_J'] > 0, 'RollWork_J'].sum()
        work_tot = sub['TotalWork_J'].sum()

        segments.append({
            'segment_id':   int(seg_id),
            'road_type':    sub['road_type'].iloc[0],
            'phase':        sub['phase'].iloc[0],
            'rows':         len(sub),
            'duration_s':   sub['dt_s'].sum(),
            'distance_m':   d_total,
            'avg_speed_kmh': (sub['speed_ms'] * sub['dt_s']).sum() / sub['dt_s'].sum() * 3.6,
            'AccECE_pct':   sub['AccECE_pct'].mean(),
            'AeroECE_pct':  sub['AeroECE_pct'].mean(),
            'WorkECE_pct':  sub['WorkECE_pct'].mean(),
        })

    seg_df = pd.DataFrame(segments)

    # Color bucket
    def bucket(pct):
        if pd.isna(pct):
            return 'gray'
        if pct < 90:   return 'green'
        if pct < 110:  return 'yellow'
        if pct < 130:  return 'orange'
        return 'red'
    seg_df['color'] = seg_df['WorkECE_pct'].apply(bucket)

    return seg_df


# =============================================================================
# Main entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', required=True, help='LCMM CSV path')
    ap.add_argument('--vehicle', default='polo',
                    choices=list(VEHICLES.keys()),
                    help='Vehicle config to use (default: polo)')
    ap.add_argument('--wltp', default='wltp_class3_reference.csv',
                    help='WLTP reference cycle CSV')
    ap.add_argument('--out-rows', default='ece_per_row.csv',
                    help='Output CSV for per-row classification + ECE')
    ap.add_argument('--out-segments', default='ece_per_segment.csv',
                    help='Output CSV for per-segment summaries')
    ap.add_argument('--no-osm', action='store_true',
                    help='Use speed-based regime classification instead of OSMnx')
    args = ap.parse_args()

    print(f"Loading {args.csv}...")
    csv = pd.read_csv(args.csv)
    n = len(csv)
    print(f"  {n} rows")

    vehicle = VEHICLES[args.vehicle]
    print(f"Vehicle: {vehicle.name}")

    # Classify road types per row
    if args.no_osm:
        print("\nClassifying road types by speed (no OSMnx)...")
        road_types = classify_road_by_speed(csv['Speed[m/s]'].values)
    else:
        print("\nClassifying road types via OSMnx...")
        road_types = classify_road_with_osmnx(
            csv['Latitude'].values, csv['Longitude'].values
        )

    # Show road-type distribution
    rt_counts = pd.Series(road_types).value_counts()
    print(f"\nRoad-type distribution:")
    for t, c in rt_counts.items():
        ph = ROAD_TYPE_TO_PHASE.get(t, '?')
        print(f"  {t:18s} → {ph:12s}  ({c} rows, {c/n*100:.1f} %)")

    # Run the energy model
    print("\nComputing per-row work components...")
    per_sec = compute_work_per_second(
        speed_ms=csv['Speed[m/s]'].values,
        altitude_m=csv['Altitude[m]'].values,
        distance_m=csv['Distance'].values,
        time_ms=csv['Time[ms]'].values,
        vehicle=vehicle,
    )

    # Per-row ECE
    per_row = compute_per_row_ece(per_sec, vehicle, road_types, args.wltp)
    per_row['Latitude']  = csv['Latitude'].values
    per_row['Longitude'] = csv['Longitude'].values
    per_row['Time_ms']   = csv['Time[ms]'].values
    # Add segment_id (consecutive rows with same road_type)
    per_row['segment_id'] = (per_row['road_type'] != per_row['road_type'].shift()).cumsum()
    per_row.to_csv(args.out_rows, index=False)
    print(f"\nWrote per-row results to {args.out_rows}")

    # Per-segment aggregation
    segments = aggregate_to_segments(per_row)
    segments.to_csv(args.out_segments, index=False)
    print(f"Wrote per-segment results to {args.out_segments}\n")

    # Print segment summary
    print(f"=== Per-segment summary ===")
    print(segments[['segment_id', 'road_type', 'phase', 'distance_m',
                    'avg_speed_kmh', 'WorkECE_pct', 'color']].to_string(index=False))

    # Distance share by color
    print(f"\n=== Trip distance by color bucket ===")
    color_dist = segments.groupby('color')['distance_m'].sum()
    total_dist = color_dist.sum()
    for c, d in color_dist.items():
        print(f"  {c:8s}: {d:7.0f} m ({d/total_dist*100:5.1f} %)")


if __name__ == '__main__':
    main()
