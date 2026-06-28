r"""
Batch validation study: run vehicle-aware eco-routing over many OD pairs and
aggregate the results to test H1/H2 robustness across a stratified sample.

Reads od_pairs_nl.csv (region, label, start, end, dem, strata...), runs the
routing engine for each pair at a fixed slack (and optionally a slack sweep),
and produces:

  batch_results.csv     one row per (pair x vehicle x route), all numbers
  batch_summary.csv     one row per pair: divergence flags, savings, strata
  divergence_by_stratum.png   divergence rate per region / relief / network
  savings_by_stratum.png      mean CO2 saving per stratum, per vehicle
  savings_vs_elevation.png    CO2 saving vs route elevation gain (H2 driver)
  savings_vs_signals.png      CO2 saving vs signal count (H2 driver)

This calls the routing functions in-process (imported from step2_routing),
loading OSM + DEM once per pair, so it is much faster than shelling out.

Usage:
    py run_batch_study.py --od-csv od_pairs_nl.csv --slack 0.50 \
        --out-dir "..\results\batch_study" --wltp "data\wltp_class3_reference.csv"

    # quick subset while testing:
    py run_batch_study.py --od-csv od_pairs_nl.csv --limit 3 ...

    # only one region:
    py run_batch_study.py --od-csv od_pairs_nl.csv --region maastricht ...
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from step2_routing import (
    load_wltp_phases, load_osmnx_network, annotate_graph_costs,
    shortest_path_by, find_eco_route, augment_route_with_signals,
    routes_are_identical, VEHICLES,
)
from elevation import (
    ElevationProvider, SyntheticElevationProvider, add_elevation_to_graph,
)


# =============================================================================
# Region-level graph reuse
# =============================================================================
# The single biggest cause of Overpass throttling is downloading a separate
# bbox per OD pair. Pairs within a region overlap heavily, so we instead
# download ONE bbox per region that spans all of its pairs, then reuse that
# graph for every pair in the region. For the NL study this cuts Overpass
# graph+signal calls from ~100 (52 pairs x 2) to ~12 (6 regions x 2).
#
# Regional graphs are cached to disk as GraphML so a rerun loads them
# locally with zero Overpass calls. We control the tag set explicitly when
# saving, so (unlike OSMnx's own cache) there is no silent missing-tag bug.
_REGION_GRAPH_CACHE: dict = {}   # region -> (G, nodes_index) in memory


def _region_bbox(region_rows, margin_deg=0.01):
    """Bbox spanning every endpoint of every pair in a region, plus margin."""
    lats, lons = [], []
    for _, row in region_rows.iterrows():
        s_lat, s_lon = [float(x) for x in row['start'].split()]
        e_lat, e_lon = [float(x) for x in row['end'].split()]
        lats += [s_lat, e_lat]; lons += [s_lon, e_lon]
    south = min(lats) - margin_deg; north = max(lats) + margin_deg
    west  = min(lons) - margin_deg; east  = max(lons) + margin_deg
    return south, north, west, east


def get_region_graph(region, region_rows, cache_dir, attempts=4, base_wait=30):
    """Return a drivable graph spanning all pairs in `region`, cached in
    memory and on disk. Downloads from Overpass only on a cold cache,
    with progressive backoff on transient failures."""
    if region in _REGION_GRAPH_CACHE:
        return _REGION_GRAPH_CACHE[region]

    import osmnx as ox
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    south, north, west, east = _region_bbox(region_rows)

    safe_region = ''.join(c if c.isalnum() else '_' for c in str(region))
    graphml = cache_dir / f"region_{safe_region}.graphml"

    if graphml.exists():
        print(f"  [region graph] loading {region} from disk cache")
        G = ox.load_graphml(graphml)
        # Coords come back as strings from GraphML — restore floats.
        for _, d in G.nodes(data=True):
            if 'x' in d: d['x'] = float(d['x'])
            if 'y' in d: d['y'] = float(d['y'])
        G.graph['request_bbox'] = (south, north, west, east)
        _REGION_GRAPH_CACHE[region] = G
        return G

    # Cold cache: download once, with retry/backoff.
    desired = ['highway', 'junction', 'railway', 'ref',
               'crossing', 'crossing:signals', 'traffic_signals']
    ox.settings.useful_tags_node = list(set(ox.settings.useful_tags_node + desired))
    # IMPORTANT: do NOT let OSMnx's own cache interfere; we manage our own.
    ox.settings.use_cache = False

    last_err = None
    for i in range(attempts):
        try:
            print(f"  [region graph] downloading {region} bbox "
                  f"[{south:.3f},{west:.3f}]→[{north:.3f},{east:.3f}] "
                  f"(spans {len(region_rows)} pairs)...")
            G = ox.graph_from_bbox(bbox=(west, south, east, north),
                                   network_type='drive')
            print(f"    {len(G.nodes)} nodes, {len(G.edges)} edges")
            G.graph['request_bbox'] = (south, north, west, east)
            try:
                ox.save_graphml(G, graphml)
                print(f"    cached -> {graphml.name}")
            except Exception as e:
                print(f"    (graph cache write failed: {e})")
            _REGION_GRAPH_CACHE[region] = G
            return G
        except Exception as ex:
            msg = str(ex).lower()
            transient = any(t in msg for t in (
                'timed out', 'timeout', 'connection', 'max retries',
                'remote end closed', '429', 'too many requests'))
            last_err = ex
            if not transient or i == attempts - 1:
                raise
            wait = base_wait * (2 ** i)
            print(f"    Overpass busy (attempt {i+1}/{attempts}); "
                  f"waiting {wait}s...")
            time.sleep(wait)
    raise last_err


def read_od_csv(path: Path) -> pd.DataFrame:
    """Read the OD CSV, skipping comment lines starting with #."""
    rows = []
    with open(path, encoding='utf-8') as f:
        header = None
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = [p.strip() for p in s.split(',')]
            if header is None:
                header = parts
                continue
            # pad/truncate to header length
            parts = (parts + [''] * len(header))[:len(header)]
            rows.append(dict(zip(header, parts)))
    return pd.DataFrame(rows)


def configure_osmnx_for_batch(overpass_endpoint=None):
    """Tune OSMnx for a long batch run against a public Overpass server.

    - Re-enable caching: a resumed run then skips already-downloaded networks,
      drastically cutting Overpass calls.
    - Longer timeout and retries ride out transient throttling.
    - Optionally switch to a less-congested Overpass mirror. The default
      server (overpass-api.de) throttles bursts hard; mirrors such as
      Kumi Systems are often far faster when the main one is busy.
    """
    try:
        import osmnx as ox
    except ImportError:
        return
    ox.settings.use_cache = True
    ox.settings.cache_folder = './cache_osmnx'
    desired = ['highway', 'junction', 'railway', 'ref',
               'crossing', 'crossing:signals', 'traffic_signals']
    ox.settings.useful_tags_node = list(set(ox.settings.useful_tags_node + desired))
    try:
        ox.settings.requests_timeout = 300
    except Exception:
        pass
    if overpass_endpoint:
        # Different OSMnx versions name this differently; set whichever exists.
        for attr in ('overpass_url', 'overpass_endpoint'):
            if hasattr(ox.settings, attr):
                setattr(ox.settings, attr, overpass_endpoint)
                print(f"  Using Overpass endpoint: {overpass_endpoint}")
                break


def load_network_with_retry(s_lat, s_lon, e_lat, e_lon,
                            attempts=4, base_wait=30):
    """Call load_osmnx_network, retrying on Overpass connection failures.

    The public Overpass server (overpass-api.de) throttles bursts of requests;
    a timed-out connection usually clears after a short wait. We back off
    progressively (30s, 60s, 120s, ...) rather than abandoning the pair.
    """
    last_err = None
    for i in range(attempts):
        try:
            return load_osmnx_network(s_lat, s_lon, e_lat, e_lon)
        except Exception as ex:
            msg = str(ex).lower()
            transient = ('timed out' in msg or 'timeout' in msg
                         or 'connection' in msg or 'max retries' in msg
                         or 'remote end closed' in msg or '429' in msg
                         or 'too many requests' in msg)
            last_err = ex
            if not transient or i == attempts - 1:
                raise
            wait = base_wait * (2 ** i)
            print(f"    Overpass busy (attempt {i+1}/{attempts}); "
                  f"waiting {wait}s and retrying...")
            time.sleep(wait)
    raise last_err


def route_breakdown(G, route):
    """Sum motion / grade / signal CO2 along a route's nodes."""
    motion = 0.0; grade = 0.0
    for u, v in zip(route.nodes[:-1], route.nodes[1:]):
        e = min(G[u][v].values(), key=lambda d: d.get('time_s', 1e9))
        grade  += e.get('co2_grade_g', 0)
        motion += e.get('co2_motion_g', 0) - e.get('co2_grade_g', 0)
    return motion, grade, getattr(route, 'signal_co2_g', 0)


def run_one_pair(row, wltp_phases, vehicle1, vehicle2, slack,
                 include_signals, fallback_dem_for_srtm, G):
    """Run routing for a single OD pair on a pre-loaded regional graph G.
    Returns (per_route_rows, summary_row)."""
    import osmnx as ox
    s_lat, s_lon = [float(x) for x in row['start'].split()]
    e_lat, e_lon = [float(x) for x in row['end'].split()]
    dem_kind = row.get('dem', 'flat') or 'flat'

    # Locate this pair's endpoints within the shared regional graph.
    src = ox.distance.nearest_nodes(G, s_lon, s_lat)
    dst = ox.distance.nearest_nodes(G, e_lon, e_lat)

    # Elevation. Use THIS PAIR's bbox (not the whole regional graph extent),
    # both because AHN can't serve a region-sized box and because we only
    # need elevation along this pair's corridor.
    try:
        if dem_kind == 'synthetic':
            dem = SyntheticElevationProvider()
            add_elevation_to_graph(G, dem)
        elif dem_kind == 'flat':
            pass  # no elevation cost
        else:
            # CRITICAL: the elevation raster must cover the nodes we actually
            # route over. We route on the REGION graph, whose nodes span the
            # whole region - not just this pair's corridor. Covering only the
            # pair's bbox leaves most region-graph nodes with NaN elevation
            # (grade silently zeroed), suppressing the grade signal that
            # drives heavy-vehicle route choice. So BOTH sources now cover
            # the FULL graph extent:
            #   - SRTM: no request-size cap, always safe.
            #   - AHN: WCS caps at ~25 km per side. The OD CSV assigns srtm
            #     to every region whose extent exceeds the cap (verified:
            #     rotterdam, utrecht, maastricht), so remaining ahn3 regions
            #     (amsterdam, denhaag, eindhoven) fit. If AHN still rejects,
            #     we retry SRTM full-extent (validated path) before flat.
            _lats = [d['y'] for _, d in G.nodes(data=True) if 'y' in d]
            _lons = [d['x'] for _, d in G.nodes(data=True) if 'x' in d]
            _bbox = (min(_lats), max(_lats), min(_lons), max(_lons))
            try:
                dem = ElevationProvider(source=dem_kind)
                dem.prepare_bbox(*_bbox)
                add_elevation_to_graph(G, dem)
            except Exception as ex_ahn:
                if dem_kind == 'ahn3':
                    print(f"    AHN full-extent failed ({str(ex_ahn)[:90]}); "
                          f"retrying SRTM full-extent...")
                    dem = ElevationProvider(source='srtm')
                    dem.prepare_bbox(*_bbox)
                    add_elevation_to_graph(G, dem)
                    dem_kind = 'srtm (ahn3 fallback)'
                else:
                    raise
    except Exception as ex:
        # DEM is non-essential: a missing/oversized DEM means "no grade cost",
        # which is the correct behaviour for flat NL corridors anyway. Rather
        # than fail the whole pair (and waste the OSM download we just did),
        # degrade to flat and record that grade was skipped.
        #   - SRTM failures fall back only if explicitly allowed (key issues).
        #   - AHN failures (e.g. corridor too long for the WCS) always fall
        #     back, since AHN precision is wasted at that scale anyway.
        if dem_kind == 'ahn3' or (dem_kind == 'srtm' and fallback_dem_for_srtm):
            print(f"    DEM '{dem_kind}' unavailable ({str(ex)[:120]}); "
                  f"continuing with flat (no grade) for this pair.")
            dem_kind = 'flat (DEM failed)'
        else:
            raise

    per_route = []
    summary = {
        'region':   row.get('region', ''),
        'od_label': row.get('od_label', ''),
        'dem':      dem_kind,
        'relief':   row.get('stratum_region_relief', ''),
        'length_stratum':  row.get('stratum_length', ''),
        'network':  row.get('stratum_network', ''),
        'slack':    slack,
    }

    veh_routes = {}
    for vkey, veh in [('v1', vehicle1), ('v2', vehicle2)]:
        annotate_graph_costs(G, veh, wltp_phases, include_grade=True)
        fastest = shortest_path_by(G, src, dst, weight='time_s')
        fastest.name = 'fastest'
        budget = (1 + slack) * fastest.time_s
        eco, pareto = find_eco_route(G, src, dst, budget)
        if include_signals:
            augment_route_with_signals(G, fastest, veh)
            for r in pareto:
                augment_route_with_signals(G, r, veh)
            feasible = [r for r in pareto if r.time_s <= budget]
            if feasible:
                eco = min(feasible, key=lambda r: r.co2_g)
        eco.name = 'eco'
        veh_routes[vkey] = {'fastest': fastest, 'eco': eco, 'veh': veh}

        for rkey, r in [('fastest', fastest), ('eco', eco)]:
            m, g, sig = route_breakdown(G, r)
            per_route.append({
                'region':   row.get('region', ''),
                'od_label': row.get('od_label', ''),
                'vehicle':  veh.name,
                'route':    rkey,
                'distance_km': r.distance_m / 1000,
                'time_min':    r.time_s / 60,
                'co2_g':       r.co2_g,
                'co2_motion_g': m,
                'co2_grade_g':  g,
                'co2_signal_g': sig,
                'signal_count': getattr(r, 'signal_count', 0),
            })

    # Summary: divergence + savings
    f1 = veh_routes['v1']['fastest']; e1 = veh_routes['v1']['eco']
    f2 = veh_routes['v2']['fastest']; e2 = veh_routes['v2']['eco']
    summary['v1'] = vehicle1.name
    summary['v2'] = vehicle2.name
    summary['fastest_eq_eco_v1'] = routes_are_identical(f1, e1)
    summary['fastest_eq_eco_v2'] = routes_are_identical(f2, e2)
    summary['eco_v1_eq_eco_v2']  = routes_are_identical(e1, e2)
    summary['v1_saving_pct'] = (f1.co2_g - e1.co2_g) / f1.co2_g * 100 if f1.co2_g else 0
    summary['v2_saving_pct'] = (f2.co2_g - e2.co2_g) / f2.co2_g * 100 if f2.co2_g else 0
    summary['eco_v1_co2_g'] = e1.co2_g
    summary['eco_v2_co2_g'] = e2.co2_g
    summary['eco_v1_grade_g'] = route_breakdown(G, e1)[1]
    summary['eco_v2_grade_g'] = route_breakdown(G, e2)[1]
    summary['eco_v1_signals'] = getattr(e1, 'signal_count', 0)
    summary['eco_v2_signals'] = getattr(e2, 'signal_count', 0)
    summary['eco_v1_dist_km'] = e1.distance_m / 1000
    summary['routes_diverge'] = not summary['eco_v1_eq_eco_v2']
    return per_route, summary


def make_plots(summary_df, out_dir):
    if summary_df.empty:
        return

    # 1. Divergence rate per stratum (region, relief, network)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in [
        (axes[0], 'region', 'by region'),
        (axes[1], 'relief', 'by relief'),
        (axes[2], 'network', 'by network type'),
    ]:
        grp = summary_df.groupby(col)['routes_diverge'].mean() * 100
        grp.plot(kind='bar', ax=ax, color='#2C5F2D', edgecolor='black')
        ax.set_ylabel('Routes diverge (% of pairs)')
        ax.set_title(f'Eco-route divergence {title}')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'divergence_by_stratum.png', dpi=140, bbox_inches='tight')
    plt.close()

    # 2. Mean CO2 saving per stratum, both vehicles
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in [(axes[0], 'region'), (axes[1], 'network')]:
        g1 = summary_df.groupby(col)['v1_saving_pct'].mean()
        g2 = summary_df.groupby(col)['v2_saving_pct'].mean()
        x = np.arange(len(g1)); w = 0.38
        ax.bar(x - w/2, g1.values, w, label=summary_df['v1'].iloc[0], color='#2C5F2D', edgecolor='black')
        ax.bar(x + w/2, g2.values, w, label=summary_df['v2'].iloc[0], color='#C73E1D', edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(g1.index, rotation=30, ha='right')
        ax.set_ylabel('Mean CO\u2082 saving (%)')
        ax.set_title(f'Eco-routing saving by {col}')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'savings_by_stratum.png', dpi=140, bbox_inches='tight')
    plt.close()

    # 3. Saving vs elevation gain (the heavy vehicle should react more)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary_df['eco_v1_grade_g'], summary_df['v1_saving_pct'],
               color='#2C5F2D', label=summary_df['v1'].iloc[0], s=60, edgecolors='black')
    ax.scatter(summary_df['eco_v2_grade_g'], summary_df['v2_saving_pct'],
               color='#C73E1D', label=summary_df['v2'].iloc[0], s=60, edgecolors='black', marker='s')
    ax.set_xlabel('Eco-route grade CO\u2082 (g)')
    ax.set_ylabel('CO\u2082 saving vs fastest (%)')
    ax.set_title('Eco-routing saving vs grade cost')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'savings_vs_elevation.png', dpi=140, bbox_inches='tight')
    plt.close()

    # 4. Saving vs signal count
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary_df['eco_v1_signals'], summary_df['v1_saving_pct'],
               color='#2C5F2D', label=summary_df['v1'].iloc[0], s=60, edgecolors='black')
    ax.scatter(summary_df['eco_v2_signals'], summary_df['v2_saving_pct'],
               color='#C73E1D', label=summary_df['v2'].iloc[0], s=60, edgecolors='black', marker='s')
    ax.set_xlabel('Signals on eco-route')
    ax.set_ylabel('CO\u2082 saving vs fastest (%)')
    ax.set_title('Eco-routing saving vs signal count')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'savings_vs_signals.png', dpi=140, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--od-csv', required=True)
    ap.add_argument('--slack', type=float, default=0.50)
    ap.add_argument('--vehicle1', default='polo', choices=list(VEHICLES.keys()))
    ap.add_argument('--vehicle2', default='volvo_fl', choices=list(VEHICLES.keys()))
    ap.add_argument('--wltp', default=r'data\wltp_class3_reference.csv')
    ap.add_argument('--out-dir', default=r'..\results\batch_study')
    ap.add_argument('--no-signals', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='Only run first N pairs (testing)')
    ap.add_argument('--region', default='', help='Only run pairs from this region')
    ap.add_argument('--fallback-flat-if-srtm-fails', action='store_true',
                    help='If SRTM download fails, continue with no grade instead of crashing')
    ap.add_argument('--pair-delay', type=float, default=8.0,
                    help='Seconds to wait between pairs, to stay under the '
                         'Overpass rate limit (default 8)')
    ap.add_argument('--overpass', default='',
                    help='Override Overpass endpoint. Try a mirror if the '
                         'default server is throttling, e.g. '
                         'https://overpass.kumi.systems/api/interpreter')
    ap.add_argument('--fresh', action='store_true',
                    help='Ignore any previous progress and rerun every pair '
                         '(default is to resume, skipping completed pairs)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    od = read_od_csv(Path(args.od_csv))
    if args.region:
        od = od[od['region'] == args.region]
    if args.limit:
        od = od.head(args.limit)

    # ---- Resume support ------------------------------------------------
    # On rerun we reload prior results and skip pairs that already
    # succeeded. This is what makes a throttled/interrupted run safe to
    # restart: only the pairs that never completed re-hit Overpass, and
    # those are also served from the OSMnx + signal disk caches if they
    # got far enough last time. Use --fresh to override.
    summary_csv = out_dir / 'batch_summary.csv'
    routes_csv  = out_dir / 'batch_results.csv'
    all_routes = []
    all_summaries = []
    done_labels = set()
    if not args.fresh and summary_csv.exists():
        try:
            prev_sum = pd.read_csv(summary_csv)
            all_summaries = prev_sum.to_dict('records')
            done_labels = set(prev_sum['od_label'].astype(str))
            if routes_csv.exists():
                all_routes = pd.read_csv(routes_csv).to_dict('records')
            print(f"Resuming: {len(done_labels)} pair(s) already complete "
                  f"(use --fresh to rerun all)")
        except Exception as e:
            print(f"Could not read prior progress ({e}); starting fresh")

    print(f"Running {len(od)} OD pairs at slack={args.slack}...")

    configure_osmnx_for_batch(args.overpass or None)
    wltp_phases = load_wltp_phases(args.wltp)
    v1 = VEHICLES[args.vehicle1]; v2 = VEHICLES[args.vehicle2]
    include_signals = not args.no_signals

    failures = []
    did_network_call = False   # whether we actually hit Overpass this loop
    graph_cache_dir = Path('./cache_region_graphs')

    # Process region by region so each region's graph is downloaded once and
    # reused across all its pairs. Within a region, the first uncached pair
    # triggers the (single) Overpass download; the rest are local.
    od = od.reset_index(drop=True)
    regions_in_order = list(dict.fromkeys(od['region'].tolist()))
    pair_counter = 0
    for region in regions_in_order:
        region_rows = od[od['region'] == region]
        # Skip the whole region if every pair in it is already done.
        remaining = [str(r.get('od_label')) for _, r in region_rows.iterrows()
                     if str(r.get('od_label')) not in done_labels]
        if not remaining:
            print(f"\n=== Region '{region}': all {len(region_rows)} pairs "
                  f"already done, skipping ===")
            pair_counter += len(region_rows)
            continue

        print(f"\n=== Region '{region}': {len(remaining)} pair(s) to run ===")
        # Load (or download) the regional graph ONCE.
        try:
            G_region = get_region_graph(region, region_rows, graph_cache_dir)
            if region not in _REGION_GRAPH_CACHE or True:
                # We only count a network call if it wasn't a disk-cache load;
                # get_region_graph prints which path it took.
                pass
        except Exception as ex:
            print(f"  Region '{region}' graph download FAILED: {ex}")
            for _, row in region_rows.iterrows():
                lbl = str(row.get('od_label'))
                if lbl not in done_labels:
                    failures.append({'od_label': lbl,
                                     'error': f'region graph failed: {ex}'})
            pair_counter += len(region_rows)
            # Save failures and move on — a later rerun retries this region.
            if failures:
                pd.DataFrame(failures).to_csv(out_dir / 'failures.csv', index=False)
            time.sleep(args.pair_delay if args.pair_delay else 0)
            continue

        for _, row in region_rows.iterrows():
            pair_counter += 1
            i = pair_counter
            label = str(row.get('od_label', f'pair{i}'))
            if label in done_labels:
                print(f"\n[{i}/{len(od)}] {label}  — already done, skipping")
                continue
            print(f"\n[{i}/{len(od)}] {label}  "
                  f"({row.get('region')}, dem={row.get('dem')})")
            t0 = time.time()
            try:
                per_route, summary = run_one_pair(
                    row, wltp_phases, v1, v2, args.slack, include_signals,
                    args.fallback_flat_if_srtm_fails, G_region,
                )
                all_routes.extend(per_route)
                all_summaries.append(summary)
                done_labels.add(label)
                dt = time.time() - t0
                print(f"    done in {dt:.0f}s | "
                      f"diverge={summary['routes_diverge']} | "
                      f"saving v1={summary['v1_saving_pct']:.1f}% "
                      f"v2={summary['v2_saving_pct']:.1f}%")
            except Exception as ex:
                print(f"    FAILED: {ex}")
                failures.append({'od_label': label, 'error': str(ex)})
                traceback.print_exc()
            # Save incrementally so a crash mid-run loses nothing.
            if all_summaries:
                pd.DataFrame(all_summaries).to_csv(summary_csv, index=False)
            if all_routes:
                pd.DataFrame(all_routes).to_csv(routes_csv, index=False)
            if failures:
                pd.DataFrame(failures).to_csv(out_dir / 'failures.csv', index=False)
        # end pairs in region
    # end regions

    summary_df = pd.DataFrame(all_summaries)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / 'failures.csv', index=False)
        print(f"\n{len(failures)} pair(s) failed — see failures.csv")

    if summary_df.empty:
        print("No successful pairs; nothing to plot.")
        return

    make_plots(summary_df, out_dir)

    # ---- Console headline numbers -------------------------------------
    n = len(summary_df)
    div_rate = summary_df['routes_diverge'].mean() * 100
    print("\n" + "=" * 64)
    print("BATCH STUDY SUMMARY")
    print("=" * 64)
    print(f"Pairs run: {n}")
    print(f"Eco-route differs between {v1.name} and {v2.name}: "
          f"{summary_df['routes_diverge'].sum()}/{n} pairs ({div_rate:.0f}%)")
    print(f"Mean CO2 saving (eco vs fastest): "
          f"{v1.name} {summary_df['v1_saving_pct'].mean():.1f}%, "
          f"{v2.name} {summary_df['v2_saving_pct'].mean():.1f}%")
    print("\nDivergence rate by relief stratum:")
    print((summary_df.groupby('relief')['routes_diverge'].mean()*100).round(0).to_string())
    print("\nDivergence rate by network stratum:")
    print((summary_df.groupby('network')['routes_diverge'].mean()*100).round(0).to_string())
    print(f"\nOutputs in {out_dir}")


if __name__ == '__main__':
    main()
