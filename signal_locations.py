"""
Visualize OSM-tagged traffic signals along a route, for ground-truth comparison
against Google Maps Street View.

Why this exists
===============
OSM signal coverage is incomplete: some real-world signals aren't tagged, and
some OSM-tagged 'traffic_signals' are pedestrian-only or decommissioned. The
thesis must disclose this. This script:

  1. Loads an OSMnx network for an OD pair.
  2. Computes a route (default: fastest by time).
  3. Plots the route on a real map (folium HTML + matplotlib PNG) with
     every signal-tagged node along the route highlighted in red.
  4. Writes a CSV of signal locations (lat, lon) along the route, so you can
     paste each into Google Maps to verify presence.

You then manually check on Google Maps (Street View or Satellite) and produce
a coverage table:
    - OSM signals confirmed real
    - OSM signals NOT real (pedestrian-only, decommissioned, false positive)
    - Real signals MISSED by OSM (false negatives)
    → coverage = TP / (TP + FN)

This validation is genuinely useful for the thesis methodology section.

Dependencies (run on your machine):
    pip install osmnx folium matplotlib pandas

Usage
-----
    python signal_locations.py \\
        --start "51.4720,5.5500" \\
        --end   "51.3830,5.4400" \\
        --route-method fastest \\
        --out-dir ./signal_check/

Outputs
-------
    signal_check/
        signals.csv               lat,lon,osm_node_id per signal on route
        route_with_signals.png    matplotlib static figure
        route_with_signals.html   interactive folium map (open in browser,
                                  click signals to copy lat/lon for Google Maps)
        verification_template.csv pre-formatted for your manual check
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def load_network(start_lat, start_lon, end_lat, end_lon, margin_deg=0.01):
    """Download a drivable OSMnx subgraph spanning both endpoints."""
    try:
        import osmnx as ox
    except ImportError:
        raise RuntimeError("pip install osmnx")

    # Retain signal-related sub-tags so we can detect all three OSM tagging
    # patterns:
    #   - 'crossing'           — legacy crossing=traffic_signals|pelican|...
    #   - 'crossing:signals'   — modern crossing:signals=yes (post-2020)
    #   - 'traffic_signals'    — sub-tag to exclude blinker/emergency
    # Without these, signal_costs.classify_signal_node() cannot fire all
    # branches and visualization will under-report coverage.
    desired = ['highway', 'junction', 'railway', 'ref',
               'crossing', 'crossing:signals', 'traffic_signals']
    ox.settings.useful_tags_node = list(set(ox.settings.useful_tags_node + desired))

    # Disable cache to avoid stale-graph bugs when tag settings change
    # (see step2_routing.load_osmnx_network for the full explanation).
    ox.settings.use_cache = False

    south = min(start_lat, end_lat) - margin_deg
    north = max(start_lat, end_lat) + margin_deg
    west  = min(start_lon, end_lon) - margin_deg
    east  = max(start_lon, end_lon) + margin_deg

    print(f"Downloading OSM network for bbox "
          f"[{south:.4f}, {west:.4f}] → [{north:.4f}, {east:.4f}]...")
    G = ox.graph_from_bbox(bbox=(west, south, east, north),
                            network_type='drive')
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    src = ox.distance.nearest_nodes(G, start_lon, start_lat)
    dst = ox.distance.nearest_nodes(G, end_lon, end_lat)
    return G, src, dst


def find_route(G, src, dst, method='fastest'):
    """Find a route in G. 'fastest' uses travel_time if present, else length."""
    import networkx as nx
    import osmnx as ox

    if method == 'fastest':
        # OSMnx adds travel_time per edge via add_edge_speeds + add_edge_travel_times
        try:
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            weight = 'travel_time'
        except Exception:
            weight = 'length'
    else:
        weight = 'length'

    path = nx.shortest_path(G, src, dst, weight=weight)
    return path


def signals_on_route(G, path) -> list:
    """Return [(osm_node_id, lat, lon, kind), ...] for signal-controlled
    nodes along the path. Uses the same detection as signal_costs to ensure
    visualization matches what the routing cost calculation sees.

    Detects three patterns (see signal_costs.classify_signal_node for full
    discussion):
      - 'traffic_signals'    : highway=traffic_signals
      - 'crossing_value'     : highway=crossing + crossing=traffic_signals
                               (or pelican/puffin/toucan)
      - 'crossing_signals'   : highway=crossing + crossing:signals=yes
    """
    from signal_costs import classify_signal_node

    signals = []
    for n in path[1:-1]:   # interior nodes only
        node_data = G.nodes[n]
        kind = classify_signal_node(node_data)
        if kind is None:
            continue

        signals.append({
            'osm_node_id': n,
            'lat': node_data.get('y'),
            'lon': node_data.get('x'),
            'kind': kind,
        })
    return signals


def render_static_map(G, path, signals, out_path):
    """matplotlib static rendering: route in blue, signals as red dots."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(11, 11))

    # All edges in light gray
    for u, v, k, d in G.edges(keys=True, data=True):
        if 'geometry' in d:
            xs, ys = d['geometry'].xy
            ax.plot(xs, ys, color='#dddddd', linewidth=0.6, zorder=1)
        else:
            x1, y1 = G.nodes[u].get('x'), G.nodes[u].get('y')
            x2, y2 = G.nodes[v].get('x'), G.nodes[v].get('y')
            ax.plot([x1, x2], [y1, y2], color='#dddddd', linewidth=0.6, zorder=1)

    # Route in blue
    route_xs, route_ys = [], []
    for u, v in zip(path[:-1], path[1:]):
        edges = G[u][v]
        edge = next(iter(edges.values()))
        if 'geometry' in edge:
            xs, ys = edge['geometry'].xy
            route_xs.extend(xs); route_ys.extend(ys)
        else:
            x1, y1 = G.nodes[u].get('x'), G.nodes[u].get('y')
            x2, y2 = G.nodes[v].get('x'), G.nodes[v].get('y')
            route_xs.extend([x1, x2]); route_ys.extend([y1, y2])
    ax.plot(route_xs, route_ys, color='#1f77b4', linewidth=2.5,
             zorder=2, label='Route')

    # Start / end markers
    x_start, y_start = G.nodes[path[0]]['x'], G.nodes[path[0]]['y']
    x_end, y_end     = G.nodes[path[-1]]['x'], G.nodes[path[-1]]['y']
    ax.scatter([x_start], [y_start], s=200, marker='*', c='green',
                edgecolors='black', linewidths=1.5, zorder=4, label='Start')
    ax.scatter([x_end], [y_end], s=200, marker='X', c='red',
                edgecolors='black', linewidths=1.5, zorder=4, label='End')

    # Signal nodes in red
    for i, s in enumerate(signals):
        ax.scatter([s['lon']], [s['lat']], s=120, c='red',
                    edgecolors='black', linewidths=1.2, zorder=5)
        ax.annotate(f"S{i+1}", (s['lon'], s['lat']),
                     fontsize=9, fontweight='bold',
                     xytext=(8, 4), textcoords='offset points',
                     zorder=6,
                     bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='white', edgecolor='red', alpha=0.9))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Route with OSM-tagged traffic signals  ({len(signals)} signals found)')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def render_folium_map(G, path, signals, out_path):
    """Interactive HTML map. Click signal markers to see their lat/lon, which
    you can paste into Google Maps for verification."""
    try:
        import folium
    except ImportError:
        print("  (folium not installed — skipping interactive map)")
        return

    # Center the map on the route midpoint
    mid_idx = len(path) // 2
    center_lat = G.nodes[path[mid_idx]]['y']
    center_lon = G.nodes[path[mid_idx]]['x']

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                    tiles='OpenStreetMap')

    # Route polyline
    route_coords = []
    for u, v in zip(path[:-1], path[1:]):
        edges = G[u][v]
        edge = next(iter(edges.values()))
        if 'geometry' in edge:
            for lon, lat in edge['geometry'].coords:
                route_coords.append([lat, lon])
        else:
            route_coords.append([G.nodes[u]['y'], G.nodes[u]['x']])
            route_coords.append([G.nodes[v]['y'], G.nodes[v]['x']])
    folium.PolyLine(route_coords, color='blue', weight=4, opacity=0.7).add_to(m)

    # Start and end
    folium.Marker(
        [G.nodes[path[0]]['y'], G.nodes[path[0]]['x']],
        popup='Start',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    folium.Marker(
        [G.nodes[path[-1]]['y'], G.nodes[path[-1]]['x']],
        popup='End',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

    # Signal markers — click to copy lat/lon for Google Maps
    for i, s in enumerate(signals):
        gmaps_url = f"https://www.google.com/maps?q={s['lat']},{s['lon']}"
        popup_html = (
            f"<b>Signal S{i+1}</b><br>"
            f"OSM node: {s['osm_node_id']}<br>"
            f"<code>{s['lat']:.6f}, {s['lon']:.6f}</code><br>"
            f"<a href='{gmaps_url}' target='_blank'>Open in Google Maps</a>"
        )
        folium.CircleMarker(
            location=[s['lat'], s['lon']],
            radius=8, color='red', fill=True, fillColor='red', fillOpacity=0.9,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"S{i+1} — click for Google Maps link"
        ).add_to(m)

    m.save(str(out_path))


def write_verification_template(signals, out_path):
    """Write a CSV pre-formatted for manual verification against Google Maps."""
    rows = []
    for i, s in enumerate(signals):
        rows.append({
            'signal_id':    f'S{i+1}',
            'osm_node_id':  s['osm_node_id'],
            'osm_tag_kind': s.get('kind', ''),  # 'traffic_signals' or 'crossing=...'
            'lat':          s['lat'],
            'lon':          s['lon'],
            'google_maps_url': f"https://www.google.com/maps?q={s['lat']},{s['lon']}",
            'verified_real': '',       # mark Y / N after checking
            'pedestrian_only': '',     # mark Y if it's a crosswalk signal only
            'notes': '',
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--start', required=True, help='Start lat,lon')
    ap.add_argument('--end',   required=True, help='End lat,lon')
    ap.add_argument('--route-method', default='fastest',
                    choices=['fastest', 'shortest'])
    ap.add_argument('--out-dir', default='./signal_check')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_lat, s_lon = [float(x) for x in args.start.split(',')]
    e_lat, e_lon = [float(x) for x in args.end.split(',')]

    G, src, dst = load_network(s_lat, s_lon, e_lat, e_lon)
    path = find_route(G, src, dst, args.route_method)
    print(f"\nRoute: {len(path)} nodes")

    # Primary detection: two-stage inventory pipeline (same as routing uses).
    # Falls back to legacy node-tag detection if OSM fetch fails.
    try:
        from signal_inventory import build_signal_inventory, match_route_to_inventory
        print("Building OSM signal inventory...")
        inv = build_signal_inventory(G)
    except ImportError:
        inv = None

    if inv is not None:
        route_signals = match_route_to_inventory(G, path, inv)
        print(f"Signal-controlled route nodes (inventory pipeline): "
              f"{len(route_signals)}")
        # Repackage as the legacy `signals` dict shape so the rendering
        # functions below don't need to change.
        signals = []
        for rs in route_signals:
            lat, lon = rs.cluster_centroid
            signals.append({
                'osm_node_id': rs.route_node,   # the route node we matched to
                'lat':         lat,
                'lon':         lon,
                'kind':        rs.sources,      # 'traffic_signals' | 'crossing_signal' | both
            })
    else:
        # Legacy fallback
        print("Inventory unavailable — using node-tag detection")
        signals = signals_on_route(G, path)
        print(f"OSM-tagged traffic signals on route (legacy): {len(signals)}")

    # Write CSV
    df = pd.DataFrame(signals)
    df.to_csv(out_dir / 'signals.csv', index=False)

    # Static figure
    render_static_map(G, path, signals, out_dir / 'route_with_signals.png')

    # Interactive HTML
    render_folium_map(G, path, signals, out_dir / 'route_with_signals.html')

    # Verification template
    write_verification_template(signals, out_dir / 'verification_template.csv')

    print(f"\nWrote outputs to {out_dir}/")
    print(f"  signals.csv                 (raw signal locations)")
    print(f"  route_with_signals.png      (static map)")
    print(f"  route_with_signals.html     (interactive — open in browser)")
    print(f"  verification_template.csv   (fill in 'verified_real' column)")
    print(f"\nNext step: open route_with_signals.html, click each red marker,")
    print(f"           follow its Google Maps link, and mark Y/N in the")
    print(f"           verification_template.csv.")


if __name__ == '__main__':
    main()
