"""
Traffic-signal cost module for vehicle-aware eco-routing.

For each OSM node tagged `highway=traffic_signals` along a route, computes
the energy and time penalty of decelerating, idling, and re-accelerating.

WHAT THE COST CAPTURES
======================
Three components per signal-tagged node (excess on top of free-flow):
    1. Idle fuel during the red phase
    2. Kinetic energy lost to brakes during deceleration
    3. Kinetic energy needed to re-accelerate to cruise speed

Whether the vehicle actually stops is stochastic in real life, so we model
this as expected value: probability of hitting red × per-stop cost.

DEFAULT PARAMETERS (with citations for the thesis)
==================================================
    expected_idle_time_per_stop = 18 s
        Source: INRIX 2022 U.S. Signals Scorecard, measured across 240,000
        signalized intersections; population-averaged delay.
        Reference: INRIX (2022), "INRIX Analyzes and Ranks Intersection
        Performance across the U.S."
        https://inrix.com/press-releases/signal-scorecard/

    probability_of_stop = 0.5
        Standard assumption for symmetric (50/50 g/r ratio) traffic signal
        cycles with random arrival. Used in HCM2010 and most traffic engineering
        eco-driving literature.

    full_stop_KE_loss
        Computed from cruise speed: KE = ½ · m · v², dissipated entirely as
        brake heat (no regen for ICE; clip negative AccWork at zero per
        ISO 23795-1 B.4 "No Perpetuum Mobile").

REFERENCES
==========
- Rakha, H., & Ding, Y. (2003). Impact of Stops on Vehicle Fuel Consumption
  and Emissions. Journal of Transportation Engineering, 129(1), 23-32.
  *Canonical reference for vehicle stop-cost decomposition.*
- INRIX (2022). U.S. Signals Scorecard.
  *Empirical idle-time-per-signal: 18 s average across 240k intersections.*
- Boriboonsomsin, K., & Barth, M. (2009). Impacts of Road Grade on Fuel
  Consumption and CO2 Emissions Evidenced by Use of Advanced Navigation
  Systems. Transportation Research Record, 2139, 21-30.
  *Validation of grade + stop-cost in real eco-routing.*
- Saerens, B., & Van den Bulck, E. (2013). Calculation of the minimum-fuel
  driving control based on Pontryagin's minimum principle. Transportation
  Research Part D, 24, 89-97.
  *Theoretical optimum stop-and-go fuel cost.*
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from iso23795_model import Vehicle


# =============================================================================
# Default parameters (documented in module docstring)
# =============================================================================
DEFAULT_EXPECTED_IDLE_S = 18.1       # INRIX 2022: population-averaged delay
                                     # per vehicle per signal (240k intersections)
DEFAULT_STOP_PROBABILITY = 0.365     # INRIX 2022: 63.5% arrive on green, so
                                     # 36.5% stop. Same source as the delay
                                     # figure -> internally consistent. Implies
                                     # conditional per-stop delay ~ 49.6 s
                                     # (upper bound: green arrivals also incur
                                     # a small slowdown delay). Replaces the
                                     # earlier symmetric-cycle assumption (0.5).


# =============================================================================
# Per-signal cost computation
# =============================================================================
@dataclass
class SignalCost:
    """Cost incurred for one signalized intersection at edge cruise speed v."""
    fuel_l: float      # litres of extra fuel due to this signal
    co2_g: float       # grams of CO2
    time_s: float      # seconds of extra time
    components: dict   # diagnostic breakdown


def compute_signal_cost(
    vehicle: Vehicle,
    cruise_speed_ms: float,
    *,
    expected_idle_s: float = DEFAULT_EXPECTED_IDLE_S,
    stop_probability: float = DEFAULT_STOP_PROBABILITY,
    deterministic: bool = False,
) -> SignalCost:
    """Compute the expected energy/time cost of one signalized intersection.

    Cost structure:
        Idle component:      expected idle equals `expected_idle_s` (the INRIX
                              figure is population-averaged, i.e. already
                              probability-weighted); per-stop conditional idle
                              is expected_idle_s / stop_probability.
        Kinetic component:   ½·m·v² is dissipated when braking, then re-supplied
                              when accelerating back to cruise. Net excess work
                              vs cruise = m·v² (one full deceleration + one full
                              acceleration; clipped at zero per ISO B.4).

    Parameters
    ----------
    vehicle : the Vehicle config (mass, idle rate, η, LHV, CO2 factor)
    cruise_speed_ms : free-flow speed at the signal's edge in m/s
    expected_idle_s : POPULATION-AVERAGED delay per signal per vehicle
                      (INRIX 2022 default 18 s; already includes stop probability)
    stop_probability : probability that the signal is red on arrival (default 0.5)
    deterministic : if True, treat every signal as a guaranteed stop (worst-case)

    Returns
    -------
    SignalCost with fuel, co2, time and a diagnostic component breakdown.
    """
    p_stop = 1.0 if deterministic else stop_probability

    # ----- Idle component -------------------------------------------------
    # SEMANTICS FIX: INRIX's 18 s is the POPULATION-AVERAGED delay per signal
    # (averaged over all vehicles, stopped and non-stopped), so the stop
    # probability is already inside it. Multiplying it by p_stop again would
    # double-count the probability and halve the idle cost (the old bug).
    # We therefore derive the CONDITIONAL per-stop delay from the population
    # average (18 s / 0.5 = 36 s given that you stop), then weight by p_stop:
    #   expectation mode    -> idle_s = 18 s   (matches INRIX exactly)
    #   deterministic mode  -> idle_s = 36 s   (true worst case, was wrongly 18)
    per_stop_idle_s = expected_idle_s / stop_probability
    idle_s = p_stop * per_stop_idle_s
    idle_fuel_l = (vehicle.idle_l_per_h / 3600) * idle_s

    # ----- Kinetic-energy component (decel + accel back to cruise) -------
    # ½·m·v² lost in braking + ½·m·v² supplied in re-acceleration.
    # Total mechanical excess = m·v². ISO B.4 clip: braking energy is dissipated
    # (no regen), so the work the engine has to do extra is m·v² (not just
    # ½·m·v²) — we still need to supply the acceleration KE from fuel.
    # However, the deceleration phase didn't consume engine fuel (it's coast/brake),
    # so the *fuel-relevant* excess is just the re-acceleration: ½·m·v².
    # Conservative model: excess engine work per stop = ½·m·v² × p_stop
    # Convert to fuel via η × LHV.
    ke_J = 0.5 * vehicle.mass_kg * cruise_speed_ms ** 2 * p_stop
    ke_fuel_l = ke_J / (vehicle.eta * vehicle.lhv_J_per_l)

    # ----- Combine ---------------------------------------------------------
    fuel_l = idle_fuel_l + ke_fuel_l
    co2_g = fuel_l * vehicle.co2_kg_per_l * 1000
    time_s = idle_s     # the KE acceleration time is already in the edge time

    return SignalCost(
        fuel_l=fuel_l,
        co2_g=co2_g,
        time_s=time_s,
        components={
            'idle_s':     idle_s,
            'idle_fuel_l': idle_fuel_l,
            'ke_J':       ke_J,
            'ke_fuel_l':  ke_fuel_l,
            'p_stop':     p_stop,
        },
    )


# =============================================================================
# Path-level signal cost: sum costs along a route
# =============================================================================
# Sub-tag values of `traffic_signals=*` that do NOT cause a normal-traffic stop
# and should therefore be excluded from the signal inventory:
#   - blinker:   flashing amber warning, no stop required
#   - emergency: green-by-default, only activates for emergency vehicles
# (ramp_meter does cause stops but with a very different cost profile;
#  it is currently still treated as a normal signal — flagged in thesis
#  limitations.)
_TRAFFIC_SIGNALS_EXCLUDE = {'blinker', 'emergency'}

# Values of the `crossing=*` tag that indicate a signal-controlled crossing
_CROSSING_SIGNAL_VALUES = {'traffic_signals', 'pelican', 'puffin', 'toucan'}


def classify_signal_node(node_data: dict) -> str | None:
    """Return a short label for the tagging pattern that marks this node as
    signal-controlled, or None if the node has no signal.

    Possible return values (in evaluation order):
        'traffic_signals'    — highway=traffic_signals (canonical)
        'crossing_value'     — highway=crossing + crossing in {traffic_signals,
                                pelican, puffin, toucan}
        'crossing_signals'   — highway=crossing + crossing:signals=yes
                                (newer OSM tagging convention, post-2020)
        None                 — no signal, OR an excluded sub-type
                                (traffic_signals=blinker / emergency)

    The label is used by the diagnostic counters in
    step2_routing/signal_locations to break down coverage by tagging pattern.
    This breakdown is what tells you, in the thesis methodology section,
    which OSM conventions are doing the work in your study area.
    """
    h = node_data.get('highway')
    if h is None:
        return None

    # Normalize to list for uniform handling (OSMnx returns either str or list)
    h_list = h if isinstance(h, list) else [h]

    # Pattern 1: highway=traffic_signals (the canonical tag)
    # Filter out blinker/emergency sub-types if the `traffic_signals` sub-tag
    # is present — these are false positives for routing purposes.
    if 'traffic_signals' in h_list:
        sub = node_data.get('traffic_signals')
        if sub in _TRAFFIC_SIGNALS_EXCLUDE:
            return None
        return 'traffic_signals'

    # Pattern 2 & 3: signal-controlled crossing
    if 'crossing' in h_list:
        # Pattern 2: legacy crossing=traffic_signals|pelican|puffin|toucan
        crossing_type = node_data.get('crossing')
        if crossing_type in _CROSSING_SIGNAL_VALUES:
            return 'crossing_value'

        # Pattern 3: modern crossing:signals=yes (separated from crossing=*)
        # OSMnx may surface this key with either ':' or '_' depending on
        # version, so check both.
        crossing_signals = (node_data.get('crossing:signals')
                            or node_data.get('crossing_signals'))
        if str(crossing_signals).lower() in ('yes', 'true'):
            return 'crossing_signals'

    return None


def has_traffic_signal(node_data: dict) -> bool:
    """Check whether an OSMnx node is a signal-controlled intersection.

    Detects three OSM tagging patterns (see classify_signal_node for details):

    1. Direct: `highway=traffic_signals` — the canonical tag.
       Excludes traffic_signals=blinker (flashing warning) and
       traffic_signals=emergency (emergency-vehicle-only).

    2. Legacy crossing sub-value: `highway=crossing` AND
       `crossing` in {traffic_signals, pelican, puffin, toucan}.

    3. Modern crossing flag: `highway=crossing` AND `crossing:signals=yes`.
       (This OSM tagging convention has been spreading since ~2020 and is
       common in recent Dutch edits. Earlier versions of this code missed
       it entirely.)

    NOTE: For OSMnx to retain the sub-tags on graph nodes, the caller must
    add these keys to ox.settings.useful_tags_node BEFORE calling
    graph_from_bbox:
        ['crossing', 'traffic_signals', 'crossing:signals']
    Handled in step2_routing.load_osmnx_network and
    signal_locations.load_network.
    """
    return classify_signal_node(node_data) is not None


def compute_path_signal_cost(
    G, path_nodes, vehicle: Vehicle,
    *,
    expected_idle_s: float = DEFAULT_EXPECTED_IDLE_S,
    stop_probability: float = DEFAULT_STOP_PROBABILITY,
    deterministic: bool = False,
) -> dict:
    """Legacy: sum signal cost using NODE-TAG detection (one tag-check per
    route node). Kept as a fallback path for when the OSM inventory pipeline
    is unavailable (no network, OSMnx error). Under-counts signals on NL
    routes because OSMnx route nodes are usually not the signal-tagged ones
    — see signal_inventory module docstring for the full discussion.

    For new code, prefer compute_path_signal_cost_from_inventory().
    """
    total_co2 = 0.0; total_fuel = 0.0; total_time = 0.0; n_signals = 0

    for i in range(1, len(path_nodes) - 1):
        node = path_nodes[i]
        node_data = G.nodes[node]
        if not has_traffic_signal(node_data):
            continue
        n_signals += 1
        # Cruise speed: from the incoming edge's maxspeed (most reliable)
        u = path_nodes[i - 1]
        edges = G[u][node]
        if hasattr(edges, 'values'):
            edge = next(iter(edges.values())) if isinstance(next(iter(edges.values())), dict) and 'maxspeed_kmh' in next(iter(edges.values())) else next(iter(edges.values()))
            v_kmh = edge.get('maxspeed_kmh', 50)
        else:
            v_kmh = edges.get('maxspeed_kmh', 50)
        v_ms = v_kmh / 3.6

        cost = compute_signal_cost(
            vehicle, v_ms,
            expected_idle_s=expected_idle_s,
            stop_probability=stop_probability,
            deterministic=deterministic,
        )
        total_co2 += cost.co2_g
        total_fuel += cost.fuel_l
        total_time += cost.time_s

    return {
        'co2_g':        total_co2,
        'fuel_l':       total_fuel,
        'time_s':       total_time,
        'signal_count': n_signals,
    }


def _edge_speed_at_route_node(G, path_nodes, route_node) -> float:
    """Return the maxspeed (km/h) of the route edge entering `route_node`.

    Falls back to 50 km/h if no incoming-edge speed is available (urban
    default, consistent with the legacy code path).
    """
    try:
        idx = path_nodes.index(route_node)
    except ValueError:
        return 50.0
    if idx == 0:
        # The matched node is the route start — use the OUTGOING edge instead
        if len(path_nodes) < 2:
            return 50.0
        u, v = path_nodes[0], path_nodes[1]
    else:
        u, v = path_nodes[idx - 1], path_nodes[idx]
    edges = G[u][v]
    if hasattr(edges, 'values'):
        edge = next(iter(edges.values()))
    else:
        edge = edges
    return float(edge.get('maxspeed_kmh', 50.0))


def compute_path_signal_cost_from_inventory(
    G, path_nodes, vehicle: Vehicle,
    route_signals: list,             # list[signal_inventory.RouteSignal]
    *,
    expected_idle_s: float = DEFAULT_EXPECTED_IDLE_S,
    stop_probability: float = DEFAULT_STOP_PROBABILITY,
    deterministic: bool = False,
) -> dict:
    """Sum signal cost from a pre-computed RouteSignal list (the new
    two-stage pipeline output).

    Each entry in `route_signals` is already deduplicated to one record
    per route node — see signal_inventory.match_clusters_to_route. We
    simply look up the cruise speed at each matched node and apply the
    per-signal cost.

    Returns the same dict shape as compute_path_signal_cost, plus a
    'matched_route_nodes' diagnostic field for the thesis methodology
    audit trail.
    """
    total_co2 = 0.0; total_fuel = 0.0; total_time = 0.0
    matched_nodes = []

    for rs in route_signals:
        v_kmh = _edge_speed_at_route_node(G, path_nodes, rs.route_node)
        v_ms = v_kmh / 3.6
        cost = compute_signal_cost(
            vehicle, v_ms,
            expected_idle_s=expected_idle_s,
            stop_probability=stop_probability,
            deterministic=deterministic,
        )
        total_co2  += cost.co2_g
        total_fuel += cost.fuel_l
        total_time += cost.time_s
        matched_nodes.append(rs.route_node)

    return {
        'co2_g':              total_co2,
        'fuel_l':             total_fuel,
        'time_s':             total_time,
        'signal_count':       len(route_signals),
        'matched_route_nodes': matched_nodes,
    }


# =============================================================================
# CLI / smoke test
# =============================================================================
if __name__ == '__main__':
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from iso23795_model import VEHICLES

    ap = argparse.ArgumentParser(description="Signal cost smoke test")
    ap.add_argument('--vehicle', default='polo', choices=list(VEHICLES.keys()))
    ap.add_argument('--cruise-kmh', type=float, default=50.0)
    args = ap.parse_args()

    vehicle = VEHICLES[args.vehicle]
    v_ms = args.cruise_kmh / 3.6
    cost = compute_signal_cost(vehicle, v_ms)

    print(f"Vehicle: {vehicle.name}")
    print(f"Cruise speed: {args.cruise_kmh} km/h ({v_ms:.2f} m/s)")
    print(f"\nSignal cost (expected value):")
    print(f"  Idle: {cost.components['idle_s']:.1f} s × {vehicle.idle_l_per_h:.1f} L/h "
          f"= {cost.components['idle_fuel_l']*1000:.2f} mL")
    print(f"  KE excess: ½·{vehicle.mass_kg:.0f}·{v_ms:.1f}² × {cost.components['p_stop']:.1f} "
          f"= {cost.components['ke_J']/1000:.2f} kJ → {cost.components['ke_fuel_l']*1000:.2f} mL")
    print(f"  Total fuel: {cost.fuel_l*1000:.2f} mL = {cost.co2_g:.1f} g CO2")
    print(f"  Total time: {cost.time_s:.1f} s")
