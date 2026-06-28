"""
Batch evaluation: run multiple vehicle configurations against multiple
real-world LCMM traces.

Workflow
--------
1. Drop your LCMM CSVs into a folder, named like:
       <regime>_<descriptor>.csv
   where regime ∈ {city, suburb, highway} (or anything else).
   Example: city_rotterdam_centrum_01.csv
            highway_a16_to_dordrecht.csv

2. Call `load_traces(folder)` to get a dict {trace_name: TraceData}.

3. Call `evaluate_grid(traces, vehicles)` to get a tidy DataFrame with one row
   per (trace × vehicle), ready for plotting and stats.

4. For sensitivity analyses, build vehicle variants with `Vehicle.with_overrides`
   or `Vehicle.with_payload`, then re-run the grid.

Trip regime is classified automatically from the trace's average moving speed
if not encoded in the filename, using thresholds aligned to WLTP phases.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import pandas as pd
import numpy as np

from iso23795_model import (
    Vehicle, VEHICLES,
    compute_work_per_second, aggregate_trip, TripResult,
    STANDSTILL_V_THRESH,
)


# Regime classification by average moving speed (km/h), aligned to WLTP phases
REGIME_BOUNDS_KMH = {
    'city':    (0,   45),    # WLTP LOW: peak 56.5, avg 18.9 km/h
    'suburb':  (45,  75),    # WLTP MEDIUM: peak 76.6, avg 39.4 km/h
    'highway': (75, 200),    # WLTP HIGH/EXTRA-HIGH: 97-131 km/h peaks
}


@dataclass
class TraceData:
    """A single real-world trip trace from LCMM."""
    name: str
    regime: str                   # 'city' | 'suburb' | 'highway' | 'unknown'
    df: pd.DataFrame              # raw CSV
    duration_s: float
    distance_km: float
    avg_moving_kmh: float
    standstill_s: float


def classify_regime(avg_moving_kmh: float) -> str:
    """Assign a WLTP-aligned regime label from average moving speed."""
    for regime, (lo, hi) in REGIME_BOUNDS_KMH.items():
        if lo <= avg_moving_kmh < hi:
            return regime
    return 'unknown'


def parse_filename_regime(name: str) -> str | None:
    """Extract regime from filename if present (e.g. 'city_xxx.csv')."""
    m = re.match(r'^(city|suburb|highway)[_\-]', name.lower())
    return m.group(1) if m else None


def load_trace(csv_path: str | Path) -> TraceData:
    """Load a single LCMM CSV and compute its summary statistics."""
    p = Path(csv_path)
    df = pd.read_csv(p)

    # Compute summary stats
    if 'Time[ms]' in df.columns:
        dt = df['Time[ms]'].diff() / 1000.0
        dt.iloc[0] = 0
    else:
        dt = pd.Series([1.0] * len(df))

    duration = float(dt.sum())
    distance = float(df['Distance'].sum() / 1000)
    moving = df['Speed[m/s]'] >= STANDSTILL_V_THRESH
    if moving.any():
        avg_moving = float((df.loc[moving, 'Speed[m/s]'] * dt[moving]).sum() / dt[moving].sum() * 3.6)
    else:
        avg_moving = 0.0
    standstill = float(dt[~moving].sum())

    # Regime: filename takes priority, else infer from speed
    regime = parse_filename_regime(p.stem) or classify_regime(avg_moving)

    return TraceData(
        name=p.stem,
        regime=regime,
        df=df,
        duration_s=duration,
        distance_km=distance,
        avg_moving_kmh=avg_moving,
        standstill_s=standstill,
    )


def load_traces(folder: str | Path) -> dict[str, TraceData]:
    """Load all CSVs in a folder."""
    folder = Path(folder)
    traces = {}
    for csv in sorted(folder.glob('*.csv')):
        try:
            t = load_trace(csv)
            traces[t.name] = t
        except Exception as e:
            print(f"  ! failed to load {csv.name}: {e}")
    return traces


def evaluate_one(trace: TraceData, vehicle: Vehicle) -> dict:
    """Run the model for one (trace, vehicle) pair and return a flat dict row."""
    per_sec = compute_work_per_second(
        speed_ms=trace.df['Speed[m/s]'].values,
        altitude_m=trace.df['Altitude[m]'].values,
        distance_m=trace.df['Distance'].values,
        time_ms=trace.df['Time[ms]'].values if 'Time[ms]' in trace.df.columns else None,
        vehicle=vehicle,
    )
    trip = aggregate_trip(per_sec, vehicle)

    return {
        'trace': trace.name,
        'regime': trace.regime,
        'vehicle': vehicle.name,
        'fuel_type': vehicle.fuel_type,
        'curb_mass_kg': vehicle.curb_mass_kg,
        'payload_kg': vehicle.payload_kg,
        'mass_total_kg': vehicle.mass_kg,
        'c_w': vehicle.c_w,
        'area_m2': vehicle.area_m2,
        'mu_roll': vehicle.mu_roll,
        'eta_engine': vehicle.eta,
        'distance_km': trip.distance_km,
        'duration_s': trip.duration_s,
        'standstill_s': trip.standstill_s,
        'avg_speed_kmh': trip.avg_speed_kmh,
        # Work components (kJ for readability)
        'aero_kJ': trip.aero_J / 1000,
        'roll_kJ': trip.roll_J / 1000,
        'grade_pos_kJ': trip.grade_J_pos / 1000,
        'grade_neg_kJ': trip.grade_J_neg / 1000,
        'acc_pos_kJ': trip.acc_J_pos / 1000,
        'acc_neg_kJ': trip.acc_J_neg / 1000,
        'total_motion_kJ': trip.total_motion_J / 1000,
        'work_MJ_per_km': trip.work_MJ_per_km,
        # Fuel & emissions
        'standstill_fuel_l': trip.standstill_fuel_l,
        'motion_fuel_l': trip.motion_fuel_l,
        'fuel_total_l': trip.fuel_total_l,
        'motion_l_per_100km': trip.motion_l_per_100km,
        'idle_l_per_100km': trip.idle_l_per_100km,
        'l_per_100km': trip.l_per_100km,
        'co2_kg': trip.co2_kg,
        'co2_g_per_km': trip.co2_g_per_km,
        # KPIs
        'EPI_l_per_100km_t': trip.EPI_l_per_100km_t,
        'API_kWh_per_100km_t': trip.API_kWh_per_100km_t,
    }


def evaluate_grid(
    traces: dict[str, TraceData] | Iterable[TraceData],
    vehicles: dict[str, Vehicle] | Iterable[Vehicle],
) -> pd.DataFrame:
    """Cartesian product: every trace × every vehicle, returning a tidy DataFrame."""
    if isinstance(traces, dict):
        traces = list(traces.values())
    if isinstance(vehicles, dict):
        vehicles = list(vehicles.values())

    rows = []
    for t in traces:
        for v in vehicles:
            rows.append(evaluate_one(t, v))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity helpers
# ---------------------------------------------------------------------------

def make_oat_variants(
    base: Vehicle,
    parameters: dict[str, list[float]],
) -> list[Vehicle]:
    """One-at-a-time sensitivity: produce vehicle variants by sweeping each
    parameter through given values, holding all others at base.

    Example
    -------
    >>> variants = make_oat_variants(VEHICLES['polo'], {
    ...     'curb_mass_kg': [1000, 1236, 1500, 2000],
    ...     'c_w':          [0.25, 0.31, 0.40],
    ...     'area_m2':      [1.8, 2.1, 2.5],
    ... })
    """
    variants = [base]  # include the baseline
    for param, values in parameters.items():
        for val in values:
            if val == getattr(base, param):
                continue  # avoid duplicating baseline
            v = base.with_overrides(**{param: val})
            v.name = f"{base.name} | {param}={val}"
            variants.append(v)
    return variants


def make_payload_sweep(base: Vehicle, payloads_kg: list[float]) -> list[Vehicle]:
    """Produce variants of the same vehicle at different payloads."""
    variants = []
    for p in payloads_kg:
        v = base.with_payload(p)
        v.name = f"{base.name} | payload={p}kg"
        variants.append(v)
    return variants
