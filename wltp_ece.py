"""
ECE percentage computation per ISO 23795-1.

The ECE indices express each work component of a real trip as a percentage of
what the same component would be on the WLTP Class 3 reference cycle, evaluated
with the SAME vehicle parameters. This is the standard's normative comparison.

  AccECE[%]  = AccWork(real)  / AccWork(WLTP_ref)  × 100
  AeroECE[%] = AeroWork(real) / AeroWork(WLTP_ref) × 100
  WorkECE[%] = TotalWork(real) / TotalWork(WLTP_ref) × 100
  STSECE[%]  = standstill_time(real) / standstill_time(WLTP_ref) × 100

These are normalised per-km to make trips of different lengths comparable.

The reference component values come from running the model on
wltp_class3_reference.csv (extracted from PKW_OCT.xlsx) for the same vehicle
configuration. We pre-compute them once per vehicle and re-use across all trips.

Key insight (validated against the LCMM CSV's own AccECE/AeroECE columns):
the percentage compares per-km work intensity, NOT trip totals — so a short
real trip can still be meaningfully compared to the 23-km WLTP cycle.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from iso23795_model import (
    Vehicle,
    compute_trip_from_wltp_reference,
    TripResult,
)


@dataclass
class WLTPReference:
    """Pre-computed per-km reference work intensities for one vehicle.

    Stored as work-per-km so they can be applied to trips of arbitrary length.
    Computed once per vehicle and cached.
    """
    vehicle_name: str
    phase: str                     # 'ALL' | 'LOW' | 'MIDDLE' | 'HIGH' | 'EXTRA-HIGH'
    aero_J_per_km: float
    roll_J_per_km: float
    acc_pos_J_per_km: float
    total_motion_J_per_km: float
    standstill_frac: float         # standstill seconds / total seconds


def build_wltp_reference(
    vehicle: Vehicle,
    wltp_csv_path: str,
    phase: str | None = None,
) -> WLTPReference:
    """Compute reference work intensities for `vehicle` on the WLTP cycle.

    If `phase` is None, uses the full 1800-second cycle. Otherwise restricts
    to one of {'LOW', 'MIDDLE', 'HIGH', 'EXTRA-HIGH'}.
    """
    per_sec, trip = compute_trip_from_wltp_reference(wltp_csv_path, vehicle, phase=phase)
    km = trip.distance_km
    return WLTPReference(
        vehicle_name=vehicle.name,
        phase=phase or 'ALL',
        aero_J_per_km=trip.aero_J / km if km > 0 else 0,
        roll_J_per_km=trip.roll_J / km if km > 0 else 0,
        acc_pos_J_per_km=trip.acc_J_pos / km if km > 0 else 0,
        total_motion_J_per_km=trip.total_motion_J / km if km > 0 else 0,
        standstill_frac=trip.standstill_s / trip.duration_s if trip.duration_s > 0 else 0,
    )


def compute_ece_percentages(
    trip: TripResult,
    ref: WLTPReference,
) -> dict:
    """Compute ECE percentages comparing this trip to the WLTP reference.

    Returns a dict with AeroECE_pct, RollECE_pct, AccECE_pct, WorkECE_pct,
    STSECE_pct. A value of 100% means the real trip imposes the same per-km
    work in that category as the WLTP reference; >100% means more demanding.
    """
    km = trip.distance_km
    if km <= 0:
        return {k: 0.0 for k in ['AeroECE_pct','RollECE_pct','AccECE_pct',
                                  'WorkECE_pct','STSECE_pct']}

    real_aero_per_km = trip.aero_J / km
    real_roll_per_km = trip.roll_J / km
    real_acc_per_km = trip.acc_J_pos / km
    real_motion_per_km = trip.total_motion_J / km
    real_sts_frac = trip.standstill_s / trip.duration_s if trip.duration_s > 0 else 0

    def safe_ratio(num, den):
        return (num / den * 100) if den > 0 else 0.0

    return {
        'AeroECE_pct': safe_ratio(real_aero_per_km, ref.aero_J_per_km),
        'RollECE_pct': safe_ratio(real_roll_per_km, ref.roll_J_per_km),
        'AccECE_pct':  safe_ratio(real_acc_per_km, ref.acc_pos_J_per_km),
        'WorkECE_pct': safe_ratio(real_motion_per_km, ref.total_motion_J_per_km),
        'STSECE_pct':  safe_ratio(real_sts_frac, ref.standstill_frac),
    }


def build_phase_references(
    vehicle: Vehicle,
    wltp_csv_path: str,
) -> dict[str, WLTPReference]:
    """Build references for all 4 phases + the full cycle."""
    refs = {'ALL': build_wltp_reference(vehicle, wltp_csv_path)}
    for phase in ['LOW', 'MIDDLE', 'HIGH', 'EXTRA-HIGH']:
        refs[phase] = build_wltp_reference(vehicle, wltp_csv_path, phase=phase)
    return refs
