r"""
Validate the ISO 23795-1 model against MULTIPLE LCMM CSVs.

This generalizes validate_dual.py's VALIDATION 2 (the LCMM real-data check)
to run over many LCMM files at once, so the validation is concrete across
several traces rather than a single recording.

For each LCMM CSV it extracts ONLY the raw inputs (Time, Speed, Distance,
Altitude), runs YOUR iso23795_model on them, and compares against LCMM's own
output columns in the same file. The comparison framing matches validate_dual:

  AeroWork, RollWork  -> identical physics, expect <1%
  GradeWork           -> ~same, different slope-filter boundary handling
  AccWork             -> ISO m·a·d vs LCMM ½m·Δv²: both valid, differ per-row.
                         Reported as a documented difference, NOT a failure.
  Fuel (per-row CSV)  -> expect ~4-5% (the AccWork-form difference propagating);
                         this matches validate_dual's +4.4% on the Polo trace.

StandStill is reported in LITRES (model outputs StandStillFuel_l), not joules,
because the model — like TEST_XLSX — keeps idle as a fuel volume and excludes
it from TotalWork. The LCMM StandStillWork[J] column is therefore not directly
comparable and is skipped (noted).

Usage:
    py validate_lcmm_multi.py --csv "data\lcmm_test.csv" --vehicle polo
    py validate_lcmm_multi.py --dir "data\lcmm_traces" --vehicle polo --out-dir "..\results\lcmm_validation"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from iso23795_model import compute_work_per_second, aggregate_trip, VEHICLES

# LCMM raw input columns
COL_SPEED = 'Speed[m/s]'
COL_DIST  = 'Distance'
COL_ALT   = 'Altitude[m]'
COL_TIME  = 'Time[ms]'

# Components we compare directly (model col -> LCMM col). These are signed
# per-row work sums. Aero/Roll are the clean "identical physics" checks.
COMPARE = [
    ('AeroWork_J',  'AeroWork[J]',  'identical physics, expect <1%'),
    ('RollWork_J',  'RollWork[J]',  'identical physics, expect <1%'),
    ('GradeWork_J', 'GradeWork[J]', '~same; different slope-filter boundary'),
    ('AccWork_J',   'AccWork[J]',   'ISO m·a·d vs LCMM ½mΔv² (documented, both valid)'),
]


def validate_one(csv_path: Path, vehicle):
    lcmm = pd.read_csv(csv_path)
    for c in (COL_SPEED, COL_DIST, COL_ALT, COL_TIME):
        if c in lcmm.columns:
            lcmm[c] = pd.to_numeric(lcmm[c], errors='coerce')

    my = compute_work_per_second(
        speed_ms=lcmm[COL_SPEED].values,
        altitude_m=lcmm[COL_ALT].values,
        distance_m=lcmm[COL_DIST].fillna(0).values,
        time_ms=lcmm[COL_TIME].values,
        vehicle=vehicle,
    )
    trip = aggregate_trip(my, vehicle)

    rows = []
    for mine, ref, note in COMPARE:
        if ref not in lcmm.columns or mine not in my.columns:
            continue
        my_t = my[mine].fillna(0).sum()
        ref_t = pd.to_numeric(lcmm[ref], errors='coerce').fillna(0).sum()
        rel = (my_t - ref_t) / ref_t * 100 if ref_t else np.nan
        rows.append({'component': mine.replace('_J',''),
                     'LCMM_J': ref_t, 'model_J': my_t,
                     'dev_pct': rel, 'note': note})
    comp = pd.DataFrame(rows)

    # Fuel: compare against LCMM per-row CSV Fuel column (AC excluded), exactly
    # as validate_dual does.
    lcmm_fuel = pd.to_numeric(lcmm.get('Fuel[l]', pd.Series(dtype=float)),
                              errors='coerce').fillna(0).sum()
    dist_km = lcmm[COL_DIST].fillna(0).sum() / 1000
    lcmm_l100 = lcmm_fuel / dist_km * 100 if dist_km else np.nan
    model_l100 = trip.l_per_100km
    fuel_gap = (model_l100 - lcmm_l100) / lcmm_l100 * 100 if lcmm_l100 else np.nan

    return {
        'file': csv_path.name, 'rows': len(lcmm),
        'components': comp,
        'model_l_per_100km': model_l100,
        'lcmm_l_per_100km': lcmm_l100,
        'fuel_gap_pct': fuel_gap,
        'model_fuel_l': trip.fuel_total_l,
        'lcmm_fuel_l': lcmm_fuel,
    }


def print_report(r):
    print(f"\n{'='*74}\n{r['file']}   ({r['rows']} rows)\n{'='*74}")
    c = r['components']
    if not c.empty:
        print(f"{'component':<12}{'LCMM (J)':>13}{'model (J)':>13}{'dev %':>9}   note")
        for _, row in c.iterrows():
            clean = row['component'] in ('AeroWork', 'RollWork')
            flag = '' if (not clean or abs(row['dev_pct']) < 1.0) else '  <-- check'
            print(f"{row['component']:<12}{row['LCMM_J']:>13,.0f}"
                  f"{row['model_J']:>13,.0f}{row['dev_pct']:>8.2f}%{flag}")
            print(f"{'':>38}   {row['note']}")
    print(f"\nFuel (vs LCMM per-row CSV column):")
    print(f"  model     {r['model_l_per_100km']:.2f} L/100km")
    print(f"  LCMM CSV  {r['lcmm_l_per_100km']:.2f} L/100km")
    print(f"  gap       {r['fuel_gap_pct']:+.1f}%   (≈+4-5% expected, AccWork-form difference)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--csv'); g.add_argument('--dir')
    ap.add_argument('--vehicle', default='polo')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    if args.vehicle not in VEHICLES:
        raise SystemExit(f"Unknown vehicle. Choose from {list(VEHICLES.keys())}")
    vehicle = VEHICLES[args.vehicle]

    files = ([Path(args.csv)] if args.csv
             else sorted(Path(args.dir).glob('*.csv')))
    files = [f for f in files if f.name != 'manifest.csv']
    if not files:
        raise SystemExit("No CSV files found.")

    summary = []
    for f in files:
        try:
            r = validate_one(f, vehicle)
        except Exception as ex:
            print(f"\n{f.name}: FAILED ({ex})")
            import traceback; traceback.print_exc()
            continue
        print_report(r)
        aero = r['components'].query("component=='AeroWork'")['dev_pct']
        roll = r['components'].query("component=='RollWork'")['dev_pct']
        summary.append({
            'file': r['file'], 'rows': r['rows'],
            'Aero_dev_pct': aero.iloc[0] if not aero.empty else np.nan,
            'Roll_dev_pct': roll.iloc[0] if not roll.empty else np.nan,
            'model_L_per_100km': r['model_l_per_100km'],
            'lcmm_L_per_100km': r['lcmm_l_per_100km'],
            'fuel_gap_pct': r['fuel_gap_pct'],
        })

    s = pd.DataFrame(summary)
    print(f"\n{'='*74}\nSUMMARY ACROSS {len(s)} FILE(S)\n{'='*74}")
    if not s.empty:
        print(s.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print(f"\nMean |Aero dev|: {s['Aero_dev_pct'].abs().mean():.2f}%  (should be <1%)")
        print(f"Mean |Roll dev|: {s['Roll_dev_pct'].abs().mean():.2f}%  (should be <1%)")
        print(f"Mean fuel gap:   {s['fuel_gap_pct'].mean():+.1f}%  (≈+4-5% expected)")
    if args.out_dir:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        s.to_csv(out / 'lcmm_validation_summary.csv', index=False)
        print(f"\nWrote {out/'lcmm_validation_summary.csv'}")


if __name__ == '__main__':
    main()
