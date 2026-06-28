"""
Step 1 worked example.

Demonstrates the full Step-1 workflow on the data you currently have:
    - 1 real-driving trace (POLO_REAL_CSV)
    - the WLTP Class 3 reference cycle extracted from PKW_OCT.xlsx
    - 4 vehicle configurations (Polo, Opel, T7 van, Volvo FL truck)

Five experiments:
    EXP 1 — WLTP reference baseline (each vehicle on the standard cycle)
    EXP 2 — Cross-vehicle comparison on the real trip
    EXP 3 — ECE percentages: real trip vs WLTP reference per phase
    EXP 4 — One-at-a-time sensitivity around the Polo baseline
    EXP 5 — Payload sweep on the van (T7)

Once you collect more real traces (city / suburb / highway), drop them into
a folder and replace `[real_trace]` with `load_traces('your/folder')` —
everything else extends without code changes.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from iso23795_model import (
    VEHICLES, compute_trip_from_wltp_reference, compute_trip_from_lcmm_csv,
)
from batch_eval import (
    load_trace, evaluate_grid, evaluate_one,
    make_oat_variants, make_payload_sweep,
)
from wltp_ece import build_wltp_reference, build_phase_references, compute_ece_percentages


# ---- Paths ------------------------------------------------------------------
WLTP_CSV = 'D:/Maastricht University/thesis/codes/data/wltp_class3_reference.csv'
REAL_CSV = 'D:/Maastricht University/thesis/codes/data/lcmm_test.csv'
OUTPUT_DIR = Path('D:/Maastricht University/thesis/codes/outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# EXP 1 — WLTP reference baseline for all vehicles
# ============================================================================
print("=" * 78)
print("EXP 1: WLTP Class 3 reference cycle, all vehicles")
print("=" * 78)
print("Standardised baseline. Compare your real trips against these numbers.\n")

rows = []
for key, veh in VEHICLES.items():
    for phase in [None, 'LOW', 'MIDDLE', 'HIGH', 'EXTRA-HIGH']:
        _, trip = compute_trip_from_wltp_reference(WLTP_CSV, veh, phase=phase)
        rows.append({
            'vehicle': veh.name,
            'fuel': veh.fuel_type,
            'phase': phase or 'ALL',
            'distance_km': round(trip.distance_km, 2),
            'duration_s': round(trip.duration_s, 0),
            'avg_kmh': round(trip.avg_speed_kmh, 1),
            'l_per_100km': round(trip.l_per_100km, 2),
            'co2_g_per_km': round(trip.co2_g_per_km, 1),
            'EPI': round(trip.EPI_l_per_100km_t, 2),
            'API': round(trip.API_kWh_per_100km_t, 2),
        })
df_wltp = pd.DataFrame(rows)
print(df_wltp.to_string(index=False))
df_wltp.to_csv(OUTPUT_DIR / 'exp1_wltp_baseline.csv', index=False)


# ============================================================================
# EXP 2 — Cross-vehicle comparison on the real trip
# ============================================================================
print("\n" + "=" * 78)
print("EXP 2: Cross-vehicle comparison on the real Polo trip")
print("=" * 78)
print("Same speed/altitude profile, four vehicle configs.")
print("(In reality a truck wouldn't accelerate like a Polo — this isolates the")
print("vehicle-parameter effect from the driving-style effect.)\n")

real_trace = load_trace(REAL_CSV)
print(f"Trace: {real_trace.regime}, {real_trace.distance_km:.2f} km, "
      f"{real_trace.duration_s:.0f} s, avg moving {real_trace.avg_moving_kmh:.1f} km/h\n")

df_compare = evaluate_grid([real_trace], VEHICLES)
cols = ['vehicle', 'fuel_type', 'mass_total_kg', 'l_per_100km', 'co2_g_per_km',
        'EPI_l_per_100km_t', 'API_kWh_per_100km_t',
        'aero_kJ', 'roll_kJ', 'grade_pos_kJ', 'acc_pos_kJ']
print(df_compare[cols].round(2).to_string(index=False))
df_compare.to_csv(OUTPUT_DIR / 'exp2_real_trip_vehicles.csv', index=False)


# ============================================================================
# EXP 3 — ECE percentages: real trip vs WLTP reference (per phase)
# ============================================================================
print("\n" + "=" * 78)
print("EXP 3: ECE percentages — real trip vs WLTP reference (Polo only)")
print("=" * 78)
print("Compares per-km work intensity of the real trip to each WLTP phase.")
print(">100% means the real trip is more demanding than the reference.\n")

polo = VEHICLES['polo']
phase_refs = build_phase_references(polo, WLTP_CSV)

# Need a TripResult for the real trip with this vehicle
_, real_trip = compute_trip_from_lcmm_csv(REAL_CSV, polo)

ece_rows = []
for phase_name, ref in phase_refs.items():
    pct = compute_ece_percentages(real_trip, ref)
    ece_rows.append({
        'phase_compared_to': phase_name,
        'AeroECE_%': round(pct['AeroECE_pct'], 1),
        'RollECE_%': round(pct['RollECE_pct'], 1),
        'AccECE_%':  round(pct['AccECE_pct'], 1),
        'WorkECE_%': round(pct['WorkECE_pct'], 1),
        'STSECE_%':  round(pct['STSECE_pct'], 1),
    })
df_ece = pd.DataFrame(ece_rows)
print(df_ece.to_string(index=False))
print()
print("Recall: LCMM CSV reports AccECE=120%, AeroECE=89%, WorkECE=124% for")
print("this same trip — those are vs the FULL cycle reference.")
df_ece.to_csv(OUTPUT_DIR / 'exp3_ece_percentages.csv', index=False)


# ============================================================================
# EXP 4 — OAT sensitivity around the Polo baseline
# ============================================================================
print("\n" + "=" * 78)
print("EXP 4: One-at-a-time sensitivity (Polo on real trip)")
print("=" * 78)
print("Each row sweeps ONE parameter; others stay at baseline.\n")

variants = make_oat_variants(polo, {
    'curb_mass_kg': [1000, 1236, 1500, 1800, 2200],
    'c_w':          [0.25, 0.31, 0.40, 0.50, 0.65],
    'area_m2':      [1.8, 2.10, 2.5, 3.5, 5.0],
    'mu_roll':      [0.008, 0.012, 0.015, 0.020],
    'eta_engine':   [0.25, 0.30, 0.35, 0.40],
})
df_oat = evaluate_grid([real_trace], variants)
baseline = df_oat[df_oat['vehicle'] == polo.name].iloc[0]
df_oat['delta_co2_pct'] = (df_oat['co2_g_per_km'] - baseline['co2_g_per_km']) / baseline['co2_g_per_km'] * 100

print(df_oat[['vehicle', 'mass_total_kg', 'c_w', 'area_m2', 'mu_roll', 'eta_engine',
              'co2_g_per_km', 'delta_co2_pct']].round(2).to_string(index=False))
df_oat.to_csv(OUTPUT_DIR / 'exp4_oat_sensitivity.csv', index=False)


# ============================================================================
# EXP 5 — Payload sweep on the van
# ============================================================================
print("\n" + "=" * 78)
print("EXP 5: Payload sweep (T7 van, 0–1000 kg cargo)")
print("=" * 78)
t7 = VEHICLES['t7']
loaded = make_payload_sweep(t7, payloads_kg=[0, 250, 500, 750, 1000])
df_payload = evaluate_grid([real_trace], loaded)
base_payload = df_payload.iloc[0]
df_payload['delta_co2_pct'] = (df_payload['co2_g_per_km'] - base_payload['co2_g_per_km']) / base_payload['co2_g_per_km'] * 100
print(df_payload[['vehicle', 'mass_total_kg', 'co2_g_per_km', 'delta_co2_pct',
                  'roll_kJ', 'acc_pos_kJ']].round(2).to_string(index=False))
df_payload.to_csv(OUTPUT_DIR / 'exp5_payload_sweep.csv', index=False)


# ============================================================================
print("\n" + "=" * 78)
print("All experiments saved to /mnt/user-data/outputs/")
print("=" * 78)
print("""
NEXT STEPS:
    1. Drive 4-5 more LCMM traces (city / suburb / highway).
    2. Drop them in a folder, named city_*.csv etc.
    3. Replace `[real_trace]` in EXP 2-5 with `load_traces('folder')`.
    4. The same OAT sensitivity per regime should reveal that:
         - city: mass dominates
         - highway: c_w dominates
       That ranking flip is your main Step-1 finding.
""")
