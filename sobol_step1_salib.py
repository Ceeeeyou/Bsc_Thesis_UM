"""
Step 1 — Final experiment: global Sobol sensitivity analysis (SALib version).

This is the SALib-based implementation. Run this on your machine after:

    pip install SALib

SALib reference (cite this in the thesis):
    Herman, J., & Usher, W. (2017). SALib: An open-source Python library for
    Sensitivity Analysis. Journal of Open Source Software, 2(9), 97.
    https://doi.org/10.21105/joss.00097

If SALib is not available in your environment, see sobol_step1_handrolled.py
for an equivalent hand-rolled implementation (same algorithm, no dependencies,
validated against the Ishigami benchmark).
 
Question: which vehicle-configurati on parameters drive trip-level CO2 emissions,
and how does that ranking change across driving regimes
(WLTP-LOW = urban, MIDDLE = suburban, HIGH = rural, EXTRA-HIGH = motorway)?

Method: Saltelli sampling with N=512 → 512×(2k+2) = 8192 model evaluations
per regime in SALib's 2nd-order scheme, or N(k+2) = 4608 in 1st-order-only mode.
We compute first-order (S1) and total-order (ST) Sobol indices.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from SALib.sample import saltelli
from SALib.analyze import sobol

sys.path.insert(0, str(Path(__file__).parent))
from iso23795_model import Vehicle, compute_work_per_second, aggregate_trip


WLTP_CSV = Path(__file__).parent / 'data/wltp_class3_reference.csv'
OUT_DIR = Path('D:/Maastricht University/thesis/codes/outputs')
OUT_DIR.mkdir(exist_ok=True)


# Pre-load WLTP traces ONCE
_wltp = pd.read_csv(WLTP_CSV)
_phase_arrays = {}
for phase in ['LOW', 'MIDDLE', 'HIGH', 'EXTRA-HIGH']:
    sub = _wltp[_wltp['phase'] == phase].reset_index(drop=True)
    speed = sub['speed_ms'].values.astype(float)
    distance = np.zeros(len(speed))
    distance[1:] = (speed[1:] + speed[:-1]) / 2
    altitude = np.zeros(len(speed))
    time_ms = (np.arange(len(speed)) * 1000).astype(float)
    _phase_arrays[phase] = (speed, altitude, distance, time_ms)


# SALib problem definition
problem = {
    'num_vars': 7,
    'names': [
        'curb_mass_kg', 'c_w', 'area_m2', 'mu_roll',
        'idle_l_per_h', 'eta_engine', 'payload_kg',
    ],
    'bounds': [
        [1000, 8000],       # curb_mass_kg : small car to small truck
        [0.25, 0.65],       # c_w          : modern sedan to box truck
        [1.8, 8.5],         # area_m2      : small car to truck
        [0.008, 0.020],     # mu_roll      : low-resistance to off-road
        [0.3, 2.0],         # idle_l_per_h : small car to large diesel
        [0.20, 0.45],       # eta_engine   : Rodrigues 2022 ICE-gasoline range
        [0, 2000],          # payload_kg   : empty to fully loaded
    ],
}


def evaluate_co2(params: np.ndarray, phase: str) -> float:
    """Evaluate trip CO2 (g/km) for one parameter vector on one WLTP phase."""
    speed, altitude, distance, time_ms = _phase_arrays[phase]
    v = Vehicle(
        name='s',
        curb_mass_kg=params[0], c_w=params[1], area_m2=params[2],
        mu_roll=params[3], idle_l_per_h=params[4],
        fuel_type='gasoline',
        eta_engine=params[5], payload_kg=params[6],
    )
    per_sec = compute_work_per_second(
        speed_ms=speed, altitude_m=altitude, distance_m=distance,
        time_ms=time_ms, vehicle=v,
    )
    trip = aggregate_trip(per_sec, v)
    return trip.co2_g_per_km


# ---------------------------------------------------------------------------
N = 512
print(f"SALib Sobol with N={N}, 7 parameters")
print(f"Total samples per regime: {N * (2 * problem['num_vars'] + 2)} (Saltelli with 2nd-order)")
print()

# Generate Saltelli sample (same matrix used for all regimes)
# calc_second_order=True is SALib default; gives us second-order S2 too if wanted
param_values = saltelli.sample(problem, N, calc_second_order=True)
print(f"Sample matrix shape: {param_values.shape}\n")

regimes = ['LOW', 'MIDDLE', 'HIGH', 'EXTRA-HIGH']
all_results = []

for phase in regimes:
    print(f"  {phase}...", end=' ', flush=True)

    # Evaluate model on every sample
    Y = np.array([evaluate_co2(p, phase) for p in param_values])

    # Compute Sobol indices (returns dict with 'S1', 'ST', 'S2', and confidence
    # intervals 'S1_conf', 'ST_conf', 'S2_conf')
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)

    # Pack into long DataFrame
    for i, name in enumerate(problem['names']):
        all_results.append({
            'phase': phase,
            'parameter': name,
            'S1': Si['S1'][i],
            'S1_conf': Si['S1_conf'][i],   # 95% CI half-width via bootstrap
            'ST': Si['ST'][i],
            'ST_conf': Si['ST_conf'][i],
            'mean_co2_g_km': float(np.mean(Y)),
            'std_co2_g_km': float(np.std(Y)),
        })

    print(f"mean CO2 = {np.mean(Y):.0f} ± {np.std(Y):.0f} g/km")

result_df = pd.DataFrame(all_results)
result_df.to_csv(OUT_DIR / 'exp6_sobol_sensitivity_salib.csv', index=False)


# Pretty-print
print("\n" + "=" * 80)
print("FIRST-ORDER (S1) with 95% bootstrap confidence intervals")
print("=" * 80)
for phase in regimes:
    sub = result_df[result_df['phase'] == phase].sort_values('ST', ascending=False)
    print(f"\n  {phase}:")
    for _, row in sub.iterrows():
        print(f"    {row['parameter']:<14s}  S1 = {row['S1']:6.3f} ± {row['S1_conf']:.3f}   "
              f"ST = {row['ST']:6.3f} ± {row['ST_conf']:.3f}")

# Pivots for the thesis
pivot_S1 = result_df.pivot(index='parameter', columns='phase', values='S1')[regimes]
pivot_ST = result_df.pivot(index='parameter', columns='phase', values='ST')[regimes]
pivot_S1.to_csv(OUT_DIR / 'exp6_sobol_S1_salib.csv')
pivot_ST.to_csv(OUT_DIR / 'exp6_sobol_ST_salib.csv')

print(f"\nSaved to {OUT_DIR}/")
print("Compare to hand-rolled output (exp6_sobol_*.csv) — values should match within ~0.02")
