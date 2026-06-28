"""
Multi-trace Step 1 evaluator.

Given a folder of LCMM CSVs and a list of vehicle configurations, produces
three deliverables:

  1) Per-trace × per-vehicle CO2/km matrix (the cross-vehicle comparison
     from EXP 2, generalised to multiple traces).
  2) Per-trace regime classification + ECE percentages against the matching
     WLTP phase (EXP 3 generalised).
  3) Per-regime concatenated-trace Sobol sensitivity indices (EXP 6
     repeated on real data instead of synthetic WLTP cycles), if SALib
     is installed.

Usage
-----
    python multi_trace_eval.py \
        --traces-dir ./my_lcmm_csvs/ \
        --out-dir ./multi_trace_results/ \
        --vehicles polo opel t7 volvo_fl \
        --sobol-n 256

If a `manifest.csv` exists in the traces directory with columns
(filename, vehicle, regime), it overrides the auto-detection. Otherwise
all traces are treated as the Polo (default), and regime is inferred
from average moving speed.

What you get
------------
    out-dir/
      per_trace_co2.csv       wide N × M matrix of CO2/km per (trace, vehicle)
      per_trace_long.csv      same data in long/tidy form
      per_trace_ece.csv       ECE percentages per trace vs matching WLTP phase
      regime_sobol_<R>.csv    Sobol indices per regime (if enough data)
      summary.md              human-readable summary
      figures/
        cross_vehicle_heatmap.png
        regime_ratio_bars.png
        sobol_comparison.png  (if Sobol was run)

Limitations to disclose in the thesis
-------------------------------------
- Cycle-replay assumption: evaluating a Polo-recorded trace under T7 or
  Volvo FL parameters assumes the same speed profile would be executed by
  those vehicles. In reality a truck driver accelerates more gently. This
  is the standard abstraction in vehicle simulation; it isolates parameter
  effects from driving-style effects.
- Regime classification by avg moving speed is coarse. A trace mixing
  motorway and urban will be misclassified. The manifest override exists
  to handle this case (label such a trace 'mixed' and exclude it from
  regime grouping).
- Concatenated regime-traces for Sobol introduce one artificial standstill
  per trace boundary (~30 s of zero-speed rows). This adds a few mL of
  idle fuel per join, which is negligible (Sobol shows idle_l_per_h is
  the lowest-importance parameter at S_T ≤ 0.01).
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from iso23795_model import (
    Vehicle, VEHICLES, compute_work_per_second, aggregate_trip,
)
from batch_eval import (
    TraceData, load_trace, classify_regime, REGIME_BOUNDS_KMH,
)
from wltp_ece import build_phase_references, compute_ece_percentages


# Map informal regime names → WLTP phase names for ECE baselining.
# 'mixed' has no canonical phase — we report ECE against the full WLTP cycle
# in that case (a single concatenation of all four phases).
REGIME_TO_WLTP_PHASE = {
    'city':    'LOW',
    'suburb':  'MIDDLE',
    'highway': 'HIGH',          # could also be EXTRA-HIGH; HIGH is a safer
                                 # middle ground for typical NL driving
}


# =============================================================================
# Manifest loading
# =============================================================================
def load_manifest(traces_dir: Path) -> dict[str, dict]:
    """Read manifest.csv if present. Returns {filename: {vehicle, regime}}.
    Missing entries are filled with defaults at evaluation time."""
    manifest_path = traces_dir / 'manifest.csv'
    if not manifest_path.exists():
        return {}

    df = pd.read_csv(manifest_path)
    # Accept either 'filename' or 'file' as the key column
    key_col = 'filename' if 'filename' in df.columns else 'file'
    out = {}
    for _, row in df.iterrows():
        entry = {}
        if 'vehicle' in df.columns and pd.notna(row.get('vehicle')):
            entry['vehicle'] = str(row['vehicle']).strip()
        if 'regime' in df.columns and pd.notna(row.get('regime')):
            entry['regime'] = str(row['regime']).strip()
        out[str(row[key_col]).strip()] = entry
    return out


# =============================================================================
# 1. Per-trace × per-vehicle matrix
# =============================================================================
def evaluate_trace_with_vehicle(trace: TraceData, vehicle: Vehicle) -> dict:
    """Run the energy model on a trace with a given vehicle, return
    one summary dict with all key metrics."""
    df = trace.df
    time_ms = df['Time[ms]'].values if 'Time[ms]' in df.columns else None
    per_sec = compute_work_per_second(
        speed_ms=df['Speed[m/s]'].values,
        altitude_m=df['Altitude[m]'].values,
        distance_m=df['Distance'].values,
        time_ms=time_ms,
        vehicle=vehicle,
    )
    trip = aggregate_trip(per_sec, vehicle)
    return {
        'trace':          trace.name,
        'vehicle':        vehicle.name,
        'distance_km':    trip.distance_km,
        'duration_min':   trip.duration_s / 60,
        'avg_moving_kmh': trace.avg_moving_kmh,
        'regime':         trace.regime,
        'fuel_l':         trip.fuel_total_l,
        'fuel_l_per_100km': trip.l_per_100km,
        'co2_kg':         trip.co2_kg,
        'co2_g_per_km':   trip.co2_g_per_km,
        'motion_J':       trip.total_motion_J,
        'standstill_l':   trip.standstill_fuel_l,
    }


def build_co2_matrix(traces: dict[str, TraceData],
                      vehicles: list[Vehicle],
                      manifest: dict) -> pd.DataFrame:
    """Run energy model on each (trace, vehicle) combination. Returns
    a long-form DataFrame with one row per pair."""
    rows = []
    for tname, trace in traces.items():
        for veh in vehicles:
            row = evaluate_trace_with_vehicle(trace, veh)
            # Override regime if manifest says so
            if tname in manifest and 'regime' in manifest[tname]:
                row['regime'] = manifest[tname]['regime']
                trace.regime = manifest[tname]['regime']
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 2. Per-trace ECE
# =============================================================================
def build_ece_per_trace(traces: dict[str, TraceData],
                         primary_vehicle: Vehicle,
                         wltp_csv: str) -> pd.DataFrame:
    """For each trace, classify regime → matching WLTP phase, then compute
    ECE percentages against that phase reference."""
    refs = build_phase_references(primary_vehicle, wltp_csv)
    rows = []
    for tname, trace in traces.items():
        regime = trace.regime
        phase = REGIME_TO_WLTP_PHASE.get(regime, 'LOW')

        df = trace.df
        time_ms = df['Time[ms]'].values if 'Time[ms]' in df.columns else None
        per_sec = compute_work_per_second(
            speed_ms=df['Speed[m/s]'].values,
            altitude_m=df['Altitude[m]'].values,
            distance_m=df['Distance'].values,
            time_ms=time_ms,
            vehicle=primary_vehicle,
        )
        trip = aggregate_trip(per_sec, primary_vehicle)
        ece = compute_ece_percentages(trip, refs[phase])

        rows.append({
            'trace':         tname,
            'regime':        regime,
            'phase_ref':     phase,
            'distance_km':   trip.distance_km,
            'AccECE_pct':    ece['AccECE_pct'],
            'AeroECE_pct':   ece['AeroECE_pct'],
            'RollECE_pct':   ece['RollECE_pct'],
            'WorkECE_pct':   ece['WorkECE_pct'],
            'STSECE_pct':    ece['STSECE_pct'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. Concatenated regime-trace Sobol
# =============================================================================
def concatenate_traces(traces_in_regime: list[TraceData],
                        gap_s: float = 30.0) -> dict:
    """Concatenate multiple traces into one continuous synthetic regime-trace.

    Inserts `gap_s` of synthetic standstill (zero-speed, 1-Hz rows) between
    consecutive traces to handle the velocity discontinuity at boundaries
    without introducing fake acceleration spikes. Timestamps are reset to
    monotonic 1-Hz.
    """
    if not traces_in_regime:
        raise ValueError("No traces to concatenate")

    speed_parts = []
    altitude_parts = []
    distance_parts = []

    for i, t in enumerate(traces_in_regime):
        speed = t.df['Speed[m/s]'].values
        altitude = t.df['Altitude[m]'].values
        distance = t.df['Distance'].values
        # Replace any NaN distances with 0 (LCMM's first row is often NaN)
        distance = np.nan_to_num(distance, nan=0.0)
        speed_parts.append(speed)
        altitude_parts.append(altitude)
        distance_parts.append(distance)
        if i < len(traces_in_regime) - 1:
            n_gap = int(gap_s)
            speed_parts.append(np.zeros(n_gap))
            altitude_parts.append(np.full(n_gap, altitude[-1]))
            distance_parts.append(np.zeros(n_gap))

    speed_arr = np.concatenate(speed_parts)
    altitude_arr = np.concatenate(altitude_parts)
    distance_arr = np.concatenate(distance_parts)
    time_ms = np.arange(len(speed_arr)) * 1000.0

    return {
        'speed_ms':   speed_arr,
        'altitude_m': altitude_arr,
        'distance_m': distance_arr,
        'time_ms':    time_ms,
        'n_traces':   len(traces_in_regime),
        'duration_s': len(speed_arr),
    }


def run_sobol_on_concatenated_traces(
    traces: dict[str, TraceData],
    base_vehicle: Vehicle,
    N: int,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Group traces by regime, concatenate each group, run Sobol on each.

    Uses SALib if available; falls back to the hand-rolled implementation
    from sobol_analysis.py otherwise.
    """
    # Group by regime
    by_regime: dict[str, list[TraceData]] = {}
    for t in traces.values():
        by_regime.setdefault(t.regime, []).append(t)

    # Try SALib; fall back to hand-rolled
    try:
        from SALib.sample.sobol import sample as salib_sample
        from SALib.analyze.sobol import analyze as salib_analyze
        use_salib = True
        print("  Sobol backend: SALib (peer-reviewed, recommended)")
    except ImportError:
        from sobol_analysis import SobolProblem, saltelli_sample, sobol_analyze
        use_salib = False
        print("  Sobol backend: hand-rolled (SALib not installed)")

    # Parameter ranges — must match sobol_step1.py for cross-comparison
    PROBLEM = {
        'num_vars': 7,
        'names':    ['curb_mass_kg', 'c_w', 'area_m2', 'mu_roll',
                     'idle_l_per_h', 'eta_engine', 'payload_kg'],
        'bounds':   [[1000, 8000],
                     [0.25, 0.65],
                     [1.8, 8.5],
                     [0.008, 0.020],
                     [0.3, 2.0],
                     [0.20, 0.45],
                     [0, 2000]],
    }

    results = {}
    for regime, trace_list in by_regime.items():
        if not trace_list:
            continue
        print(f"\n  Regime '{regime}': {len(trace_list)} trace(s), "
              f"total {sum(t.duration_s for t in trace_list):.0f} s")

        concat = concatenate_traces(trace_list)

        # Sample parameters
        if use_salib:
            samples = salib_sample(PROBLEM, N, calc_second_order=False, seed=seed)
        else:
            problem = SobolProblem(names=PROBLEM['names'],
                                    bounds=[tuple(b) for b in PROBLEM['bounds']])
            samples = saltelli_sample(problem, N, seed=seed)

        # Evaluate each sample on the concatenated trace
        Y = np.zeros(samples.shape[0])
        for j in range(samples.shape[0]):
            params = dict(zip(PROBLEM['names'], samples[j]))
            v_sample = base_vehicle.with_overrides(**params)
            per_sec = compute_work_per_second(
                speed_ms=concat['speed_ms'],
                altitude_m=concat['altitude_m'],
                distance_m=concat['distance_m'],
                time_ms=concat['time_ms'],
                vehicle=v_sample,
            )
            trip = aggregate_trip(per_sec, v_sample)
            Y[j] = (trip.co2_kg * 1000 / trip.distance_km
                    if trip.distance_km > 0 else 0)

        # Analyze
        if use_salib:
            si = salib_analyze(PROBLEM, Y, calc_second_order=False, print_to_console=False)
            df = pd.DataFrame({
                'parameter': PROBLEM['names'],
                'S1':        si['S1'],
                'S1_conf':   si['S1_conf'],
                'ST':        si['ST'],
                'ST_conf':   si['ST_conf'],
            })
        else:
            df = sobol_analyze(problem, Y, N)
        results[regime] = df
        print(f"    Top-3 ST: " + ", ".join(
            f"{df.iloc[i]['parameter']}={df.iloc[i]['ST']:.2f}"
            for i in df['ST'].argsort()[::-1][:3].tolist()
        ))
    return results


# =============================================================================
# Plotting
# =============================================================================
def plot_cross_vehicle_heatmap(co2_long: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of CO2/km, traces × vehicles, with traces sorted by regime."""
    import matplotlib.pyplot as plt

    pivot = co2_long.pivot_table(index='trace', columns='vehicle',
                                   values='co2_g_per_km')
    # Sort traces by their (regime, avg_speed)
    trace_meta = co2_long.groupby('trace').agg(
        regime=('regime', 'first'),
        avg_speed=('avg_moving_kmh', 'first')
    )
    regime_order = ['city', 'suburb', 'highway', 'mixed']
    trace_meta['regime_idx'] = trace_meta['regime'].apply(
        lambda r: regime_order.index(r) if r in regime_order else 99
    )
    trace_meta = trace_meta.sort_values(['regime_idx', 'avg_speed'])
    pivot = pivot.loc[trace_meta.index]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{i}  [{trace_meta.loc[i, 'regime']}]" for i in pivot.index])
    # Cell labels
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                     color='white' if val > pivot.values.mean() else 'black',
                     fontsize=9)
    plt.colorbar(im, ax=ax, label='CO₂ (g/km)')
    ax.set_title('Per-trace × per-vehicle CO₂ emissions (g/km)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def plot_regime_ratio_bars(co2_long: pd.DataFrame, out_path: Path) -> None:
    """For each regime, average CO2/km per vehicle. Shows truck-to-car ratio
    by regime — the empirical analog of Sobol's regime ranking flip."""
    import matplotlib.pyplot as plt

    grouped = co2_long.groupby(['regime', 'vehicle'])['co2_g_per_km'].mean().reset_index()
    regime_order = [r for r in ['city', 'suburb', 'highway', 'mixed']
                     if r in grouped['regime'].unique()]
    veh_order = co2_long['vehicle'].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: absolute CO2 per regime per vehicle
    x = np.arange(len(regime_order))
    w = 0.8 / len(veh_order)
    colors = ['#2C5F2D', '#88B04B', '#F7B538', '#C73E1D']
    for i, v in enumerate(veh_order):
        vals = [grouped[(grouped.regime == r) & (grouped.vehicle == v)]['co2_g_per_km'].mean()
                for r in regime_order]
        ax1.bar(x + i * w - (len(veh_order) - 1) * w / 2, vals, w,
                label=v, color=colors[i % len(colors)], edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_order)
    ax1.set_ylabel('Mean CO₂ (g/km)')
    ax1.set_title('Mean CO₂ per regime, per vehicle')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: truck-to-car ratio per regime
    polo_like = veh_order[0]   # assume first is lightest
    truck_like = veh_order[-1]  # last is heaviest
    ratios = [grouped[(grouped.regime == r) & (grouped.vehicle == truck_like)]['co2_g_per_km'].mean() /
              grouped[(grouped.regime == r) & (grouped.vehicle == polo_like)]['co2_g_per_km'].mean()
              for r in regime_order]
    ax2.bar(regime_order, ratios, color='#21295C', edgecolor='black')
    for i, v in enumerate(ratios):
        ax2.text(i, v + 0.05, f'{v:.1f}×', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'{truck_like} / {polo_like} CO₂ ratio')
    ax2.set_title(f'Heavy/light vehicle ratio by regime')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def plot_sobol_comparison(sobol_results: dict[str, pd.DataFrame],
                            out_path: Path) -> None:
    """Compare regime-trace Sobol ST values across regimes."""
    import matplotlib.pyplot as plt

    if not sobol_results:
        return
    params = sobol_results[list(sobol_results.keys())[0]]['parameter'].tolist()
    regimes = list(sobol_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(params))
    w = 0.8 / len(regimes)
    colors = ['#2C5F2D', '#F7B538', '#C73E1D', '#21295C']
    for i, r in enumerate(regimes):
        st_vals = sobol_results[r]['ST'].values
        ax.bar(x + i * w - (len(regimes) - 1) * w / 2, st_vals, w,
                label=r, color=colors[i % len(colors)], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=20)
    ax.set_ylabel('Total-order Sobol index $S_T$')
    ax.set_title('Per-regime Sobol indices on concatenated real traces')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces-dir', required=True,
                    help='Folder of LCMM CSVs (optionally with manifest.csv)')
    ap.add_argument('--out-dir', default='./multi_trace_results',
                    help='Where to write outputs')
    ap.add_argument('--vehicles', nargs='+', default=['polo', 'opel', 't7', 'volvo_fl'],
                    choices=list(VEHICLES.keys()),
                    help='Vehicle configs to evaluate against each trace')
    ap.add_argument('--primary-vehicle', default='polo',
                    choices=list(VEHICLES.keys()),
                    help='Vehicle for ECE percentages (default: polo)')
    ap.add_argument('--wltp', default='d:/Maastricht University/thesis/codes/data/wltp_class3_reference.csv',
                    help='WLTP reference CSV')
    ap.add_argument('--sobol-n', type=int, default=256,
                    help='Sobol sample size N. 0 = skip Sobol.')
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)

    # ---- Load traces ------------------------------------------------------
    print(f"Loading traces from {traces_dir}...")
    manifest = load_manifest(traces_dir)
    if manifest:
        print(f"  Found manifest with {len(manifest)} entries")

    traces = {}
    for csv in sorted(traces_dir.glob('*.csv')):
        if csv.name == 'manifest.csv':
            continue
        t = load_trace(csv)
        # Apply manifest regime override
        if csv.name in manifest and 'regime' in manifest[csv.name]:
            t.regime = manifest[csv.name]['regime']
        traces[csv.name] = t
        print(f"  {csv.name}: {t.distance_km:.2f} km, "
              f"{t.duration_s/60:.1f} min, "
              f"avg moving {t.avg_moving_kmh:.1f} km/h → {t.regime}")

    if not traces:
        print("No traces found.")
        return

    # ---- (1) Cross-vehicle matrix ----------------------------------------
    print(f"\nEvaluating {len(traces)} traces × {len(args.vehicles)} vehicles...")
    vehicles = [VEHICLES[v] for v in args.vehicles]
    co2_long = build_co2_matrix(traces, vehicles, manifest)
    co2_long.to_csv(out_dir / 'per_trace_long.csv', index=False)
    # Wide pivot of CO2/km
    co2_wide = co2_long.pivot_table(
        index='trace', columns='vehicle', values='co2_g_per_km'
    ).round(1)
    co2_wide.to_csv(out_dir / 'per_trace_co2.csv')
    print(f"  Wrote per_trace_co2.csv ({len(co2_wide)} traces × {len(co2_wide.columns)} vehicles)")

    plot_cross_vehicle_heatmap(co2_long, out_dir / 'figures/cross_vehicle_heatmap.png')
    plot_regime_ratio_bars(co2_long, out_dir / 'figures/regime_ratio_bars.png')

    # ---- (2) ECE per trace -----------------------------------------------
    print(f"\nComputing ECE percentages (primary vehicle: {args.primary_vehicle})...")
    primary = VEHICLES[args.primary_vehicle]
    ece_df = build_ece_per_trace(traces, primary, args.wltp)
    ece_df.to_csv(out_dir / 'per_trace_ece.csv', index=False)
    print(f"  Wrote per_trace_ece.csv")

    # ---- (3) Regime-Sobol ------------------------------------------------
    sobol_results = {}
    if args.sobol_n > 0:
        print(f"\nRunning Sobol on regime-concatenated traces (N={args.sobol_n})...")
        sobol_results = run_sobol_on_concatenated_traces(
            traces, primary, args.sobol_n
        )
        for regime, df in sobol_results.items():
            df.to_csv(out_dir / f'regime_sobol_{regime}.csv', index=False)
        if sobol_results:
            plot_sobol_comparison(sobol_results, out_dir / 'figures/sobol_comparison.png')
            print(f"  Wrote regime_sobol_*.csv and figures/sobol_comparison.png")

    # ---- Summary -----------------------------------------------------------
    write_summary(out_dir, traces, co2_long, ece_df, sobol_results)
    print(f"\nAll outputs in: {out_dir}")


def write_summary(out_dir: Path, traces: dict, co2_long: pd.DataFrame,
                   ece_df: pd.DataFrame, sobol_results: dict) -> None:
    """Write a human-readable summary.md."""
    lines = ["# Multi-trace Step 1 evaluation", ""]

    # Traces section
    lines.append("## Input traces")
    lines.append(f"- {len(traces)} traces loaded")
    by_regime: dict[str, int] = {}
    for t in traces.values():
        by_regime[t.regime] = by_regime.get(t.regime, 0) + 1
    for r, n in sorted(by_regime.items()):
        lines.append(f"  - {r}: {n} trace(s)")
    lines.append("")

    # Cross-vehicle headline
    lines.append("## Cross-vehicle CO₂ (g/km), regime means")
    lines.append("")
    by_regime_veh = co2_long.groupby(['regime', 'vehicle'])['co2_g_per_km'].mean().reset_index()
    pivot = by_regime_veh.pivot(index='regime', columns='vehicle', values='co2_g_per_km').round(0)
    lines.append(pivot.to_markdown())
    lines.append("")

    # Truck-to-car ratio per regime
    veh_order = co2_long['vehicle'].unique().tolist()
    if len(veh_order) >= 2:
        light, heavy = veh_order[0], veh_order[-1]
        lines.append(f"## {heavy} / {light} CO₂ ratio by regime")
        lines.append("")
        for r in pivot.index:
            ratio = pivot.loc[r, heavy] / pivot.loc[r, light]
            lines.append(f"- {r}: **{ratio:.2f}×**")
        lines.append("")

    # ECE
    lines.append("## ECE percentages (per trace, vs matching WLTP phase)")
    lines.append("")
    lines.append(ece_df.round(1).to_markdown(index=False))
    lines.append("")

    # Sobol
    if sobol_results:
        lines.append("## Sobol total-order indices (per-regime, concatenated traces)")
        lines.append("")
        all_params = sobol_results[list(sobol_results.keys())[0]]['parameter'].tolist()
        rows = []
        for r, df in sobol_results.items():
            d = dict(zip(df['parameter'], df['ST']))
            rows.append({'regime': r, **{p: round(d.get(p, 0), 3) for p in all_params}})
        st_table = pd.DataFrame(rows).set_index('regime')
        lines.append(st_table.to_markdown())
        lines.append("")

    (out_dir / 'summary.md').write_text("\n".join(lines))


if __name__ == '__main__':
    main()
