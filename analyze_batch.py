"""
Analysis / wrap-up for the batch study.

Reads the batch outputs (batch_summary.csv, batch_results.csv) and produces
thesis-ready aggregate tables and hypothesis tests. This is the layer on top
of run_batch_study's plots: the plots show patterns, this gives the numbers
(means, spreads, and significance) you cite in the Results chapter.

Outputs (written to <results-dir>/analysis/):
  overall_summary.txt         headline numbers in plain language
  savings_by_stratum.csv      mean / std / n CO2 saving per stratum & vehicle
  divergence_by_stratum.csv   divergence rate per stratum
  hypothesis_tests.txt        H1/H2/H3 tests with statistics and p-values
  per_pair_table.csv          tidy per-pair table for an appendix

Usage:
    py analyze_batch.py --results-dir "..\\results\\batch_study"
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(results_dir: Path):
    s = pd.read_csv(results_dir / 'batch_summary.csv')
    r_path = results_dir / 'batch_results.csv'
    r = pd.read_csv(r_path) if r_path.exists() else None
    return s, r


def ci95(x):
    """95% confidence interval half-width for a mean (t-based, small-n safe)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float('nan')
    from math import sqrt
    sd = x.std(ddof=1)
    # t critical ~ use 1.96 for n>30 else a rough small-sample bump
    tcrit = 1.96 if n >= 30 else 2.0 + 1.0 / n
    return tcrit * sd / sqrt(n)


def overall_summary(s: pd.DataFrame, out_dir: Path):
    v1 = s['v1'].iloc[0]; v2 = s['v2'].iloc[0]
    lines = []
    lines.append("=== OVERALL SUMMARY ===\n")
    lines.append(f"Vehicles compared: {v1} (light) vs {v2} (heavy)")
    lines.append(f"OD pairs analysed: {len(s)}")
    lines.append(f"Regions: {', '.join(sorted(s['region'].unique()))}\n")

    lines.append("Mean CO2 saving from eco-routing vs fastest route:")
    for v, col in [(v1, 'v1_saving_pct'), (v2, 'v2_saving_pct')]:
        m = s[col].mean(); c = ci95(s[col]); mx = s[col].max()
        lines.append(f"  {v:<12}: {m:5.2f}% (95% CI +/- {c:.2f}), "
                     f"max {mx:.1f}%")
    lines.append("")

    div = s['routes_diverge'].mean() * 100
    lines.append(f"Eco-route differs from fastest for the two vehicles "
                 f"(routes_diverge) in {div:.1f}% of pairs.")
    f1 = (~s['fastest_eq_eco_v1']).mean() * 100
    f2 = (~s['fastest_eq_eco_v2']).mean() * 100
    lines.append(f"Eco-route differs from fastest route:")
    lines.append(f"  {v1:<12}: {f1:.1f}% of pairs")
    lines.append(f"  {v2:<12}: {f2:.1f}% of pairs")
    lines.append("  (If the heavy vehicle diverges more often, that supports H1:")
    lines.append("   mass changes the optimal route.)\n")

    txt = '\n'.join(lines)
    (out_dir / 'overall_summary.txt').write_text(txt, encoding='utf-8')
    print(txt)


def stratum_tables(s: pd.DataFrame, out_dir: Path):
    v1 = s['v1'].iloc[0]; v2 = s['v2'].iloc[0]
    rows = []
    for col in ['region', 'length_stratum', 'network', 'relief']:
        if col not in s.columns:
            continue
        for level, g in s.groupby(col):
            rows.append({
                'stratum_type': col,
                'level': level,
                'n': len(g),
                f'{v1}_mean_saving_%': round(g['v1_saving_pct'].mean(), 2),
                f'{v1}_ci95': round(ci95(g['v1_saving_pct']), 2),
                f'{v2}_mean_saving_%': round(g['v2_saving_pct'].mean(), 2),
                f'{v2}_ci95': round(ci95(g['v2_saving_pct']), 2),
                'divergence_%': round(g['routes_diverge'].mean() * 100, 1),
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / 'savings_by_stratum.csv', index=False)
    print("\n=== SAVINGS & DIVERGENCE BY STRATUM ===")
    print(tbl.to_string(index=False))
    return tbl


def add_avoidable_grade(s: pd.DataFrame, r: pd.DataFrame):
    """Derive AVOIDABLE grade per vehicle: grade-CO2(fastest) - grade-CO2(eco).

    This is the climbing the eco route avoided relative to the fastest route -
    the correct predictor of grade-driven savings. The previously used metric
    (absolute accumulated grade on the eco route) scales with trip length and
    total relief, not with the *opportunity* to avoid climbing, which is why
    it correlated poorly. Derived from the long-format batch_results, so no
    batch re-run is needed.
    """
    if r is None or 'co2_grade_g' not in r.columns:
        return s
    v1 = s['v1'].iloc[0]; v2 = s['v2'].iloc[0]
    for vname, col in [(v1, 'avoid_grade_v1_g'), (v2, 'avoid_grade_v2_g')]:
        sub = r[r['vehicle'] == vname]
        piv = sub.pivot_table(index='od_label', columns='route',
                              values='co2_grade_g', aggfunc='first')
        if 'fastest' in piv.columns and 'eco' in piv.columns:
            av = (piv['fastest'] - piv['eco']).rename(col).reset_index()
            s = s.merge(av, on='od_label', how='left')
    return s


def hypothesis_tests(s: pd.DataFrame, out_dir: Path):
    """H1: heavy vehicle saves more / diverges more than light.
       H2: grade cost correlates with saving (relief matters).
       H3: signal count correlates with saving."""
    try:
        from scipy import stats
        have_scipy = True
    except ImportError:
        have_scipy = False

    v1 = s['v1'].iloc[0]; v2 = s['v2'].iloc[0]
    lines = ["=== HYPOTHESIS TESTS ===\n"]

    # ---- H1: paired difference in saving, heavy vs light --------------
    lines.append("H1 — Mass affects routing benefit")
    lines.append(f"  Paired comparison of CO2 saving: {v2} (heavy) vs {v1} (light),")
    lines.append(f"  same OD pairs.")
    d = s['v2_saving_pct'] - s['v1_saving_pct']
    lines.append(f"  Mean difference (heavy - light): {d.mean():.2f} pp "
                 f"(95% CI +/- {ci95(d):.2f})")
    if have_scipy and len(d.dropna()) >= 2:
        t, p = stats.wilcoxon(s['v2_saving_pct'], s['v1_saving_pct'])
        lines.append(f"  Wilcoxon signed-rank: W={t:.1f}, p={p:.4f} "
                     f"({'significant' if p < 0.05 else 'not significant'} at a=0.05)")
    lines.append("")

    # ---- H2: saving vs AVOIDABLE grade (primary) -----------------------
    lines.append("H2 — Terrain (grade) drives savings")
    lines.append("  Primary predictor: AVOIDABLE grade = grade-CO2(fastest) - grade-CO2(eco),")
    lines.append("  i.e. the climbing the optimizer dodged. (The absolute accumulated grade")
    lines.append("  on the chosen route scales with trip length, not opportunity, and is")
    lines.append("  reported below only for comparison.)")
    for v, scol, acol in [(v1, 'v1_saving_pct', 'avoid_grade_v1_g'),
                          (v2, 'v2_saving_pct', 'avoid_grade_v2_g')]:
        if acol not in s.columns:
            lines.append(f"  {v:<12}: avoidable grade unavailable (no batch_results)")
            continue
        sub = s[[scol, acol]].dropna()
        if have_scipy and len(sub) >= 3 and sub[acol].std() > 0:
            rho, p = stats.spearmanr(sub[acol], sub[scol])
            lines.append(f"  {v:<12}: Spearman rho(AVOIDABLE grade, saving)="
                         f"{rho:.3f}, p={p:.4f}")
        else:
            lines.append(f"  {v:<12}: insufficient variation/scipy for test")
    lines.append("  Secondary (legacy) metric — absolute grade on the eco route:")
    for v, scol, gcol in [(v1, 'v1_saving_pct', 'eco_v1_grade_g'),
                          (v2, 'v2_saving_pct', 'eco_v2_grade_g')]:
        if gcol not in s.columns:
            continue
        sub = s[[scol, gcol]].dropna()
        if have_scipy and len(sub) >= 3 and sub[gcol].std() > 0:
            rho, p = stats.spearmanr(sub[gcol], sub[scol])
            lines.append(f"  {v:<12}: rho(absolute grade, saving)={rho:.3f}, p={p:.4f}")
    lines.append("")

    # ---- H3: saving vs signal count -----------------------------------
    lines.append("H3 — Traffic signals drive savings")
    for v, scol, sigcol in [(v1, 'v1_saving_pct', 'eco_v1_signals'),
                            (v2, 'v2_saving_pct', 'eco_v2_signals')]:
        if sigcol not in s.columns:
            continue
        sub = s[[scol, sigcol]].dropna()
        if have_scipy and len(sub) >= 3 and sub[sigcol].std() > 0:
            rho, p = stats.spearmanr(sub[sigcol], sub[scol])
            lines.append(f"  {v:<12}: Spearman rho(signals, saving)="
                         f"{rho:.3f}, p={p:.4f}")
        else:
            lines.append(f"  {v:<12}: insufficient variation/scipy for test")
    lines.append("")

    if not have_scipy:
        lines.append("NOTE: scipy not installed — correlation/test statistics")
        lines.append("skipped. Install with: pip install scipy")

    txt = '\n'.join(lines)
    (out_dir / 'hypothesis_tests.txt').write_text(txt, encoding='utf-8')
    print("\n" + txt)


def per_pair_table(s: pd.DataFrame, out_dir: Path, r: pd.DataFrame = None):
    s = s.copy()

    # The batch summary records eco_v1_dist_km (Polo) but not the truck's
    # eco distance. Derive eco_v2_dist_km from the long-format batch_results
    # (one row per vehicle-route) so the table carries both vehicles' eco
    # distances. Falls back gracefully if batch_results is unavailable.
    if 'eco_v2_dist_km' not in s.columns and r is not None:
        truck = r[r['vehicle'].str.contains('Volvo', case=False, na=False)
                  & (r['route'] == 'eco')][['od_label', 'distance_km']]
        truck = truck.rename(columns={'distance_km': 'eco_v2_dist_km'})
        s = s.merge(truck, on='od_label', how='left')

    cols = ['region', 'od_label', 'length_stratum', 'network', 'relief',
            'v1_saving_pct', 'v2_saving_pct', 'routes_diverge',
            'eco_v1_signals', 'eco_v2_signals',
            'eco_v1_grade_g', 'eco_v2_grade_g',
            'eco_v1_dist_km', 'eco_v2_dist_km',
            'avoid_grade_v1_g', 'avoid_grade_v2_g']
    cols = [c for c in cols if c in s.columns]
    s[cols].to_csv(out_dir / 'per_pair_table.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    s, r = load(results_dir)
    print(f"Loaded {len(s)} summary rows from {results_dir}\n")

    overall_summary(s, out_dir)
    stratum_tables(s, out_dir)
    s = add_avoidable_grade(s, r)
    hypothesis_tests(s, out_dir)
    per_pair_table(s, out_dir, r)

    print(f"\nAll analysis outputs in: {out_dir}")
    print("  overall_summary.txt      headline numbers")
    print("  savings_by_stratum.csv   mean/CI/n per stratum")
    print("  hypothesis_tests.txt     H1/H2/H3 with p-values")
    print("  per_pair_table.csv       appendix table")


if __name__ == '__main__':
    main()
