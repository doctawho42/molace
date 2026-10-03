#!/usr/bin/env bash
# Runs increment 1's measurements in the order the pre-registration fixes.
#
# The order is not cosmetic. The spine result must exist on disk before the holdout is opened, which
# the holdout loader enforces with a RuntimeError, because opening it first would turn it into a
# second training set.
set -uo pipefail
cd "$(dirname "$0")/.."

step () { echo; echo "=== $* ==="; }

step "1/5 spine sweep (skipped if results/spine.csv exists)"
[ -f results/spine.csv ] || uv run python -c "
from molace.analysis.sweep import run_sweep
run_sweep(seeds=(0,)).to_csv('results/spine.csv', index=False)
print('spine.csv written')
"

step "2/5 frozen analysis on the spine"
uv run python -c "
import pandas as pd
from molace.analysis.correlate import report
from molace.analysis.prereg import prereg_fingerprint
df = pd.read_csv('results/spine.csv')
r = report(df)
print('prereg:', prereg_fingerprint())
print('n =', r['meta']['n'], '| n_eff =', r['meta']['n_eff'],
      '| clusters =', r['meta']['n_clusters'],
      '| assumed intra-class rho =', r['meta']['rho_intra_assumed'])
print()
for k in ('positive_control','primary','incremental_over_rogi','decisive_secondary'):
    b = r[k]
    print(f'{k:24s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  excludes 0: {b.excludes_zero}  n={b.n}')
print()
print('mean correction:', round(r['correction_mean'], 4))
print('PRIMARY CLAIM SUPPORTED:', r['supported'])
print()
print('Read the positive control first. If it does not exclude zero, the pipeline is suspect and')
print('nothing below it is interpreted.')
" 2>&1 | tee results/report_spine.txt

step "3/5 holdout, opened once"
uv run python -c "
import numpy as np, pandas as pd
from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.analysis.prereg import load_prereg
from molace.data.tdc_admet import TDC_REGRESSION, assert_spine_recorded, load_tdc
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
from molace.models.gap import evaluate_target
assert_spine_recorded()
k = load_prereg()['graphs']['primary']['k']
rows = []
for name in TDC_REGRESSION:
    df = load_tdc(name)
    g = knn.knn_graph(tanimoto_matrix(ecfp4(df['smiles'].tolist())), k=k)
    r = evaluate_target(df, seed=0)
    rows.append({'dataset': name, 'n': len(df), 'receptor_class': name,
                 'assortativity': target_assortativity(g, df['y'].to_numpy(float)).value,
                 'gap': r.gap, 'access': r.access, 'correction': r.correction})
    print(' ', name, len(df), round(rows[-1]['assortativity'], 3), round(r.gap, 3), flush=True)
t = pd.DataFrame(rows); t.to_csv('results/holdout_tdc.csv', index=False)
print()
for col in ('access','gap','correction'):
    b = cluster_bootstrap_spearman(t.assortativity, t[col], t.receptor_class, 10000, 0)
    print(f'{col:12s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]')
print()
print('n = 9. Reported once. No tuning follows this.')
" 2>&1 | tee results/report_holdout.txt

step "4/5 Hodge controls, the criterion, and the readout ablation"
uv run python -c "
import numpy as np, pandas as pd
from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.hodge import complex as cx, controls
from molace.hodge.energy import readout_ablation
k = load_prereg()['graphs']['primary']['k']
rows, abl = [], []
for name in ma.DATASETS:
    df = ma.load_target(name)
    try:
        e = controls.energy_controls(df, seed=0, k=k, triangle_budget=200000)
        c = controls.curl_versus_dispersion(df, seed=0, k=k, triangle_budget=200000)
    except ValueError as exc:
        print(' skip', name, exc, flush=True); continue
    rows.append({'dataset': name, 'receptor_class': ma.receptor_class(name),
                 'curl_trained': e['trained']['curl'], 'harm_trained': e['trained']['harmonic'],
                 'grad_trained': e['trained']['gradient'],
                 'curl_shuffled': e['shuffled']['curl'], 'curl_floor': e['pointwise_floor']['curl'],
                 **c})
    fp = ecfp4(df['smiles'].tolist())
    cc = cx.build(knn.knn_graph(tanimoto_matrix(fp), k=k), triangle_budget=50000)
    a = readout_ablation(fp, df['y'].to_numpy(float), cc, seed=0)
    abl.append({'dataset': name, 'curl_linear': a['linear']['curl'], 'curl_mlp': a['mlp']['curl']})
    print(f\"  {name:16s} curl={rows[-1]['curl_trained']:.4f} harm={rows[-1]['harm_trained']:.4f} \"
          f\"rho_diff={rows[-1]['rho_difference']:+.3f}\", flush=True)
t = pd.DataFrame(rows); t.to_csv('results/hodge_controls.csv', index=False)
pd.DataFrame(abl).to_csv('results/readout_ablation.csv', index=False)
print()
print('prereg:', prereg_fingerprint())
print('VALIDITY CHECK, read first -- max curl fraction of the pointwise floor (must be ~0):',
      float(t.curl_floor.max()))
print('median curl fraction, trained:', round(float(t.curl_trained.median()), 4))
print('median harmonic fraction, trained:', round(float(t.harm_trained.median()), 4))
print('median curl fraction, shuffled labels:', round(float(t.curl_shuffled.median()), 4))
print()
print('mean rho difference (curl minus dispersion):', round(float(t.rho_difference.mean()), 4))
boot = [float(np.mean(np.random.default_rng(i).choice(t.rho_difference, len(t)))) for i in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f'95% bootstrap CI on the mean difference: [{lo:+.4f}, {hi:+.4f}]')
print('CRITERION PASSED (curl beats anchor dispersion):', bool(lo > 0))
" 2>&1 | tee results/report_hodge.txt

step "5/5 figures"
uv run python -c "
from molace.analysis.figures import make_figures
for p in make_figures():
    print(' ', p)
"

step "done"
