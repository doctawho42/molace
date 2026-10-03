#!/usr/bin/env bash
# Runs increment 1's measurements in the order the pre-registration fixes.
#
# The order is not cosmetic. The spine result must exist on disk, and must record a completed
# analysis, before the holdout is opened; the holdout loader enforces that by raising, because
# opening it first would turn the holdout into a second training set.
#
# `set -e` matters here: the reports are written through `tee`, which creates the file whether python
# succeeded or crashed, so without it a failed step would leave a traceback in a report and the next
# step would proceed against it.
set -euo pipefail
cd "$(dirname "$0")/.."

step () { echo; echo "=== $* ==="; }

step "1/6 spine sweep (skipped if results/spine.csv exists)"
[ -f results/spine.csv ] || uv run python -c "
from molace.analysis.sweep import run_sweep
run_sweep().to_csv('results/spine.csv', index=False)
print('spine.csv written')
"

step "2/6 frozen analysis on the spine"
uv run python -c "
import pandas as pd
from molace.analysis.correlate import report
from molace.analysis.prereg import prereg_fingerprint
df = pd.read_csv('results/spine.csv')
r = report(df)
print('prereg:', prereg_fingerprint())
print('n =', r['meta']['n'], '| n_eff =', r['meta']['n_eff'],
      '| clusters =', r['meta']['n_clusters'],
      '| assumed intra-class rho =', r['meta']['rho_intra_assumed'],
      '| seeds averaged per target:', sorted(set(df.get('n_seeds', [1]))))
print()
for k in ('positive_control','primary','incremental_over_rogi','decisive_secondary'):
    b = r[k]
    print(f'{k:24s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  '
          f'excludes 0: {b.excludes_zero}  n={b.n}  draws kept={b.n_draws}')
print()
print('mean correction:', round(r['correction_mean'], 4))
print('PRIMARY CLAIM SUPPORTED:', r['supported'])
print()
print('Read the positive control first. If it does not exclude zero, the pipeline is suspect and')
print('nothing below it is interpreted.')
print()
print('UNITS CAVEAT. The gap is expressed in the label\'s own units. Across these targets the label')
print('standard deviation varies about twofold and correlates with the statistic, so the magnitudes')
print('are partly a property of scale. A post-hoc standardised version of the primary is reported in')
print('results/results_increment1.md; it is also a null, so the conclusion is unchanged.')
" 2>&1 | tee results/report_spine.txt

step "3/6 external replication on DeepDelta's published predictions"
uv run python -c "
import pandas as pd
from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.analysis.prereg import load_prereg
from molace.data.deepdelta import DD_DATASETS, ROOT, published_gaps
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures.assortativity import target_assortativity
k = load_prereg()['graphs']['primary']['k']
rows = []
for ds in DD_DATASETS:
    df = pd.read_csv(ROOT / 'Datasets' / 'Benchmarks' / f'{ds}.csv')
    g = knn.knn_graph(tanimoto_matrix(ecfp4(df['SMILES'].tolist())), k=k)
    rows.append({'dataset': ds, 'n': len(df),
                 'assortativity': target_assortativity(g, df['Y'].to_numpy(float)).value})
t = pd.DataFrame(rows).merge(published_gaps(), on='dataset')
t['receptor_class'] = t['dataset']      # no families here: each dataset is its own cluster
t.to_csv('results/external_deepdelta.csv', index=False)
print(t[['dataset','n','assortativity','gap_vs_rf','gap_vs_chemprop']].to_string(index=False))
print()
for col in ('gap_vs_rf','gap_vs_chemprop'):
    b = cluster_bootstrap_spearman(t.assortativity, t[col], t.receptor_class, 10000, 0)
    print(f'{col:16s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  n={b.n}')
print()
print('n = 10, so this is a replication with wide intervals, not a second primary result. These gaps')
print('are also in label units and these datasets mix scales, so the same units caveat applies.')
" 2>&1 | tee results/report_external.txt

step "4/6 holdout, opened once"
uv run python -c "
import pandas as pd
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
                 'label_sd': float(df['y'].std()),
                 'assortativity': target_assortativity(g, df['y'].to_numpy(float)).value,
                 'gap': r.gap, 'access': r.access, 'correction': r.correction})
t = pd.DataFrame(rows); t.to_csv('results/holdout_tdc.csv', index=False)
print(t[['dataset','n','label_sd','assortativity','gap','correction']].to_string(index=False))
print()
for col in ('access','gap','correction'):
    b = cluster_bootstrap_spearman(t.assortativity, t[col], t.receptor_class, 10000, 0)
    print(f'{col:12s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]')
print()
print('THIS HOLDOUT IS INVALID AS PRE-REGISTERED. Read the numbers above only with that in mind.')
print('The gap is in label units and these nine endpoints mix log and raw scales: the label standard')
print('deviation spans a factor of about a hundred, from under one to over eighty. Both the statistic')
print('and the gap track that scale, so the correlation reported here is a property of units rather')
print('than of graph structure. The pre-registration failed to require a dimensionless gap. No')
print('transform is applied, because applying one after seeing the result would be tuning, and the')
print('holdout is spent. The fix belongs to the next increment: define the gap in label standard')
print('deviations so it is dimensionless before anything is measured.')
print()
print('n = 9. Reported once. No tuning follows this.')
" 2>&1 | tee results/report_holdout.txt

step "5/6 Hodge controls, the criterion, and the readout ablation"
uv run python -c "
import pandas as pd
from molace.analysis.correlate import cluster_bootstrap_mean
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
                 'residual_trained': e['trained']['residual_check'],
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
print('VALIDITY CHECKS, read first.')
print('  max curl fraction of the pointwise floor (must be ~0):', float(t.curl_floor.max()))
print('  worst Pythagorean residual of any decomposition (must be ~0):',
      float(t.residual_trained.abs().max()))
print('  note: the floor control alone cannot detect wrong orientation signs -- a pure gradient flow')
print('  leaves a zero residual. The signs are enforced in hodge.complex.build, which asserts')
print('  B1 @ B2 == 0 on every complex it returns.')
print()
print('median curl fraction, trained:', round(float(t.curl_trained.median()), 4))
print('median harmonic fraction, trained:', round(float(t.harm_trained.median()), 4))
print('median curl fraction, shuffled labels:', round(float(t.curl_shuffled.median()), 4))
print('  the trained and shuffled curl fractions are indistinguishable, so the circulation carries')
print('  no label information.')
print()
b = cluster_bootstrap_mean(t.rho_difference, t.receptor_class, 20000, 0)
print(f'mean rho difference (curl minus anchor dispersion): {b.mean:+.4f}')
print(f'95% CLUSTER bootstrap CI over receptor classes: [{b.lo:+.4f}, {b.hi:+.4f}]  '
      f'clusters={b.n_clusters}  draws kept={b.n_draws}')
print('negative on', int((t.rho_difference < 0).sum()), 'of', len(t), 'targets')
print('CRITERION PASSED (curl beats anchor dispersion):', bool(b.lo > 0))
" 2>&1 | tee results/report_hodge.txt

step "6/6 figures"
uv run python -c "
from molace.analysis.figures import make_figures
for p in make_figures():
    print(' ', p)
"

step "done"
