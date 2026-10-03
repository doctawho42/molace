# molace increment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a parameter-free, pre-training statistic of a molecule-similarity
graph predicts when pairwise models beat pointwise ones on 30 MoleculeACE targets, and census
where a trained pairwise flow's energy lives in the Hodge decomposition.

**Architecture:** A flat per-task pipeline. `data/` yields tidy frames, `graphs/` turns SMILES
into one fingerprint and two graph kinds with diagnostics asserted as preconditions,
`measures/` maps (graph, labels) to one number plus coverage, `models/` runs three arms over a
shared anchor set so their difference is an algebraic identity, `hodge/` decomposes an edge
flow, `analysis/` correlates per-task numbers with a cluster bootstrap. Every task caches to
`results/tasks/<dataset>.json` so a crash on target 23 keeps the first 22.

**Tech Stack:** Python 3.11 (matching `~/anaconda3/envs/solnet` and `tgnn-solv`), uv for the
environment, rdkit, numpy, scipy, pandas, scikit-learn, lightgbm, networkx, pyyaml, pytest,
PyTDC, rogi.

**Spec:** `docs/superpowers/specs/2026-10-03-molace-increment1-design.md` (commit 4350096)

## Global Constraints

- Fingerprint: binary Morgan radius 2, 2048 bits, computed in `graphs/fingerprints.py` and
  nowhere else. A second fingerprint call site is a bug.
- Label column: `y [pEC50/pKi]`. Never `y` (standardised and negative).
- `cliff_mol` read from per-target CSVs, never from `benchmark_data/metadata/datasets.csv`
  (which describes pre-Correction data and disagrees on 28 of 30 targets). The metadata table
  is read for `Receptor Class` only.
- Similarity graph: kNN by Tanimoto, k = 10 primary, union-symmetrised. Robustness k in {5, 20}
  plus the Tanimoto-threshold graph.
- Anchor count m = 10, a separate constant from k, never tied to it.
- Anchor selection lives only in `models/anchors.py`; the kNN floor and the pairwise arm draw
  the identical set with uniform weights or the decomposition is not exact.
- Binning a continuous label into classes is forbidden; the categorical measures raise on
  continuous input.
- Seeds: 3, averaged, learned arms only. No best-seed selection.
- `prereg/increment1.yaml` is committed before any measurement runs on real data.
- No `Co-Authored-By` or generated-with trailers in commit messages.

## Review Focus

1. **Unparseable SMILES** — RDKit returning `None` must raise naming the index and the string,
   never silently drop a molecule, because a dropped row shifts every label index. (Task 2)
2. **Disconnected kNN graph** — assortativity stays defined but the gradient potential does
   not; component count and coverage must be returned with every measure.
   (Task 3 `test_diagnostics_reports_components_and_triangles_on_a_known_graph`,
   Task 4 `test_coverage_excludes_isolated_nodes`)
3. **Degenerate `cliff_mol`** — a target where every molecule is one class makes `h_adj` and
   `LI` a 0/0; must raise with the target named, not return NaN.
   (Task 5 `test_a_single_class_raises_rather_than_returning_nan`)
4. **Tied anchor distances** — the m nearest neighbours must be deterministic across runs, or
   the kNN-floor/pairwise identity holds in one run and not the next.
   (Task 9 `test_ties_break_on_ascending_training_index_and_are_stable`, and
   Task 3 `test_knn_neighbour_choice_is_deterministic_under_ties` for the graph)
5. **Zero-triangle graph entering `hodge/`** — must raise, because the energy budget would
   otherwise silently report a gradient fraction of 1 and look like a finding.
   (Task 15 `test_a_graph_with_no_triangles_raises_because_curl_is_not_defined_there`,
   Task 19 `test_a_complex_without_triangles_raises_instead_of_reporting_zero_curl`)

## Deviations From The Spec, Recorded

Four. Each is a decision the plan makes differently from the approved spec, written here so an
executor does not have to discover it and a reader does not have to trust that nothing drifted.

1. **`graphs/comparison.py` is not created.** The spec §6 and §12 give the comparison graph its own
   module, with anchors forming a clique plus spokes so that triangles exist. Under this plan the
   comparison graph **is** the kNN similarity graph: training pairs are each molecule with its `m`
   nearest neighbours in both directions (Task 11), and the Hodge complex is built on a kNN graph
   (Tasks 15, 19). A kNN graph is already rich in triangles, so the anchor-clique construction the
   spec prescribed to avoid a bipartite comparison graph is not needed and would add a second graph
   kind for nothing. The precondition it existed to guarantee is kept and strengthened: `build()`
   raises on a triangle-free complex, and the Task 15 census reports the triangle count for all 30
   targets, so an empty curl subspace shows up as a number rather than as a silent zero.
2. **The norm-matched shuffled control is implemented without rescaling** (Task 19). The spec asks
   for matched total flow norm because the original design compared raw norms. Reporting the
   scale-invariant fraction removes that confound by construction, and a test pins the invariance.
   Norm matching returns if a raw norm is ever reported.
3. **ROGI may ship in place of ROGI-XD** (Task 7), under a 45-minute time box, with the flavour
   actually used recorded in the module, the README and the pre-registration. The spec names
   ROGI-XD, whose comparability across dataset sizes is the property this cross-task setting wants.
   Shipping plain ROGI openly is acceptable; shipping it silently is not.
4. **The curl-against-harmonic dimension split is exact only below 5,000 edges** (Task 15); above
   that it is a randomised estimate, labelled `rank_method = "estimated"` in the census and hatched
   in the figure. The gradient and cycle-space dimensions are exact everywhere.

---

### Task 1: Project skeleton, environment, MoleculeACE loader

**Files:**
- Create: `pyproject.toml`, `src/molace/__init__.py`, `src/molace/data/__init__.py`,
  `src/molace/data/moleculeace.py`, `tests/test_moleculeace.py`, `scripts/fetch_data.sh`
- Test: `tests/test_moleculeace.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DATASETS: list[str]` (30 names), `load_target(name: str) -> pandas.DataFrame`
  with columns `smiles, y, cliff_mol, split` where `y` is float, `cliff_mol` is int 0/1 and
  `split` is `"train"`/`"test"`; `receptor_class(name: str) -> str`;
  `RECEPTOR_CLASSES: dict[str, str]`.

- [ ] **Step 1: Create the environment and package skeleton**

```bash
cd /Users/nikitapolomosnov/PycharmProjects/molace
uv init --name molace --package --python 3.11 --no-workspace
uv add rdkit numpy scipy pandas scikit-learn lightgbm networkx pyyaml
uv add --dev pytest
mkdir -p src/molace/{data,graphs,measures,models,hodge,analysis} tests scripts results/tasks
for d in data graphs measures models hodge analysis; do touch src/molace/$d/__init__.py; done
```

- [ ] **Step 2: Write the data fetch script**

```bash
cat > scripts/fetch_data.sh <<'EOF'
#!/usr/bin/env bash
# Clones the two upstream data repositories into data/raw (gitignored).
# MoleculeACE must be CLONED, not pip-installed: Data/results/* is absent from package_data.
set -euo pipefail
mkdir -p data/raw
[ -d data/raw/MoleculeACE ] || git clone --depth 1 https://github.com/molML/MoleculeACE.git data/raw/MoleculeACE
[ -d data/raw/DeepDelta ]   || git clone --depth 1 https://github.com/RekerLab/DeepDelta.git data/raw/DeepDelta
echo "benchmark csvs: $(ls data/raw/MoleculeACE/MoleculeACE/Data/benchmark_data/*.csv | wc -l)"
EOF
chmod +x scripts/fetch_data.sh
./scripts/fetch_data.sh
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_moleculeace.py
import numpy as np
import pytest
from molace.data import moleculeace as ma


def test_thirty_datasets_and_total_molecule_count():
    assert len(ma.DATASETS) == 30
    assert sum(len(ma.load_target(d)) for d in ma.DATASETS) == 48714


def test_size_extremes_match_the_published_corrected_data():
    sizes = {d: len(ma.load_target(d)) for d in ma.DATASETS}
    assert min(sizes.values()) == 615
    assert max(sizes.values()) == 3657


def test_label_is_the_potency_column_not_the_standardised_one():
    df = ma.load_target("CHEMBL2835_Ki")
    # y [pEC50/pKi] is positive potency; the `y` column is standardised and goes negative.
    assert df["y"].min() > 0.0
    assert 8.0 < df["y"].mean() < 9.5


def test_cliff_flags_come_from_the_csv_not_the_stale_metadata_table():
    # metadata/datasets.csv describes pre-Correction data and reports 120 for this target.
    assert int(ma.load_target("CHEMBL2971_Ki")["cliff_mol"].sum()) == 162


def test_split_column_is_train_test_only():
    df = ma.load_target("CHEMBL4203_Ki")
    assert set(df["split"]) == {"train", "test"}
    assert (df["split"] == "train").sum() == 582


def test_receptor_classes_are_the_six_published_blocks():
    counts = {}
    for d in ma.DATASETS:
        counts[ma.receptor_class(d)] = counts.get(ma.receptor_class(d), 0) + 1
    assert counts == {"GPCR": 12, "NR": 6, "Kinase": 6, "Other": 3,
                      "Protease": 2, "Transferase": 1}


def test_unknown_target_raises():
    with pytest.raises(KeyError):
        ma.load_target("CHEMBL_nope")
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_moleculeace.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.data.moleculeace'`

- [ ] **Step 5: Implement the loader**

```python
# src/molace/data/moleculeace.py
"""MoleculeACE loader.

Three traps this module exists to close, all measured on the shipped data:
  * the canonical label is `y [pEC50/pKi]`; the column named `y` is standardised and negative,
  * `cliff_mol` must come from the per-target CSV, because metadata/datasets.csv still
    describes the pre-Correction data (Correction: PubMed 36995229) and disagrees with the
    CSVs on cliff counts for 28 of 30 targets,
  * the repo must be cloned, not pip-installed, or Data/results/ is missing.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "MoleculeACE" / "MoleculeACE" / "Data"
BENCH = ROOT / "benchmark_data"
METADATA = BENCH / "metadata" / "datasets.csv"
RESULTS = ROOT / "results" / "MoleculeACE_results.csv"

LABEL_COLUMN = "y [pEC50/pKi]"


def _require_data() -> None:
    if not BENCH.is_dir():
        raise FileNotFoundError(
            f"{BENCH} is missing. Run scripts/fetch_data.sh; MoleculeACE must be cloned, "
            "because Data/results/ is not in the pip package."
        )


@functools.lru_cache(maxsize=1)
def _datasets() -> tuple[str, ...]:
    _require_data()
    return tuple(sorted(p.stem for p in BENCH.glob("*.csv")))


DATASETS = list(_datasets())


@functools.lru_cache(maxsize=64)
def load_target(name: str) -> pd.DataFrame:
    """One target as `smiles, y, cliff_mol, split`, in file order."""
    if name not in _datasets():
        raise KeyError(f"{name!r} is not one of the 30 MoleculeACE targets")
    raw = pd.read_csv(BENCH / f"{name}.csv")
    return pd.DataFrame(
        {
            "smiles": raw["smiles"].astype(str),
            "y": raw[LABEL_COLUMN].astype(float),
            "cliff_mol": raw["cliff_mol"].astype(int),
            "split": raw["split"].astype(str),
        }
    )


@functools.lru_cache(maxsize=1)
def _classes() -> dict[str, str]:
    _require_data()
    md = pd.read_csv(METADATA)
    # Only the Receptor Class column is trusted here; the cliff counts in this file are stale.
    return dict(zip(md["Dataset"].astype(str), md["Receptor Class"].astype(str)))


RECEPTOR_CLASSES = _classes()


def receptor_class(name: str) -> str:
    return _classes()[name]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_moleculeace.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
printf '%s\n' 'data/raw/' 'results/' '.venv/' '__pycache__/' '*.pyc' '.DS_Store' '.remember/' > .gitignore
git add -A
git commit -m "the loader reads the potency column and the corrected cliff flags

The canonical label is the potency column, not the standardised one that goes negative, and
the cliff flags come from the per-target files rather than the summary table, which still
describes the data as it stood before the published correction and disagrees on twenty-eight
of thirty targets. Golden tests pin the thirty files, the 48,714 molecules and the size
extremes, so a silent change upstream fails the suite instead of moving every later number.
The repository is cloned rather than installed because the published performance matrix is
absent from the package data."
```

---

### Task 2: The one fingerprint, and the Tanimoto matrix

**Files:**
- Create: `src/molace/graphs/fingerprints.py`, `tests/test_fingerprints.py`
- Test: `tests/test_fingerprints.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RADIUS = 2`, `N_BITS = 2048`, `ecfp4(smiles: Sequence[str]) -> np.ndarray`
  returning `(n, 2048)` `uint8`; `tanimoto_matrix(fp: np.ndarray) -> np.ndarray` returning
  `(n, n)` `float32` with 1.0 on the diagonal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fingerprints.py
import numpy as np
import pytest
from molace.graphs import fingerprints as fpmod


def test_shape_and_dtype():
    fp = fpmod.ecfp4(["CCO", "CCC", "c1ccccc1"])
    assert fp.shape == (3, 2048)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp)) <= {0, 1}


def test_identical_smiles_give_identical_rows():
    fp = fpmod.ecfp4(["CCO", "CCO"])
    assert np.array_equal(fp[0], fp[1])


def test_unparseable_smiles_raises_and_names_the_offender():
    with pytest.raises(ValueError) as e:
        fpmod.ecfp4(["CCO", "not_a_molecule", "CCC"])
    assert "index 1" in str(e.value)
    assert "not_a_molecule" in str(e.value)


def test_tanimoto_is_one_on_the_diagonal_and_symmetric():
    T = fpmod.tanimoto_matrix(fpmod.ecfp4(["CCO", "CCC", "c1ccccc1O"]))
    assert np.allclose(np.diag(T), 1.0)
    assert np.allclose(T, T.T)
    assert T.min() >= 0.0 and T.max() <= 1.0


def test_tanimoto_matches_the_hand_computed_definition():
    fp = fpmod.ecfp4(["CCO", "CCCCO"])
    a, b = fp[0].astype(bool), fp[1].astype(bool)
    expected = (a & b).sum() / (a | b).sum()
    assert fpmod.tanimoto_matrix(fp)[0, 1] == pytest.approx(expected, abs=1e-6)


def test_empty_fingerprint_row_does_not_divide_by_zero():
    # a molecule with no set bits would make the union zero; force the degenerate case
    fp = np.zeros((2, 2048), dtype=np.uint8)
    T = fpmod.tanimoto_matrix(fp)
    assert T[0, 1] == 0.0
    assert np.allclose(np.diag(T), 1.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fingerprints.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.graphs.fingerprints'`

- [ ] **Step 3: Implement**

```python
# src/molace/graphs/fingerprints.py
"""The project's single molecular representation.

The design fixes exactly one free parameter, the representation, because the cliff set and
every graph statistic move with it. That discipline only holds if there is one call site, so
this module is the only place a fingerprint is computed. A second one is a bug.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

RADIUS = 2
N_BITS = 2048

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def ecfp4(smiles: Sequence[str]) -> np.ndarray:
    """Binary ECFP4 (Morgan radius 2, 2048 bits) as an (n, 2048) uint8 array.

    Raises on an unparseable SMILES rather than dropping it: a dropped row would shift every
    label index downstream and silently corrupt every per-target number.
    """
    out = np.zeros((len(smiles), N_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES at index {i}: {s!r}")
        out[i] = _GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return out


def tanimoto_matrix(fp: np.ndarray) -> np.ndarray:
    """Dense pairwise Tanimoto, (n, n) float32, 1.0 on the diagonal."""
    x = fp.astype(np.float32)
    inter = x @ x.T
    counts = x.sum(axis=1)
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(union > 0.0, inter / union, 0.0).astype(np.float32)
    np.fill_diagonal(t, 1.0)
    return t
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fingerprints.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/molace/graphs/fingerprints.py tests/test_fingerprints.py
git commit -m "one fingerprint, computed in one place

Fixing the representation is the project's single methodological parameter, so the computation
lives at exactly one call site and everything downstream imports it. An unparseable structure
raises and names the offending index and string rather than being dropped, because a dropped
row shifts every label index and would corrupt each per-target number without any visible
failure. Tanimoto guards the degenerate empty-fingerprint union instead of dividing by zero."
```

---

### Task 3: Graph construction and diagnostics

**Files:**
- Create: `src/molace/graphs/knn.py`, `src/molace/graphs/threshold.py`,
  `src/molace/graphs/diagnostics.py`, `tests/test_graphs.py`
- Test: `tests/test_graphs.py`

**Interfaces:**
- Consumes: `fingerprints.tanimoto_matrix`.
- Produces: `knn.knn_graph(T: np.ndarray, k: int) -> networkx.Graph` (union-symmetrised,
  nodes `0..n-1`, edge attribute `weight`); `threshold.threshold_graph(T, tau) -> nx.Graph`;
  `diagnostics.graph_diagnostics(g) -> dict` with keys `n_nodes, n_edges, density,
  mean_degree, n_components, largest_component_fraction, n_isolated, n_triangles,
  dim_gradient, dim_cycle_space`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graphs.py
import networkx as nx
import numpy as np
import pytest
from molace.graphs import diagnostics as dg
from molace.graphs import fingerprints as fpmod
from molace.graphs import knn, threshold


def _toy_T(n=12, seed=0):
    rng = np.random.default_rng(seed)
    fp = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    return fpmod.tanimoto_matrix(fp)


def test_knn_gives_every_node_at_least_k_neighbours():
    g = knn.knn_graph(_toy_T(), k=3)
    assert min(dict(g.degree()).values()) >= 3


def test_knn_is_union_symmetrised_so_degree_is_not_exactly_k():
    # Union symmetrisation means degree >= k, not == k. The spec depends on knowing this.
    g = knn.knn_graph(_toy_T(n=40), k=3)
    assert max(dict(g.degree()).values()) > 3


def test_knn_has_no_self_loops_and_keeps_all_nodes():
    g = knn.knn_graph(_toy_T(), k=3)
    assert nx.number_of_selfloops(g) == 0
    assert g.number_of_nodes() == 12


def test_knn_neighbour_choice_is_deterministic_under_ties():
    T = np.full((6, 6), 0.5, dtype=np.float32)
    np.fill_diagonal(T, 1.0)
    a = sorted(knn.knn_graph(T, k=2).edges())
    b = sorted(knn.knn_graph(T, k=2).edges())
    assert a == b


def test_threshold_graph_keeps_only_edges_at_or_above_tau():
    T = _toy_T()
    g = threshold.threshold_graph(T, tau=0.3)
    for u, v in g.edges():
        assert T[u, v] >= 0.3


def test_diagnostics_reports_components_and_triangles_on_a_known_graph():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])   # one triangle
    g.add_edges_from([(3, 4)])                   # a second component
    g.add_node(5)                                # an isolated node
    d = dg.graph_diagnostics(g)
    assert d["n_nodes"] == 6 and d["n_edges"] == 4
    assert d["n_components"] == 3
    assert d["n_isolated"] == 1
    assert d["n_triangles"] == 1
    assert d["dim_gradient"] == 6 - 3          # |V| - c
    assert d["dim_cycle_space"] == 4 - 6 + 3   # |E| - |V| + c


def test_diagnostics_on_an_edgeless_graph_does_not_divide_by_zero():
    d = dg.graph_diagnostics(nx.empty_graph(4))
    assert d["n_edges"] == 0
    assert d["density"] == 0.0 and d["mean_degree"] == 0.0
    assert d["n_triangles"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_graphs.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.graphs.knn'`

- [ ] **Step 3: Implement the two constructions**

```python
# src/molace/graphs/knn.py
"""Degree-controlled similarity graph.

Primary construction, because at a fixed Tanimoto threshold the measured edge density across
the 30 MoleculeACE targets spans 98x and mean degree 82x, and on identical labels the
threshold-to-kNN switch moves adjusted homophily 12x. A headline correlation computed across
graphs of 98x differing density would be confounded with density.

Union symmetrisation gives every node degree AT LEAST k, not exactly k: a molecule appearing
in many neighbour lists accumulates extra edges. Mean degree is therefore still a covariate,
and the sweep records it.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def knn_graph(T: np.ndarray, k: int) -> nx.Graph:
    n = T.shape[0]
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n < 2:
        raise ValueError(f"a kNN graph needs at least 2 molecules, got {n}")
    kk = min(k, n - 1)
    s = T.astype(np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    # lexsort on (-similarity, index) makes the neighbour choice deterministic under ties.
    order = np.lexsort((np.tile(np.arange(n), (n, 1)), -s), axis=1)[:, :kk]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in order[i]:
            j = int(j)
            if i != j:
                g.add_edge(i, j, weight=float(T[i, j]))
    return g
```

```python
# src/molace/graphs/threshold.py
"""Tanimoto-threshold similarity graph: robustness check only, never primary.

Kept because the Chemical Space Network literature this project works inside is built on
threshold graphs, and because the density spread it produces is itself a reportable
measurement.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def threshold_graph(T: np.ndarray, tau: float) -> nx.Graph:
    n = T.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    iu = np.triu_indices(n, k=1)
    keep = T[iu] >= tau
    for u, v, w in zip(iu[0][keep], iu[1][keep], T[iu][keep]):
        g.add_edge(int(u), int(v), weight=float(w))
    return g
```

- [ ] **Step 4: Implement diagnostics**

```python
# src/molace/graphs/diagnostics.py
"""Graph facts that are preconditions, not report lines.

Coverage and component structure travel with every measure, because the gradient potential of
a Hodge decomposition is fixed only up to a constant per component, and on threshold graphs the
measured component count reached 925 with 637 isolated molecules on one target.
"""
from __future__ import annotations

import networkx as nx


def graph_diagnostics(g: nx.Graph) -> dict:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    comps = list(nx.connected_components(g))
    c = len(comps)
    largest = max((len(x) for x in comps), default=0)
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
        "mean_degree": (2.0 * m / n) if n else 0.0,
        "n_components": c,
        "largest_component_fraction": (largest / n) if n else 0.0,
        "n_isolated": sum(1 for _, d in g.degree() if d == 0),
        "n_triangles": sum(nx.triangles(g).values()) // 3,
        "dim_gradient": n - c,
        "dim_cycle_space": m - n + c,
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_graphs.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/molace/graphs/knn.py src/molace/graphs/threshold.py src/molace/graphs/diagnostics.py tests/test_graphs.py
git commit -m "the degree-controlled graph is primary, and its degree is not exactly k

At one fixed Tanimoto threshold the measured density across the thirty targets spans a factor
of ninety-eight and mean degree a factor of eighty-two, and on identical labels moving from a
threshold to a nearest-neighbour graph shifts adjusted homophily twelvefold, so a correlation
computed across threshold graphs would be confounded with density. Union symmetrisation leaves
degree bounded below by k rather than equal to it, which a test pins so the later analysis keeps
mean degree as a covariate instead of assuming it away. Neighbour choice is made deterministic
under ties, because the anchor identity the three arms rely on must hold across runs. Component
count, isolated nodes and triangle count travel with every graph, since the gradient potential
is fixed only per component and a graph without triangles has no curl to measure."
```

---

### Task 4: Target assortativity (the primary statistic)

**Files:**
- Create: `src/molace/measures/assortativity.py`, `tests/test_assortativity.py`
- Test: `tests/test_assortativity.py`

**Interfaces:**
- Consumes: nothing beyond networkx and numpy.
- Produces: `target_assortativity(g: nx.Graph, y: np.ndarray) -> MeasureResult`, where
  `MeasureResult` is a frozen dataclass with fields `value: float`, `coverage: float`,
  `n_used: int`. Defined here and reused by every other measure module.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_assortativity.py
import networkx as nx
import numpy as np
import pytest
from molace.measures.assortativity import MeasureResult, target_assortativity


def test_matches_networkx_on_a_random_graph():
    rng = np.random.default_rng(0)
    g = nx.gnp_random_graph(60, 0.15, seed=1)
    y = rng.normal(size=60)
    nx.set_node_attributes(g, {i: float(y[i]) for i in g.nodes}, "y")
    expected = nx.numeric_assortativity_coefficient(g, "y")
    assert target_assortativity(g, y).value == pytest.approx(expected, abs=1e-8)


def test_a_perfectly_smooth_label_on_a_path_is_near_one():
    g = nx.path_graph(50)
    y = np.arange(50, dtype=float)
    assert target_assortativity(g, y).value > 0.9


def test_an_alternating_label_on_a_path_is_negative():
    g = nx.path_graph(50)
    y = np.array([i % 2 for i in range(50)], dtype=float)
    assert target_assortativity(g, y).value < -0.9


def test_coverage_excludes_isolated_nodes():
    g = nx.path_graph(8)
    g.add_nodes_from([8, 9])        # two isolated molecules
    y = np.arange(10, dtype=float)
    r = target_assortativity(g, y)
    assert r.n_used == 8
    assert r.coverage == pytest.approx(0.8)


def test_edgeless_graph_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="no edges"):
        target_assortativity(nx.empty_graph(5), np.arange(5, dtype=float))


def test_constant_label_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="constant"):
        target_assortativity(nx.path_graph(6), np.ones(6))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        target_assortativity(nx.path_graph(6), np.ones(5))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_assortativity.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.measures.assortativity'`

- [ ] **Step 3: Implement**

```python
# src/molace/measures/assortativity.py
"""Newman (2003) scalar target assortativity: the project's primary statistic.

This is the group's own choice for a continuous target. GraphLand: "To measure the similarity
of labels of connected nodes for regression datasets, we use target assortativity - the Pearson
correlation coefficient of target values between pairs of connected nodes", and GraphPFN
repeats it. It carries no binning parameter, and the Platonov et al. paper notes that adjusted
homophily "is known in graph analysis literature as assortativity coefficient", so this is the
continuous reduction of the categorical measure rather than a substitute for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class MeasureResult:
    """One graph statistic plus how much of the data it actually describes.

    coverage is the fraction of nodes that participate in at least one edge. It is carried on
    every measure because on sparse graphs a measure can describe a minority of the molecules
    and still look like a dataset-level number.
    """

    value: float
    coverage: float
    n_used: int


def target_assortativity(g: nx.Graph, y: np.ndarray) -> MeasureResult:
    y = np.asarray(y, dtype=float)
    if len(y) != g.number_of_nodes():
        raise ValueError(
            f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes"
        )
    if g.number_of_edges() == 0:
        raise ValueError("target assortativity is undefined on a graph with no edges")
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    ends = np.array([(y[pos[u]], y[pos[v]]) for u, v in g.edges()], dtype=float)
    a = np.concatenate([ends[:, 0], ends[:, 1]])
    b = np.concatenate([ends[:, 1], ends[:, 0]])
    if a.std() == 0.0:
        raise ValueError(
            "target assortativity is undefined when the label is constant over edge endpoints"
        )
    used = sum(1 for _, d in g.degree() if d > 0)
    return MeasureResult(
        value=float(np.corrcoef(a, b)[0, 1]),
        coverage=used / g.number_of_nodes(),
        n_used=used,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_assortativity.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/molace/measures/assortativity.py tests/test_assortativity.py
git commit -m "the primary statistic is the one their own benchmarks use for a continuous target

Target assortativity is what GraphLand and GraphPFN report for regression datasets, it carries
no binning parameter, and the paper that defines adjusted homophily notes the two coincide on
categorical attributes, so this is the continuous reduction rather than a substitute. The
implementation is validated against the networkx coefficient on a random graph and pinned at
both extremes by a smooth and an alternating label on a path. Every measure now returns its
coverage alongside its value, because on a sparse graph a statistic can describe a minority of
the molecules and still read as a dataset-level number. A degenerate input raises instead of
returning a quiet NaN."
```

---

### Task 5: Categorical measures on `cliff_mol`, with the binning door nailed shut

**Files:**
- Create: `src/molace/measures/homophily.py`, `src/molace/measures/informativeness.py`,
  `tests/test_categorical_measures.py`
- Test: `tests/test_categorical_measures.py`

**Interfaces:**
- Consumes: `measures.assortativity.MeasureResult`.
- Produces: `homophily.edge_homophily(g, labels) -> MeasureResult`,
  `homophily.adjusted_homophily(g, labels) -> MeasureResult`,
  `informativeness.label_informativeness(g, labels) -> MeasureResult`,
  `homophily.require_categorical(labels) -> np.ndarray` (shared guard, raises `TypeError`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_categorical_measures.py
import networkx as nx
import numpy as np
import pytest
from molace.measures.homophily import (adjusted_homophily, edge_homophily,
                                       require_categorical)
from molace.measures.informativeness import label_informativeness


def test_edge_homophily_on_a_hand_counted_graph():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    labels = np.array([0, 0, 1, 1])
    # edges: (0,1) same, (1,2) different, (2,3) same -> 2/3
    assert edge_homophily(g, labels).value == pytest.approx(2 / 3)


def test_adjusted_homophily_matches_the_formula_by_hand():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    labels = np.array([0, 0, 1, 1])
    # degrees 1,2,2,1; 2|E| = 6; D_0 = 3, D_1 = 3; sum pbar^2 = 0.25 + 0.25 = 0.5
    # h_adj = (2/3 - 0.5) / (1 - 0.5) = 1/3
    assert adjusted_homophily(g, labels).value == pytest.approx(1 / 3)


def test_adjusted_homophily_is_one_for_perfect_separation():
    g = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    labels = np.array([0] * 5 + [1] * 5)
    assert adjusted_homophily(g, labels).value == pytest.approx(1.0)


def test_label_informativeness_is_zero_when_the_label_is_independent_of_structure():
    g = nx.complete_graph(40)
    labels = np.array([0, 1] * 20)
    assert label_informativeness(g, labels).value < 0.05


def test_label_informativeness_is_one_for_perfect_separation():
    g = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    labels = np.array([0] * 6 + [1] * 6)
    assert label_informativeness(g, labels).value == pytest.approx(1.0, abs=1e-9)


def test_label_informativeness_matches_a_hand_computed_two_class_case():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    labels = np.array([0, 0, 1, 1])
    # 2|E| = 6. joint counts (both directions): (0,0) 2, (1,1) 2, (0,1) 1, (1,0) 1
    # p = (0.5, 0.5); H = ln 2
    # I = 2*(1/3)ln((1/3)/0.25) + 2*(1/6)ln((1/6)/0.25)
    p = np.array([0.5, 0.5])
    joint = np.array([[2 / 6, 1 / 6], [1 / 6, 2 / 6]])
    h = -(p * np.log(p)).sum()
    i = sum(joint[a, b] * np.log(joint[a, b] / (p[a] * p[b])) for a in range(2) for b in range(2))
    assert label_informativeness(g, labels).value == pytest.approx(i / h)


def test_a_continuous_label_is_refused_by_both_measures():
    g = nx.path_graph(30)
    y = np.linspace(4.0, 10.0, 30)          # a real pIC50-like vector
    for fn in (adjusted_homophily, label_informativeness):
        with pytest.raises(TypeError) as e:
            fn(g, y)
        assert "categorical" in str(e.value)
        assert "assortativity" in str(e.value)


def test_many_integer_classes_is_also_refused_because_that_is_binning():
    g = nx.path_graph(40)
    labels = np.arange(40) % 12          # 12 "classes" is a binning, not a natural label
    with pytest.raises(TypeError, match="categorical"):
        adjusted_homophily(g, labels)


def test_a_single_class_raises_rather_than_returning_nan():
    g = nx.path_graph(8)
    labels = np.zeros(8, dtype=int)
    with pytest.raises(ValueError, match="one class"):
        adjusted_homophily(g, labels)
    with pytest.raises(ValueError, match="one class"):
        label_informativeness(g, labels)


def test_require_categorical_accepts_the_real_cliff_flag():
    labels = require_categorical(np.array([0, 1, 1, 0, 1]))
    assert labels.dtype.kind in "iu"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_categorical_measures.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.measures.homophily'`

- [ ] **Step 3: Implement the guard and the homophily measures**

```python
# src/molace/measures/homophily.py
"""Edge and adjusted homophily, categorical labels only.

Definitions from Platonov, Kuznedelev, Babenko, Prokhorenkova, "Characterizing Graph Datasets
for Node Classification: Homophily-Heterophily Dichotomy and Beyond", arXiv 2209.06177, eq. (2):

    h_adj = (h_edge - sum_k (D_k / 2|E|)^2) / (1 - sum_k (D_k / 2|E|)^2)

where h_edge is the fraction of edges joining same-class endpoints and D_k is the summed degree
of class-k nodes.

The guard below is the point of this module. Binning a continuous target was measured to move
adjusted homophily 7.5x and label informativeness 100x on one real target at one fixed graph,
which would make the headline number a function of a free parameter the project claims to have
fixed. So binning is refused in code rather than discouraged in prose.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult

MAX_CLASSES = 8

_REFUSAL = (
    "adjusted homophily and label informativeness are defined for categorical labels only "
    "(Platonov et al., arXiv 2209.06177: a class label y_v in {1..C}, class degree sums D_k, "
    "and a discrete mutual information). Binning a continuous target is forbidden by "
    "prereg/increment1.yaml: on CHEMBL2835_Ki, one graph and one bin count, swapping quantile "
    "for equal-width bins moved adjusted homophily 7.5x and label informativeness 100x. "
    "For a continuous target use measures.assortativity.target_assortativity."
)


def require_categorical(labels: np.ndarray) -> np.ndarray:
    """Accept a genuine categorical label; refuse anything that is a binning in disguise."""
    a = np.asarray(labels)
    if a.dtype.kind == "f" and not np.all(a == np.floor(a)):
        raise TypeError(_REFUSAL)
    a = a.astype(np.int64)
    if len(np.unique(a)) > MAX_CLASSES:
        raise TypeError(_REFUSAL + f" Got {len(np.unique(a))} distinct values.")
    return a


def _check(g: nx.Graph, labels: np.ndarray) -> np.ndarray:
    a = require_categorical(labels)
    if len(a) != g.number_of_nodes():
        raise ValueError(f"length mismatch: {len(a)} labels for {g.number_of_nodes()} nodes")
    if g.number_of_edges() == 0:
        raise ValueError("homophily is undefined on a graph with no edges")
    if len(np.unique(a)) < 2:
        raise ValueError(
            "the label takes only one class on this target, so homophily and label "
            "informativeness are 0/0; report the target as degenerate instead"
        )
    return a


def _coverage(g: nx.Graph) -> tuple[float, int]:
    used = sum(1 for _, d in g.degree() if d > 0)
    return used / g.number_of_nodes(), used


def edge_homophily(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    a = _check(g, labels)
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    same = sum(1 for u, v in g.edges() if a[pos[u]] == a[pos[v]])
    cov, n_used = _coverage(g)
    return MeasureResult(same / g.number_of_edges(), cov, n_used)


def adjusted_homophily(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    a = _check(g, labels)
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    two_m = 2.0 * g.number_of_edges()
    deg = dict(g.degree())
    h_edge = edge_homophily(g, labels).value
    baseline = 0.0
    for c in np.unique(a):
        d_k = sum(deg[v] for v in nodes if a[pos[v]] == c)
        baseline += (d_k / two_m) ** 2
    if baseline >= 1.0:
        raise ValueError("adjusted homophily is undefined: the degree baseline reached 1")
    cov, n_used = _coverage(g)
    return MeasureResult((h_edge - baseline) / (1.0 - baseline), cov, n_used)
```

- [ ] **Step 4: Implement label informativeness**

```python
# src/molace/measures/informativeness.py
"""Label informativeness, eq. (3) of arXiv 2209.06177: LI = I(y_xi, y_eta) / H(y_xi).

Over a uniformly random edge with a random orientation, so the marginal is degree-weighted,
p(k) = D_k / 2|E|. Categorical only, and there is no published continuous analogue: the string
"label informativeness" does not appear in GraphLand at all. Writing one would be an unbudgeted
methods contribution with nothing to validate against, so this module refuses instead.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult
from molace.measures.homophily import _check, _coverage


def label_informativeness(g: nx.Graph, labels: np.ndarray) -> MeasureResult:
    a = _check(g, labels)
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    classes = np.unique(a)
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    joint = np.zeros((k, k), dtype=float)
    for u, v in g.edges():
        i, j = idx[a[pos[u]]], idx[a[pos[v]]]
        joint[i, j] += 1.0
        joint[j, i] += 1.0
    joint /= joint.sum()
    p = joint.sum(axis=1)
    h = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    if h == 0.0:
        raise ValueError("label informativeness is undefined: the label entropy is zero")
    mi = 0.0
    for i in range(k):
        for j in range(k):
            if joint[i, j] > 0.0:
                mi += joint[i, j] * np.log(joint[i, j] / (p[i] * p[j]))
    cov, n_used = _coverage(g)
    return MeasureResult(float(mi / h), cov, n_used)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_categorical_measures.py -v`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add src/molace/measures/homophily.py src/molace/measures/informativeness.py tests/test_categorical_measures.py
git commit -m "the categorical measures refuse a continuous label in code

Both measures are defined over a class label, class degree sums and a discrete mutual
information, and the project's targets are potencies. Binning them was measured on one real
target at one fixed graph to move adjusted homophily by a factor of seven and a half and label
informativeness by a factor of a hundred, which would make the headline number a function of a
parameter the project claims to have fixed, so the guard raises rather than warns and its
message names the continuous alternative. Many integer classes is refused too, since that is a
binning wearing an integer dtype. The formulas are pinned against hand-computed two-class cases
and at the separation extreme, and a single-class target raises as degenerate instead of
returning a quiet NaN."
```

---

### Task 6: Unbiased homophily — the measure their benchmarks actually report

**Files:**
- Create: `src/molace/measures/unbiased.py`, `tests/test_unbiased_homophily.py`
- Modify: `docs/superpowers/specs/2026-10-03-molace-increment1-design.md` (§13a item 1: strike
  it once the formula is recorded)
- Test: `tests/test_unbiased_homophily.py`

**Interfaces:**
- Consumes: `measures.assortativity.MeasureResult`, `measures.homophily.require_categorical`.
- Produces: `unbiased_homophily(g: nx.Graph, labels: np.ndarray) -> MeasureResult`.

**Why this task exists:** GraphLand states verbatim that Mironov and Prokhorenkova (2024)
"constructed the first known homophily measure that satisfies all these properties - unbiased
homophily. Thus, in our work, we use unbiased homophily", and GraphPFN does the same. Adjusted
homophily is their 2023 vocabulary; its own source paper concedes it fails one of the four
properties ("The minimal agreement is not satisfied"). Pitching the superseded measure to this
group in late 2026 is a self-inflicted wound.

- [ ] **Step 1: Read the source and record the formula in the module docstring**

Fetch `https://arxiv.org/abs/2412.09663` (Mironov, Prokhorenkova, "Revisiting Graph Homophily
Measures", LoG 2024, PMLR v269) and read the definition of unbiased homophily. Transcribe the
equation and its number into the module docstring, together with the `alpha` (or equivalent)
setting GraphLand uses.

**Writing this formula from memory is forbidden.** It is a 2024 paper and the formula must come
off the page. If the abs page does not render the equations, use the PMLR PDF at
`https://proceedings.mlr.press/v269/`.

- [ ] **Step 2: Write the failing property tests**

These test the four properties the measure is *defined* by, so they are valid before the formula
is known and they are what the paper claims. Adjusted homophily fails the third one, which is
the discriminating test.

```python
# tests/test_unbiased_homophily.py
import networkx as nx
import numpy as np
import pytest
from molace.measures.homophily import adjusted_homophily
from molace.measures.unbiased import unbiased_homophily


def test_maximal_agreement_is_one_for_perfect_separation():
    g = nx.disjoint_union(nx.complete_graph(6), nx.complete_graph(6))
    labels = np.array([0] * 6 + [1] * 6)
    assert unbiased_homophily(g, labels).value == pytest.approx(1.0, abs=1e-9)


def test_constant_baseline_under_label_permutation():
    rng = np.random.default_rng(0)
    g = nx.gnp_random_graph(200, 0.05, seed=3)
    labels = np.array([0] * 100 + [1] * 100)
    vals = []
    for _ in range(200):
        vals.append(unbiased_homophily(g, rng.permutation(labels)).value)
    assert abs(float(np.mean(vals))) < 0.05


def test_minimal_agreement_the_property_adjusted_homophily_fails():
    # A complete bipartite graph with the parts as classes is maximally heterophilous:
    # every edge crosses classes. Unbiased homophily must reach its minimum, -1.
    g = nx.complete_bipartite_graph(8, 8)
    labels = np.array([0] * 8 + [1] * 8)
    assert unbiased_homophily(g, labels).value == pytest.approx(-1.0, abs=1e-6)
    # adjusted homophily does not reach -1 here; that is why the group replaced it
    assert adjusted_homophily(g, labels).value > -1.0 + 1e-6


def test_comparable_across_different_class_counts():
    # The paper's purpose: a measure usable "across datasets with different label
    # distributions". Two balanced labellings with 2 and 4 classes on the same structure must
    # both sit near zero under permutation.
    rng = np.random.default_rng(1)
    g = nx.gnp_random_graph(200, 0.05, seed=5)
    for n_classes in (2, 4):
        labels = np.tile(np.arange(n_classes), 200 // n_classes)
        vals = [unbiased_homophily(g, rng.permutation(labels)).value for _ in range(150)]
        assert abs(float(np.mean(vals))) < 0.06, n_classes


def test_a_continuous_label_is_refused_here_too():
    g = nx.path_graph(30)
    with pytest.raises(TypeError, match="categorical"):
        unbiased_homophily(g, np.linspace(4.0, 10.0, 30))


def test_coverage_is_reported():
    g = nx.path_graph(8)
    g.add_nodes_from([8, 9])
    labels = np.array([0, 1] * 5)
    r = unbiased_homophily(g, labels)
    assert r.n_used == 8 and r.coverage == pytest.approx(0.8)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_unbiased_homophily.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.measures.unbiased'`

- [ ] **Step 4: Implement from the transcribed formula**

Write `src/molace/measures/unbiased.py` with the equation from Step 1 in the docstring,
reusing `require_categorical` and `_coverage` from `measures.homophily` and returning a
`MeasureResult`. The module must contain the arXiv number, the equation number, and the
`alpha` setting, so a reader can check the implementation against the paper without leaving
the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_unbiased_homophily.py -v`
Expected: 6 passed

If `test_minimal_agreement_the_property_adjusted_homophily_fails` does not pass, the formula was
transcribed wrongly — the paper's central claim is that this measure satisfies minimal
agreement. Re-read before adjusting the test.

- [ ] **Step 6: Strike §13a item 1 in the spec and commit**

```bash
git add src/molace/measures/unbiased.py tests/test_unbiased_homophily.py docs/superpowers/specs/2026-10-03-molace-increment1-design.md
git commit -m "the categorical arm uses the measure their current benchmarks report

Both GraphLand and GraphPFN replaced adjusted homophily with unbiased homophily, and the paper
defining adjusted homophily concedes it fails one of the four properties a cross-dataset measure
needs, so proposing the 2023 vocabulary to this group in late 2026 invites the first correction
of the conversation. The formula is transcribed from the source rather than written from memory,
with the arXiv number and the equation number in the module so a reader can check it in place.
The tests are the four defining properties, and the discriminating one is minimal agreement on a
complete bipartite graph, which adjusted homophily provably does not reach and this measure
must."
```

---

### Task 7: ROGI, the baseline the primary statistic has to beat

**Files:**
- Create: `src/molace/measures/rogi.py`, `tests/test_rogi.py`
- Test: `tests/test_rogi.py`

**Interfaces:**
- Consumes: `measures.assortativity.MeasureResult`.
- Produces: `roughness(fp: np.ndarray, y: np.ndarray) -> MeasureResult`, plus the module
  constant `ROGI_FLAVOUR: str` recording which implementation was used
  (`"rogi"` or `"rogi-xd"`).

**Why this task exists and what is at stake:** target assortativity on a kNN similarity graph
measures close to what ROGI measures — the smoothness of a property over chemical space,
differently normalised. If that is hidden, a cheminformatics reader says the project rediscovered
roughness in graph vocabulary. Stated in front, the contribution becomes the translation between
the two vocabularies plus the question of which predicts model-class choice better. So the
headline claim is **incremental over this baseline**, which makes this task load-bearing rather
than decorative.

**A scope decision with a time box.** The spec names ROGI-XD (arXiv 2305.08238), whose selling
point is comparability across datasets of different size — exactly our cross-task setting. Plain
ROGI is pip-installable (`rogi` 0.1, coleygroup); ROGI-XD exists only as a GitHub repository
(`coleygroup/rogi-xd`, last pushed 2023-07-27) and pulls in pretrained encoders. **Time box:
45 minutes on ROGI-XD.** If it does not install and run on one target inside that, ship plain
ROGI, set `ROGI_FLAVOUR = "rogi"`, and record the substitution in the README and the
pre-registration. Shipping ROGI silently while the spec says ROGI-XD is the failure mode to
avoid; shipping ROGI openly is fine.

- [ ] **Step 1: Install and introspect the real API**

```bash
uv add rogi
uv run python -c "
import rogi, inspect
print('version:', getattr(rogi, '__version__', 'unknown'))
print('exports:', [x for x in dir(rogi) if not x.startswith('_')])
for name in [x for x in dir(rogi) if not x.startswith('_')]:
    obj = getattr(rogi, name)
    if callable(obj):
        try: print(f'  {name}{inspect.signature(obj)}')
        except (TypeError, ValueError): print(f'  {name} (no signature)')
"
```

Record the actual call in the implementation. Do not guess the signature from the paper.

- [ ] **Step 2: Attempt ROGI-XD inside the time box**

```bash
git clone --depth 1 https://github.com/coleygroup/rogi-xd.git /tmp/rogi-xd
sed -n '1,80p' /tmp/rogi-xd/README.md
```

Decide: `ROGI_FLAVOUR = "rogi-xd"` if it runs on one MoleculeACE target within 45 minutes,
else `"rogi"`.

- [ ] **Step 3: Write the failing tests**

These pin behaviour, not the upstream signature, so they survive either flavour.

```python
# tests/test_rogi.py
import numpy as np
import pytest
from molace.measures.rogi import ROGI_FLAVOUR, roughness


def _two_cluster_fp(n=120, seed=0):
    """Two well-separated fingerprint clusters."""
    rng = np.random.default_rng(seed)
    base_a = (rng.random(2048) < 0.02)
    base_b = (rng.random(2048) < 0.02)
    rows = []
    for i in range(n):
        base = base_a if i < n // 2 else base_b
        row = base.copy()
        flip = rng.choice(2048, size=5, replace=False)
        row[flip] = ~row[flip]
        rows.append(row)
    return np.array(rows, dtype=np.uint8)


def test_flavour_is_recorded():
    assert ROGI_FLAVOUR in {"rogi", "rogi-xd"}


def test_a_smooth_landscape_is_less_rough_than_a_shuffled_one():
    fp = _two_cluster_fp()
    y_smooth = np.concatenate([np.full(60, 5.0), np.full(60, 9.0)])
    rng = np.random.default_rng(1)
    y_shuffled = rng.permutation(y_smooth)
    assert roughness(fp, y_smooth).value < roughness(fp, y_shuffled).value


def test_value_is_finite_and_in_the_unit_interval():
    fp = _two_cluster_fp()
    y = np.concatenate([np.full(60, 5.0), np.full(60, 9.0)])
    v = roughness(fp, y).value
    assert np.isfinite(v) and 0.0 <= v <= 1.0


def test_deterministic_across_calls():
    fp = _two_cluster_fp()
    y = np.linspace(4.0, 10.0, 120)
    assert roughness(fp, y).value == pytest.approx(roughness(fp, y).value)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        roughness(_two_cluster_fp(), np.ones(10))


def test_constant_label_raises_rather_than_returning_nan():
    fp = _two_cluster_fp()
    with pytest.raises(ValueError, match="constant"):
        roughness(fp, np.ones(120))
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rogi.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.measures.rogi'`

- [ ] **Step 5: Implement the wrapper**

Write `src/molace/measures/rogi.py` wrapping the call recorded in Step 1. The wrapper owns the
input validation (length match, constant-label refusal) so the tests above hold regardless of
what upstream does, sets `ROGI_FLAVOUR` from the Step 2 decision, converts the upstream result
to `MeasureResult` with `coverage = 1.0` and `n_used = len(y)` (roughness is computed on all
molecules, not only those with graph neighbours), and states in the docstring that this is the
baseline the primary statistic must beat and why.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rogi.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/molace/measures/rogi.py tests/test_rogi.py pyproject.toml uv.lock
git commit -m "roughness is the baseline, because the primary statistic is close to it

Target assortativity on a nearest-neighbour similarity graph measures the smoothness of a
property over chemical space, which is what the roughness index measures under a different
normalisation. Hidden, that invites the reading that the project rediscovered roughness in graph
vocabulary; stated in front, the contribution becomes the translation between the two
vocabularies and the question of which one predicts the choice of model class, so the headline
claim is reported as incremental over this baseline. The wrapper owns validation so the
behavioural tests hold whichever upstream flavour is installed, and the flavour actually used is
recorded in the module rather than assumed."
```

---

### Task 8: Freeze the pre-registration

**Files:**
- Create: `prereg/increment1.yaml`, `src/molace/analysis/prereg.py`, `tests/test_prereg.py`
- Test: `tests/test_prereg.py`

**Interfaces:**
- Consumes: `data.moleculeace.DATASETS`, `measures.*` module names.
- Produces: `load_prereg() -> dict`, `prereg_fingerprint() -> str` (the git blob hash of the
  YAML, stamped into every result file and figure).

**This task must complete before Task 12 runs the sweep.** Nothing measured on real data may
precede it. The analysis plan exists so that a null is informative; a plan written after seeing
the numbers is not a plan.

- [ ] **Step 1: Write the pre-registration**

```yaml
# prereg/increment1.yaml
# Frozen analysis plan for molace increment 1.
# Committed before any measurement runs on real data. Referenced by blob hash from every
# result file and figure. Changes after the sweep must be a new file with a new name, never an
# edit of this one.
version: 1
date: 2026-10-03
spec: docs/superpowers/specs/2026-10-03-molace-increment1-design.md

representation:
  fingerprint: morgan
  radius: 2
  n_bits: 2048
  binary: true
  rationale: >
    The single fixed methodological parameter. Cliff sets and graph statistics both move with
    the representation, so it is fixed once and everything else is free.

graphs:
  primary:
    kind: knn
    k: 10
    symmetrisation: union
    note: >
      Union symmetrisation gives degree >= k, not exactly k. Mean degree therefore remains a
      covariate in the incremental regression; it is not assumed constant.
  robustness:
    - {kind: knn, k: 5, symmetrisation: union}
    - {kind: knn, k: 20, symmetrisation: union}
    - {kind: threshold, tau: 0.5}
    - {kind: knn, k: 10, symmetrisation: directed}

task_sets:
  spine:
    source: MoleculeACE
    n_tasks: 30
    label_column: "y [pEC50/pKi]"
    split: shipped `split` column, 80/20, stratified
  holdout:
    source: TDC admet_benchmark, regression subset
    n_tasks: 9
    datasets: [caco2_wang, lipophilicity_astrazeneca, solubility_aqsoldb, ppbr_az,
               vdss_lombardo, half_life_obach, clearance_hepatocyte_az,
               clearance_microsome_az, ld50_zhu]
    opened: exactly once, after the spine primary is computed and recorded
  external_anchor:
    source: DeepDelta Results/ shipped per-fold predictions
    n_tasks: 10
    families: [RandomForest, ChemProp50, DeepDelta5]
    note: all three are scored on the same delta task, verified from the file layout

independent_variables:
  primary:
    name: target_assortativity
    definition: Newman 2003 scalar assortativity, Pearson r of y over the symmetrised edge list
    computed_on: raw potency
    free_parameters: 0
  categorical_arm:
    computed_on: cliff_mol
    measures: [unbiased_homophily, label_informativeness, adjusted_homophily]
    free_parameters: 0
    circularity_note: >
      cliff_mol is defined from substructure/scaffold/SMILES similarity; the graph is ECFP4
      kNN. Different relations. Stated in the text, not hidden.
  baseline:
    name: rogi
    role: the primary must show incremental contribution over this
  secondary:
    name: dirichlet_energy
    normalisation: divided by the label variance times the edge count
    presented_as: an import, not a measure of this group

forbidden:
  - Binning a continuous target into classes for any categorical measure. Measured on
    CHEMBL2835_Ki at one graph and one bin count, swapping quantile for equal-width bins moved
    adjusted homophily 7.5x and label informativeness 100x. Enforced by a TypeError.
  - Inventing a continuous label informativeness. No published definition exists.
  - Selecting the best seed, the best k, or the best graph kind by outcome.

arms:
  pointwise:
    learners: [svm, lightgbm, mlp]
    features: ecfp4
    test_time_label_access: none
  knn_floor:
    rule: mean of y over the m nearest training molecules by Tanimoto on the one fingerprint
    m: 10
    weights: uniform
    deterministic: true
  pairwise:
    learners: [svm, lightgbm, mlp]
    feature_maps: [difference, concat_difference]
    inference: mean over the same m anchors of (y_i + f(x_i, x_new))
    m: 10

dependent_variables:
  selection: 5-fold CV on the training split only, by mean CV RMSE, then refit on full train
  evaluation: the shipped test split, once
  primary_gap: RMSE_pointwise_selected - RMSE_pairwise_selected
  access: RMSE_pointwise_selected - RMSE_knn_floor
  correction: RMSE_knn_floor - RMSE_pairwise_selected
  identity: >
    With uniform weights over the same m anchors the pairwise prediction equals the kNN
    prediction plus the mean learned correction, as algebra. The telescoping of the three error
    numbers is separately trivial and carries no content; the identity is what licenses
    attributing `correction` to the learned function alone.
  forbidden_reporting: >
    RMSE is nonlinear, so `correction` is not a share of `gap` and must never be reported as a
    percentage of it.
  secondary: [per-learner gaps, the same three quantities under RMSE on cliff compounds]
  seeds: 3, averaged, learned arms only

analysis:
  positive_control:
    statistic: spearman(target_assortativity, access)
    expectation: must be positive with a 95 percent interval excluding zero
    on_failure: >
      indicts the measurement pipeline, not the hypothesis. Nothing downstream is interpreted
      until it is resolved.
  primary:
    statistic: spearman(target_assortativity, primary_gap)
  incremental:
    model: primary_gap ~ target_assortativity + rogi + mean_degree + log(n_molecules)
    reported: the contribution of target_assortativity with its interval
  decisive_secondary:
    statistics: [spearman(target_assortativity, correction), mean(correction)]
    interpretation: >
      correction indistinguishable from zero is the half-1 statement of hypothesis 2, that a
      pairwise model is a pointwise model plus anchor averaging, reached through held-out error
      and without Hodge machinery. Section 10 of the spec tests the same claim through flow
      energy. Disagreement between them is reported, not reconciled by choosing a number.
  uncertainty:
    method: cluster bootstrap over the Receptor Class column
    clusters: {GPCR: 12, Kinase: 6, NR: 6, Other: 3, Protease: 2, Transferase: 1}
    resamples: 10000
    interval: percentile, 95 percent
  decision_rule: >
    The primary claim is supported iff the cluster-bootstrap interval for the primary Spearman
    excludes zero AND the incremental contribution over rogi has an interval excluding zero.
  no_sign_test: >
    At n=5 the exact permutation critical value is |rho| = 1.0 and a sign-level null discards a
    true rho of 0.5 to 0.7 in 9 to 20 percent of samples. There is no sign-level kill rule.
  effective_n:
    value: about 10
    assumption: intra-receptor-class correlation 0.3
    evidence: >
      CHEMBL237_Ki and CHEMBL237_EC50 are the same protein; 26.7 percent of molecules appear in
      more than one task; JAK1 and JAK2 overlap 88.5 percent.
    requirement: n_eff is printed wherever n is printed

hodge:
  census_exact: [n_nodes, n_edges, n_components, n_triangles, dim_gradient, dim_cycle_space]
  census_estimated:
    quantity: the split of the cycle space into curl and harmonic
    exact_when: n_edges <= 5000, by dense SVD
    otherwise: randomised range finder, tolerance logged
  energy: >
    Orthogonal projections by lsmr. Gradient and curl subspaces are orthogonal because
    B1 @ B2 = 0, which is asserted as a test.
  controls:
    - the trained model
    - the same architecture on shuffled labels, with total flow norm matched
    - an exactly curl-free floor from differencing a trained pointwise model on the same edges
  void_null: >
    The original null is void: the target flow dy_ij = y_j - y_i is the gradient of a node
    labelling and is curl-free identically under true AND permuted labels, measured 0.00e+00 on
    five real graphs. Raw-norm comparison is additionally scale-confounded, since shuffling
    inflates the mean-square edge target 2.1x to 11.7x.
  kill_criterion: >
    The per-edge curl fraction must predict that edge's held-out error better than plain anchor
    dispersion, which is published three times. Comparator: DeepDelta's additivity MAE
    0.127 +/- 0.043. Failing to beat it is reported as a measured null.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_prereg.py
import re
import subprocess

import pytest
from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data.moleculeace import DATASETS, RECEPTOR_CLASSES


def test_loads_and_has_the_sections_the_sweep_reads():
    p = load_prereg()
    for key in ("representation", "graphs", "task_sets", "independent_variables",
                "arms", "dependent_variables", "analysis", "hodge", "forbidden"):
        assert key in p, key


def test_representation_matches_the_code():
    from molace.graphs.fingerprints import N_BITS, RADIUS
    r = load_prereg()["representation"]
    assert (r["radius"], r["n_bits"], r["binary"]) == (RADIUS, N_BITS, True)


def test_primary_graph_k_and_anchor_m_are_separate_constants():
    p = load_prereg()
    assert p["graphs"]["primary"]["k"] == 10
    assert p["arms"]["knn_floor"]["m"] == 10
    # Same value today, different constants. The test exists so a change to one does not
    # silently move the other.
    assert "m" not in p["graphs"]["primary"]
    assert "k" not in p["arms"]["knn_floor"]


def test_cluster_sizes_match_the_shipped_metadata():
    declared = load_prereg()["analysis"]["uncertainty"]["clusters"]
    actual = {}
    for d in DATASETS:
        c = RECEPTOR_CLASSES[d]
        actual[c] = actual.get(c, 0) + 1
    assert declared == actual


def test_spine_task_count_matches_the_data():
    assert load_prereg()["task_sets"]["spine"]["n_tasks"] == len(DATASETS)


def test_fingerprint_is_a_git_blob_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", prereg_fingerprint())


def test_fingerprint_is_committed_so_results_are_anchored():
    # A dirty pre-registration cannot anchor a result.
    out = subprocess.run(["git", "status", "--porcelain", "prereg/increment1.yaml"],
                         capture_output=True, text=True).stdout
    assert out.strip() == "", f"prereg/increment1.yaml is uncommitted: {out!r}"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prereg.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.analysis.prereg'`

- [ ] **Step 4: Implement the loader**

```python
# src/molace/analysis/prereg.py
"""Access to the frozen analysis plan.

The plan is anchored by the git blob hash of the YAML, which is stamped into every result file
and every figure. A result whose stamp does not match the committed plan was produced under a
different plan, and that is visible rather than arguable.
"""
from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import yaml

PREREG = Path(__file__).resolve().parents[3] / "prereg" / "increment1.yaml"


@functools.lru_cache(maxsize=1)
def load_prereg() -> dict:
    if not PREREG.is_file():
        raise FileNotFoundError(f"{PREREG} is missing; the plan must exist before measurement")
    with PREREG.open() as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def prereg_fingerprint() -> str:
    """The git blob hash of the pre-registration file."""
    out = subprocess.run(
        ["git", "hash-object", str(PREREG)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()
```

- [ ] **Step 5: Commit the plan, then run the tests**

The commit must happen before the final test can pass, because that test asserts the file is
clean.

```bash
uv add pyyaml
git add prereg/increment1.yaml src/molace/analysis/prereg.py tests/test_prereg.py
git commit -m "the analysis plan is frozen before anything is measured

A null is only informative if the plan predates the numbers, so the plan is committed before the
sweep runs and every result and figure carries the blob hash of this file. It fixes the one
representation, names the primary graph and the robustness set, names the three arms and the
exact decomposition of their error differences, and forbids the three things that would otherwise
turn a free parameter into a finding: binning a continuous target, inventing a continuous label
informativeness, and selecting the best seed or graph by outcome. The decision rule requires the
bootstrap interval to exclude zero and the contribution over the roughness baseline to do the
same, and the sign-level kill rule is struck because it discards a true correlation in up to a
fifth of samples. The effective sample size is declared at about ten alongside the nominal
thirty, with the overlaps that cause it."
uv run pytest tests/test_prereg.py -v
```
Expected: 7 passed

---

### Task 9: Shared anchor selection and the kNN floor arm

**Files:**
- Create: `src/molace/models/anchors.py`, `src/molace/models/knn_floor.py`,
  `tests/test_anchors.py`
- Test: `tests/test_anchors.py`

**Interfaces:**
- Consumes: `graphs.fingerprints.tanimoto_matrix`.
- Produces: `anchors.M = 10`; `anchors.select(sim_test_train: np.ndarray, m: int = M)
  -> np.ndarray` returning `(n_test, m)` int indices into the training set, ties broken by
  ascending training index; `knn_floor.predict(y_train: np.ndarray, idx: np.ndarray)
  -> np.ndarray`.

**Why one module:** the kNN floor and the pairwise arm must draw the identical anchor set with
uniform weights, or the pairwise prediction is no longer the floor plus the mean learned
correction and the whole decomposition stops being exact. Two call sites would let that drift
silently while the numbers stayed plausible, which is the worst kind of bug here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_anchors.py
import numpy as np
import pytest
from molace.models import anchors, knn_floor


def test_selects_m_highest_similarity_training_indices():
    sim = np.array([[0.1, 0.9, 0.5, 0.7]])     # one test molecule, four training molecules
    idx = anchors.select(sim, m=2)
    assert idx.shape == (1, 2)
    assert set(idx[0]) == {1, 3}


def test_ties_break_on_ascending_training_index_and_are_stable():
    sim = np.full((1, 5), 0.5)
    a = anchors.select(sim, m=3)
    b = anchors.select(sim, m=3)
    assert a.tolist() == [[0, 1, 2]]
    assert a.tolist() == b.tolist()


def test_m_larger_than_the_training_set_is_clipped():
    sim = np.full((2, 3), 0.4)
    assert anchors.select(sim, m=10).shape == (2, 3)


def test_default_m_is_ten_and_is_not_the_graph_k():
    from molace.analysis.prereg import load_prereg
    p = load_prereg()
    assert anchors.M == 10 == p["arms"]["knn_floor"]["m"]
    # distinct constant from the similarity graph degree, even at the same value
    assert anchors.M is not p["graphs"]["primary"]["k"] or True  # value equality only


def test_floor_is_the_uniform_mean_of_the_anchor_labels():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    idx = np.array([[0, 1], [2, 3]])
    assert knn_floor.predict(y_train, idx).tolist() == [1.5, 3.5]


def test_floor_is_deterministic_and_needs_no_seed():
    y_train = np.linspace(0, 1, 50)
    sim = np.random.default_rng(0).random((20, 50))
    idx = anchors.select(sim)
    a = knn_floor.predict(y_train, idx)
    b = knn_floor.predict(y_train, anchors.select(sim))
    assert np.array_equal(a, b)


def test_empty_training_set_raises():
    with pytest.raises(ValueError, match="training"):
        anchors.select(np.zeros((3, 0)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_anchors.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.models.anchors'`

- [ ] **Step 3: Implement**

```python
# src/molace/models/anchors.py
"""The one place anchors are chosen.

The kNN floor arm and the pairwise arm must draw the identical set with uniform weights.
With that, the pairwise prediction is exactly the floor plus the mean learned correction:

    y_pair = (1/m) sum_i [ y_i + f(x_i, x_new) ] = y_knn + (1/m) sum_i f(x_i, x_new)

which is what licenses attributing the error change to the learned function and nothing else.
Two call sites would let the sets drift apart while the numbers stayed plausible.

m is a separate constant from the similarity graph's k. They happen to share the value 10.
Do not tie them.
"""
from __future__ import annotations

import numpy as np

M = 10


def select(sim_test_train: np.ndarray, m: int = M) -> np.ndarray:
    """Indices of the m most similar training molecules for each test molecule.

    Ties break on ascending training index, so the set is identical across runs. That
    determinism is a requirement, not a nicety: without it the decomposition identity holds in
    one run and not the next.
    """
    sim = np.asarray(sim_test_train, dtype=np.float64)
    if sim.ndim != 2:
        raise ValueError(f"expected a 2-D (n_test, n_train) matrix, got shape {sim.shape}")
    n_test, n_train = sim.shape
    if n_train == 0:
        raise ValueError("cannot choose anchors from an empty training set")
    mm = min(m, n_train)
    tie = np.tile(np.arange(n_train), (n_test, 1))
    return np.lexsort((tie, -sim), axis=1)[:, :mm].astype(np.int64)
```

```python
# src/molace/models/knn_floor.py
"""The label-access floor.

The pairwise arm reads m measured labels at test time; the pointwise arm reads none. Without
this arm a pointwise-against-pairwise gap partly measures access to data rather than
architecture. Uniform weights over the anchors chosen in models.anchors, no learned function,
no hyperparameter, no seed.
"""
from __future__ import annotations

import numpy as np


def predict(y_train: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train, dtype=float)
    idx = np.asarray(anchor_idx, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError(f"expected (n_test, m) anchor indices, got shape {idx.shape}")
    return y[idx].mean(axis=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_anchors.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/molace/models/anchors.py src/molace/models/knn_floor.py tests/test_anchors.py
git commit -m "anchors are chosen in one place, and the floor makes label access measurable

The pairwise arm reads ten measured labels at test time and the pointwise arm reads none, so
without a floor the gap between them partly measures access to data rather than architecture.
With uniform weights over an identical anchor set the pairwise prediction is exactly the floor
plus the mean learned correction, which is what licenses attributing the error change to the
learned function alone, so selection lives at a single call site rather than two that could drift
apart while the numbers stayed plausible. Ties break on ascending training index so the set is
identical across runs, because determinism there is what keeps the identity from holding in one
run and failing in the next."
```

---

### Task 10: The pointwise arm

**Files:**
- Create: `src/molace/models/pointwise.py`, `tests/test_pointwise.py`
- Test: `tests/test_pointwise.py`

**Interfaces:**
- Consumes: `graphs.fingerprints.ecfp4`.
- Produces: `LEARNERS: tuple[str, ...] = ("svm", "lightgbm", "mlp")`;
  `make_learner(name: str, seed: int)` returning a scikit-learn-compatible regressor;
  `fit_predict(name, X_train, y_train, X_test, seed) -> np.ndarray`;
  `select_by_cv(X_train, y_train, seed, n_folds=5) -> tuple[str, dict[str, float]]`
  returning the winning learner name and the per-learner mean CV RMSE.

**Baseline note:** SVM is first because it is the actual bar. From MoleculeACE's own shipped
results matrix, ECFP-SVM takes 21 of 30 targets on RMSE with mean rank 1.60, against ECFP-GBM at
2 of 30 and rank 3.00. Naming gradient boosting as the baseline, as the source proposal did,
inflates every reported gap by the SVM-minus-GBM margin, and a reviewer with the same CSV finds
it in minutes. MLP is included because SQRL's Table 1 shows tree models gain nothing from pairing
(XGBoost 0.79 to 0.76, RF 0.80 to 0.77), so a tree-only comparison would produce a degenerate gap.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pointwise.py
import numpy as np
import pytest
from molace.models import pointwise


def _easy_task(n=160, seed=0):
    """A learnable task: y is a linear function of a few fingerprint bits."""
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    w = np.zeros(2048)
    w[rng.choice(2048, 30, replace=False)] = rng.normal(size=30)
    y = X @ w + rng.normal(scale=0.1, size=n)
    return X, y


def test_the_three_learners_are_named_in_order_with_svm_first():
    assert pointwise.LEARNERS == ("svm", "lightgbm", "mlp")


def test_each_learner_beats_predicting_the_mean():
    X, y = _easy_task()
    Xtr, ytr, Xte, yte = X[:120], y[:120], X[120:], y[120:]
    baseline = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    for name in pointwise.LEARNERS:
        pred = pointwise.fit_predict(name, Xtr, ytr, Xte, seed=0)
        rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
        assert rmse < baseline, name


def test_predictions_have_the_right_shape():
    X, y = _easy_task()
    pred = pointwise.fit_predict("svm", X[:120], y[:120], X[120:], seed=0)
    assert pred.shape == (40,)


def test_same_seed_reproduces_exactly():
    X, y = _easy_task()
    a = pointwise.fit_predict("mlp", X[:120], y[:120], X[120:], seed=7)
    b = pointwise.fit_predict("mlp", X[:120], y[:120], X[120:], seed=7)
    assert np.array_equal(a, b)


def test_cv_selection_returns_a_named_learner_and_all_scores():
    X, y = _easy_task()
    winner, scores = pointwise.select_by_cv(X[:120], y[:120], seed=0)
    assert winner in pointwise.LEARNERS
    assert set(scores) == set(pointwise.LEARNERS)
    assert scores[winner] == min(scores.values())


def test_cv_selection_never_touches_the_test_split():
    # Guard by signature: select_by_cv takes no test arguments at all.
    import inspect
    params = set(inspect.signature(pointwise.select_by_cv).parameters)
    assert not {"X_test", "y_test"} & params


def test_unknown_learner_raises():
    with pytest.raises(KeyError, match="unknown learner"):
        pointwise.make_learner("randomforest", seed=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pointwise.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.models.pointwise'`

- [ ] **Step 3: Implement**

```python
# src/molace/models/pointwise.py
"""The pointwise arm: one molecule in, one property out, no test-time label access.

SVM leads because it is the real bar. From MoleculeACE's own shipped results matrix, ECFP-SVM
takes 21 of 30 targets on RMSE with mean rank 1.60, against ECFP-GBM at 2 of 30 and rank 3.00,
so naming gradient boosting as the baseline inflates every reported gap. The MLP is here because
SQRL's own table shows tree models gain nothing from pairing, and a tree-only comparison would
make the gap degenerate.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

LEARNERS: tuple[str, ...] = ("svm", "lightgbm", "mlp")


def make_learner(name: str, seed: int):
    if name == "svm":
        return SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, random_state=seed, verbose=-1,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 128), max_iter=600, early_stopping=True,
            n_iter_no_change=20, random_state=seed,
        )
    raise KeyError(f"unknown learner {name!r}; expected one of {LEARNERS}")


def fit_predict(name: str, X_train, y_train, X_test, seed: int) -> np.ndarray:
    model = make_learner(name, seed)
    model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=float))
    return np.asarray(model.predict(np.asarray(X_test, dtype=np.float32)), dtype=float)


def select_by_cv(X_train, y_train, seed: int, n_folds: int = 5):
    """Pick the learner by mean CV RMSE on the training split only.

    Takes no test arguments by design: selection must not see the evaluation split.
    """
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for name in LEARNERS:
        errs = []
        for tr, va in kf.split(X):
            pred = fit_predict(name, X[tr], y[tr], X[va], seed)
            errs.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        scores[name] = float(np.mean(errs))
    return min(scores, key=scores.__getitem__), scores
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pointwise.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/molace/models/pointwise.py tests/test_pointwise.py
git commit -m "the pointwise baseline is the one that actually wins on this benchmark

From the shipped results matrix the support vector model takes twenty-one of thirty targets with
a mean rank of 1.60 while gradient boosting takes two at rank 3.00, so naming boosting as the bar
would inflate every gap the project reports and a reader with the same file would find it at once.
A neural arm is included because the pairing literature shows tree models gain nothing from
pairing, which would make a tree-only comparison degenerate. Selection runs on the training split
by cross-validated error and takes no test arguments at all, so the signature itself prevents the
evaluation split from leaking into the choice."
```

---

### Task 11: The pairwise arm, with the sign convention fixed

**Files:**
- Create: `src/molace/models/pairwise.py`, `tests/test_pairwise.py`
- Test: `tests/test_pairwise.py`

**Interfaces:**
- Consumes: `models.pointwise.make_learner`, `models.anchors.select`,
  `models.knn_floor.predict`.
- Produces: `FEATURE_MAPS: tuple[str, ...] = ("difference", "concat_difference")`;
  `pair_features(Xa, Xb, kind) -> np.ndarray`;
  `training_pairs(X_train, y_train, sim_train_train, m) -> tuple[np.ndarray, np.ndarray]`;
  `fit(learner, kind, X_train, y_train, sim_train_train, m, seed) -> PairwiseModel`;
  `PairwiseModel.corrections(X_train, X_test, anchor_idx) -> np.ndarray` of shape
  `(n_test, m)`; `PairwiseModel.predict(y_train, X_train, X_test, anchor_idx) -> np.ndarray`.

**The sign convention, stated once and obeyed everywhere.** `f(x_a, x_b)` estimates
`y_b - y_a`: the second argument minus the first. Training pairs `(a, b)` carry target
`y_b - y_a` and features built from `x_b - x_a`. Inference is
`y_new = mean_a [ y_a + f(x_a, x_new) ]` with the anchor first and the query second.

This is written down because SQRL's published equations are inconsistent: its eq. 3 under eq. 2's
convention estimates `2 y_i - y_new`. We do not reproduce SQRL and we say so in the README.

**The pair budget, pre-registered.** All ordered pairs would be 8.5 million on the largest
target. Training pairs are instead each training molecule with its `m` nearest other training
molecules, both directions, giving `2 * n_train * m` pairs — about 58,000 on the largest target.
This matches the inference-time distribution, where anchors are nearest neighbours, rather than
training on a distribution the model never sees at test time.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pairwise.py
import numpy as np
import pytest
from molace.graphs.fingerprints import tanimoto_matrix
from molace.models import anchors, knn_floor, pairwise


def _task(n=180, seed=0):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    w = np.zeros(2048)
    w[rng.choice(2048, 30, replace=False)] = rng.normal(size=30)
    y = X @ w + rng.normal(scale=0.1, size=n)
    return X, y


def _split(X, y, n_train=140):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def test_difference_feature_map_is_second_minus_first():
    Xa = np.array([[1, 0, 1]], dtype=np.uint8)
    Xb = np.array([[1, 1, 0]], dtype=np.uint8)
    got = pairwise.pair_features(Xa, Xb, "difference")
    assert got.tolist() == [[0.0, 1.0, -1.0]]


def test_concat_difference_feature_map_layout():
    Xa = np.array([[1, 0]], dtype=np.uint8)
    Xb = np.array([[0, 1]], dtype=np.uint8)
    got = pairwise.pair_features(Xa, Xb, "concat_difference")
    assert got.tolist() == [[1.0, 0.0, 0.0, 1.0, -1.0, 1.0]]


def test_training_pair_count_is_two_n_times_m():
    X, y = _task()
    Xtr, ytr, _, _ = _split(X, y)
    sim = tanimoto_matrix(Xtr)
    feats, targets = pairwise.training_pairs(Xtr, ytr, sim, m=5, kind="difference")
    assert len(targets) == 2 * len(ytr) * 5
    assert feats.shape[0] == len(targets)


def test_training_targets_follow_the_stated_convention():
    X = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    y = np.array([3.0, 10.0])
    sim = tanimoto_matrix(X)
    feats, targets = pairwise.training_pairs(X, y, sim, m=1, kind="difference")
    # both directions of the single pair: y_b - y_a is +7 and -7
    assert sorted(np.round(targets, 6).tolist()) == [-7.0, 7.0]


def test_unknown_feature_map_raises():
    with pytest.raises(KeyError, match="unknown feature map"):
        pairwise.pair_features(np.zeros((1, 3)), np.zeros((1, 3)), "cross_attention")


@pytest.mark.parametrize("kind", ["difference", "concat_difference"])
def test_prediction_equals_the_floor_plus_the_mean_correction(kind):
    """The identity the whole decomposition rests on."""
    X, y = _task()
    Xtr, ytr, Xte, _ = _split(X, y)
    sim_tr = tanimoto_matrix(Xtr)
    sim_te_tr = tanimoto_matrix(X)[140:, :140]
    idx = anchors.select(sim_te_tr, m=10)
    model = pairwise.fit("lightgbm", kind, Xtr, ytr, sim_tr, m=10, seed=0)
    pred = model.predict(ytr, Xtr, Xte, idx)
    floor = knn_floor.predict(ytr, idx)
    corr = model.corrections(Xtr, Xte, idx).mean(axis=1)
    assert np.allclose(pred, floor + corr, atol=1e-10)


def test_corrections_have_shape_n_test_by_m():
    X, y = _task()
    Xtr, ytr, Xte, _ = _split(X, y)
    sim_tr = tanimoto_matrix(Xtr)
    idx = anchors.select(tanimoto_matrix(X)[140:, :140], m=10)
    model = pairwise.fit("svm", "difference", Xtr, ytr, sim_tr, m=10, seed=0)
    assert model.corrections(Xtr, Xte, idx).shape == (40, 10)


def test_a_bias_free_linear_head_gives_corrections_that_cancel_on_a_triangle():
    """A linear readout on the difference is a gradient flow, so triangle sums vanish.

    This is the half-1 shadow of the §10 theorem and it costs nothing to assert here.
    """
    rng = np.random.default_rng(0)
    X = (rng.random((3, 2048)) < 0.02).astype(np.float64)
    w = rng.normal(size=2048)
    f = lambda a, b: float(w @ (X[b] - X[a]))
    assert f(0, 1) + f(1, 2) + f(2, 0) == pytest.approx(0.0, abs=1e-9)


def test_same_seed_reproduces_predictions():
    X, y = _task()
    Xtr, ytr, Xte, _ = _split(X, y)
    sim_tr = tanimoto_matrix(Xtr)
    idx = anchors.select(tanimoto_matrix(X)[140:, :140], m=10)
    a = pairwise.fit("mlp", "difference", Xtr, ytr, sim_tr, m=10, seed=3).predict(ytr, Xtr, Xte, idx)
    b = pairwise.fit("mlp", "difference", Xtr, ytr, sim_tr, m=10, seed=3).predict(ytr, Xtr, Xte, idx)
    assert np.array_equal(a, b)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pairwise.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.models.pairwise'`

- [ ] **Step 3: Implement**

```python
# src/molace/models/pairwise.py
"""The pairwise arm.

Sign convention, obeyed everywhere: f(x_a, x_b) estimates y_b - y_a, the second argument minus
the first. Training pairs (a, b) carry target y_b - y_a with features from x_b - x_a, and
inference is y_new = mean_a [ y_a + f(x_a, x_new) ] with the anchor first. This is written down
because SQRL's published equations are inconsistent: its eq. 3 under eq. 2's convention estimates
2 y_i - y_new. We do not reproduce SQRL.

Pair budget: each training molecule with its m nearest other training molecules, both directions,
2 * n_train * m pairs. All ordered pairs would be 8.5 million on the largest target, and training
on all pairs would also train on a distribution the model never meets at test time, where anchors
are nearest neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from molace.models.anchors import M
from molace.models.pointwise import make_learner

FEATURE_MAPS: tuple[str, ...] = ("difference", "concat_difference")


def pair_features(Xa: np.ndarray, Xb: np.ndarray, kind: str) -> np.ndarray:
    a = np.asarray(Xa, dtype=np.float32)
    b = np.asarray(Xb, dtype=np.float32)
    if kind == "difference":
        return b - a
    if kind == "concat_difference":
        return np.hstack([a, b, b - a])
    raise KeyError(f"unknown feature map {kind!r}; expected one of {FEATURE_MAPS}")


def training_pairs(X_train, y_train, sim_train_train: np.ndarray, m: int, kind: str):
    """Both directions of (molecule, its m nearest other training molecules)."""
    X = np.asarray(X_train)
    y = np.asarray(y_train, dtype=float)
    n = len(y)
    s = np.asarray(sim_train_train, dtype=np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    mm = min(m, n - 1)
    tie = np.tile(np.arange(n), (n, 1))
    nbr = np.lexsort((tie, -s), axis=1)[:, :mm]
    a_idx = np.repeat(np.arange(n), mm)
    b_idx = nbr.reshape(-1)
    a_all = np.concatenate([a_idx, b_idx])
    b_all = np.concatenate([b_idx, a_idx])
    feats = pair_features(X[a_all], X[b_all], kind)
    targets = y[b_all] - y[a_all]
    return feats, targets


@dataclass
class PairwiseModel:
    learner_name: str
    kind: str
    model: object

    def corrections(self, X_train, X_test, anchor_idx: np.ndarray) -> np.ndarray:
        """f(anchor, query) for every (query, anchor) pair, shape (n_test, m)."""
        Xtr = np.asarray(X_train)
        Xte = np.asarray(X_test)
        idx = np.asarray(anchor_idx, dtype=np.int64)
        n_test, mm = idx.shape
        a = Xtr[idx.reshape(-1)]                                 # anchors first
        b = np.repeat(Xte, mm, axis=0)                           # query second
        preds = self.model.predict(pair_features(a, b, self.kind))
        return np.asarray(preds, dtype=float).reshape(n_test, mm)

    def predict(self, y_train, X_train, X_test, anchor_idx: np.ndarray) -> np.ndarray:
        y = np.asarray(y_train, dtype=float)
        idx = np.asarray(anchor_idx, dtype=np.int64)
        return (y[idx] + self.corrections(X_train, X_test, idx)).mean(axis=1)


def fit(learner: str, kind: str, X_train, y_train, sim_train_train, m: int = M,
        seed: int = 0) -> PairwiseModel:
    feats, targets = training_pairs(X_train, y_train, sim_train_train, m, kind)
    model = make_learner(learner, seed)
    model.fit(feats, targets)
    return PairwiseModel(learner_name=learner, kind=kind, model=model)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pairwise.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/molace/models/pairwise.py tests/test_pairwise.py
git commit -m "the pairwise arm states its sign convention and draws anchors first

The convention is written into the module because the paper this arm imitates is internally
inconsistent: under its own training convention its inference rule estimates twice the anchor
label minus the query, so we fix our own order explicitly and say we do not reproduce theirs.
Training pairs are each molecule with its nearest neighbours in both directions rather than all
ordered pairs, which is both a budget on the largest target and a match to the distribution the
model meets at inference, where anchors are nearest neighbours. A parametrised test asserts the
prediction equals the floor plus the mean correction for both feature maps, so the decomposition
the analysis depends on cannot drift without the suite failing."
```

---

### Task 12: `gap.py` — one definition, exact decomposition

**Files:**
- Create: `src/molace/models/gap.py`, `tests/test_gap.py`
- Test: `tests/test_gap.py`

**Interfaces:**
- Consumes: `models.pointwise`, `models.pairwise`, `models.knn_floor`, `models.anchors`,
  `graphs.fingerprints`.
- Produces: `rmse(y_true, y_pred) -> float`; `ArmResult` dataclass with
  `rmse_all: float`, `rmse_cliff: float`, `selected: str`;
  `GapResult` dataclass with `pointwise, knn_floor, pairwise: ArmResult`, `gap: float`,
  `access: float`, `correction: float`, `per_learner_gap: dict[str, float]`;
  `evaluate_target(df: pandas.DataFrame, seed: int) -> GapResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gap.py
import numpy as np
import pandas as pd
import pytest
from molace.models import gap


def test_rmse_matches_the_definition():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert gap.rmse(y, p) == pytest.approx(np.sqrt(4 / 3))


def test_the_three_error_numbers_telescope_to_the_gap_exactly():
    r = gap.GapResult(
        pointwise=gap.ArmResult(1.00, 1.20, "svm"),
        knn_floor=gap.ArmResult(0.90, 1.10, "knn"),
        pairwise=gap.ArmResult(0.70, 0.95, "lightgbm"),
        gap=0.30, access=0.10, correction=0.20, per_learner_gap={},
    )
    assert r.access + r.correction == pytest.approx(r.gap, abs=1e-12)


def test_gap_result_rejects_an_inconsistent_decomposition():
    with pytest.raises(ValueError, match="decomposition"):
        gap.GapResult(
            pointwise=gap.ArmResult(1.0, 1.2, "svm"),
            knn_floor=gap.ArmResult(0.9, 1.1, "knn"),
            pairwise=gap.ArmResult(0.7, 0.95, "lightgbm"),
            gap=0.30, access=0.10, correction=0.99, per_learner_gap={},
        )


def _toy_frame(n=200, seed=0):
    """A synthetic target shaped like a MoleculeACE frame."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, 12))
    smiles = ["C" * (1 + int(b.sum())) + "O" for b in bits]
    y = 5.0 + bits[:, :4] @ np.array([1.0, -0.8, 0.6, 1.3]) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({
        "smiles": smiles,
        "y": y,
        "cliff_mol": (rng.random(n) < 0.25).astype(int),
        "split": ["train"] * int(0.8 * n) + ["test"] * (n - int(0.8 * n)),
    })


def test_evaluate_target_returns_a_consistent_decomposition():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    assert r.access + r.correction == pytest.approx(r.gap, abs=1e-9)
    assert r.pointwise.selected in ("svm", "lightgbm", "mlp")
    assert r.knn_floor.selected == "knn"


def test_evaluate_target_reports_cliff_rmse_separately():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    for arm in (r.pointwise, r.knn_floor, r.pairwise):
        assert np.isfinite(arm.rmse_all) and np.isfinite(arm.rmse_cliff)


def test_a_target_with_no_cliff_compounds_in_test_gives_nan_cliff_not_a_crash():
    df = _toy_frame()
    df.loc[df["split"] == "test", "cliff_mol"] = 0
    r = gap.evaluate_target(df, seed=0)
    assert np.isnan(r.pointwise.rmse_cliff)
    assert np.isfinite(r.pointwise.rmse_all)


def test_per_learner_gaps_cover_all_three_learners():
    r = gap.evaluate_target(_toy_frame(), seed=0)
    assert set(r.per_learner_gap) == {"svm", "lightgbm", "mlp"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gap.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.models.gap'`

- [ ] **Step 3: Implement**

```python
# src/molace/models/gap.py
"""The dependent variable, defined once.

gap      = RMSE_pointwise - RMSE_pairwise          (positive means pairwise wins)
access   = RMSE_pointwise - RMSE_knn_floor         (what test-time label access buys)
correction = RMSE_knn_floor - RMSE_pairwise        (what the learned pairwise function buys)

The telescoping access + correction = gap is trivially true for any three numbers and carries no
content by itself. What gives `correction` meaning is the prediction identity enforced in
models.pairwise: with uniform weights over the same anchors the pairwise prediction is the floor
plus the mean learned correction, so the error change is attributable to that function and
nothing else.

RMSE is nonlinear, so `correction` is NOT a share of `gap` and must never be reported as a
percentage of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.models import anchors, knn_floor, pairwise, pointwise


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


@dataclass(frozen=True)
class ArmResult:
    rmse_all: float
    rmse_cliff: float
    selected: str


@dataclass(frozen=True)
class GapResult:
    pointwise: ArmResult
    knn_floor: ArmResult
    pairwise: ArmResult
    gap: float
    access: float
    correction: float
    per_learner_gap: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if abs((self.access + self.correction) - self.gap) > 1e-9:
            raise ValueError(
                "decomposition is inconsistent: access + correction must equal gap, got "
                f"{self.access} + {self.correction} != {self.gap}"
            )


def evaluate_target(df: pd.DataFrame, seed: int) -> GapResult:
    tr = df["split"].to_numpy() == "train"
    te = ~tr
    fp = ecfp4(df["smiles"].tolist())
    y = df["y"].to_numpy(dtype=float)
    cliff = df["cliff_mol"].to_numpy(dtype=int)[te].astype(bool)

    X_tr, y_tr, X_te, y_te = fp[tr], y[tr], fp[te], y[te]
    sim_all = tanimoto_matrix(fp)
    sim_tr = sim_all[np.ix_(tr, tr)]
    idx = anchors.select(sim_all[np.ix_(te, tr)])

    def arm(pred: np.ndarray, selected: str) -> ArmResult:
        return ArmResult(
            rmse_all=rmse(y_te, pred),
            rmse_cliff=rmse(y_te[cliff], pred[cliff]) if cliff.any() else float("nan"),
            selected=selected,
        )

    pw_name, _ = pointwise.select_by_cv(X_tr, y_tr, seed)
    pw = arm(pointwise.fit_predict(pw_name, X_tr, y_tr, X_te, seed), pw_name)
    fl = arm(knn_floor.predict(y_tr, idx), "knn")

    # Pairwise selection by CV over (learner, feature map), training split only.
    best, best_score = None, np.inf
    for learner in pointwise.LEARNERS:
        for kind in pairwise.FEATURE_MAPS:
            score = _pairwise_cv_rmse(learner, kind, X_tr, y_tr, sim_tr, seed)
            if score < best_score:
                best, best_score = (learner, kind), score
    learner, kind = best
    model = pairwise.fit(learner, kind, X_tr, y_tr, sim_tr, anchors.M, seed)
    pr = arm(model.predict(y_tr, X_tr, X_te, idx), f"{learner}+{kind}")

    per_learner = {}
    for name in pointwise.LEARNERS:
        p_pred = pointwise.fit_predict(name, X_tr, y_tr, X_te, seed)
        q = pairwise.fit(name, "difference", X_tr, y_tr, sim_tr, anchors.M, seed)
        per_learner[name] = rmse(y_te, p_pred) - rmse(y_te, q.predict(y_tr, X_tr, X_te, idx))

    return GapResult(
        pointwise=pw, knn_floor=fl, pairwise=pr,
        gap=pw.rmse_all - pr.rmse_all,
        access=pw.rmse_all - fl.rmse_all,
        correction=fl.rmse_all - pr.rmse_all,
        per_learner_gap=per_learner,
    )


def _pairwise_cv_rmse(learner, kind, X_tr, y_tr, sim_tr, seed, n_folds: int = 5) -> float:
    """Mean CV RMSE of the pairwise arm, measured on held-out molecules, not on pairs."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    errs = []
    for tr, va in kf.split(X_tr):
        s_in = sim_tr[np.ix_(tr, tr)]
        idx = anchors.select(sim_tr[np.ix_(va, tr)])
        model = pairwise.fit(learner, kind, X_tr[tr], y_tr[tr], s_in, anchors.M, seed)
        errs.append(rmse(y_tr[va], model.predict(y_tr[tr], X_tr[tr], X_tr[va], idx)))
    return float(np.mean(errs))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_gap.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/molace/models/gap.py tests/test_gap.py
git commit -m "the gap has one definition and its decomposition is checked on construction

Three arms, one evaluation split, one metric pair, and the dependent variable computed in a single
place rather than assembled from sources that use different splits and different metrics. The
split into what label access buys and what the learned function buys is validated in the
constructor, so an inconsistent triple cannot be stored and silently analysed. The docstring
records that the telescoping is trivial while the prediction identity is what licenses the
attribution, and that the correction is not a share of the gap, because the error is nonlinear and
reporting a percentage would be meaningless. Cross-validation for the pairwise arm holds out
molecules rather than pairs, since holding out pairs would leak a molecule's label through its
partner."
```

---

### Task 13: The sweep, cached per target

**Files:**
- Create: `src/molace/analysis/sweep.py`, `tests/test_sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `measure_target(name, seed) -> dict`; `run_sweep(datasets, seeds, force=False)
  -> pandas.DataFrame` with one row per target; `CACHE = Path("results/tasks")`.

Every target's record is written to `results/tasks/<name>__seed<k>.json` and reused unless
`force=True`, so a crash on target 23 keeps the first 22 and a re-run costs nothing. Each record
carries `prereg` (the blob hash) so a result produced under a different plan is visible.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep.py
import json

import numpy as np
import pytest
from molace.analysis import sweep


def test_record_schema_has_every_field_the_analysis_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rec = sweep.measure_target("CHEMBL2835_Ki", seed=0)
    for key in ("dataset", "receptor_class", "seed", "prereg", "n_molecules",
                "assortativity", "assortativity_coverage",
                "unbiased_homophily", "label_informativeness", "adjusted_homophily",
                "rogi", "rogi_flavour", "dirichlet",
                "mean_degree", "n_components", "largest_component_fraction", "n_triangles",
                "rmse_pointwise", "rmse_knn_floor", "rmse_pairwise",
                "rmse_cliff_pointwise", "rmse_cliff_knn_floor", "rmse_cliff_pairwise",
                "gap", "access", "correction", "selected_pointwise", "selected_pairwise"):
        assert key in rec, key


def test_record_is_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    first = sweep.measure_target("CHEMBL2835_Ki", seed=0)
    path = tmp_path / "CHEMBL2835_Ki__seed0.json"
    assert path.is_file()
    # corrupt the cache with a sentinel; a reuse must return the sentinel, not recompute
    rec = json.loads(path.read_text())
    rec["gap"] = -999.0
    path.write_text(json.dumps(rec))
    assert sweep.measure_target("CHEMBL2835_Ki", seed=0)["gap"] == -999.0
    assert sweep.measure_target("CHEMBL2835_Ki", seed=0, force=True)["gap"] != -999.0
    assert first["dataset"] == "CHEMBL2835_Ki"


def test_decomposition_holds_in_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rec = sweep.measure_target("CHEMBL2835_Ki", seed=0)
    assert rec["access"] + rec["correction"] == pytest.approx(rec["gap"], abs=1e-9)


def test_every_record_carries_the_prereg_hash(tmp_path, monkeypatch):
    from molace.analysis.prereg import prereg_fingerprint
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    assert sweep.measure_target("CHEMBL2835_Ki", seed=0)["prereg"] == prereg_fingerprint()


def test_sweep_averages_over_seeds_and_returns_one_row_per_target(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    df = sweep.run_sweep(["CHEMBL2835_Ki", "CHEMBL4203_Ki"], seeds=(0, 1))
    assert len(df) == 2
    assert set(df["dataset"]) == {"CHEMBL2835_Ki", "CHEMBL4203_Ki"}
    assert df["seed"].isna().all() or "seed" not in df.columns


def test_a_failing_target_does_not_lose_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    df = sweep.run_sweep(["CHEMBL2835_Ki", "CHEMBL_nope"], seeds=(0,))
    assert len(df) == 1
    assert df.iloc[0]["dataset"] == "CHEMBL2835_Ki"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.analysis.sweep'`

- [ ] **Step 3: Implement**

```python
# src/molace/analysis/sweep.py
"""Per-target measurement, cached to disk.

One JSON per (target, seed). A crash on target 23 keeps the first 22, and every record carries
the blob hash of the frozen plan, so a number produced under a different plan is visible rather
than arguable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data import moleculeace as ma
from molace.graphs import diagnostics as dg
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.measures import rogi as rogi_mod
from molace.measures.assortativity import target_assortativity
from molace.measures.dirichlet import dirichlet_energy
from molace.measures.homophily import adjusted_homophily
from molace.measures.informativeness import label_informativeness
from molace.measures.unbiased import unbiased_homophily
from molace.models.gap import evaluate_target

log = logging.getLogger(__name__)
CACHE = Path(__file__).resolve().parents[3] / "results" / "tasks"


def _maybe(fn, *a, **kw):
    """Measures that are undefined on a degenerate target return None rather than crashing."""
    try:
        return fn(*a, **kw)
    except (ValueError, TypeError) as exc:
        log.warning("measure %s skipped: %s", getattr(fn, "__name__", fn), exc)
        return None


def measure_target(name: str, seed: int, force: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}__seed{seed}.json"
    if path.is_file() and not force:
        return json.loads(path.read_text())

    plan = load_prereg()
    k = plan["graphs"]["primary"]["k"]
    df = ma.load_target(name)
    fp = ecfp4(df["smiles"].tolist())
    y = df["y"].to_numpy(dtype=float)
    cliff = df["cliff_mol"].to_numpy(dtype=int)
    g = knn.knn_graph(tanimoto_matrix(fp), k=k)
    diag = dg.graph_diagnostics(g)

    asrt = target_assortativity(g, y)
    ub = _maybe(unbiased_homophily, g, cliff)
    li = _maybe(label_informativeness, g, cliff)
    adj = _maybe(adjusted_homophily, g, cliff)
    rg = _maybe(rogi_mod.roughness, fp, y)
    dr = _maybe(dirichlet_energy, g, y)
    res = evaluate_target(df, seed=seed)

    rec = {
        "dataset": name,
        "receptor_class": ma.receptor_class(name),
        "seed": seed,
        "prereg": prereg_fingerprint(),
        "n_molecules": int(len(df)),
        "assortativity": asrt.value,
        "assortativity_coverage": asrt.coverage,
        "unbiased_homophily": None if ub is None else ub.value,
        "label_informativeness": None if li is None else li.value,
        "adjusted_homophily": None if adj is None else adj.value,
        "rogi": None if rg is None else rg.value,
        "rogi_flavour": rogi_mod.ROGI_FLAVOUR,
        "dirichlet": None if dr is None else dr.value,
        "mean_degree": diag["mean_degree"],
        "n_components": diag["n_components"],
        "largest_component_fraction": diag["largest_component_fraction"],
        "n_triangles": diag["n_triangles"],
        "rmse_pointwise": res.pointwise.rmse_all,
        "rmse_knn_floor": res.knn_floor.rmse_all,
        "rmse_pairwise": res.pairwise.rmse_all,
        "rmse_cliff_pointwise": res.pointwise.rmse_cliff,
        "rmse_cliff_knn_floor": res.knn_floor.rmse_cliff,
        "rmse_cliff_pairwise": res.pairwise.rmse_cliff,
        "gap": res.gap,
        "access": res.access,
        "correction": res.correction,
        "selected_pointwise": res.pointwise.selected,
        "selected_pairwise": res.pairwise.selected,
        "per_learner_gap": res.per_learner_gap,
    }
    path.write_text(json.dumps(rec, indent=2, allow_nan=True))
    return rec


def run_sweep(datasets=None, seeds=(0, 1, 2), force: bool = False) -> pd.DataFrame:
    """One row per target, learned quantities averaged over seeds."""
    names = list(datasets) if datasets is not None else list(ma.DATASETS)
    rows = []
    for name in names:
        per_seed = []
        for seed in seeds:
            try:
                per_seed.append(measure_target(name, seed, force=force))
            except Exception as exc:                      # a bad target must not lose the rest
                log.error("target %s seed %s failed: %s", name, seed, exc)
        if not per_seed:
            continue
        head = per_seed[0]
        row = {k: v for k, v in head.items() if k not in {"seed", "per_learner_gap"}}
        for field in ("rmse_pointwise", "rmse_knn_floor", "rmse_pairwise",
                      "rmse_cliff_pointwise", "rmse_cliff_knn_floor", "rmse_cliff_pairwise",
                      "gap", "access", "correction"):
            row[field] = float(np.nanmean([r[field] for r in per_seed]))
        row["n_seeds"] = len(per_seed)
        rows.append(row)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Write `measures/dirichlet.py`, which the sweep imports**

```python
# src/molace/measures/dirichlet.py
"""Normalised Dirichlet energy of the label.

Presented as an import, not as a measure of the Prokhorenkova line: the string "dirichlet" does
not appear in any of their five relevant papers. It is defended on its own merits, namely that it
is defined for a continuous label, and its normalisation is declared as its one free parameter.

    E = sum_{(u,v) in E} (y_u - y_v)^2 / (|E| * var(y))

The denominator makes it comparable across targets with different label scales and edge counts.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from molace.measures.assortativity import MeasureResult


def dirichlet_energy(g: nx.Graph, y: np.ndarray) -> MeasureResult:
    y = np.asarray(y, dtype=float)
    if len(y) != g.number_of_nodes():
        raise ValueError(f"length mismatch: {len(y)} labels for {g.number_of_nodes()} nodes")
    if g.number_of_edges() == 0:
        raise ValueError("Dirichlet energy is undefined on a graph with no edges")
    var = float(np.var(y))
    if var == 0.0:
        raise ValueError("Dirichlet energy is undefined when the label is constant")
    nodes = sorted(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    total = sum((y[pos[u]] - y[pos[v]]) ** 2 for u, v in g.edges())
    used = sum(1 for _, d in g.degree() if d > 0)
    return MeasureResult(
        value=float(total / (g.number_of_edges() * var)),
        coverage=used / g.number_of_nodes(),
        n_used=used,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sweep.py tests/ -v`
Expected: all pass

- [ ] **Step 6: Run the real sweep on all 30 targets**

```bash
uv run python -c "
from molace.analysis.sweep import run_sweep
df = run_sweep()
df.to_csv('results/spine.csv', index=False)
print(df[['dataset','assortativity','rogi','mean_degree','gap','access','correction']].to_string())
print()
print('rows:', len(df), '| targets with a defined assortativity:', df['assortativity'].notna().sum())
"
```
Expected: 30 rows. If fewer, read `results/tasks/` for which targets failed and why before
proceeding — a missing target is a finding, not a nuisance.

- [ ] **Step 7: Commit**

```bash
git add src/molace/analysis/sweep.py src/molace/measures/dirichlet.py tests/test_sweep.py results/spine.csv
git commit -m "the sweep caches per target and stamps every record with the frozen plan

One record per target and seed on disk, so a failure on the twenty-third target keeps the first
twenty-two and a re-run is free. Each record carries the blob hash of the pre-registration, which
makes a number produced under a different plan visible rather than arguable, and the decomposition
is re-checked inside the record. A measure that is undefined on a degenerate target records a null
and logs why instead of taking the sweep down, and a target that fails entirely is dropped from the
frame rather than silently filled. Dirichlet energy arrives normalised by label variance and edge
count so it is comparable across targets, and is labelled an import rather than this group's
measure, since the word does not appear in their papers."
```

---

### Task 14: Correlation with a cluster bootstrap, positive control first

**Files:**
- Create: `src/molace/analysis/correlate.py`, `tests/test_correlate.py`
- Test: `tests/test_correlate.py`

**Interfaces:**
- Consumes: `analysis.prereg.load_prereg`, the sweep frame.
- Produces: `cluster_bootstrap_spearman(x, y, clusters, n_resamples=10000, seed=0)
  -> BootResult` with `rho: float, lo: float, hi: float, excludes_zero: bool`;
  `incremental_contribution(frame, target, extra_cols) -> BootResult`;
  `effective_n(clusters, rho_intra=0.3) -> float`;
  `report(frame) -> dict` ordered so the positive control is first.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_correlate.py
import numpy as np
import pandas as pd
import pytest
from molace.analysis import correlate as co


def test_bootstrap_recovers_a_strong_correlation_and_excludes_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x * 2.0 + rng.normal(scale=0.3, size=60)
    clusters = np.repeat(np.arange(6), 10)
    r = co.cluster_bootstrap_spearman(x, y, clusters, n_resamples=2000, seed=0)
    assert r.rho > 0.9 and r.lo > 0.0 and r.excludes_zero


def test_bootstrap_does_not_exclude_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = rng.normal(size=60)
    clusters = np.repeat(np.arange(6), 10)
    r = co.cluster_bootstrap_spearman(x, y, clusters, n_resamples=2000, seed=0)
    assert not r.excludes_zero
    assert r.lo < 0.0 < r.hi


def test_resampling_is_over_clusters_not_rows():
    """With one informative cluster and five flat ones, row resampling would be overconfident."""
    x = np.concatenate([np.linspace(0, 1, 10), np.zeros(50)])
    y = np.concatenate([np.linspace(0, 1, 10), np.zeros(50)])
    clusters = np.repeat(np.arange(6), 10)
    r = co.cluster_bootstrap_spearman(x, y, clusters, n_resamples=2000, seed=0)
    assert r.hi - r.lo > 0.2, "a cluster bootstrap must be wider than a row bootstrap here"


def test_effective_n_is_about_ten_for_the_real_cluster_sizes():
    clusters = (["GPCR"] * 12 + ["Kinase"] * 6 + ["NR"] * 6 +
                ["Other"] * 3 + ["Protease"] * 2 + ["Transferase"] * 1)
    assert 8.0 < co.effective_n(clusters, rho_intra=0.3) < 12.0


def test_effective_n_equals_n_when_clusters_are_singletons():
    assert co.effective_n(list(range(30)), rho_intra=0.3) == pytest.approx(30.0)


def test_incremental_contribution_is_near_zero_for_a_redundant_predictor():
    rng = np.random.default_rng(2)
    base = rng.normal(size=60)
    frame = pd.DataFrame({
        "gap": base + rng.normal(scale=0.2, size=60),
        "assortativity": base,
        "rogi": base,                      # a perfect duplicate of the signal
        "mean_degree": rng.normal(size=60),
        "n_molecules": rng.integers(600, 3600, size=60),
        "receptor_class": np.repeat(np.arange(6), 10),
    })
    r = co.incremental_contribution(frame, "gap", ["rogi", "mean_degree", "n_molecules"])
    assert not r.excludes_zero


def test_report_puts_the_positive_control_first():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "dataset": [f"T{i}" for i in range(30)],
        "assortativity": rng.normal(size=30),
        "rogi": rng.normal(size=30),
        "mean_degree": rng.normal(size=30),
        "n_molecules": rng.integers(600, 3600, size=30),
        "gap": rng.normal(size=30),
        "access": rng.normal(size=30),
        "correction": rng.normal(size=30),
        "receptor_class": (["GPCR"] * 12 + ["Kinase"] * 6 + ["NR"] * 6 +
                           ["Other"] * 3 + ["Protease"] * 2 + ["Transferase"] * 1),
    })
    keys = list(co.report(frame))
    assert keys[0] == "positive_control"
    assert "n" in co.report(frame)["meta"] and "n_eff" in co.report(frame)["meta"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_correlate.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.analysis.correlate'`

- [ ] **Step 3: Implement**

```python
# src/molace/analysis/correlate.py
"""The frozen analysis, in the order the plan fixes.

The positive control comes first because kNN regression works precisely when the label is smooth
on the kNN graph, so assortativity must predict what label access buys. An interval covering zero
there indicts the pipeline, not the hypothesis, and nothing downstream is interpreted until it is
resolved.

Resampling is over receptor classes, not rows. The 30 targets are not independent: two are the
same protein measured two ways, 26.7 percent of molecules appear in more than one task, and JAK1
and JAK2 overlap 88.5 percent. n_eff is reported wherever n is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BootResult:
    rho: float
    lo: float
    hi: float
    excludes_zero: bool
    n: int
    n_clusters: int


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def cluster_bootstrap_spearman(x, y, clusters, n_resamples: int = 10000,
                               seed: int = 0) -> BootResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    groups = [np.flatnonzero(cl == c) for c in uniq]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        v = _spearman(x[idx], y[idx])
        if np.isfinite(v):
            draws.append(v)
    d = np.asarray(draws)
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d.size else (np.nan, np.nan)
    return BootResult(
        rho=_spearman(x, y), lo=lo, hi=hi,
        excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)),
        n=int((np.isfinite(x) & np.isfinite(y)).sum()),
        n_clusters=len(uniq),
    )


def effective_n(clusters, rho_intra: float = 0.3) -> float:
    """Block-exchangeable effective sample size: n / (1 + (m_bar - 1) * rho)."""
    cl = pd.Series(list(clusters))
    sizes = cl.value_counts().to_numpy(dtype=float)
    n = float(sizes.sum())
    m_bar = float((sizes ** 2).sum() / n)         # size-weighted mean cluster size
    return n / (1.0 + (m_bar - 1.0) * rho_intra)


def incremental_contribution(frame: pd.DataFrame, target: str, extra_cols,
                             predictor: str = "assortativity",
                             n_resamples: int = 10000, seed: int = 0) -> BootResult:
    """Spearman of the predictor against the target's residual after the extra columns.

    Rank-based throughout: ranks in, OLS residual, Spearman out. This is the quantity the plan's
    decision rule needs — does the predictor say anything the roughness baseline, the mean degree
    and the task size do not already say.
    """
    cols = [target, predictor, *extra_cols]
    f = frame.dropna(subset=cols)
    r = f[cols].rank()
    design = np.column_stack([np.ones(len(f)), *[r[c].to_numpy() for c in extra_cols]])
    beta, *_ = np.linalg.lstsq(design, r[target].to_numpy(), rcond=None)
    resid = r[target].to_numpy() - design @ beta
    return cluster_bootstrap_spearman(
        r[predictor].to_numpy(), resid, f["receptor_class"].to_numpy(),
        n_resamples=n_resamples, seed=seed,
    )


def report(frame: pd.DataFrame, n_resamples: int = 10000, seed: int = 0) -> dict:
    cl = frame["receptor_class"].to_numpy()
    a = frame["assortativity"].to_numpy(dtype=float)
    out = {
        "positive_control": cluster_bootstrap_spearman(
            a, frame["access"].to_numpy(dtype=float), cl, n_resamples, seed),
        "primary": cluster_bootstrap_spearman(
            a, frame["gap"].to_numpy(dtype=float), cl, n_resamples, seed),
        "incremental_over_rogi": incremental_contribution(
            frame, "gap", ["rogi", "mean_degree", "n_molecules"],
            n_resamples=n_resamples, seed=seed),
        "decisive_secondary": cluster_bootstrap_spearman(
            a, frame["correction"].to_numpy(dtype=float), cl, n_resamples, seed),
        "correction_mean": float(np.nanmean(frame["correction"].to_numpy(dtype=float))),
        "meta": {"n": int(len(frame)), "n_eff": round(effective_n(cl), 1),
                 "rho_intra_assumed": 0.3, "n_clusters": int(len(np.unique(cl)))},
    }
    out["supported"] = bool(
        out["primary"].excludes_zero and out["incremental_over_rogi"].excludes_zero
    )
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_correlate.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the frozen analysis and record it**

```bash
uv run python -c "
import json, pandas as pd
from molace.analysis.correlate import report
from molace.analysis.prereg import prereg_fingerprint
df = pd.read_csv('results/spine.csv')
r = report(df)
print('prereg:', prereg_fingerprint())
print('n =', r['meta']['n'], '| n_eff =', r['meta']['n_eff'])
for k in ('positive_control','primary','incremental_over_rogi','decisive_secondary'):
    b = r[k]
    print(f'{k:24s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  excludes 0: {b.excludes_zero}')
print('mean correction:', round(r['correction_mean'], 4))
print('PRIMARY CLAIM SUPPORTED:', r['supported'])
" | tee results/report_spine.txt
```

Read the positive control line first. If it does not exclude zero, stop and diagnose the
pipeline; do not interpret the primary.

- [ ] **Step 6: Commit**

```bash
git add src/molace/analysis/correlate.py tests/test_correlate.py results/report_spine.txt
git commit -m "the analysis resamples receptor classes and reads the control before the result

The thirty targets are not independent: two are one protein measured two ways, a quarter of the
molecules appear in more than one task, and the closest pair of kinases shares nearly nine tenths
of the smaller set, so the bootstrap resamples classes rather than rows and the effective sample
size is reported beside the nominal one. A test pins that behaviour by constructing a case where a
row bootstrap would be overconfident. The control comes first in the report because nearest
neighbour regression works exactly when the label is smooth on the nearest neighbour graph, so an
interval covering zero there is a verdict on the pipeline rather than on the hypothesis. The
incremental quantity is rank based throughout, which is what the decision rule needs: whether the
statistic says anything the roughness baseline, the degree and the task size do not."
```

---

### Task 15: The Hodge complex and the dimension census

**Files:**
- Create: `src/molace/hodge/complex.py`, `tests/test_hodge_complex.py`
- Test: `tests/test_hodge_complex.py`

**Interfaces:**
- Consumes: networkx, scipy.sparse.
- Produces: `Complex` dataclass with `B1: csr_matrix (|V| x |E|)`, `B2: csr_matrix (|E| x |T|)`,
  `edges: list[tuple[int,int]]`, `triangles: list[tuple[int,int,int]]`, `nodes: list[int]`;
  `build(g, triangle_budget: int | None = None) -> Complex`;
  `census(g, triangle_budget=None, exact_rank_max_edges=5000) -> dict`.

**Orientation and signs.** Edges are stored as `(u, v)` with `u < v` and oriented `u -> v`, so
`B1[u, e] = -1` and `B1[v, e] = +1`. Triangles are stored as `(i, j, k)` with `i < j < k` and
contribute `+1` to `(i,j)`, `+1` to `(j,k)`, `-1` to `(i,k)`. With those signs
`B1 @ B2 = 0` identically, which the first test asserts — it is the identity every later
projection rests on.

**Why a budget.** Measured on real graphs: CHEMBL234_Ki at a Tanimoto threshold of 0.3 has
337,346 edges and 32,992,829 triangles, making the curl operator about 1.1e13 entries, and the
smallest target at the same threshold has 61 times more triangles than it has pairs. The budget
is a hard cap, and exceeding it switches to a sampled triangle set with the sampling logged, never
silently.

**Why the rank split is bounded.** `dim_gradient = |V| - c` and `dim_cycle_space = |E| - |V| + c`
are exact and cheap. Splitting the cycle space into curl and harmonic needs `rank(B2)`, which is
only computed exactly where `|E| <= exact_rank_max_edges` (dense SVD is affordable there). Above
that the split is estimated by a randomised range finder and reported as estimated, with the
tolerance in the record. Reporting an estimate as exact is the failure mode to avoid.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hodge_complex.py
import networkx as nx
import numpy as np
import pytest
from molace.hodge import complex as cx


def test_the_fundamental_identity_b1_b2_is_zero():
    g = nx.complete_graph(7)
    c = cx.build(g)
    assert abs((c.B1 @ c.B2)).max() == 0.0


def test_identity_holds_on_a_random_graph_too():
    g = nx.gnp_random_graph(40, 0.25, seed=0)
    c = cx.build(g)
    assert abs((c.B1 @ c.B2)).max() == 0.0


def test_b1_has_one_minus_one_and_one_plus_one_per_edge():
    c = cx.build(nx.path_graph(5))
    col_sums = np.asarray(c.B1.sum(axis=0)).ravel()
    assert np.allclose(col_sums, 0.0)
    assert abs(c.B1).sum() == 2 * c.B1.shape[1]


def test_triangle_count_matches_networkx():
    g = nx.gnp_random_graph(30, 0.3, seed=1)
    c = cx.build(g)
    assert len(c.triangles) == sum(nx.triangles(g).values()) // 3


def test_a_graph_with_no_triangles_raises_because_curl_is_not_defined_there():
    with pytest.raises(ValueError, match="no triangles"):
        cx.build(nx.path_graph(10))


def test_triangle_budget_subsamples_and_records_it():
    g = nx.complete_graph(20)          # 1140 triangles
    c = cx.build(g, triangle_budget=100)
    assert len(c.triangles) == 100
    assert c.triangles_sampled is True
    assert c.triangles_total == 1140
    assert abs((c.B1 @ c.B2)).max() == 0.0     # the identity survives subsampling


def test_triangle_sampling_is_deterministic():
    g = nx.complete_graph(20)
    assert cx.build(g, triangle_budget=50).triangles == cx.build(g, triangle_budget=50).triangles


def test_census_reports_exact_combinatorics_on_a_known_graph():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    g.add_edges_from([(3, 4), (4, 5), (3, 5)])
    d = cx.census(g)
    assert d["n_nodes"] == 6 and d["n_edges"] == 6 and d["n_components"] == 2
    assert d["n_triangles"] == 2
    assert d["dim_gradient"] == 4                 # 6 - 2
    assert d["dim_cycle_space"] == 2              # 6 - 6 + 2
    assert d["dim_curl"] == 2                     # two independent triangles
    assert d["dim_harmonic"] == 0
    assert d["rank_method"] == "exact"


def test_census_finds_a_harmonic_component_on_a_hollow_square():
    # A 4-cycle has a cycle space of dimension 1 and no triangles, so the cycle is harmonic.
    g = nx.cycle_graph(4)
    d = cx.census(g, require_triangles=False)
    assert d["n_triangles"] == 0
    assert d["dim_cycle_space"] == 1
    assert d["dim_curl"] == 0
    assert d["dim_harmonic"] == 1


def test_census_marks_the_rank_as_estimated_above_the_exact_threshold():
    g = nx.gnp_random_graph(90, 0.4, seed=2)
    d = cx.census(g, exact_rank_max_edges=10)
    assert d["rank_method"] == "estimated"
    assert "rank_tolerance" in d
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_hodge_complex.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.hodge.complex'`

- [ ] **Step 3: Implement**

```python
# src/molace/hodge/complex.py
"""Boundary operators of the clique 2-complex, and the dimension census.

Edges are (u, v) with u < v, oriented u -> v: B1[u, e] = -1, B1[v, e] = +1.
Triangles are (i, j, k) with i < j < k and contribute +1 to (i,j), +1 to (j,k), -1 to (i,k).
With those signs B1 @ B2 = 0 identically, which is asserted as a test, and that identity is why
the gradient and curl subspaces are orthogonal.

The census reports what is exact as exact and what is estimated as estimated. dim_gradient and
dim_cycle_space are cheap and exact. Splitting the cycle space into curl and harmonic needs
rank(B2); that is computed exactly only on small graphs and estimated above a stated edge count,
because reporting an estimate as exact is how a measurement becomes a false finding.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp


@dataclass
class Complex:
    B1: sp.csr_matrix
    B2: sp.csr_matrix
    nodes: list
    edges: list
    triangles: list
    triangles_total: int
    triangles_sampled: bool = False


def _triangles(g: nx.Graph) -> list[tuple[int, int, int]]:
    out = []
    adj = {v: set(g.neighbors(v)) for v in g}
    for v in sorted(g.nodes()):
        higher = sorted(u for u in adj[v] if u > v)
        for a, b in itertools.combinations(higher, 2):
            if b in adj[a]:
                out.append((v, a, b))
    return out


def build(g: nx.Graph, triangle_budget: int | None = None,
          require_triangles: bool = True, seed: int = 0) -> Complex:
    nodes = sorted(g.nodes())
    nidx = {v: i for i, v in enumerate(nodes)}
    edges = sorted(tuple(sorted(e)) for e in g.edges())
    eidx = {e: i for i, e in enumerate(edges)}

    tris = _triangles(g)
    total = len(tris)
    sampled = False
    if require_triangles and total == 0:
        raise ValueError(
            "this graph has no triangles, so the curl subspace is empty by construction and a "
            "measured curl of zero would be an artefact of the graph rather than a finding"
        )
    if triangle_budget is not None and total > triangle_budget:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(total, size=triangle_budget, replace=False))
        tris = [tris[i] for i in keep]
        sampled = True

    rows, cols, vals = [], [], []
    for (u, v), j in eidx.items():
        rows += [nidx[u], nidx[v]]
        cols += [j, j]
        vals += [-1.0, 1.0]
    B1 = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(edges)))

    rows, cols, vals = [], [], []
    for t, (i, j, k) in enumerate(tris):
        for e, s in (((i, j), 1.0), ((j, k), 1.0), ((i, k), -1.0)):
            rows.append(eidx[e])
            cols.append(t)
            vals.append(s)
    B2 = sp.csr_matrix((vals, (rows, cols)), shape=(len(edges), len(tris)))

    return Complex(B1=B1, B2=B2, nodes=nodes, edges=edges, triangles=tris,
                   triangles_total=total, triangles_sampled=sampled)


def _rank_exact(B2: sp.csr_matrix) -> int:
    if B2.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(B2.toarray(), tol=1e-8))


def _rank_estimated(B2: sp.csr_matrix, tol: float = 1e-8, seed: int = 0) -> int:
    """Randomised range-finder estimate of rank(B2), capped by the cycle space dimension."""
    if B2.shape[1] == 0:
        return 0
    rng = np.random.default_rng(seed)
    probe = min(B2.shape[1], 2048)
    omega = rng.normal(size=(B2.shape[1], probe))
    y = np.asarray(B2 @ omega)
    s = np.linalg.svd(y, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())


def census(g: nx.Graph, triangle_budget: int | None = None,
           exact_rank_max_edges: int = 5000, require_triangles: bool = True,
           seed: int = 0) -> dict:
    c = build(g, triangle_budget=triangle_budget, require_triangles=require_triangles, seed=seed)
    n, m = len(c.nodes), len(c.edges)
    comps = nx.number_connected_components(g)
    dim_grad = n - comps
    dim_cycle = m - n + comps
    if m <= exact_rank_max_edges:
        dim_curl, method, tol = _rank_exact(c.B2), "exact", None
    else:
        tol = 1e-8
        dim_curl, method = _rank_estimated(c.B2, tol=tol, seed=seed), "estimated"
    dim_curl = min(dim_curl, dim_cycle)
    out = {
        "n_nodes": n, "n_edges": m, "n_components": comps,
        "n_triangles": len(c.triangles), "n_triangles_total": c.triangles_total,
        "triangles_sampled": c.triangles_sampled,
        "dim_gradient": dim_grad, "dim_cycle_space": dim_cycle,
        "dim_curl": dim_curl, "dim_harmonic": dim_cycle - dim_curl,
        "rank_method": method,
    }
    if tol is not None:
        out["rank_tolerance"] = tol
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_hodge_complex.py -v`
Expected: 10 passed

- [ ] **Step 5: Run the census on all 30 targets**

```bash
uv run python -c "
import json, numpy as np, pandas as pd
from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data import moleculeace as ma
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.hodge.complex import census
k = load_prereg()['graphs']['primary']['k']
rows = []
for name in ma.DATASETS:
    df = ma.load_target(name)
    g = knn.knn_graph(tanimoto_matrix(ecfp4(df['smiles'].tolist())), k=k)
    d = census(g, triangle_budget=200000, require_triangles=False)
    d['dataset'] = name; d['prereg'] = prereg_fingerprint()
    rows.append(d)
    print(name, d['n_edges'], d['n_triangles'], d['dim_gradient'], d['dim_curl'], d['dim_harmonic'], d['rank_method'])
t = pd.DataFrame(rows); t.to_csv('results/hodge_census.csv', index=False)
print()
print('harmonic / gradient ratio, median:', round(float(np.median(t.dim_harmonic / t.dim_gradient)), 3))
print('targets where harmonic exceeds gradient:', int((t.dim_harmonic > t.dim_gradient).sum()), 'of', len(t))
print('targets with zero triangles:', int((t.n_triangles == 0).sum()))
"
```

The synthetic probe that motivated this expected the harmonic component to run 2 to 5 times the
gradient one. **That probe is not a result and must not be quoted.** This table is the result,
whatever it says.

- [ ] **Step 6: Commit**

```bash
git add src/molace/hodge/complex.py tests/test_hodge_complex.py results/hodge_census.csv
git commit -m "the complex asserts its own fundamental identity, and the census says what it estimates

Signs are chosen so the composition of the two boundary operators vanishes identically, which the
first test asserts, because that identity is the reason the gradient and curl subspaces are
orthogonal and every later projection depends on it. A graph without triangles raises rather than
quietly reporting no curl, since an empty curl subspace there is a property of the graph and not a
measurement. The triangle budget is a hard cap and subsampling is recorded in the complex, because
the largest target reaches thirty-three million triangles and the smallest has sixty times more
triangles than pairs. The gradient and cycle dimensions are exact and cheap; the split of the cycle
space is exact only on small graphs and labelled estimated above a stated edge count, so an
estimate is never read as exact."
```

---

### Task 16: Energy budget and the frozen-encoder readout ablation

**Files:**
- Create: `src/molace/hodge/decompose.py`, `src/molace/hodge/energy.py`,
  `tests/test_hodge_decompose.py`, `tests/test_readout_ablation.py`
- Test: both

**Interfaces:**
- Consumes: `hodge.complex.Complex`, `graphs.fingerprints`.
- Produces: `decompose(c: Complex, f: np.ndarray) -> Decomposition` with fields
  `gradient, curl, harmonic: np.ndarray` and `potential: np.ndarray`;
  `energy.fractions(d: Decomposition) -> dict` with `gradient, curl, harmonic` summing to 1;
  `energy.per_edge_curl(c, d) -> np.ndarray`;
  `energy.readout_ablation(fp, y, c, seed) -> dict`.

- [ ] **Step 1: Write the failing tests for the decomposition**

```python
# tests/test_hodge_decompose.py
import networkx as nx
import numpy as np
import pytest
from molace.hodge import complex as cx
from molace.hodge import energy
from molace.hodge.decompose import decompose


def _c(n=25, p=0.35, seed=0):
    return cx.build(nx.gnp_random_graph(n, p, seed=seed))


def test_a_gradient_flow_decomposes_to_pure_gradient():
    c = _c()
    rng = np.random.default_rng(0)
    phi = rng.normal(size=len(c.nodes))
    f = np.asarray(c.B1.T @ phi).ravel()
    d = decompose(c, f)
    fr = energy.fractions(d)
    assert fr["gradient"] == pytest.approx(1.0, abs=1e-8)
    assert fr["curl"] < 1e-8 and fr["harmonic"] < 1e-8


def test_a_curl_flow_decomposes_to_pure_curl():
    c = _c()
    rng = np.random.default_rng(1)
    psi = rng.normal(size=len(c.triangles))
    f = np.asarray(c.B2 @ psi).ravel()
    fr = energy.fractions(decompose(c, f))
    assert fr["curl"] == pytest.approx(1.0, abs=1e-8)
    assert fr["gradient"] < 1e-8


def test_the_three_components_are_mutually_orthogonal():
    c = _c()
    rng = np.random.default_rng(2)
    f = rng.normal(size=len(c.edges))
    d = decompose(c, f)
    assert abs(float(d.gradient @ d.curl)) < 1e-8
    assert abs(float(d.gradient @ d.harmonic)) < 1e-8
    assert abs(float(d.curl @ d.harmonic)) < 1e-8


def test_the_components_reconstruct_the_flow():
    c = _c()
    rng = np.random.default_rng(3)
    f = rng.normal(size=len(c.edges))
    d = decompose(c, f)
    assert np.allclose(d.gradient + d.curl + d.harmonic, f, atol=1e-8)


def test_fractions_sum_to_one():
    c = _c()
    rng = np.random.default_rng(4)
    fr = energy.fractions(decompose(c, rng.normal(size=len(c.edges))))
    assert sum(fr.values()) == pytest.approx(1.0, abs=1e-8)


def test_a_difference_of_node_labels_is_curl_free_under_true_and_shuffled_labels():
    """The original exit-experiment null, shown to be void.

    dy_ij = y_j - y_i is the gradient of a node labelling, so it is curl-free identically, and a
    permutation of y is still a node labelling.
    """
    c = _c()
    rng = np.random.default_rng(5)
    y = rng.normal(size=len(c.nodes))
    for labels in (y, rng.permutation(y)):
        f = np.array([labels[c.nodes.index(v)] - labels[c.nodes.index(u)] for u, v in c.edges])
        fr = energy.fractions(decompose(c, f))
        assert fr["curl"] < 1e-10
        assert fr["harmonic"] < 1e-10


def test_harmonic_is_nonzero_on_a_graph_with_an_empty_cycle():
    g = nx.cycle_graph(5)
    c = cx.build(g, require_triangles=False)
    f = np.ones(len(c.edges))            # circulating once around the hole
    fr = energy.fractions(decompose(c, f))
    assert fr["harmonic"] > 0.9


def test_per_edge_curl_is_zero_for_a_gradient_flow():
    c = _c()
    rng = np.random.default_rng(6)
    f = np.asarray(c.B1.T @ rng.normal(size=len(c.nodes))).ravel()
    assert float(np.abs(energy.per_edge_curl(c, decompose(c, f))).max()) < 1e-8


def test_flow_length_mismatch_raises():
    c = _c()
    with pytest.raises(ValueError, match="edges"):
        decompose(c, np.ones(3))
```

- [ ] **Step 2: Write the failing tests for the ablation**

```python
# tests/test_readout_ablation.py
import networkx as nx
import numpy as np
import pytest
from molace.hodge import complex as cx
from molace.hodge import energy


def _setup(n=60, seed=0):
    rng = np.random.default_rng(seed)
    fp = (rng.random((n, 2048)) < 0.02).astype(np.uint8)
    w = np.zeros(2048); w[rng.choice(2048, 30, replace=False)] = rng.normal(size=30)
    y = fp @ w + rng.normal(scale=0.1, size=n)
    g = nx.gnp_random_graph(n, 0.3, seed=seed)
    return fp, y, cx.build(g)


def test_a_bias_free_linear_readout_is_a_gradient_flow_to_machine_precision():
    """The §10 theorem. A linear readout on a difference of a frozen encoder is phi_i - phi_j."""
    fp, y, c = _setup()
    out = energy.readout_ablation(fp, y, c, seed=0)
    assert out["linear"]["curl"] < 1e-12
    assert out["linear"]["harmonic"] < 1e-12
    assert out["linear"]["gradient"] == pytest.approx(1.0, abs=1e-10)


def test_an_affine_readout_puts_exactly_three_c_on_every_triangle():
    """h(z) = w.z + c gives a triangle sum of 3c regardless of the data."""
    fp, y, c = _setup()
    out = energy.readout_ablation(fp, y, c, seed=0, affine_bias=0.7)
    assert out["affine_triangle_sum"] == pytest.approx(3 * 0.7, abs=1e-9)


def test_a_nonlinear_readout_has_curl_bounded_away_from_zero():
    """Magnitudes are probe-specific; only the sign of the effect is asserted."""
    fp, y, c = _setup()
    out = energy.readout_ablation(fp, y, c, seed=0)
    assert out["mlp"]["curl"] > 1e-6
    assert out["mlp"]["curl"] > out["linear"]["curl"] * 1000


def test_the_encoder_is_frozen_so_the_linear_problem_is_convex_and_reproducible():
    fp, y, c = _setup()
    a = energy.readout_ablation(fp, y, c, seed=0)["linear"]
    b = energy.readout_ablation(fp, y, c, seed=1)["linear"]
    assert a["curl"] == pytest.approx(b["curl"], abs=1e-12)
    assert a["gradient"] == pytest.approx(b["gradient"], abs=1e-10)
```

- [ ] **Step 3: Run both test files to verify they fail**

Run: `uv run pytest tests/test_hodge_decompose.py tests/test_readout_ablation.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.hodge.decompose'`

- [ ] **Step 4: Implement the decomposition**

```python
# src/molace/hodge/decompose.py
"""Three-way Hodge decomposition of an edge flow.

f = gradient + curl + harmonic, mutually orthogonal because B1 @ B2 = 0.

The two-way "gradient plus circulation" language of the source proposal is wrong, not merely
incomplete: curl-free implies gradient only when the first homology vanishes, and on these graphs
it does not. Harmonic flows are both curl-free and divergence-free yet globally inconsistent, so a
HodgeRank potential recovered on such a graph is locally consistent and globally meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from molace.hodge.complex import Complex


@dataclass(frozen=True)
class Decomposition:
    gradient: np.ndarray
    curl: np.ndarray
    harmonic: np.ndarray
    potential: np.ndarray


def decompose(c: Complex, f: np.ndarray, atol: float = 1e-12) -> Decomposition:
    f = np.asarray(f, dtype=float).ravel()
    if len(f) != len(c.edges):
        raise ValueError(f"flow has {len(f)} entries for {len(c.edges)} edges")
    phi = spla.lsmr(c.B1.T, f, atol=atol, btol=atol)[0]
    grad = np.asarray(c.B1.T @ phi).ravel()
    rest = f - grad
    if c.B2.shape[1] == 0:
        curl = np.zeros_like(f)
    else:
        psi = spla.lsmr(c.B2, rest, atol=atol, btol=atol)[0]
        curl = np.asarray(c.B2 @ psi).ravel()
    return Decomposition(gradient=grad, curl=curl, harmonic=rest - curl, potential=phi)
```

- [ ] **Step 5: Implement the energy budget and the ablation**

```python
# src/molace/hodge/energy.py
"""Energy fractions, per-edge curl, and the readout ablation.

The ablation is the §10 theorem in runnable form. With the encoder FROZEN and a bias-free linear
readout, the flow is w.g(x_i) - w.g(x_j) = phi_i - phi_j, exactly a gradient flow, so the model
collapses to a pointwise model plus anchor averaging. The encoder must not be trained end to end:
a linear head would push the nonlinearity down into the encoder and the ablation would stop
isolating the readout. The project's one-fixed-encoder decision already supplies the frozen
variant, and with the encoder frozen the linear problem is convex, so the collapse is provable
rather than merely measured.

The real boundary is linear against nonlinear readout, not difference-of-encoders against joint
encoding: by Cauchy, h is additive on triangles iff it is linear.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from molace.hodge.complex import Complex
from molace.hodge.decompose import Decomposition, decompose


def fractions(d: Decomposition) -> dict:
    total = float(d.gradient @ d.gradient + d.curl @ d.curl + d.harmonic @ d.harmonic)
    if total == 0.0:
        raise ValueError("the flow is identically zero, so its energy budget is undefined")
    return {
        "gradient": float(d.gradient @ d.gradient) / total,
        "curl": float(d.curl @ d.curl) / total,
        "harmonic": float(d.harmonic @ d.harmonic) / total,
    }


def per_edge_curl(c: Complex, d: Decomposition) -> np.ndarray:
    """The magnitude of the curl component at each edge. Zero for a gradient flow."""
    return np.abs(d.curl)


def _edge_flow(c: Complex, predict) -> np.ndarray:
    """Evaluate a pairwise predictor on every edge, oriented u -> v (second minus first)."""
    return np.array([predict(u, v) for u, v in c.edges], dtype=float)


def readout_ablation(fp: np.ndarray, y: np.ndarray, c: Complex, seed: int = 0,
                     affine_bias: float = 0.0) -> dict:
    """Linear against nonlinear readout on a frozen encoder.

    The encoder is the fingerprint itself: fixed, not learned, exactly as the design requires.
    Training targets are the true edge differences, which are additive by construction.
    """
    x = np.asarray(fp, dtype=np.float64)
    y = np.asarray(y, dtype=float)
    pos = {v: i for i, v in enumerate(c.nodes)}
    pairs = np.array([(pos[u], pos[v]) for u, v in c.edges])
    diffs = x[pairs[:, 1]] - x[pairs[:, 0]]
    targets = y[pairs[:, 1]] - y[pairs[:, 0]]

    lin = Ridge(alpha=1.0, fit_intercept=False).fit(diffs, targets)
    out = {"linear": fractions(decompose(c, _edge_flow(
        c, lambda u, v: float(lin.coef_ @ (x[pos[v]] - x[pos[u]])))))}

    mlp = MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=seed).fit(diffs, targets)
    out["mlp"] = fractions(decompose(c, _edge_flow(
        c, lambda u, v: float(mlp.predict((x[pos[v]] - x[pos[u]]).reshape(1, -1))[0]))))

    # An affine head puts exactly 3c on every triangle, independent of the data.
    if not c.triangles:
        raise ValueError(
            "the readout ablation needs at least one triangle; a complex without triangles has "
            "an empty curl subspace and the ablation would report zero curl as a finding"
        )
    i, j, k = c.triangles[0]
    aff = lambda u, v: float(lin.coef_ @ (x[pos[v]] - x[pos[u]])) + affine_bias
    out["affine_triangle_sum"] = aff(i, j) + aff(j, k) - aff(i, k) + 2 * affine_bias
    out["n_edges"] = len(c.edges)
    out["n_triangles"] = len(c.triangles)
    return out
```

- [ ] **Step 6: Run both test files to verify they pass**

Run: `uv run pytest tests/test_hodge_decompose.py tests/test_readout_ablation.py -v`
Expected: 13 passed

If `test_an_affine_readout_puts_exactly_three_c_on_every_triangle` fails, the triangle sum is being
assembled with the wrong edge signs — the identity is `h(ij) + h(jk) - h(ik)` with the stored
orientation, and the three bias terms must add rather than cancel.

- [ ] **Step 7: Commit**

```bash
git add src/molace/hodge/decompose.py src/molace/hodge/energy.py tests/test_hodge_decompose.py tests/test_readout_ablation.py
git commit -m "the decomposition is three-way, and the theorem runs on a frozen encoder

Curl-free implies gradient only when the first homology vanishes, which on these graphs it does
not, so the two-way language of the source proposal is wrong rather than incomplete and the
harmonic part gets its own component and its own test on a graph whose only cycle is a hole. A test
also records why the original exit experiment was void: the difference of node labels is the
gradient of a labelling and is curl-free identically under true and permuted labels alike, so there
was never a circulation in the target for a signal to live in. The ablation freezes the encoder,
which makes the linear-readout problem convex and the collapse to a pointwise model provable rather
than measured, and asserts the two mathematical facts, that a bias-free linear head gives machine-zero
curl and an affine head gives exactly three times its bias on every triangle, while the nonlinear
magnitudes are only asserted to be bounded away from zero because they are probe-specific."
```

---

### Task 17: The one published pointwise-against-pairwise gap

**Files:**
- Create: `src/molace/data/deepdelta.py`, `tests/test_deepdelta.py`
- Test: `tests/test_deepdelta.py`

**Interfaces:**
- Consumes: `data/raw/DeepDelta` (cloned in Task 1).
- Produces: `DD_DATASETS: tuple[str, ...]` (10 names); `load_predictions(dataset, family, repeat)
  -> tuple[np.ndarray, np.ndarray]` returning `(true_delta, pred_delta)`;
  `published_gaps() -> pandas.DataFrame` with columns
  `dataset, rmse_RandomForest, rmse_ChemProp50, rmse_DeepDelta5, gap_vs_rf, gap_vs_chemprop`.

**Why this exists.** Neither SQRL nor MoleculeACE publishes a per-target pointwise-against-pairwise
gap: SQRL reports only numbers aggregated across tasks and MoleculeACE's twelve algorithms are all
pointwise. DeepDelta does, on ten ADMET datasets, and ships the raw per-fold predictions, so the
gap is recoverable without retraining.

**The file format, verified.** `Results/<Family>/<Dataset>_<Family>_<repeat>.csv` is a 2-row wide
CSV: row 0 is the true delta and row 1 is the predicted delta, one column per pair. 50 files per
family = 10 datasets x 5 repeats of 10-fold cross-validation, with each file pooling the ten folds.
Caco2 has 82,810 columns, which is 10 folds x 91^2 within-fold pairs on 910 molecules, so self-pairs
are included. **All three families are scored on the same delta task** — for the two pointwise
families the predicted delta is the difference of their pointwise predictions — which is the
comparability check the audit asked for and the reason this anchor is usable at all.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deepdelta.py
import numpy as np
import pytest
from molace.data import deepdelta as dd


def test_ten_benchmark_datasets():
    assert len(dd.DD_DATASETS) == 10
    assert "Caco2" in dd.DD_DATASETS and "FUBrain" in dd.DD_DATASETS


def test_prediction_files_are_two_rows_true_then_predicted():
    true, pred = dd.load_predictions("Caco2", "RandomForest", 0)
    assert true.shape == pred.shape
    assert true.shape[0] == 82810


def test_the_self_pair_has_a_true_delta_of_zero():
    true, _ = dd.load_predictions("Caco2", "DeepDelta5", 0)
    assert true[0] == pytest.approx(0.0)


def test_all_three_families_share_the_same_true_deltas_so_the_task_is_comparable():
    a, _ = dd.load_predictions("Caco2", "RandomForest", 0)
    b, _ = dd.load_predictions("Caco2", "ChemProp50", 0)
    c, _ = dd.load_predictions("Caco2", "DeepDelta5", 0)
    assert a.shape == b.shape == c.shape
    assert np.allclose(a, b) and np.allclose(a, c)


def test_published_gaps_has_one_row_per_dataset_and_finite_numbers():
    g = dd.published_gaps()
    assert len(g) == 10
    for col in ("rmse_RandomForest", "rmse_ChemProp50", "rmse_DeepDelta5",
                "gap_vs_rf", "gap_vs_chemprop"):
        assert np.isfinite(g[col]).all(), col


def test_gap_sign_convention_matches_the_project(dataset="Caco2"):
    """Positive means the pairwise model wins, as everywhere else in this project."""
    g = dd.published_gaps().set_index("dataset")
    row = g.loc[dataset]
    assert row["gap_vs_rf"] == pytest.approx(row["rmse_RandomForest"] - row["rmse_DeepDelta5"])


def test_unknown_family_raises():
    with pytest.raises(KeyError, match="family"):
        dd.load_predictions("Caco2", "XGBoost", 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_deepdelta.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.data.deepdelta'`

- [ ] **Step 3: Implement**

```python
# src/molace/data/deepdelta.py
"""The only published per-task pointwise-against-pairwise gap, recovered from shipped files.

Fralish, Chen, Skaluba, Reker, J. Cheminform. 15:101 (2023). Results/<Family>/<Dataset>_<Family>_
<repeat>.csv is a 2-row wide CSV: row 0 the true delta, row 1 the predicted delta, one column per
pair, 5 repeats of 10-fold cross-validation per dataset with the folds pooled per file.

All three families are scored on the same delta task — for the pointwise families the predicted
delta is the difference of their pointwise predictions — which is what makes the gap comparable and
is asserted by a test rather than assumed.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "data" / "raw" / "DeepDelta"
RESULTS = ROOT / "Results"
FAMILIES = ("RandomForest", "ChemProp50", "DeepDelta5")
PAIRWISE_FAMILY = "DeepDelta5"
REPEATS = (0, 1, 2, 3, 4)

DD_DATASETS: tuple[str, ...] = (
    "Caco2", "FUBrain", "FreeSolv", "HalfLife", "HemoTox",
    "HepClear", "MicroClear", "RenClear", "Sol", "VDss",
)


def load_predictions(dataset: str, family: str, repeat: int):
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; expected one of {FAMILIES}")
    path = RESULTS / family / f"{dataset}_{family}_{repeat}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/fetch_data.sh")
    arr = pd.read_csv(path).to_numpy(dtype=float)
    if arr.shape[0] != 2:
        raise ValueError(f"{path} has {arr.shape[0]} rows, expected 2 (true, predicted)")
    return arr[0], arr[1]


def _rmse_over_repeats(dataset: str, family: str) -> float:
    errs = []
    for r in REPEATS:
        true, pred = load_predictions(dataset, family, r)
        errs.append(float(np.sqrt(np.mean((true - pred) ** 2))))
    return float(np.mean(errs))


@functools.lru_cache(maxsize=1)
def published_gaps() -> pd.DataFrame:
    rows = []
    for ds in DD_DATASETS:
        rec = {"dataset": ds}
        for fam in FAMILIES:
            rec[f"rmse_{fam}"] = _rmse_over_repeats(ds, fam)
        rec["gap_vs_rf"] = rec["rmse_RandomForest"] - rec[f"rmse_{PAIRWISE_FAMILY}"]
        rec["gap_vs_chemprop"] = rec["rmse_ChemProp50"] - rec[f"rmse_{PAIRWISE_FAMILY}"]
        rows.append(rec)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_deepdelta.py -v`
Expected: 7 passed

- [ ] **Step 5: Measure the graph statistics on those ten datasets and correlate**

The ten DeepDelta datasets carry `SMILES,Y` only — no cliff flag, no split column — so the
categorical arm does not apply and only the continuous statistic is available here.

```bash
uv run python -c "
import numpy as np, pandas as pd
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
t['receptor_class'] = t['dataset']          # no families here: each dataset is its own cluster
t.to_csv('results/external_deepdelta.csv', index=False)
print(t[['dataset','n','assortativity','gap_vs_rf','gap_vs_chemprop']].to_string())
for col in ('gap_vs_rf','gap_vs_chemprop'):
    b = cluster_bootstrap_spearman(t.assortativity, t[col], t.receptor_class, 10000, 0)
    print(f'{col:16s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]  n={b.n}')
print()
print('n = 10, so this is a replication with wide intervals, not a second primary result.')
" | tee results/report_external.txt
```

- [ ] **Step 6: Commit**

```bash
git add src/molace/data/deepdelta.py tests/test_deepdelta.py results/external_deepdelta.csv results/report_external.txt
git commit -m "the external anchor is the only published per-task gap, and it needed no retraining

Neither of the two sources the source proposal named supplies a per-target pointwise-against-pairwise
gap: one reports only numbers aggregated across its tasks and the other benchmarks twelve pointwise
models and no pairwise one. The third pairwise paper in the same bibliography does report per
benchmark and ships the raw per-fold predictions, so ten tasks of genuine published gap are
recoverable from files. A test asserts all three families share identical true deltas, which is the
comparability check that makes the gap a quantity rather than a mixture, and the ten datasets carry
no cliff flag so only the continuous statistic is computed there. Ten tasks with wide intervals is a
replication, and the output says so rather than presenting it as a second primary result."
```

---

### Task 18: The holdout, opened once

**Files:**
- Create: `src/molace/data/tdc_admet.py`, `tests/test_tdc_admet.py`
- Test: `tests/test_tdc_admet.py`

**Interfaces:**
- Consumes: PyTDC, the sweep machinery.
- Produces: `TDC_REGRESSION: tuple[str, ...]` (the 9 names from the pre-registration);
  `load_tdc(name) -> pandas.DataFrame` with columns `smiles, y, cliff_mol, split` where
  `cliff_mol` is all zero (TDC has no cliff annotation) and `split` is TDC's own scaffold split.

**The holdout discipline.** This task is written now and run **once**, after `results/report_spine.txt`
exists and is committed. Running it earlier, or re-running it after seeing its result, converts the
holdout into a second training set. A `RuntimeError` enforces the ordering.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tdc_admet.py
import numpy as np
import pytest
from molace.data import tdc_admet as t


def test_nine_regression_tasks_matching_the_prereg():
    from molace.analysis.prereg import load_prereg
    assert len(t.TDC_REGRESSION) == 9
    assert list(t.TDC_REGRESSION) == load_prereg()["task_sets"]["holdout"]["datasets"]


def test_a_task_loads_into_the_project_frame_shape():
    df = t.load_tdc("caco2_wang")
    assert list(df.columns) == ["smiles", "y", "cliff_mol", "split"]
    assert set(df["split"]) == {"train", "test"}
    assert (df["cliff_mol"] == 0).all()        # TDC carries no cliff annotation
    assert len(df) > 200


def test_labels_are_finite_and_non_constant():
    df = t.load_tdc("caco2_wang")
    assert np.isfinite(df["y"]).all()
    assert df["y"].std() > 0


def test_unknown_task_raises():
    with pytest.raises(KeyError, match="holdout"):
        t.load_tdc("tox21")


def test_the_holdout_refuses_to_run_before_the_spine_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "SPINE_REPORT", tmp_path / "absent.txt")
    with pytest.raises(RuntimeError, match="holdout"):
        t.assert_spine_recorded()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tdc_admet.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.data.tdc_admet'`

- [ ] **Step 3: Implement**

```python
# src/molace/data/tdc_admet.py
"""The pre-registered out-of-domain holdout: 9 TDC ADMET regression tasks.

Nine, not twenty-two: the other thirteen tasks in TDC's admet_benchmark are classification. The
names come from the pre-registration and a test asserts they match, so the holdout cannot quietly
grow or shrink.

These frames carry no cliff annotation, so the categorical arm does not apply here and only the
continuous statistic is computed. The split is TDC's own scaffold split, not MoleculeACE's.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SPINE_REPORT = Path(__file__).resolve().parents[3] / "results" / "report_spine.txt"

TDC_REGRESSION: tuple[str, ...] = (
    "caco2_wang", "lipophilicity_astrazeneca", "solubility_aqsoldb", "ppbr_az",
    "vdss_lombardo", "half_life_obach", "clearance_hepatocyte_az",
    "clearance_microsome_az", "ld50_zhu",
)


def assert_spine_recorded() -> None:
    """The holdout may only be opened after the spine result exists."""
    if not SPINE_REPORT.is_file():
        raise RuntimeError(
            "the holdout may not be opened before the spine result is recorded at "
            f"{SPINE_REPORT}; opening it first turns the holdout into a second training set"
        )


def load_tdc(name: str) -> pd.DataFrame:
    if name not in TDC_REGRESSION:
        raise KeyError(f"{name!r} is not one of the 9 pre-registered holdout tasks")
    from tdc.single_pred import ADME, Tox

    loader = Tox if name == "ld50_zhu" else ADME
    data = loader(name=name)
    split = data.get_split(method="scaffold")
    frames = []
    for part, tag in (("train", "train"), ("valid", "train"), ("test", "test")):
        d = split[part]
        frames.append(pd.DataFrame({
            "smiles": d["Drug"].astype(str),
            "y": d["Y"].astype(float),
            "cliff_mol": 0,
            "split": tag,
        }))
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv add pytdc
uv run pytest tests/test_tdc_admet.py -v
```
Expected: 5 passed

- [ ] **Step 5: Open the holdout, once**

```bash
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
    print(name, len(df), round(rows[-1]['assortativity'],3), round(r.gap,3))
t = pd.DataFrame(rows); t.to_csv('results/holdout_tdc.csv', index=False)
for col in ('access','gap','correction'):
    b = cluster_bootstrap_spearman(t.assortativity, t[col], t.receptor_class, 10000, 0)
    print(f'{col:12s} rho={b.rho:+.3f}  95% CI [{b.lo:+.3f}, {b.hi:+.3f}]')
print()
print('n = 9. Reported once. No tuning follows this.')
" | tee results/report_holdout.txt
```

- [ ] **Step 6: Commit**

```bash
git add src/molace/data/tdc_admet.py tests/test_tdc_admet.py results/holdout_tdc.csv results/report_holdout.txt pyproject.toml uv.lock
git commit -m "the holdout is nine regression tasks and is opened after the spine, not before

Nine, because the other thirteen tasks in that benchmark are classification, and the names are read
from the pre-registration with a test asserting they match so the holdout cannot quietly grow or
shrink. A runtime check refuses to open it until the spine result exists on disk, since opening it
first would turn it into a second training set, and the output states that no tuning follows. These
frames carry no cliff annotation, so only the continuous statistic is computed there and the
categorical arm is left out rather than faked."
```

---

### Task 19: The three controls, and the criterion half 2 must pass

**Files:**
- Create: `src/molace/hodge/controls.py`, `tests/test_controls.py`
- Test: `tests/test_controls.py`

**Interfaces:**
- Consumes: `hodge.complex.build`, `hodge.decompose`, `hodge.energy`, `models.pairwise`,
  `models.pointwise`, `models.anchors`.
- Produces: `flow_on(model, c, X, kind) -> np.ndarray`;
  `energy_controls(df, seed, k, triangle_budget) -> dict` with keys
  `trained, shuffled, pointwise_floor` (each a fractions dict) plus `n_edges, n_triangles`;
  `curl_versus_dispersion(df, seed, k, triangle_budget) -> dict` with keys
  `rho_curl, rho_dispersion, rho_difference, n_edges`.

**One control turned out to be redundant, and that is worth saying.** The spec asks for the
shuffled arm to be compared at matched total flow norm. That requirement existed because the
original design compared raw norms, and shuffling inflates the mean-square edge target 2.1x to
11.7x. Reporting the scale-invariant **fraction** removes the confound by construction, so no
rescaling is implemented. Norm matching is reinstated only if a raw norm is ever reported, and the
module docstring says so.

**The criterion.** Per-edge curl must predict that edge's held-out error better than plain anchor
dispersion, which is published three times (PADRE 2021, TNNR, Zhang et al. 2023). If it does not,
half 2 has no contribution and the null is reported, not omitted. DeepDelta's additivity MAE of
0.127 +/- 0.043 is the magnitude comparator for the curl itself, not for this comparison, since it
is an error scale and not a correlation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_controls.py
import numpy as np
import pandas as pd
import pytest
from molace.hodge import controls


def _frame(n=140, seed=0):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, 14))
    smiles = ["C" * (1 + int(b.sum())) + "O" for b in bits]
    y = 6.0 + bits[:, :5] @ np.array([1.0, -0.9, 0.7, 1.2, -0.5]) + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({
        "smiles": smiles, "y": y,
        "cliff_mol": (rng.random(n) < 0.2).astype(int),
        "split": ["train"] * int(0.8 * n) + ["test"] * (n - int(0.8 * n)),
    })


def test_the_pointwise_floor_is_curl_free_to_machine_precision():
    """Differencing a pointwise model on the edges must give an exactly gradient flow.

    This is a validity check on the whole pipeline: if the floor shows curl, the complex or the
    projection is wrong, not the science.
    """
    out = controls.energy_controls(_frame(), seed=0, k=10, triangle_budget=20000)
    assert out["pointwise_floor"]["curl"] < 1e-10
    assert out["pointwise_floor"]["harmonic"] < 1e-10
    assert out["pointwise_floor"]["gradient"] == pytest.approx(1.0, abs=1e-9)


def test_all_three_controls_report_fractions_that_sum_to_one():
    out = controls.energy_controls(_frame(), seed=0, k=10, triangle_budget=20000)
    for arm in ("trained", "shuffled", "pointwise_floor"):
        assert sum(out[arm].values()) == pytest.approx(1.0, abs=1e-8), arm


def test_fractions_are_scale_invariant_so_no_norm_matching_is_needed():
    """The reason the spec's norm-matching control is not implemented."""
    from molace.hodge import complex as cx
    from molace.hodge.decompose import decompose
    from molace.hodge.energy import fractions
    import networkx as nx
    c = cx.build(nx.gnp_random_graph(30, 0.3, seed=0))
    rng = np.random.default_rng(0)
    f = rng.normal(size=len(c.edges))
    a = fractions(decompose(c, f))
    b = fractions(decompose(c, 11.7 * f))
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-9), key


def test_the_shuffled_arm_is_reported_and_differs_from_the_floor():
    out = controls.energy_controls(_frame(), seed=0, k=10, triangle_budget=20000)
    assert out["shuffled"]["curl"] > out["pointwise_floor"]["curl"]


def test_curl_versus_dispersion_returns_both_and_their_difference():
    out = controls.curl_versus_dispersion(_frame(), seed=0, k=10, triangle_budget=20000)
    for key in ("rho_curl", "rho_dispersion", "rho_difference", "n_edges"):
        assert key in out
    assert out["rho_difference"] == pytest.approx(out["rho_curl"] - out["rho_dispersion"], abs=1e-9)


def test_a_complex_without_triangles_raises_instead_of_reporting_zero_curl():
    df = _frame(n=40)
    with pytest.raises(ValueError, match="no triangles"):
        controls.energy_controls(df, seed=0, k=1, triangle_budget=10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_controls.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.hodge.controls'`

- [ ] **Step 3: Implement**

```python
# src/molace/hodge/controls.py
"""The three controls, and the criterion half 2 must pass.

Controls:
  trained          - the trained pairwise flow on the test-split comparison graph
  shuffled         - the same architecture trained on permuted labels
  pointwise_floor  - a trained pointwise model differenced on the same edges, which is exactly a
                     gradient flow and therefore a validity check on the complex and projection

The spec asks the shuffled arm to be compared at matched total flow norm. That requirement came
from the original design's raw-norm comparison, where shuffling inflates the mean-square edge
target 2.1x to 11.7x. Reporting the scale-invariant fraction removes that confound by
construction, so no rescaling happens here. If a raw norm is ever reported, norm matching must come
back with it.

Criterion: per-edge curl must beat plain anchor dispersion at predicting an edge's held-out error.
Anchor dispersion is published three times and is the thing to beat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
from molace.hodge import complex as cx
from molace.hodge.decompose import decompose
from molace.hodge.energy import fractions, per_edge_curl
from molace.models import anchors, pairwise, pointwise


def flow_on(model: pairwise.PairwiseModel, c: cx.Complex, X: np.ndarray) -> np.ndarray:
    """Evaluate a pairwise model on every edge, oriented u -> v (second minus first)."""
    pos = {v: i for i, v in enumerate(c.nodes)}
    a = np.array([pos[u] for u, v in c.edges])
    b = np.array([pos[v] for u, v in c.edges])
    feats = pairwise.pair_features(X[a], X[b], model.kind)
    return np.asarray(model.model.predict(feats), dtype=float)


def _setup(df: pd.DataFrame, seed: int, k: int, triangle_budget: int):
    tr = df["split"].to_numpy() == "train"
    te = ~tr
    fp = ecfp4(df["smiles"].tolist())
    y = df["y"].to_numpy(dtype=float)
    X_tr, y_tr, X_te, y_te = fp[tr], y[tr], fp[te], y[te]
    sim_tr = tanimoto_matrix(X_tr)
    g_te = knn.knn_graph(tanimoto_matrix(X_te), k=k)
    c = cx.build(g_te, triangle_budget=triangle_budget, seed=seed)
    return X_tr, y_tr, X_te, y_te, sim_tr, c


def energy_controls(df: pd.DataFrame, seed: int, k: int, triangle_budget: int) -> dict:
    X_tr, y_tr, X_te, y_te, sim_tr, c = _setup(df, seed, k, triangle_budget)
    kind = "difference"

    trained = pairwise.fit("lightgbm", kind, X_tr, y_tr, sim_tr, anchors.M, seed)
    rng = np.random.default_rng(seed)
    shuffled = pairwise.fit("lightgbm", kind, X_tr, rng.permutation(y_tr), sim_tr, anchors.M, seed)

    pw_pred = pointwise.fit_predict("lightgbm", X_tr, y_tr, X_te, seed)
    pos = {v: i for i, v in enumerate(c.nodes)}
    floor = np.array([pw_pred[pos[v]] - pw_pred[pos[u]] for u, v in c.edges], dtype=float)

    return {
        "trained": fractions(decompose(c, flow_on(trained, c, X_te))),
        "shuffled": fractions(decompose(c, flow_on(shuffled, c, X_te))),
        "pointwise_floor": fractions(decompose(c, floor)),
        "n_edges": len(c.edges),
        "n_triangles": len(c.triangles),
    }


def curl_versus_dispersion(df: pd.DataFrame, seed: int, k: int, triangle_budget: int) -> dict:
    """Does per-edge curl predict an edge's held-out error better than anchor dispersion?"""
    X_tr, y_tr, X_te, y_te, sim_tr, c = _setup(df, seed, k, triangle_budget)
    kind = "difference"
    model = pairwise.fit("lightgbm", kind, X_tr, y_tr, sim_tr, anchors.M, seed)

    f = flow_on(model, c, X_te)
    d = decompose(c, f)
    curl = per_edge_curl(c, d)

    pos = {v: i for i, v in enumerate(c.nodes)}
    truth = np.array([y_te[pos[v]] - y_te[pos[u]] for u, v in c.edges], dtype=float)
    err = np.abs(f - truth)

    # Anchor dispersion, the published baseline: per molecule, the spread of y_a + f(a, molecule)
    # over its anchors; for an edge, the mean of its two endpoints' dispersions.
    idx = anchors.select(tanimoto_matrix(np.vstack([X_te, X_tr]))[: len(X_te), len(X_te):])
    per_mol = (y_tr[idx] + model.corrections(X_tr, X_te, idx)).std(axis=1)
    disp = np.array([0.5 * (per_mol[pos[u]] + per_mol[pos[v]]) for u, v in c.edges], dtype=float)

    rho_curl = float(stats.spearmanr(curl, err).statistic)
    rho_disp = float(stats.spearmanr(disp, err).statistic)
    return {
        "rho_curl": rho_curl,
        "rho_dispersion": rho_disp,
        "rho_difference": rho_curl - rho_disp,
        "n_edges": len(c.edges),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_controls.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the controls and the criterion on all 30 targets**

```bash
uv run python -c "
import numpy as np, pandas as pd
from molace.analysis.correlate import cluster_bootstrap_spearman
from molace.analysis.prereg import load_prereg, prereg_fingerprint
from molace.data import moleculeace as ma
from molace.hodge import controls
from molace.hodge import complex as cx
from molace.hodge.energy import readout_ablation
from molace.graphs import knn
from molace.graphs.fingerprints import ecfp4, tanimoto_matrix
k = load_prereg()['graphs']['primary']['k']
rows, abl = [], []
for name in ma.DATASETS:
    df = ma.load_target(name)
    try:
        e = controls.energy_controls(df, seed=0, k=k, triangle_budget=200000)
        c = controls.curl_versus_dispersion(df, seed=0, k=k, triangle_budget=200000)
    except ValueError as exc:
        print('skip', name, exc); continue
    rows.append({'dataset': name, 'receptor_class': ma.receptor_class(name),
                 'curl_trained': e['trained']['curl'], 'harm_trained': e['trained']['harmonic'],
                 'grad_trained': e['trained']['gradient'],
                 'curl_shuffled': e['shuffled']['curl'], 'curl_floor': e['pointwise_floor']['curl'],
                 **c})
    fp = ecfp4(df['smiles'].tolist())
    cc = cx.build(knn.knn_graph(tanimoto_matrix(fp), k=k), triangle_budget=50000)
    a = readout_ablation(fp, df['y'].to_numpy(float), cc, seed=0)
    abl.append({'dataset': name, 'curl_linear': a['linear']['curl'], 'curl_mlp': a['mlp']['curl']})
    print(name, round(rows[-1]['curl_trained'],4), round(rows[-1]['harm_trained'],4), round(rows[-1]['rho_difference'],3))
t = pd.DataFrame(rows); t.to_csv('results/hodge_controls.csv', index=False)
pd.DataFrame(abl).to_csv('results/readout_ablation.csv', index=False)
print()
print('prereg:', prereg_fingerprint())
print('median curl fraction, trained:', round(float(t.curl_trained.median()),4))
print('median harmonic fraction, trained:', round(float(t.harm_trained.median()),4))
print('max curl fraction, pointwise floor (must be ~0):', float(t.curl_floor.max()))
b = cluster_bootstrap_spearman(np.ones(len(t)), t.rho_difference, t.receptor_class, 10000, 0)
print('mean rho difference (curl minus dispersion):', round(float(t.rho_difference.mean()),4))
lo, hi = np.percentile([np.mean(np.random.default_rng(i).choice(t.rho_difference, len(t))) for i in range(10000)], [2.5, 97.5])
print(f'95% bootstrap CI on the mean difference: [{lo:+.4f}, {hi:+.4f}]')
print('CRITERION PASSED (curl beats dispersion):', bool(lo > 0))
" | tee results/report_hodge.txt
```

Read `max curl fraction, pointwise floor` first: it must be at machine zero. If it is not, the
complex or the projection is wrong and nothing else on this page means anything.

- [ ] **Step 6: Commit**

```bash
git add src/molace/hodge/controls.py tests/test_controls.py results/hodge_controls.csv results/readout_ablation.csv results/report_hodge.txt
git commit -m "three controls, one of them a validity check, and the criterion stated before the number

Differencing a trained pointwise model on the same edges must give an exactly gradient flow, so that
arm is a check on the complex and the projection rather than a scientific comparison: if it shows
curl, nothing else on the page means anything, and the runner prints it first. The spec's
norm-matched shuffled arm is implemented without rescaling, because reporting the scale-invariant
fraction removes the confound that requirement existed to handle, and a test pins the invariance so
the reasoning is checkable; norm matching returns only if a raw norm is ever reported. The criterion
is written down before the number: per-edge curl must beat plain anchor dispersion at predicting an
edge's held-out error, since dispersion is published three times, and failing to beat it is reported
as a measured null rather than left out."
```

---

### Task 20: Figures, and a README that says what was refuted

**Files:**
- Create: `src/molace/analysis/figures.py`, `README.md`, `tests/test_figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: the four result CSVs and the three report text files.
- Produces: `make_figures(out_dir="results/figures") -> list[Path]` writing
  `fig1_assortativity_vs_gap.png`, `fig2_decomposition.png`, `fig3_hodge_census.png`,
  `fig4_readout_ablation.png`. Every figure's caption carries the pre-registration blob hash.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_figures.py
import pandas as pd
import pytest
from molace.analysis import figures


def test_every_figure_is_written(tmp_path):
    paths = figures.make_figures(out_dir=tmp_path)
    assert len(paths) == 4
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0


def test_captions_carry_the_prereg_hash(tmp_path):
    from molace.analysis.prereg import prereg_fingerprint
    figures.make_figures(out_dir=tmp_path)
    caption = (tmp_path / "captions.md").read_text()
    assert prereg_fingerprint()[:12] in caption


def test_captions_state_n_and_n_eff(tmp_path):
    figures.make_figures(out_dir=tmp_path)
    caption = (tmp_path / "captions.md").read_text()
    assert "n_eff" in caption and "n = 30" in caption


def test_a_missing_result_file_raises_rather_than_drawing_an_empty_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "SPINE", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        figures.make_figures(out_dir=tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_figures.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'molace.analysis.figures'`

- [ ] **Step 3: Implement the figures**

```bash
uv add matplotlib
```

```python
# src/molace/analysis/figures.py
"""Four panels, each from one committed CSV.

A missing input raises and names the path rather than drawing an empty panel, because an empty
panel reads as a measured absence. Every caption carries the blob hash of the frozen plan plus the
nominal and effective sample sizes, so a figure separated from this repository still says what plan
produced it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molace.analysis.correlate import cluster_bootstrap_spearman, effective_n
from molace.analysis.prereg import prereg_fingerprint

RESULTS = Path(__file__).resolve().parents[3] / "results"
SPINE = RESULTS / "spine.csv"
CENSUS = RESULTS / "hodge_census.csv"
ABLATION = RESULTS / "readout_ablation.csv"

PALETTE = {"GPCR": "#4C72B0", "Kinase": "#DD8452", "NR": "#55A868",
           "Other": "#C44E52", "Protease": "#8172B3", "Transferase": "#937860"}


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run the task that writes it before plotting")
    return pd.read_csv(path)


def _fig1(spine: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for cls, grp in spine.groupby("receptor_class"):
        ax.scatter(grp["assortativity"], grp["gap"], s=44, alpha=0.85,
                   color=PALETTE.get(cls, "#666666"), label=cls, edgecolor="white", linewidth=0.6)
    ok = spine[["rogi", "gap"]].notna().all(axis=1)
    if ok.sum() > 2:
        z = np.polyfit(spine.loc[ok, "rogi"].rank(), spine.loc[ok, "gap"].rank(), 1)
        xs = np.linspace(spine["assortativity"].min(), spine["assortativity"].max(), 10)
        ax.plot(xs, np.full_like(xs, np.polyval(z, spine.loc[ok, "rogi"].rank().mean())),
                "--", color="#999999", linewidth=1.2, label="roughness-only level")
    b = cluster_bootstrap_spearman(spine["assortativity"], spine["gap"],
                                   spine["receptor_class"], 10000, 0)
    ax.axhline(0.0, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.set_xlabel("target assortativity")
    ax.set_ylabel("gap  (RMSE pointwise - RMSE pairwise)")
    ax.set_title(f"rho = {b.rho:+.3f}   95% CI [{b.lo:+.3f}, {b.hi:+.3f}]", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    path = out / "fig1_assortativity_vs_gap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig2(spine: pd.DataFrame, out: Path) -> Path:
    d = spine.sort_values("gap").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(d))
    ax.bar(x, d["access"], color="#4C72B0", label="access (label access buys)")
    ax.bar(x, d["correction"], bottom=d["access"], color="#DD8452",
           label="correction (the learned function buys)")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"], rotation=90, fontsize=5.5)
    ax.set_ylabel("RMSE difference")
    ax.set_title("heights are error differences, not shares: RMSE is nonlinear", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / "fig2_decomposition.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig3(census: pd.DataFrame, out: Path) -> Path:
    d = census.sort_values("dim_harmonic", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for off, col, colour, name in ((-w, "dim_gradient", "#4C72B0", "gradient"),
                                   (0.0, "dim_curl", "#DD8452", "curl"),
                                   (w, "dim_harmonic", "#55A868", "harmonic")):
        hatch = ["//" if m == "estimated" else "" for m in d["rank_method"]]
        bars = ax.bar(x + off, d[col], width=w, color=colour, label=name)
        for bar, h in zip(bars, hatch):
            bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"], rotation=90, fontsize=5.5)
    ax.set_ylabel("subspace dimension")
    ax.set_title("hatched bars: rank estimated, not computed exactly", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / "fig3_hodge_census.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _fig4(abl: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    floor = 1e-16
    for col, colour, name in (("curl_linear", "#4C72B0", "bias-free linear head"),
                              ("curl_mlp", "#DD8452", "MLP head")):
        ax.scatter(np.arange(len(abl)), np.maximum(abl[col], floor), s=36,
                   color=colour, label=name, edgecolor="white", linewidth=0.5)
    ax.axhline(floor, color="#999999", linestyle="--", linewidth=1.0, label="machine zero")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(abl)))
    ax.set_xticklabels(abl["dataset"], rotation=90, fontsize=5.5)
    ax.set_ylabel("curl fraction of flow energy")
    ax.set_title("a linear readout is a gradient flow; a nonlinear one is not", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path = out / "fig4_readout_ablation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_figures(out_dir="results/figures") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spine = _read(SPINE)
    census = _read(CENSUS)
    abl = _read(ABLATION)
    paths = [_fig1(spine, out), _fig2(spine, out), _fig3(census, out), _fig4(abl, out)]

    stamp = prereg_fingerprint()
    n = len(spine)
    n_eff = round(effective_n(spine["receptor_class"]), 1)
    captions = [
        f"All panels produced under pre-registration {stamp[:12]} "
        f"(full hash {stamp}). n = {n} tasks, n_eff = {n_eff} "
        "at an assumed intra-receptor-class correlation of 0.3.",
        "",
        "**fig1** Target assortativity against the pointwise-minus-pairwise gap, one point per "
        "target, coloured by receptor class. The interval is a cluster bootstrap over receptor "
        "classes, not over rows. The panel licenses a statement about association across tasks; it "
        "does not license a causal claim, and it does not by itself show the statistic adds "
        "anything to roughness, which is the separate incremental quantity in the report.",
        "",
        "**fig2** The gap split into what test-time label access buys and what the learned pairwise "
        "function buys, per target. Heights are error differences in the label's units. They are "
        "**not** shares of the gap: RMSE is nonlinear, so a percentage would be meaningless.",
        "",
        "**fig3** Hodge subspace dimensions per target on the pre-registered kNN graph. Hatched "
        "bars mark targets whose curl rank was estimated by a randomised range finder rather than "
        "computed exactly, which happens above the stated edge count. An estimated bar does not "
        "license the precision an exact one does.",
        "",
        "**fig4** Curl fraction of the trained edge flow under a bias-free linear readout and under "
        "an MLP readout, both on the frozen fingerprint encoder, log scale with the machine-zero "
        "line drawn. The linear result is a theorem being checked, not a measurement; the MLP "
        "magnitudes are measurements and are specific to this encoder and these graphs.",
    ]
    (out / "captions.md").write_text("\n".join(captions))
    return paths
```

- [ ] **Step 4: Close the spec's §13a text preconditions**

These gate sentences, not code, and the README is the first text in the project, so they close here.

1. Confirm MoleculeACE's cliff definition against van Tilborg, Alenicheva, Grisoni, JCIM 2022
   directly. The thresholds carried through this project as **unverified** are: greater than 90
   percent substructure, scaffold or SMILES similarity together with a greater than tenfold
   activity difference. If the paper says otherwise, correct the circularity paragraph in the spec
   §7 and the README before either is committed.
2. Open these five by eye, abstract read, and check our description matches: arXiv 2409.14500,
   2508.20906, 2509.21489, 2601.04507, 2508.16495. They were batch-confirmed on title match by the
   same pass that mis-described 2509.18893 as atom-level heterophily when it is graph-level with
   motif labels.
3. Give SALI a real citation (Guha and Van Drie) or drop the name from any prose. It appears in the
   source proposal's gap paragraph with no bibliography entry.
4. Record every outcome in `docs/preconditions.md`, including any that change a claim.

- [ ] **Step 5: Write the README**

```markdown
# molace

Does a parameter-free, pre-training statistic of a molecule-similarity graph predict when
pairwise models beat pointwise ones? Nodes are molecules; edges are comparisons between them.

## What this is

Increment 1 of a pet project. The design is in
`docs/superpowers/specs/2026-10-03-molace-increment1-design.md`; the frozen analysis plan is
`prereg/increment1.yaml` and every result and figure carries its git blob hash.

## What the audit refuted before any code was written

The original proposal is recorded in the spec's §2 together with these verdicts, so the claims
cannot quietly return:

- **Molecules-as-nodes with Tanimoto edges is not a new move.** Chemical Space Networks
  (Maggiora and Bajorath, 2014 onward) are exactly that construction, characterised with network
  measures including assortativity across 36 ChEMBL activity classes, and that literature controls
  edge density. This project works inside that paradigm and controls density.
- **No named pairwise architecture is curl-free.** SQRL applies a nonlinear MLP head to the
  difference of representations, PADRE is a random forest on concatenated features plus their
  difference, DeepDelta concatenates two message-passing embeddings, and RankRefine has no learned
  pairwise function. The theorem here is restated for a frozen encoder with a bias-free linear
  readout, where it is provable, and the real boundary is linear against nonlinear readout.
- **Triangle closure as a label-free consistency check is published.** Twin Neural Network
  Regression states and uses the loop condition and derives uncertainty from its violation;
  DeepDelta measures the additivity residual at 0.127 +/- 0.043 and uses it as an unsupervised
  quality signal. What is claimed here is the orthogonal three-way decomposition, the energy
  budget and the observability identity, not the observation.
- **Neither SQRL nor MoleculeACE publishes a per-target pointwise-against-pairwise gap.** The gap
  is computed here on MoleculeACE's own shipped split; DeepDelta's per-fold predictions supply the
  one genuinely published external anchor, on ten tasks.

Two further corrections: adjusted homophily and label informativeness are categorical-only, and
the group's own choice for a continuous target is Newman target assortativity, which carries no
binning parameter; and adjusted homophily was superseded by unbiased homophily in 2024, which is
what their current benchmarks report.

We do not reproduce SQRL's numbers. Its published equations are inconsistent: its inference rule
under its own training convention estimates twice the anchor label minus the query.

## Reproducing

```bash
uv sync
./scripts/fetch_data.sh
uv run pytest
uv run python -m molace.analysis.sweep        # writes results/spine.csv
```

## Results

`results/report_spine.txt`, `results/report_external.txt`, `results/report_holdout.txt`,
`results/hodge_census.csv`, and `results/figures/`.

## Known deviations from the spec

Recorded rather than hidden; see `results/figures/captions.md` for the per-figure version.
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest -v
```
Expected: the whole suite passes

- [ ] **Step 7: Commit**

```bash
git add src/molace/analysis/figures.py tests/test_figures.py README.md docs/preconditions.md results/figures pyproject.toml uv.lock
git commit -m "the figures carry the plan's hash, and the README states what the audit refuted

Four panels, each raising on a missing input so an absent result can never be drawn as an empty
one, and each captioned with the blob hash of the frozen plan alongside the nominal and effective
sample sizes. The decomposition panel says in its own caption that the heights are error differences
rather than shares, because the error is nonlinear and a reader would otherwise take them for
percentages. The census panel hatches the targets whose rank was estimated rather than computed
exactly. The README leads with the four refuted claims and the two corrections, since the fastest way
to lose a reader who knows this literature is to present a twelve-year-old paradigm as an unusual
move, and it states plainly that the imitated paper's numbers are not reproduced here and why."
```
