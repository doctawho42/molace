# Text preconditions (spec §13a), and what checking them changed

These gated sentences, not code. Recorded here with their outcomes, including the ones that changed a
claim.

## 1. Unbiased homophily formula — CLOSED, no claim changed

Equation (3) of arXiv 2412.09663, transcribed into `src/molace/measures/unbiased.py`:

    h_unb(C) = sum_{i<j} ( sqrt(c_ii c_jj) - c_ij ) / sum_{i<j} ( sqrt(c_ii c_jj) + c_ij )

Recommended form, `alpha = 0`, so no free parameter in our hands. All four defining properties pass as
tests. One thing the paper leaves implicit and the formula depends on: `C` is normalised by counting
both orientations of every edge and dividing by `2|E|`, which is the convention under which the paper's
own stated values come out right.

## 2. MoleculeACE cliff definition — CLOSED, and it CORRECTS the spec

Read from the code that generated the shipped `cliff_mol`, which is better evidence than the paper's
prose: `data/raw/MoleculeACE/MoleculeACE/benchmark/cliffs.py`.

    find_cliffs(self, similarity: float = 0.9, potency_fold: float = 10, ...)
    moleculeace_similarity(): m_tani | m_scaff | m_leve, each >= similarity
    get_tanimoto_matrix(smiles, radius: int = 2, nBits: int = 1024)
    fc = (get_fc(self.bioactivity) > potency_fold)

So a molecule is flagged a cliff when it has a partner at **>= 0.9 similarity in at least one of three
channels** -- Morgan r=2/1024 Tanimoto, generic-Murcko-scaffold Morgan Tanimoto, SMILES Levenshtein --
**and** a **> 10-fold** potency difference. The proposal's description of the thresholds was right.

**The spec was wrong about the relation.** Spec §7 said our graph is built on "a different relation"
from the cliff definition. It is not: the first of the three channels is Morgan r=2 Tanimoto, the same
relation our graph uses, differing only in bit width (1024 against our 2048, a folding difference that
changes little). The circularity is real and must be stated as real.

**What limits it, measured rather than asserted.** The thresholds differ sharply. Cliff edges need
Tanimoto >= 0.9; our kNN graph at k = 10 has a median edge similarity of 0.30 to 0.79, and the share
of our edges reaching 0.9 is:

| target | edges | median edge Tanimoto | share >= 0.9 |
|---|---|---|---|
| CHEMBL4203_Ki | 5,209 | 0.301 | 0.02% |
| CHEMBL4792_Ki | 9,831 | 0.667 | 0.06% |
| CHEMBL234_Ki | 25,032 | 0.655 | 1.51% |
| CHEMBL287_Ki | 9,197 | 0.569 | 2.22% |
| CHEMBL2835_Ki | 4,335 | 0.785 | 2.75% |

So the overlap is real in kind and small in extent: between 0.02% and 2.75% of the edges we measure
homophily on could be cliff-defining edges. That is the honest version of the paragraph, and it is a
number rather than a reassurance.

## 3. Five bibliography entries — CLOSED, all five check out

Verified against the arXiv API (`export.arxiv.org/api/query?id_list=...`), which returns the
authoritative title and abstract, rather than against a search-result title.

| id | real title | our description |
|---|---|---|
| 2409.14500 | GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data | matches |
| 2508.20906 | Turning Tabular Foundation Models into Graph Foundation Models | matches (the proposal's "G2T-FM") |
| 2509.21489 | GraphPFN: A Prior-Data Fitted Graph Foundation Model | matches |
| 2601.04507 | A Semi-supervised Molecular Learning Framework for Activity Cliff Estimation | matches ("semi-supervised learning for activity cliffs") |
| 2508.16495 | Post Hoc Regression Refinement via Pairwise Rankings | matches RankRefine; the abstract's "model-agnostic, plug-and-play post hoc method" is consistent with it having no learned pairwise function |

So the batch confirmation that misdescribed 2509.18893 did not mislead on these five. **2509.18893
itself remains misdescribed in the source proposal**: it is graph-level heterophily with motif-based
labels, not atom-level heterophily inside molecular graphs, and any text using it must say so.

## 4. SALI citation — CLOSED

Verified through Crossref: Guha, R.; Van Drie, J. H. "Structure-Activity Landscape Index: Identifying
and Quantifying Activity Cliffs." *Journal of Chemical Information and Modeling* **2008**, 48 (3),
646-658. DOI `10.1021/ci7004093` (published 2008-02-28). The name can now appear in prose with a real
reference instead of being dropped.
