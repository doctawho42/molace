"""Curate independent regression targets from ChEMBL, to settle the ROGI question.

Pre-registration: prereg/increment3_chembl.yaml, blob 7519ef31c4374637483bae167221f9b7567ece7f,
frozen before any activity data was downloaded.

increment2_rogi could not resolve whether target assortativity adds anything over the roughness
index, because 30 of its 40 datasets were the set the claim was found on and the other 10 had no
power. These targets share no ChEMBL id with MoleculeACE, so they are independent of the discovery
set by construction.

One pass over the API collects every pchembl-valued Ki and EC50 measurement on human single
proteins with an exact relation, which is enough to rank targets by count AND to build the datasets,
with no second download. The pull is cached so a rerun costs nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
OUT = Path("data/raw/chembl_targets")
PAGE = 1000
TYPES = ("Ki", "EC50")
MIN_COMPOUNDS = 200
N_TARGETS = 40
CHECKPOINT_PAGES = 25      # flush every 25,000 rows
MAX_RETRIES = 8            # backing off to two minutes, for outages longer than a few seconds


def pull(standard_type: str) -> pd.DataFrame:
    """Page through one activity type, checkpointing so a dropped connection costs minutes.

    The first version of this saved only at the end and lost 303,000 rows to a DNS failure at
    offset 303,000. Pages are now flushed to a partial file every CHECKPOINT_PAGES, and a rerun
    resumes from the offset already on disk. Retries back off to two minutes, because the failure
    that happened was a name-resolution outage lasting longer than a twelve-second wait.
    """
    cache = OUT / f"raw_{standard_type}.csv"
    if cache.is_file():
        print(f"  {standard_type}: cached", flush=True)
        return pd.read_csv(cache)

    partial = OUT / f"partial_{standard_type}.csv"
    rows, offset = [], 0
    if partial.is_file():
        done = pd.read_csv(partial)
        rows = done.to_dict("records")
        offset = (len(rows) // PAGE) * PAGE
        print(f"  {standard_type}: resuming from offset {offset:,} "
              f"({len(rows):,} rows on disk)", flush=True)
        rows = rows[:offset]

    s = requests.Session()
    s.headers["User-Agent"] = "molace-research/1.0 (academic; graph homophily study)"
    t0, since_flush = time.time(), 0
    while True:
        params = {
            "standard_type": standard_type, "pchembl_value__isnull": "false",
            "target_organism": "Homo sapiens", "standard_relation": "=",
            "limit": PAGE, "offset": offset,
            "only": "target_chembl_id,canonical_smiles,pchembl_value",
        }
        data = None
        for attempt in range(MAX_RETRIES):
            try:
                r = s.get(BASE, params=params, timeout=120)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    pd.DataFrame(rows).to_csv(partial, index=False)
                    print(f"    giving up at offset {offset}; {len(rows):,} rows kept in "
                          f"{partial.name}, rerun to resume", flush=True)
                    raise
                wait = min(120, 5 * 2 ** attempt)
                print(f"    retry {attempt+1}/{MAX_RETRIES} at offset {offset} after {wait}s: "
                      f"{type(exc).__name__}", flush=True)
                time.sleep(wait)
        acts = data["activities"]
        if not acts:
            break
        rows.extend(acts)
        offset += PAGE
        since_flush += 1
        if since_flush >= CHECKPOINT_PAGES:
            pd.DataFrame(rows).to_csv(partial, index=False)
            since_flush = 0
            print(f"    {standard_type}: {offset:,} of {data['page_meta']['total_count']:,} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if data["page_meta"]["next"] is None:
            break

    df = pd.DataFrame(rows)
    df["standard_type"] = standard_type
    df.to_csv(cache, index=False)
    partial.unlink(missing_ok=True)
    print(f"  {standard_type}: {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("pulling activities (cached after the first run)", flush=True)
    df = pd.concat([pull(t) for t in TYPES], ignore_index=True)
    df = df.dropna(subset=["canonical_smiles", "pchembl_value", "target_chembl_id"])
    df["pchembl_value"] = pd.to_numeric(df.pchembl_value, errors="coerce")
    df = df.dropna(subset=["pchembl_value"])
    print(f"total usable rows: {len(df):,}", flush=True)

    counts = df.groupby(["target_chembl_id", "standard_type"]).size().rename("n").reset_index()
    dominant = counts.sort_values("n", ascending=False).drop_duplicates("target_chembl_id")

    from molace.data import moleculeace as ma
    excluded = {n.rsplit("_", 1)[0] for n in ma.DATASETS}
    print(f"excluding {len(excluded)} ChEMBL ids that appear in MoleculeACE", flush=True)
    cand = dominant[~dominant.target_chembl_id.isin(excluded)].sort_values("n", ascending=False)

    selected, skipped = [], []
    for _, row in cand.iterrows():
        if len(selected) >= N_TARGETS:
            break
        tid, st = row.target_chembl_id, row.standard_type
        sub = df[(df.target_chembl_id == tid) & (df.standard_type == st)]
        agg = sub.groupby("canonical_smiles").pchembl_value.median().reset_index()
        if len(agg) < MIN_COMPOUNDS:
            skipped.append([tid, st, int(len(agg))])
            continue
        agg.columns = ["smiles", "y"]
        name = f"{tid}_{st}"
        agg.to_csv(OUT / f"{name}.csv", index=False)
        selected.append({"dataset": name, "target_chembl_id": tid, "standard_type": st,
                         "n_activities": int(row.n), "n_compounds": int(len(agg))})
        print(f"  {name:22s} {int(row.n):6d} activities -> {len(agg):5d} compounds", flush=True)

    meta = pd.DataFrame(selected)
    meta.to_csv(OUT / "selected.csv", index=False)
    (OUT / "skipped.json").write_text(json.dumps(skipped, indent=1))
    print(flush=True)
    print(f"selected {len(meta)} targets; {len(skipped)} dropped for fewer than "
          f"{MIN_COMPOUNDS} unique compounds, recorded in skipped.json")
    print(f"compounds per target: min {meta.n_compounds.min()}, median "
          f"{int(meta.n_compounds.median())}, max {meta.n_compounds.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
