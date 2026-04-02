"""
build_graph.py  —  MandiFlow v2.0
====================================
Builds a geographic (district-based) adjacency matrix.

Role in the pipeline:
  This is the GEOGRAPHIC FALLBACK graph builder. It connects mandis
  that share the same district — a fast, data-free baseline.

  For production models, use infer_supply_routes.py instead, which
  builds a data-driven correlation graph from actual price history.

  Use build_graph.py when:
    - You have a new commodity with insufficient price history for
      correlation inference (< 365 days of data)
    - You want a quick sanity-check graph for a new region
    - infer_supply_routes.py hasn't been run yet for a commodity

Key fix from v1:
  v1 indexed the matrix by Market_ID integers directly. Market_IDs
  are non-contiguous (gaps from deleted/merged mandis) and can be
  very large (e.g. Market_ID=6400 in a 1000-node graph), creating
  a 6400×6400 sparse matrix where 5400 rows/cols are empty.

  v2 re-indexes to a dense 0-based scheme matching the order in
  the commodity index file (mandi_adjacency_index_<commodity>.txt),
  so the matrix dimensions exactly match what model.py expects.

Usage:
    # Build geographic graph for a commodity that has an index file
    python build_graph.py --commodity ONION

    # Build a generic all-commodity graph (no index file required)
    python build_graph.py --all
"""

import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sparse


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MASTER_FILE = "mandi_master_data.parquet"


# ---------------------------------------------------------------------------
# CORE BUILDER
# ---------------------------------------------------------------------------

def build_geographic_graph(
    commodity: str = None,
    output_path: str = None,
) -> sparse.csr_matrix:
    """
    Builds a symmetric binary adjacency matrix connecting mandis
    that share the same district.

    If `commodity` is given, the matrix is filtered to mandis in the
    commodity's index file and ordered to match it exactly.

    If `commodity` is None, builds a generic matrix over all mandis.

    Returns the sparse adjacency matrix (also saved to disk).
    """
    if not os.path.exists(MASTER_FILE):
        print(f"❌ {MASTER_FILE} not found.")
        return None

    print(f"📖 Loading district mappings from Parquet...")
    df = pd.read_parquet(
        MASTER_FILE,
        columns=["Market", "District"],
    )
    df["Market"]   = df["Market"].astype(str).str.strip()
    df["District"] = df["District"].astype(str).str.strip()

    # Optional: filter to commodity-specific mandis
    if commodity:
        idx_path = f"mandi_adjacency_index_{commodity.lower()}.txt"
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                index_markets = [line.strip() for line in f if line.strip()]
            index_set = set(m.upper() for m in index_markets)
            df = df[df["Market"].str.upper().isin(index_set)]
            print(f"   Filtered to {len(index_markets)} markets from {idx_path}")
        else:
            print(f"   ⚠️  No index file for {commodity}. Using all mandis.")
            index_markets = None
    else:
        index_markets = None

    # Deduplicate to one row per market
    mandi_mapping = (
        df[["Market", "District"]]
        .drop_duplicates(subset=["Market"])
        .copy()
    )

    # Build dense 0-based index
    if index_markets:
        # Use the exact order from the index file so matrix[i] = index_markets[i]
        market_upper_to_orig = {
            m.upper(): m for m in mandi_mapping["Market"].tolist()
        }
        ordered_markets = []
        for m in index_markets:
            # Find original-case version in our data
            orig = market_upper_to_orig.get(m.upper(), m)
            ordered_markets.append(orig)
        mandi_mapping = (
            pd.DataFrame({"Market": ordered_markets})
            .merge(mandi_mapping, on="Market", how="left")
        )
        mandi_mapping["District"] = mandi_mapping["District"].fillna("Unknown")
    else:
        mandi_mapping = mandi_mapping.reset_index(drop=True)

    market_to_idx = {
        row["Market"]: i
        for i, row in mandi_mapping.iterrows()
    }
    N = len(mandi_mapping)
    print(f"   Building {N}×{N} geographic graph...")

    # Group mandis by district, connect all pairs within same district
    rows, cols = [], []
    districts = mandi_mapping.groupby("District")["Market"].apply(list)

    edge_count = 0
    for district, market_list in districts.items():
        if len(market_list) < 2:
            continue   # isolated mandi — no geographic neighbors
        for m_i in market_list:
            for m_j in market_list:
                if m_i == m_j:
                    continue
                i = market_to_idx.get(m_i)
                j = market_to_idx.get(m_j)
                if i is not None and j is not None:
                    rows.append(i)
                    cols.append(j)
                    edge_count += 1

    if not rows:
        print("⚠️  No edges found — all mandis are in unique districts.")
        adj = sparse.eye(N, format="csr", dtype=np.float32)
    else:
        data = np.ones(len(rows), dtype=np.float32)
        adj  = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    # Save
    if output_path is None:
        if commodity:
            output_path = f"mandi_adjacency_{commodity.lower()}_geographic.npz"
        else:
            output_path = "mandi_adjacency_geographic.npz"

    sparse.save_npz(output_path, adj)

    # Stats
    degrees = np.array(adj.sum(axis=1)).flatten()
    isolated = (degrees == 0).sum()

    print(f"\n✅ Geographic graph saved: {output_path}")
    print(f"   Nodes:           {N}")
    print(f"   Edges:           {adj.nnz}")
    print(f"   Graph density:   {adj.nnz / (N*N) * 100:.2f}%")
    print(f"   Degree — min: {degrees.min():.0f}, "
          f"max: {degrees.max():.0f}, "
          f"mean: {degrees.mean():.1f}")
    print(f"   Isolated nodes:  {isolated}"
          + (" ⚠️" if isolated > 0 else " ✅"))

    return adj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MandiFlow v2.0 — Geographic Graph Builder"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--commodity", type=str,
        help="Build graph for a specific commodity index (e.g. ONION)"
    )
    group.add_argument(
        "--all", action="store_true",
        help="Build a generic graph over all mandis in the Parquet"
    )

    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output path for the .npz file"
    )

    args = parser.parse_args()

    if args.all:
        build_geographic_graph(commodity=None, output_path=args.output)
    else:
        build_geographic_graph(
            commodity=args.commodity.upper(),
            output_path=args.output,
        )