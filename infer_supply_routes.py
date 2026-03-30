"""
infer_supply_routes.py  —  MandiFlow v2.0
==========================================
Builds a data-driven supply route graph for a commodity using lagged
price correlation across the FULL historical record (25 years).

Key improvements over v1:
  - Uses full history (best available start → 2024) for correlation — more
    data = more reliable supply route detection
  - Threshold raised 0.5 → 0.65 to reduce over-connectivity
  - Per-node degree cap of 50 edges — prevents hub nodes from dominating
    GCN message passing
  - Symmetrises the graph (if A→B, also B→A) with averaged weights
  - Saves a separate corr_matrix.npy for diagnostics
  - Supports all Tier 1 + Tier 2 commodities

Usage:
    python infer_supply_routes.py --commodity ONION
    python infer_supply_routes.py --commodity TOMATO
    python infer_supply_routes.py --commodity POTATO
    python infer_supply_routes.py --commodity WHEAT
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sparse


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Minimum correlation to create an edge
CORR_THRESHOLD = 0.65           # was 0.5 in v1

# Max edges per node — prevents hub over-dominance in GCN
MAX_DEGREE = 50

# Minimum trading days for a mandi to be included in graph inference
MIN_DAYS_FOR_GRAPH = 365        # 1 full year minimum across 25yr window

# Lag in days for correlation (1 = look for "A today → B tomorrow" pattern)
LAG_DAYS = 1

# For the correlation computation, use this many days of rolling window
# to compute normalisation (prevents very old low-variance periods dominating)
ROLLING_NORM_WINDOW = 90        # days


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_full_history(commodity: str) -> pd.DataFrame:
    """
    Loads the complete available price history for a commodity.
    Uses PyArrow column pushdown for memory efficiency on 75M row Parquet.
    """
    commodity_variants = [commodity, commodity.title(), commodity.lower()]

    print(f"📖 Loading full history for {commodity} (all available years)...")

    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset("mandi_master_data.parquet", format="parquet")
        table = dataset.to_table(
            filter=ds.field("Commodity").isin(commodity_variants),
            columns=["Arrival_Date", "Market", "Modal_Price"],
        )
        df = table.to_pandas()
    except Exception as e:
        print(f"❌ Failed to load Parquet: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")
    df = df.dropna(subset=["Modal_Price"])
    df = df[df["Modal_Price"] > 0]

    date_range = f"{df['Arrival_Date'].min().date()} → {df['Arrival_Date'].max().date()}"
    print(f"   Records: {len(df):,} | Date range: {date_range}")
    return df


# ---------------------------------------------------------------------------
# PIVOT + FILTER
# ---------------------------------------------------------------------------

def build_pivot(df: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """
    Builds a daily price pivot and filters to mandis with enough history.
    """
    print("📊 Aggregating daily prices...")
    daily = (
        df.groupby(["Arrival_Date", "Market"])["Modal_Price"]
        .mean()
        .reset_index()
    )
    pivot = daily.pivot(index="Arrival_Date", columns="Market", values="Modal_Price")
    pivot = pivot.sort_index()

    # Filter to mandis with minimum trading days
    counts = pivot.notna().sum(axis=0)
    active = counts[counts >= min_days].index.tolist()
    pivot = pivot[active]

    print(f"   Active mandis (≥{min_days} days): {len(active)}")
    print(f"   Date range in pivot: {pivot.index[0].date()} → {pivot.index[-1].date()}")

    # Interpolate short gaps, forward/backward fill edges
    pivot = (
        pivot
        .interpolate(method="time", limit=14)
        .ffill(limit=7)
        .bfill(limit=7)
    )

    # Drop any column still mostly NaN after fill (shouldn't happen with min_days filter)
    pivot = pivot.dropna(axis=1, thresh=int(0.50 * len(pivot)))

    return pivot


# ---------------------------------------------------------------------------
# LAGGED CORRELATION
# ---------------------------------------------------------------------------

def compute_lagged_correlation(pivot: pd.DataFrame, lag: int = 1) -> np.ndarray:
    """
    Computes the lagged cross-correlation matrix.

    corr_matrix[i, j] = correlation(price_i at t, price_j at t+lag)

    This captures the "i leads j" supply chain relationship — e.g.
    Lasalgaon (wholesale origin) leads Azadpur (retail destination) by 1 day.

    Returns:
        corr_matrix: (N, N) float32 array
    """
    data = pivot.values.astype(np.float32)          # (T, N)
    T, N = data.shape

    print(f"🔢 Computing lagged correlation matrix ({N}×{N}) with lag={lag}...")
    print(f"   This may take a minute for large N...")

    # Split into lagged and current
    lagged_data   = data[:-lag]     # t   — "origin" prices
    current_data  = data[lag:]      # t+1 — "destination" prices

    # Z-score normalise each column (zero mean, unit variance)
    # Using nanmean/nanstd to handle any residual NaNs
    lag_mean   = np.nanmean(lagged_data, axis=0)
    lag_std    = np.nanstd(lagged_data,  axis=0) + 1e-8
    curr_mean  = np.nanmean(current_data, axis=0)
    curr_std   = np.nanstd(current_data,  axis=0) + 1e-8

    lag_norm   = (lagged_data  - lag_mean)  / lag_std
    curr_norm  = (current_data - curr_mean) / curr_std

    # Replace any NaN (from constant columns) with 0
    lag_norm  = np.nan_to_num(lag_norm,  nan=0.0)
    curr_norm = np.nan_to_num(curr_norm, nan=0.0)

    # Pearson correlation via matrix multiply: (N, T-lag) × (T-lag, N) → (N, N)
    T_eff = lag_norm.shape[0]
    corr_matrix = (lag_norm.T @ curr_norm) / T_eff   # (N, N)

    # Clip to [-1, 1] (floating point can drift slightly outside)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    print(f"   Correlation stats: min={corr_matrix.min():.3f}, "
          f"max={corr_matrix.max():.3f}, mean={corr_matrix.mean():.3f}")

    above_thresh = (corr_matrix > CORR_THRESHOLD).sum()
    print(f"   Edges above threshold ({CORR_THRESHOLD}): {above_thresh:,} "
          f"({100*above_thresh/(N*N):.1f}% density)")

    return corr_matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# BUILD SPARSE ADJACENCY WITH DEGREE CAP
# ---------------------------------------------------------------------------

def build_adjacency(
    corr_matrix: np.ndarray,
    market_names: list,
    threshold: float = CORR_THRESHOLD,
    max_degree: int = MAX_DEGREE,
) -> sparse.csr_matrix:
    """
    Converts the correlation matrix to a sparse adjacency matrix.

    Steps:
      1. Threshold: keep only corr > threshold
      2. Degree cap: each node keeps its top-K strongest edges
      3. Symmetrise: if A→B exists, add B→A with averaged weight
      4. Remove self-loops
    """
    N = len(market_names)

    print(f"🕸️  Building adjacency (threshold={threshold}, max_degree={max_degree})...")

    # Step 1 + 2: Per-node top-K filtering
    rows, cols, weights = [], [], []
    node_edges = defaultdict(list)

    for i in range(N):
        # Get all edges from node i above threshold (excluding self)
        targets = [
            (corr_matrix[i, j], j)
            for j in range(N)
            if i != j and corr_matrix[i, j] > threshold
        ]
        # Keep only top max_degree by weight
        targets.sort(reverse=True)
        for w, j in targets[:max_degree]:
            node_edges[i].append((i, j, w))

    # Step 3: Symmetrise — add reverse edge with averaged weight
    edge_dict = {}                          # (i,j) → weight
    for i, edge_list in node_edges.items():
        for (src, dst, w) in edge_list:
            key_fwd = (src, dst)
            key_rev = (dst, src)
            if key_fwd not in edge_dict:
                edge_dict[key_fwd] = w
            if key_rev not in edge_dict:
                edge_dict[key_rev] = w
            else:
                # Average with existing reverse weight
                edge_dict[key_rev] = (edge_dict[key_rev] + w) / 2.0

    for (i, j), w in edge_dict.items():
        rows.append(i)
        cols.append(j)
        weights.append(float(w))

    # Step 4: Build sparse matrix
    if not rows:
        print("⚠️  No edges found above threshold — returning identity matrix.")
        return sparse.eye(N, format="csr", dtype=np.float32)

    adj = sparse.csr_matrix(
        (weights, (rows, cols)),
        shape=(N, N),
        dtype=np.float32,
    )

    # Stats
    degrees = np.array(adj.sum(axis=1)).flatten()
    print(f"   Edges (symmetric): {adj.nnz:,}")
    print(f"   Graph density:     {adj.nnz / (N*N) * 100:.2f}%  (was 65% in v1)")
    print(f"   Degree — min: {degrees.min():.1f}, max: {degrees.max():.1f}, "
          f"mean: {degrees.mean():.1f}")
    isolated = (degrees == 0).sum()
    if isolated > 0:
        print(f"   ⚠️  Isolated nodes (no edges): {isolated} — will use self-loop fallback")

    return adj


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def infer_routes(commodity: str):
    commodity = commodity.upper()

    print(f"\n{'='*60}")
    print(f"  MandiFlow v2.0 — Graph Inference: {commodity}")
    print(f"{'='*60}\n")

    # 1. Load full history (25 years)
    df = load_full_history(commodity)
    if df.empty:
        print(f"❌ No data for {commodity}. Exiting.")
        return

    # 2. Build pivot + filter active mandis
    pivot = build_pivot(df, min_days=MIN_DAYS_FOR_GRAPH)
    if pivot.empty or pivot.shape[1] < 2:
        print("❌ Not enough mandis after filtering. Exiting.")
        return

    market_names = pivot.columns.tolist()
    N = len(market_names)

    # 3. Save the index file
    idx_path = f"mandi_adjacency_index_{commodity.lower()}.txt"
    with open(idx_path, "w") as f:
        for name in market_names:
            f.write(f"{name}\n")
    print(f"\n✅ Index saved: {idx_path}  ({N} mandis)")

    # 4. Compute lagged correlation
    corr_matrix = compute_lagged_correlation(pivot, lag=LAG_DAYS)

    # Save raw correlation matrix for diagnostics / future use
    corr_path = f"{commodity.lower()}_corr_matrix.npy"
    np.save(corr_path, corr_matrix)
    print(f"✅ Correlation matrix saved: {corr_path}  shape={corr_matrix.shape}")

    # 5. Build sparse adjacency with degree cap
    adj = build_adjacency(corr_matrix, market_names)

    # 6. Save adjacency
    adj_path = f"mandi_adjacency_{commodity.lower()}.npz"
    sparse.save_npz(adj_path, adj)
    print(f"✅ Adjacency saved: {adj_path}")

    # 7. Quick sanity: top 10 best-connected mandis
    degrees = np.array(adj.sum(axis=1)).flatten()
    top10_idx = np.argsort(degrees)[::-1][:10]
    print(f"\n🔝 Top 10 best-connected mandis (supply hubs):")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"   {rank:>2}. {market_names[idx]:<40} degree={degrees[idx]:.1f}")

    print(f"\n{'='*60}")
    print(f"  ✅ Graph inference complete for {commodity}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MandiFlow v2.0 — Supply Route Graph Inference"
    )
    parser.add_argument(
        "--commodity", type=str, required=True,
        help="Commodity: ONION, TOMATO, POTATO, WHEAT, GARLIC, MAIZE"
    )
    args = parser.parse_args()
    infer_routes(args.commodity)