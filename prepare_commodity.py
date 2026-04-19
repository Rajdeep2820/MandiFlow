"""
prepare_commodity.py  —  MandiFlow v2.0
=======================================
Builds the training matrix for a single commodity from the master Parquet.

What changed from v1:
  - Filters mandis to ≥60% temporal coverage (no more all-1500 matrices)
  - All mandis still included in graph; sparse ones get spatial imputation
  - Saves BOTH ratio matrix AND anchor prices (raw Rs) for correct denorm
  - Uses 2010-2024 for model training; full history for graph inference
  - Writes a clean updated index file matching the actual active mandis
  - Regime-aware: flags the 4 structural break years so loss can be
    down-weighted on those samples during training

Usage:
    python prepare_commodity.py --commodity ONION
    python prepare_commodity.py --commodity TOMATO
    python prepare_commodity.py --commodity POTATO
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TIER_1 = ["ONION", "TOMATO", "POTATO"]
TIER_2 = ["WHEAT", "GARLIC", "MAIZE", "PADDY"]

# Structural break years — samples from these years get a flag so train.py
# can optionally down-weight them (markets behaved atypically)
REGIME_BREAK_YEARS = {2016, 2017, 2020, 2021}

# Coverage threshold: a mandi must have reported prices on at least this
# fraction of all trading days in the training window to be an "anchor" node
MIN_ANCHOR_COVERAGE = 0.60

# A mandi with less coverage than this is completely excluded even from the
# graph (truly dead / renamed / data artifact)
MIN_GRAPH_COVERAGE = 0.05

# Training window (pre-training on 2010-2020 foundation decade + fine-tune years)
PRETRAIN_START = "2010-01-01"
PRETRAIN_END   = "2020-12-31"
FINETUNE_START = "2021-01-01"
FINETUNE_END   = "2024-12-31"

# Ratio clamp — daily price can't realistically move more than 3× or less than 0.3×
RATIO_CLIP_LOW  = 0.30
RATIO_CLIP_HIGH = 3.00

# Fallback fill limit — only fill short gaps (≤7 days) via interpolation
MAX_GAP_FILL_DAYS = 7


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_commodity_data(commodity: str, start: str, end: str) -> pd.DataFrame:
    """
    Reads Arrival_Date, Market, Modal_Price for one commodity from Parquet.
    Handles all common capitalisation variants (ONION / Onion / onion).
    """
    commodity_variants = [commodity, commodity.title(), commodity.lower()]

    dataset = ds.dataset("mandi_master_data.parquet", format="parquet")

    table = dataset.to_table(
        filter=(
            ds.field("Commodity").isin(commodity_variants) &
            (ds.field("Arrival_Date") >= pd.Timestamp(start)) &
            (ds.field("Arrival_Date") <= pd.Timestamp(end))
        ),
        columns=["Arrival_Date", "Market", "Modal_Price"],
    )

    df = table.to_pandas()
    if df.empty:
        return df

    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")
    df = df.dropna(subset=["Modal_Price"])
    df = df[df["Modal_Price"] > 0]
    return df


def build_daily_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates to one price per (date, market) then pivots.
    Multiple variety entries on the same day are averaged.
    """
    daily = (
        df.groupby(["Arrival_Date", "Market"])["Modal_Price"]
        .mean()
        .reset_index()
    )
    pivot = daily.pivot(index="Arrival_Date", columns="Market", values="Modal_Price")
    pivot = pivot.sort_index()
    return pivot


def compute_coverage(pivot: pd.DataFrame) -> pd.Series:
    """Returns fraction of non-NaN days per mandi column."""
    return pivot.notna().mean(axis=0)


def spatial_impute(pivot: pd.DataFrame, anchor_cols: list, all_cols: list) -> pd.DataFrame:
    """
    For non-anchor mandis (sparse), fill missing values using the median of
    anchor mandis on the same day. This is a simple 0th-order spatial prior —
    better than a global constant like 1500.

    anchor_cols: mandis with ≥ MIN_ANCHOR_COVERAGE (used as donors)
    all_cols:    all mandis to keep in the matrix (including sparse ones)

    Both lists are intersected with pivot.columns so this works correctly
    across windows — the finetune pivot may not contain every mandi that
    was active in the pretrain window.
    """
    pivot_cols = set(pivot.columns)

    # Only use anchor donors that actually exist in this window's pivot
    available_anchors = [c for c in anchor_cols if c in pivot_cols]

    # Only keep graph mandis that exist in this window's pivot
    available_all = [c for c in all_cols if c in pivot_cols]

    # Mandis in graph_mandis that are missing from this window entirely
    # get a column of NaN added — spatial median will fill them
    missing_cols = [c for c in all_cols if c not in pivot_cols]

    if not available_anchors:
        # Extreme edge case: no anchor mandis in this window at all
        # Fall back to the global median of whatever we have
        print(f"   ⚠️  No anchor mandis found in this window. Using global median.")
        anchor_daily_median = pivot.median(axis=1)
    else:
        anchor_daily_median = pivot[available_anchors].median(axis=1)

    if len(missing_cols) > 0:
        print(f"   ℹ️  {len(missing_cols)} mandis absent from this window "
              f"— filled with spatial median.")

    # Build result with all graph_mandis columns in the correct order
    result = pivot[available_all].copy()

    # Add missing mandi columns filled entirely with spatial median
    for col in missing_cols:
        result[col] = anchor_daily_median

    # Reorder to match the original all_cols order (important for matrix consistency)
    result = result[all_cols]

    # Fill gaps per column
    anchor_set = set(available_anchors)
    for col in all_cols:
        if col in anchor_set:
            result[col] = (
                result[col]
                .interpolate(method="time", limit=MAX_GAP_FILL_DAYS)
                .fillna(anchor_daily_median)
            )
        else:
            result[col] = result[col].fillna(anchor_daily_median)

    return result


def build_regime_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Returns a boolean array (T,) — True where the date falls in a known
    structural break year. train.py uses this to optionally down-weight loss.
    """
    return np.array([d.year in REGIME_BREAK_YEARS for d in index], dtype=bool)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def prep_data(commodity: str, mode: str = "both"):
    """
    mode = 'pretrain'  → builds matrix for 2010-2020
    mode = 'finetune'  → builds matrix for 2021-2024
    mode = 'both'      → builds both and saves separately (recommended)
    """
    commodity = commodity.upper()

    if commodity not in TIER_1 + TIER_2:
        print(f"⚠️  {commodity} is not in a recognised tier. Proceeding anyway.")

    index_file  = f"mandi_adjacency_index_{commodity.lower()}.txt"
    if not os.path.exists(index_file):
        print(f"❌ {index_file} not found. Run infer_supply_routes.py first.")
        return

    print(f"\n{'='*60}")
    print(f"  MandiFlow v2.0 — Preparing: {commodity}")
    print(f"{'='*60}")

    windows = []
    if mode in ("pretrain", "both"):
        windows.append(("pretrain", PRETRAIN_START, PRETRAIN_END))
    if mode in ("finetune", "both"):
        windows.append(("finetune", FINETUNE_START, FINETUNE_END))

    # We compute coverage using the full pretrain window so the same mandi
    # set is used for both matrices (consistency for the graph)
    print(f"\n📖 Loading data for coverage analysis ({PRETRAIN_START} → {PRETRAIN_END})...")
    coverage_df = load_commodity_data(commodity, PRETRAIN_START, PRETRAIN_END)

    if coverage_df.empty:
        print(f"❌ No data found for {commodity}. Check Parquet commodity name.")
        return

    print(f"   Raw records: {len(coverage_df):,}")

    coverage_pivot = build_daily_pivot(coverage_df)
    coverage = compute_coverage(coverage_pivot)

    anchor_mandis = coverage[coverage >= MIN_ANCHOR_COVERAGE].index.tolist()
    graph_mandis  = coverage[coverage >= MIN_GRAPH_COVERAGE].index.tolist()
    dead_mandis   = coverage[coverage <  MIN_GRAPH_COVERAGE].index.tolist()

    print(f"\n📊 Mandi Coverage Report:")
    print(f"   Total mandis found in data:        {len(coverage):>5}")
    print(f"   Anchor mandis (≥{MIN_ANCHOR_COVERAGE*100:.0f}% coverage):    {len(anchor_mandis):>5}  ← loss computed here")
    print(f"   Graph mandis  (≥{MIN_GRAPH_COVERAGE*100:.0f}% coverage):     {len(graph_mandis):>5}  ← all in graph")
    print(f"   Dead mandis   (<{MIN_GRAPH_COVERAGE*100:.0f}% coverage):      {len(dead_mandis):>5}  ← excluded entirely")

    # Write updated index (graph mandis — superset of anchor mandis)
    with open(index_file, "w") as f:
        for m in graph_mandis:
            f.write(f"{m}\n")
    print(f"\n✅ Updated {index_file} → {len(graph_mandis)} nodes")

    # Save anchor mask so train.py knows which nodes to include in loss
    anchor_mask = np.array([m in set(anchor_mandis) for m in graph_mandis], dtype=bool)
    anchor_mask_path = f"{commodity.lower()}_anchor_mask.npy"
    np.save(anchor_mask_path, anchor_mask)
    print(f"✅ Anchor mask saved → {anchor_mask_path}")
    print(f"   ({anchor_mask.sum()} anchor nodes / {len(anchor_mask)} total nodes)")

    # Build matrices for each window
    for window_name, start, end in windows:
        print(f"\n{'─'*50}")
        print(f"  Building {window_name} matrix  ({start} → {end})")
        print(f"{'─'*50}")

        df = load_commodity_data(commodity, start, end)
        if df.empty:
            print(f"   ⚠️  No data in this window — skipping.")
            continue

        pivot = build_daily_pivot(df)

        # Report how many graph mandis are present in this specific window
        present_in_window = [m for m in graph_mandis if m in pivot.columns]
        missing_in_window = len(graph_mandis) - len(present_in_window)
        print(f"   Mandis present in this window: {len(present_in_window)} / {len(graph_mandis)}"
              + (f"  ({missing_in_window} absent → spatial fill)" if missing_in_window else ""))

        # Spatially impute: anchor mandis fill sparse ones.
        # spatial_impute handles the case where finetune pivot has fewer
        # mandis than the pretrain-derived anchor/graph lists.
        print(f"   Imputing sparse mandis using spatial median...")
        pivot_full = spatial_impute(pivot, anchor_mandis, graph_mandis)

        T = len(pivot_full)
        N = len(graph_mandis)
        print(f"   Matrix shape before ratio: ({T}, {N})")

        raw_prices = pivot_full.values.astype(np.float32)  # (T, N)

        # --- Ratio normalization: Price_t / Price_{t-1} ---
        # Shape becomes (T-1, N)
        denom = raw_prices[:-1].copy()
        denom = np.where(denom < 1.0, 1.0, denom)          # guard zero/tiny denominators
        ratio_matrix = raw_prices[1:] / denom
        ratio_matrix = np.clip(ratio_matrix, RATIO_CLIP_LOW, RATIO_CLIP_HIGH)

        # Anchor prices = raw price at t (i.e. Price_{t-1} for the t+1 prediction)
        anchor_prices = raw_prices[:-1]                     # (T-1, N)

        # Regime flags aligned to ratio matrix (same T-1 length)
        regime_flags = build_regime_flags(pivot_full.index[1:])

        # --- Save ---
        matrix_path  = f"{commodity.lower()}_{window_name}_matrix.npy"
        anchor_path  = f"{commodity.lower()}_{window_name}_anchors.npy"
        regime_path  = f"{commodity.lower()}_{window_name}_regime_flags.npy"
        dates_path   = f"{commodity.lower()}_{window_name}_dates.npy"

        np.save(matrix_path, ratio_matrix)
        np.save(anchor_path, anchor_prices)
        np.save(regime_path, regime_flags)

        # Save the actual date index aligned to the ratio matrix (starts at day 1,
        # not day 0, because ratio[t] = price[t]/price[t-1])
        # data_loader.py uses this for accurate shock event labeling
        dates_array = pivot_full.index[1:].to_numpy()   # (T-1,) — matches ratio_matrix
        np.save(dates_path, dates_array)

        # Verify
        nan_check = np.isnan(ratio_matrix).sum()

        print(f"\n   ✅ {window_name} matrix saved:")
        print(f"      Ratio matrix:   {matrix_path}  shape={ratio_matrix.shape}")
        print(f"      Anchor prices:  {anchor_path}   shape={anchor_prices.shape}")
        print(f"      Regime flags:   {regime_path}   shape={regime_flags.shape}")
        print(f"      Dates index:    {dates_path}   "
              f"({dates_array[0]} → {dates_array[-1]})")
        print(f"      Ratio range:    [{ratio_matrix.min():.4f}, {ratio_matrix.max():.4f}]")
        print(f"      Ratio mean:     {ratio_matrix.mean():.6f}  (should be ~1.0)")
        print(f"      NaN count:      {nan_check}  (should be 0)")
        print(f"      Regime days:    {regime_flags.sum()} / {len(regime_flags)}")

    print(f"\n{'='*60}")
    print(f"  ✅ {commodity} preparation complete.")
    print(f"{'='*60}\n")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MandiFlow v2.0 — Commodity Data Preparation")
    parser.add_argument(
        "--commodity", type=str, required=True,
        help="Commodity name: ONION, TOMATO, POTATO, WHEAT, GARLIC, MAIZE"
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["pretrain", "finetune", "both"],
        help="Which time window to build (default: both)"
    )
    args = parser.parse_args()
    prep_data(args.commodity, mode=args.mode)