import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import scipy.stats as stats
import scipy.sparse as sparse
import os
import argparse

def infer_routes(commodity="ONION"):
    print(f"🚀 Starting graph inference for: {commodity}")
    
    # Specify columns to save memory
    columns = ["Arrival_Date", "Market", "Commodity", "Modal_Price"]
    print("Loading parquet data...")
    table = pq.read_table("mandi_master_data.parquet", columns=columns)
    df = table.to_pandas()
    
    print(f"Filtering strictly for {commodity}...")
    df = df[df["Commodity"].str.upper() == commodity.upper()]
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    
    # Drop rows with missing prices or dates
    df = df.dropna(subset=["Modal_Price", "Arrival_Date"])
    
    print("Aggregating daily modal prices by Market...")
    df_daily = df.groupby(["Arrival_Date", "Market"])["Modal_Price"].mean().reset_index()
    
    print("Creating time-series pivot table...")
    pivot = df_daily.pivot(index="Arrival_Date", columns="Market", values="Modal_Price")
    
    # Keep only markets with enough data (e.g., at least 500 days of records)
    print("Filtering active markets...")
    min_days = 500
    active_markets = pivot.count()[pivot.count() > min_days].index
    pivot = pivot[active_markets]
    
    # Fill missing values with interpolation/forward fill
    pivot = pivot.interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    market_names = pivot.columns.tolist()
    
    # Save index for model reference
    index_file = f"mandi_adjacency_index_{commodity.lower()}.txt"
    with open(index_file, "w") as f:
        for name in market_names:
            f.write(f"{name}\n")
    
    # Correlation Math
    print(f"Calculating lagged correlations for {len(market_names)} markets...")
    data = pivot.values
    num_markets = data.shape[1]
    
    # We'll use a sparse matrix to save memory
    rows, cols, weights = [], [], []
    
    # Optimization: Calculate correlation matrix using numpy/multiplication for speed 
    # then threshold it. Pearson correlation is more scalable than loop-based Granger for 2k nodes.
    # We shift one series by 1 day to find "supply followers"
    current_prices = data[1:] # t
    lagged_prices = data[:-1] # t-1
    
    # Normalize for correlation
    curr_norm = (current_prices - current_prices.mean(axis=0)) / (current_prices.std(axis=0) + 1e-6)
    lag_norm = (lagged_prices - lagged_prices.mean(axis=0)) / (lagged_prices.std(axis=0) + 1e-6)
    
    # Correlation Matrix = (Lagged^T * Current) / T
    print("Computing matrix multiplication...")
    corr_matrix = np.dot(lag_norm.T, curr_norm) / len(curr_norm)
    
    threshold = 0.5 # Minimum correlation to consider it a "supply route"
    for i in range(num_markets):
        for j in range(num_markets):
            if i == j: continue
            val = corr_matrix[i, j]
            if val > threshold:
                rows.append(i)
                cols.append(j)
                weights.append(val)
                
    print(f"Found {len(weights)} edges above threshold.")
    
    print("Saving sparse adjacency matrix...")
    adj = sparse.csr_matrix((weights, (rows, cols)), shape=(num_markets, num_markets))
    adj_file = f"mandi_adjacency_{commodity.lower()}.npz"
    sparse.save_npz(adj_file, adj)
    
    print(f"✅ Data-driven graph inference complete for {commodity}.")

if __name__ == "__main__":
    infer_routes()
