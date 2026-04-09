import pandas as pd
import numpy as np
import scipy.sparse as sparse
import os

def create_commodity_resources_v2(search_term):
    print(f"📦 Searching Parquet for: {search_term}")
    
    # 1. Load data
    df = pd.read_parquet("mandi_master_data.parquet", columns=["Market", "Commodity"])
    
    # 2. Case-insensitive substring match (Finds "Wheat", "Wheat(Common)", etc.)
    mask = df["Commodity"].str.contains(search_term, case=False, na=False)
    matches = df[mask]
    
    if matches.empty:
        print(f"❌ Zero records found for '{search_term}'. Available commodities include: {df['Commodity'].unique()[:10]}")
        return

    # Use the official name found in the first row for the filename
    official_name = matches["Commodity"].iloc[0].upper().replace(" ", "_")
    commodity_markets = sorted(matches["Market"].unique())
    
    # 3. Save Index
    idx_path = f"mandi_adjacency_index_{search_term.lower()}.txt"
    with open(idx_path, "w") as f:
        for market in commodity_markets:
            f.write(f"{market}\n")
            
    # 4. Save Identity Matrix
    num_nodes = len(commodity_markets)
    adj_matrix = sparse.eye(num_nodes).tocsr()
    adj_path = f"mandi_adjacency_{search_term.lower()}.npz"
    sparse.save_npz(adj_path, adj_matrix)
    
    print(f"✅ SUCCESS for {search_term}!")
    print(f"   - Official name in data: {official_name}")
    print(f"   - Mandis found: {len(commodity_markets)}")
    print(f"   - Files created: {idx_path}, {adj_path}")

# Run specifically for Wheat
create_commodity_resources_v2("Wheat")