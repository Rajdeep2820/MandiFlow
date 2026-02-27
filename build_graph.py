import pandas as pd
import numpy as np
import scipy.sparse as sparse
import os

# --- PATHS ---
MASTER_FILE = "mandi_master_data.parquet"
ADJ_MATRIX_FILE = "mandi_adjacency.npz"

def build_adjacency_matrix():
    if not os.path.exists(MASTER_FILE):
        print("❌ Master file not found! Run preprocess.py first.")
        return

    print("📖 Reading Master Data (IDs only)...")
    # Loading only necessary columns to save memory
    df = pd.read_parquet(MASTER_FILE, columns=['Market_ID', 'District_ID'])
    
    # Get unique market-district pairings
    mandi_mapping = df.drop_duplicates(subset=['Market_ID'])
    num_markets = mandi_mapping['Market_ID'].max() + 1
    
    print(f"🕸️ Constructing network for {num_markets} unique Mandis...")

    rows = []
    cols = []
    
    # Group markets by their District
    # Every market in the same district gets a bidirectional link
    districts = mandi_mapping.groupby('District_ID')['Market_ID'].apply(list)

    for market_list in districts:
        for i in market_list:
            for j in market_list:
                rows.append(i)
                cols.append(j)

    # Math: Create Sparse CSR Matrix
    # We use float32 to keep the matrix lightweight
    data = np.ones(len(rows), dtype=np.float32)
    adj_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_markets, num_markets))

    # Save as compressed NumPy format
    sparse.save_npz(ADJ_MATRIX_FILE, adj_matrix)
    
    print(f"✅ SUCCESS: Adjacency Matrix saved to {ADJ_MATRIX_FILE}")
    print(f"Total Spatial Connections (Edges) created: {adj_matrix.nnz}")

if __name__ == "__main__":
    build_adjacency_matrix()