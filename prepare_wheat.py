import pandas as pd
import pyarrow.dataset as ds
import numpy as np

def prep_wheat():
    print("🧹 Extracting Wheat data from 75M row master...")
    dataset = ds.dataset("mandi_master_data.parquet", format="parquet")
    
    # Load only necessary columns for the last 2 years of Wheat
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
    table = dataset.to_table(
        filter=(ds.field("Commodity").isin(["WHEAT", "Wheat", "Wheat(Common)"])) & 
               (ds.field("Arrival_Date") >= cutoff),
        columns=["Arrival_Date", "Market", "Modal_Price"]
    )
    df = table.to_pandas()
    
    # Standardize and Pivot
    df['Market'] = df['Market'].astype(str).str.upper().str.strip()
    daily = df.groupby(["Arrival_Date", "Market"])["Modal_Price"].mean().reset_index()
    pivot = daily.pivot(index="Arrival_Date", columns="Market", values="Modal_Price")
    
    # Load your specific market index to ensure columns match the GCN
    with open("mandi_adjacency_index_wheat.txt", "r") as f:
        market_names = [line.strip() for line in f]
    
    pivot = pivot.reindex(columns=market_names)
    pivot = pivot.ffill().bfill().fillna(1500.0)
    
    # Save as a lightweight NumPy file for instant training
    np.save("wheat_training_matrix.npy", pivot.values.astype(np.float32))
    print(f"✅ Success! Matrix saved: {pivot.shape} (Days x Markets)")

if __name__ == "__main__":
    prep_wheat()