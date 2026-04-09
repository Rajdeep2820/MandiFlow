import pandas as pd
import pyarrow.dataset as ds
import numpy as np
import argparse
import os

def prep_data(commodity):
    commodity = commodity.upper()
    output_file = f"{commodity.lower()}_training_matrix.npy"
    index_file = f"mandi_adjacency_index_{commodity.lower()}.txt"
    
    if not os.path.exists(index_file):
        print(f"❌ Error: {index_file} not found. Run generate_graphs.py first!")
        return

    print(f"🧹 Extracting {commodity} data (Last 2 years)...")
    dataset = ds.dataset("mandi_master_data.parquet", format="parquet")
    
    # Filter for the specific commodity
    comm_filter = [commodity, commodity.title(), commodity.lower()]
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
    
    try:
        table = dataset.to_table(
            filter=(ds.field("Commodity").isin(comm_filter)) & 
                   (ds.field("Arrival_Date") >= cutoff),
            columns=["Arrival_Date", "Market", "Modal_Price"]
        )
        df = table.to_pandas()
        
        if df.empty:
            print(f"❌ No records found for {commodity} in the last 2 years.")
            return

        # Pivot logic
        df['Market'] = df['Market'].astype(str).str.upper().str.strip()
        daily = df.groupby(["Arrival_Date", "Market"])["Modal_Price"].mean().reset_index()
        pivot = daily.pivot(index="Arrival_Date", columns="Market", values="Modal_Price")
        
        # Match with graph index
        with open(index_file, "r") as f:
            market_names = [line.strip() for line in f]
        
        pivot = pivot.reindex(columns=market_names)
        pivot = pivot.ffill().bfill().fillna(1500.0)
        
        np.save(output_file, pivot.values.astype(np.float32))
        print(f"✅ Success! Matrix saved as {output_file} | Shape: {pivot.shape}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", type=str, required=True)
    args = parser.parse_args()
    prep_data(args.commodity)