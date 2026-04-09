import pandas as pd
import pyarrow.dataset as ds
import os

parquet_path = '/Users/rajdeepsinghpanwar/Downloads/MandiFlow/mandi_master_data.parquet'
output_path = '/Users/rajdeepsinghpanwar/Downloads/MandiFlow/all_mandis.txt'

def extract_mandis():
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return

    print(f"Reading {parquet_path}...")
    dataset = ds.dataset(parquet_path, format="parquet")
    
    unique_markets = set()
    
    # Read in batches to be memory efficient
    for batch in dataset.to_batches(columns=['Market']):
        markets = batch.to_pandas()['Market'].dropna().unique()
        unique_markets.update(markets)
    
    sorted_markets = sorted(list(unique_markets))
    
    with open(output_path, 'w') as f:
        for market in sorted_markets:
            f.write(f"{market}\n")
            
    print(f"Extracted {len(sorted_markets)} unique mandis.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    extract_mandis()
