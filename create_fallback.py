import pandas as pd
import os

def create_mini_backup():
    source = "mandi_master_data.parquet"
    target = "mini_fallback.csv"
    
    if os.path.exists(source):
        print(f"⚡ Reading latest records from {source}...")
        # We only load the columns the dashboard actually needs
        df = pd.read_parquet(source, columns=['Commodity', 'Market', 'District', 'Modal_Price'])
        
        # Take the last 5,000 records (most recent)
        mini_df = df.tail(5000)
        
        # Standardize columns to lowercase so they match the API format
        mini_df.columns = [c.lower() for c in mini_df.columns]
        
        mini_df.to_csv(target, index=False)
        print(f"✅ {target} created successfully! (Size: {len(mini_df)} rows)")
    else:
        print(f"❌ Error: {source} not found. Please ensure you are in the MandiFlow folder.")

if __name__ == "__main__":
    create_mini_backup()