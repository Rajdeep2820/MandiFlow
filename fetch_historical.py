import os
import time
import requests
import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

start_date = datetime.date(2026, 2, 11)
end_date = datetime.date(2026, 3, 19)

delta = datetime.timedelta(days=1)
curr_date = start_date

print(f"🚀 Starting Historical Backfill: {start_date} to {end_date}")

all_records = []

while curr_date <= end_date:
    date_str = curr_date.strftime("%d/%m/%Y")
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 10000,
        "filters[Arrival_Date]": date_str
    }
    
    try:
        resp = requests.get(URL, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            records = resp.json().get('records', [])
            all_records.extend(records)
            print(f"✅ Extracted {len(records):<5} rows for {date_str}")
        else:
            print(f"❌ Failed {date_str} - Code: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Error {date_str}: {e}")
        
    curr_date += delta
    time.sleep(0.3)

if not all_records:
    print("No data fetched.")
    exit()

print(f"\n🔄 Download compiled. Total {len(all_records)} new rows.")
new_df = pd.DataFrame(all_records)

# 1. Match Schema Formatting
rename_map = {
    'state': 'State', 'district': 'District', 'market': 'Market', 
    'commodity': 'Commodity', 'variety': 'Variety', 'grade': 'Grade',
    'min_price': 'Min_Price', 'max_price': 'Max_Price', 'modal_price': 'Modal_Price',
    'arrival_date': 'Arrival_Date'
}
new_df = new_df.rename(columns=rename_map)

# 2. Math & Feature Engineering (Mirroring preprocess.py)
for float_col in ['Min_Price', 'Max_Price', 'Modal_Price']:
    new_df[float_col] = pd.to_numeric(new_df[float_col], errors='coerce')

new_df = new_df.dropna(subset=['Modal_Price'])

new_df['Arrival_Date'] = pd.to_datetime(new_df['Arrival_Date'], dayfirst=True)
month = new_df['Arrival_Date'].dt.month
new_df['month_sin'] = np.sin(2 * np.pi * month / 12)
new_df['month_cos'] = np.cos(2 * np.pi * month / 12)

# 3. Label Encoded IDs mapping (to preserve Adjacency Matrix integrity)
master_file = "mandi_master_data.parquet"
if os.path.exists(master_file):
    print("📖 Loading Master Parquet Dictionary to preserve Matrix IDs...")
    master_df = pd.read_parquet(master_file)
    
    # Map old IDs directly
    for col in ['State', 'District', 'Market', 'Commodity']:
        if f"{col}_ID" in master_df.columns:
            # Create a lookup dictionary from the master file
            lookup = master_df.drop_duplicates(subset=[col]).set_index(col)[f'{col}_ID']
            new_df[f'{col}_ID'] = new_df[col].map(lookup)
            
            # Map new unseen values safely without nonlocal closures
            unmapped_mask = new_df[f'{col}_ID'].isna()
            unmapped_vals = new_df.loc[unmapped_mask, col].unique()
            if len(unmapped_vals) > 0:
                max_id = int(master_df[f'{col}_ID'].max() if pd.notna(master_df[f'{col}_ID'].max()) else 0)
                new_map = {v: max_id + 1 + i for i, v in enumerate(unmapped_vals)}
                new_df.loc[unmapped_mask, f'{col}_ID'] = new_df.loc[unmapped_mask, col].map(new_map)
                
    # Concatenate seamlessly
    print("🔗 Appending to Master Data...")
    final_df = pd.concat([master_df, new_df], ignore_index=True)
    
    final_df.to_parquet(master_file, index=False, engine='pyarrow')
    print(f"🎉 SUCCESS! Dataset updated. New Total Rows: {len(final_df)}")
else:
    print("❌ Cannot find mandi_master_data.parquet!")
