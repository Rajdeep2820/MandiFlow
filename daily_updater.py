import os
import sys
import logging
import requests
import datetime
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Set up logging for the Cron Job to output to a safe file
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Ensure CWD is always correct in Cron
log_filename = "daily_updater.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# Daily Target Date
target_date = datetime.date.today()
date_str = target_date.strftime("%d/%m/%Y")
logging.info(f"🚀 Waking up for automated fetch: {date_str}")

# Retrieve Data
params = {
    "api-key": API_KEY,
    "format": "json",
    "limit": 10000,
    "filters[Arrival_Date]": date_str
}

try:
    resp = requests.get(URL, params=params, headers=HEADERS, timeout=30)
    if resp.status_code == 200:
        records = resp.json().get('records', [])
        logging.info(f"✅ Extracted {len(records)} official records from data.gov.in")
    else:
        logging.error(f"❌ Failed to reach API. Status: {resp.status_code}")
        sys.exit(1)
except Exception as e:
    logging.error(f"⚠️ Network Exception: {e}")
    sys.exit(1)

if not records:
    logging.info("ℹ️ No records found on government server for today. Markets might be closed.")
    sys.exit(0)

new_df = pd.DataFrame(records)

# 1. Match Schema Formatting Identically
rename_map = {
    'state': 'State', 'district': 'District', 'market': 'Market', 
    'commodity': 'Commodity', 'variety': 'Variety', 'grade': 'Grade',
    'min_price': 'Min_Price', 'max_price': 'Max_Price', 'modal_price': 'Modal_Price',
    'arrival_date': 'Arrival_Date'
}
new_df = new_df.rename(columns=rename_map)

for float_col in ['Min_Price', 'Max_Price', 'Modal_Price']:
    new_df[float_col] = pd.to_numeric(new_df[float_col], errors='coerce')
new_df = new_df.dropna(subset=['Modal_Price'])

new_df['Arrival_Date'] = pd.to_datetime(new_df['Arrival_Date'], dayfirst=True)
month = new_df['Arrival_Date'].dt.month
new_df['month_sin'] = np.sin(2 * np.pi * month / 12)
new_df['month_cos'] = np.cos(2 * np.pi * month / 12)

# 3. Intelligent Label Encoded mapping
master_file = "mandi_master_data.parquet"
if not os.path.exists(master_file):
    logging.error("❌ Cannot find mandi_master_data.parquet! Halting.")
    sys.exit(1)
    
master_df = pd.read_parquet(master_file)

# Duplication check: 
if target_date in master_df['Arrival_Date'].dt.date.values:
    logging.info("⚠️ This exact date has already been appended previously. Halting to avoid duplication.")
    sys.exit(0)

# Map IDs efficiently
for col in ['State', 'District', 'Market', 'Commodity']:
    if f"{col}_ID" in master_df.columns:
        lookup = master_df.drop_duplicates(subset=[col]).set_index(col)[f'{col}_ID']
        new_df[f'{col}_ID'] = new_df[col].map(lookup)
        
        unmapped_mask = new_df[f'{col}_ID'].isna()
        unmapped_vals = new_df.loc[unmapped_mask, col].unique()
        if len(unmapped_vals) > 0:
            max_id = int(master_df[f'{col}_ID'].max() if pd.notna(master_df[f'{col}_ID'].max()) else 0)
            new_map = {v: max_id + 1 + i for i, v in enumerate(unmapped_vals)}
            new_df.loc[unmapped_mask, f'{col}_ID'] = new_df.loc[unmapped_mask, col].map(new_map)
            logging.info(f"➕ Auto-mapped {len(unmapped_vals)} brand new '{col}' entries into Adjacency Matrix IDs!")

# Concatenate & Save
final_df = pd.concat([master_df, new_df], ignore_index=True)
final_df.to_parquet(master_file, index=False, engine='pyarrow')

logging.info(f"🎉 SUCCESS! Parquet locked. System grew by {len(new_df)} rows. Goodnight!")
