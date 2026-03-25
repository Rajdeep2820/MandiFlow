import os
import sys
import time
import logging
import requests
import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
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
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

master_file = "mandi_master_data.parquet"
if not os.path.exists(master_file):
    logging.error("❌ Cannot find mandi_master_data.parquet! Halting.")
    sys.exit(1)

# 1. Read ONLY Arrival_Date + ID columns — never load full 75M rows
logging.info("📖 Reading Parquet metadata to determine catch-up window (memory-safe)...")
date_df = pd.read_parquet(master_file, columns=['Arrival_Date'])
date_df['Arrival_Date'] = pd.to_datetime(date_df['Arrival_Date'], errors='coerce')
last_date = date_df['Arrival_Date'].max().date()
del date_df  # Free immediately

target_date = datetime.date.today()

if last_date >= target_date:
    logging.info("✅ Database is entirely up-to-date! No action needed.")
    sys.exit(0)

# 2. Iterate and Fetch from `last_date + 1` to `target_date`
current_fetch_date = last_date + datetime.timedelta(days=1)
logging.info(f"🚀 Waking up for Catch-Up fetch! ({current_fetch_date.strftime('%d/%m/%Y')} -> {target_date.strftime('%d/%m/%Y')})")

all_records = []

LIMIT = 10000  # API page size

while current_fetch_date <= target_date:
    date_str = current_fetch_date.strftime("%d/%m/%Y")
    offset = 0
    day_total = 0

    while True:  # Paginate until all records fetched
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": LIMIT,
            "offset": offset,
            "filters[arrival_date]": date_str   # new resource uses lowercase filter keys
        }
        try:
            resp = requests.get(URL, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                records = resp.json().get('records', [])
                all_records.extend(records)
                day_total += len(records)
                if len(records) < LIMIT:
                    break  # Last page — done for this date
                offset += LIMIT
                time.sleep(0.3)  # Be polite between pages
            else:
                logging.error(f"❌ API error for {date_str} offset={offset}. Status: {resp.status_code}")
                break
        except Exception as e:
            logging.error(f"⚠️ Network Exception for {date_str} offset={offset}: {e}")
            break

    logging.info(f"✅ Fetched {day_total:<6} total records for {date_str} ({offset//LIMIT + 1} page(s))")
    current_fetch_date += datetime.timedelta(days=1)
    time.sleep(0.5)

if not all_records:
    logging.info("ℹ️ No records found on government server for any new days. Markets might be closed.")
    sys.exit(0)

new_df = pd.DataFrame(all_records)

# 3. Match Schema Formatting
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

if new_df.empty:
    logging.info("ℹ️ Fetched records had no valid Modal_Price. Appending skipped.")
    sys.exit(0)

new_df['Arrival_Date'] = pd.to_datetime(new_df['Arrival_Date'], dayfirst=True)
month = new_df['Arrival_Date'].dt.month
new_df['month_sin'] = np.sin(2 * np.pi * month / 12)
new_df['month_cos'] = np.cos(2 * np.pi * month / 12)

# 4. Memory-safe ID mapping — only load the 4 ID columns
logging.info("📖 Loading only ID columns for label-encoding lookup...")
id_cols = ['State', 'State_ID', 'District', 'District_ID',
           'Market', 'Market_ID', 'Commodity', 'Commodity_ID']
id_df = pd.read_parquet(master_file, columns=id_cols)

for col in ['State', 'District', 'Market', 'Commodity']:
    id_col = f'{col}_ID'
    if id_col in id_df.columns:
        lookup = id_df.drop_duplicates(subset=[col]).set_index(col)[id_col]
        new_df[id_col] = new_df[col].map(lookup)

        unmapped_mask = new_df[id_col].isna()
        unmapped_vals = new_df.loc[unmapped_mask, col].unique()
        if len(unmapped_vals) > 0:
            max_id = int(id_df[id_col].max() if pd.notna(id_df[id_col].max()) else 0)
            new_map = {v: max_id + 1 + i for i, v in enumerate(unmapped_vals)}
            new_df.loc[unmapped_mask, id_col] = new_df.loc[unmapped_mask, col].map(new_map)
            logging.info(f"➕ Auto-mapped {len(unmapped_vals)} new '{col}' entries into Matrix IDs!")

del id_df  # Free memory immediately

# 5. Memory-safe streaming append — never load full dataset into RAM
logging.info(f"🔗 Streaming {len(new_df)} new rows into Parquet (batch append)...")

existing_schema = pq.read_schema(master_file)
new_table = pa.Table.from_pandas(new_df, schema=existing_schema, preserve_index=False)

writer = pq.ParquetWriter(master_file + ".tmp", schema=existing_schema)
pf = pq.ParquetFile(master_file)
for batch in pf.iter_batches(batch_size=500_000):
    writer.write_batch(batch)
writer.write_table(new_table)
writer.close()

# Atomic replace — safe against corruption on crash
os.replace(master_file + ".tmp", master_file)

total_rows = pq.read_metadata(master_file).num_rows
logging.info(f"🎉 SUCCESS! Parquet updated. Appended {len(new_df)} rows. Total: {total_rows}")
