import os
import time
import requests
import datetime
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

start_date = datetime.date(2026, 3, 20)
end_date = datetime.date(2026, 3, 22)

delta = datetime.timedelta(days=1)
curr_date = start_date

print(f"🚀 Starting Historical Backfill: {start_date} to {end_date}")

all_records = []

LIMIT = 10000  # API page size

while curr_date <= end_date:
    date_str = curr_date.strftime("%d/%m/%Y")
    offset = 0
    day_total = 0

    while True:  # Paginate until all records for this date are fetched
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": LIMIT,
            "offset": offset,
            "filters[arrival_date]": date_str
        }
        try:
            resp = requests.get(URL, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                records = resp.json().get('records', [])
                all_records.extend(records)
                day_total += len(records)
                if len(records) < LIMIT:
                    break  # Last page — done for this date
                offset += LIMIT
                time.sleep(0.3)  # Be polite between pages
            else:
                print(f"❌ Failed {date_str} offset={offset} - Code: {resp.status_code}")
                break
        except Exception as e:
            print(f"⚠️ Error {date_str} offset={offset}: {e}")
            break

    print(f"✅ Fetched {day_total:<6} total records for {date_str} ({offset//LIMIT + 1} page(s))")
    curr_date += delta
    time.sleep(0.5)


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

# 3. Label Encoded IDs mapping — MEMORY SAFE: only load ID columns, not full dataset
master_file = "mandi_master_data.parquet"
if os.path.exists(master_file):
    print("📖 Loading only ID columns from Master Parquet (memory-safe)...")
    id_cols = ['State', 'State_ID', 'District', 'District_ID',
               'Market', 'Market_ID', 'Commodity', 'Commodity_ID']
    # Read only the columns we need for ID mapping
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

    del id_df  # Free memory immediately

    # 4. APPEND with date-deduplication — drop existing rows for re-downloaded dates first
    download_dates = set(new_df['Arrival_Date'].dt.date)
    print(f"🗑️  Removing any existing rows for dates: {sorted(download_dates)}")
    print(f"🔗 Appending {len(new_df)} fresh rows to Master Parquet...")

    existing_schema = pq.read_schema(master_file)
    new_table = pa.Table.from_pandas(new_df, schema=existing_schema, preserve_index=False)

    writer = pq.ParquetWriter(master_file + ".tmp", schema=existing_schema)
    pf = pq.ParquetFile(master_file)
    removed = 0
    for batch in pf.iter_batches(batch_size=500_000):
        batch_df = batch.to_pandas()
        batch_df['Arrival_Date'] = pd.to_datetime(batch_df['Arrival_Date'])
        # Drop rows for dates we're about to replace with fresh data
        mask = batch_df['Arrival_Date'].dt.date.isin(download_dates)
        removed += mask.sum()
        clean = batch_df[~mask]
        if not clean.empty:
            writer.write_table(pa.Table.from_pandas(clean, schema=existing_schema, preserve_index=False))
    writer.write_table(new_table)
    writer.close()

    os.replace(master_file + ".tmp", master_file)
    total_rows = pq.read_metadata(master_file).num_rows
    print(f"🗑️  Removed {removed} stale rows for those dates")
    print(f"🎉 SUCCESS! Dataset updated. New Total Rows: {total_rows}")
else:
    print("❌ Cannot find mandi_master_data.parquet!")

