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

# --- 1. CONFIGURATION & LOGGING ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_filename = "daily_updater.log"
lock_filename = "updater.lock"

# Setup dual logging (File + Terminal)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

fh = logging.FileHandler(log_filename)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(ch)

# --- 2. SINGLE INSTANCE LOCK ---
if os.path.exists(lock_filename):
    logging.warning("⚠️ Script is already running or a previous run crashed. Exiting.")
    sys.exit(0)
open(lock_filename, 'w').close()

try:
    API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
    URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    HEADERS = {'User-Agent': 'Mozilla/5.0'}
    master_file = "mandi_master_data.parquet"

    if not os.path.exists(master_file):
        logging.error("❌ Cannot find master Parquet! Halting.")
        sys.exit(1)

    # 3. DETERMINE CATCH-UP WINDOW
    logging.info("📖 Scanning Parquet metadata for last update...")
    # Optimization: Read only the last row to find the date
    metadata_df = pd.read_parquet(master_file, columns=['Arrival_Date'])
    metadata_df['Arrival_Date'] = pd.to_datetime(metadata_df['Arrival_Date'], errors='coerce')
    last_date = metadata_df['Arrival_Date'].max().date()
    del metadata_df

    target_date = datetime.date.today()
    if last_date >= target_date:
        logging.info("✅ Database is entirely up-to-date.")
        sys.exit(0)

    current_fetch_date = last_date + datetime.timedelta(days=1)
    logging.info(f"🚀 Starting Catch-Up: {current_fetch_date} -> {target_date}")

    all_records = []
    LIMIT = 10000

    # 4. FETCH WITH RETRIES
    while current_fetch_date <= target_date:
        date_str = current_fetch_date.strftime("%d/%m/%Y")
        offset = 0
        day_total = 0
        
        while True:
            params = {
                "api-key": API_KEY, "format": "json", 
                "limit": LIMIT, "offset": offset, 
                "filters[Arrival_Date]": date_str
            }
            
            success = False
            for attempt in range(3): # 3 Retries per page
                try:
                    resp = requests.get(URL, params=params, headers=HEADERS, timeout=45)
                    if resp.status_code == 200:
                        records = resp.json().get('records', [])
                        all_records.extend(records)
                        day_total += len(records)
                        success = True
                        break
                    else:
                        logging.warning(f"⚠️ API Status {resp.status_code} on {date_str}. Attempt {attempt+1}")
                except Exception as e:
                    logging.warning(f"⚠️ Timeout/Network Error on {date_str}. Attempt {attempt+1}")
                time.sleep(5)

            if not success:
                logging.error(f"❌ Failed to fetch data for {date_str} after 3 attempts. Halting to prevent gaps.")
                sys.exit(1)

            if len(records) < LIMIT:
                break
            offset += LIMIT
        
        logging.info(f"📅 {date_str}: Fetched {day_total} records.")
        current_fetch_date += datetime.timedelta(days=1)

    # 5. DATA VALIDATION (WEEKEND SAFEGUARD)
    if not all_records or len(all_records) < 500:
        logging.info("ℹ️ Gov server returned negligible records (Weekend/Holiday). No update performed.")
        sys.exit(0)

    # 6. PROCESSING & MAPPING
    new_df = pd.DataFrame(all_records)
    for float_col in ['Min_Price', 'Max_Price', 'Modal_Price']:
        new_df[float_col] = pd.to_numeric(new_df[float_col], errors='coerce')
    new_df = new_df.dropna(subset=['Modal_Price'])

    new_df['Arrival_Date'] = pd.to_datetime(new_df['Arrival_Date'], dayfirst=True)
    month = new_df['Arrival_Date'].dt.month
    new_df['month_sin'] = np.sin(2 * np.pi * month / 12)
    new_df['month_cos'] = np.cos(2 * np.pi * month / 12)

    logging.info("📖 Mapping Matrix IDs (Memory-Safe)...")
    id_cols = ['State', 'State_ID', 'District', 'District_ID', 'Market', 'Market_ID', 'Commodity', 'Commodity_ID']
    id_df = pd.read_parquet(master_file, columns=id_cols)

    for col in ['State', 'District', 'Market', 'Commodity']:
        id_col = f'{col}_ID'
        lookup = id_df.drop_duplicates(subset=[col]).set_index(col)[id_col]
        new_df[id_col] = new_df[col].map(lookup)
        
        # Auto-map new entries if any
        unmapped = new_df[new_df[id_col].isna()][col].unique()
        if len(unmapped) > 0:
            current_max = int(id_df[id_col].max() or 0)
            new_map = {v: current_max + 1 + i for i, v in enumerate(unmapped)}
            new_df.loc[new_df[id_col].isna(), id_col] = new_df[col].map(new_map)
            logging.info(f"➕ New mapping: Added {len(unmapped)} new IDs for {col}.")
    
    del id_df

    # 7. ATOMIC PARQUET APPEND
    logging.info(f"🔗 Appending {len(new_df)} rows to {master_file}...")
    
    # Get the existing schema
    existing_schema = pq.read_schema(master_file)
    
    # 🟢 NEW LOGIC: Align columns and let PyArrow handle the type casting
    # Ensure all columns exist in new_df
    for col in existing_schema.names:
        if col not in new_df.columns:
            new_df[col] = None
            
    # Reorder columns to match Parquet schema exactly
    new_df = new_df[existing_schema.names]

    # Convert to PyArrow Table and EXPLICITLY cast to the existing schema
    # This solves the DataType(string) conflict
    new_table = pa.Table.from_pandas(new_df, preserve_index=False)
    new_table = new_table.cast(target_schema=existing_schema)

    # Proceed with the atomic streaming append
    writer = pq.ParquetWriter(master_file + ".tmp", schema=existing_schema)
    pf = pq.ParquetFile(master_file)
    for batch in pf.iter_batches(batch_size=200_000):
        writer.write_batch(batch)
    writer.write_table(new_table)
    writer.close()

    os.replace(master_file + ".tmp", master_file)
    logging.info(f"🎉 SUCCESS! Database updated. New Total: {pq.read_metadata(master_file).num_rows}")

finally:
    if os.path.exists(lock_filename):
        os.remove(lock_filename)