import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder

# --- UPDATED PATH FOR YOUR MACHINE ---
# Based on your scan, the dataset folder is 'MinorP Dataset'
INPUT_DIR = "MinorP Dataset/Parquet" 
OUTPUT_FILE = "mandi_master_data.parquet"

def run_preprocessing():
    # 1. Verification
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: Cannot find folder: {INPUT_DIR}")
        print("Please ensure 'Parquet' folder is inside 'MinorP Dataset'")
        return

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not files:
        print(f"❌ ERROR: No files found in {INPUT_DIR}")
        return

    print(f"📂 Found {len(files)} files in {INPUT_DIR}. Processing...")
    
    all_dfs = []

    for file in files:
        year_name = os.path.basename(file)
        print(f"⌛ Processing {year_name}...", end="\r")
        
        try:
            # 1. ATTEMPT PARQUET LOAD
            df = pd.read_parquet(file)
            print(f"⌛ Processing {year_name} (Parquet)...", end="\r")

        except Exception as e:
            # 2. FALLBACK TO CSV IF PARQUET FAILS (Fix for 2025/2026)
            print(f"\n⚠️ Parquet Error in {year_name}: {e}")
            
            # Logic: Convert 'Parquet/2025.parquet' -> 'MinorP Dataset/csv/2025.csv'
            # (Adjust the 'csv' folder name below if yours is named differently)
            csv_backup = file.replace("Parquet", "csv").replace(".parquet", ".csv")
            
            if os.path.exists(csv_backup):
                print(f"🔄 Fallback: Loading {os.path.basename(csv_backup)} instead...")
                df = pd.read_csv(csv_backup)
            else:
                print(f"❌ Failed to find CSV backup at {csv_backup}. Skipping year.")
                continue

        # --- MATH: CLEANING (Applied to whichever file loaded) ---
        # Z-Score Math: Removing values outside 3 standard deviations
        mu = df['Modal_Price'].mean()
        sigma = df['Modal_Price'].std()
        df = df[(df['Modal_Price'] >= mu - 3*sigma) & (df['Modal_Price'] <= mu + 3*sigma)]
        
        # --- MATH: SEASONALITY ---
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
        month = df['Arrival_Date'].dt.month
        
        # Cyclical Encoding: Maps months to coordinates on a 2D circle
        # Impact: Ensures Dec (12) and Jan (1) are mathematically adjacent.
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        all_dfs.append(df)

    # 2. CONSOLIDATION
    print("\nMerging all years into Master Brain...")
    master_df = pd.concat(all_dfs, ignore_index=True)

    # 3. LABEL ENCODING (Math: Numerical Mapping)
    # We turn text into numbers so the Graph Neural Network can process them
    for col in ['State', 'District', 'Market', 'Commodity']:
        le = LabelEncoder()
        master_df[f'{col}_ID'] = le.fit_transform(master_df[col].astype(str))

    # 4. FINAL SAVE
    master_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    print(f"✅ SUCCESS: Created {OUTPUT_FILE} with {len(master_df)} rows.")

if __name__ == "__main__":
    run_preprocessing()