import pandas as pd
import numpy as np
from rapidfuzz import process, utils

def run_smart_geocoder():
    print("📖 Loading datasets...")
    try:
        # Loading your master file
        master_df = pd.read_parquet("mandi_master_data.parquet", columns=['Market_ID', 'District', 'Market'])
    except Exception as e:
        print(f"❌ Error: Could not find master parquet. ({e})")
        return

    # Load the Pincode CSV
    pincode_df = pd.read_csv("indian_pincodes.csv")

    # 1. MATH FIX: Convert Latitude/Longitude to numeric (Float)
    # 'errors=coerce' turns any non-numeric text (like "NA") into NaN
    pincode_df['latitude'] = pd.to_numeric(pincode_df['latitude'], errors='coerce')
    pincode_df['longitude'] = pd.to_numeric(pincode_df['longitude'], errors='coerce')
    
    # Drop rows that don't have coordinates
    pincode_df = pincode_df.dropna(subset=['latitude', 'longitude'])
    
    # Standardize district names
    pincode_df['district'] = pincode_df['district'].astype(str).str.upper().str.strip()
    
    # 2. MATH: Group by District to find the average (Centroid)
    # Now that they are numeric, the mean() function will work perfectly
    district_coords = pincode_df.groupby('district').agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    pincode_districts = district_coords['district'].tolist()

    # 3. Process Mandi Districts
    mandi_mapping = master_df.drop_duplicates(subset=['Market_ID']).copy()
    mandi_mapping['District_Clean'] = mandi_mapping['District'].astype(str).str.upper().str.strip()

    # 4. SMART FUZZY MATCHING LOGIC
    def find_best_match(name):
        if not name or name == 'NAN':
            return None
            
        manual_map = {
            "ALLEPPEY": "ALAPPUZHA",
            "SHIMOGA": "SHIVAMOGGA",
            "FEROZPUR": "FIROZPUR",
            "WEST CHAMBARAN": "WEST CHAMPARAN",
            "HISSAR": "HISAR",
            "DEEG": "BHARATPUR" # Mapping new district to parent for coordinates
        }
        if name in manual_map:
            return manual_map[name]

        match = process.extractOne(name, pincode_districts, processor=utils.default_process)
        
        if match and match[1] > 80:
            return match[0]
        return None

    print(f"🧩 Matching {len(mandi_mapping['District_Clean'].unique())} districts using Fuzzy Logic...")
    mandi_mapping['Matched_District'] = mandi_mapping['District_Clean'].apply(find_best_match)

    # 5. Merge coordinates
    final_coords = pd.merge(
        mandi_mapping, 
        district_coords, 
        left_on='Matched_District', 
        right_on='district', 
        how='left'
    )

    # 6. Save for the Dashboard
    output_cols = ['Market_ID', 'Market', 'District', 'latitude', 'longitude']
    final_coords[output_cols].to_csv("market_coords.csv", index=False)
    
    matched = final_coords['latitude'].notnull().sum()
    total = len(final_coords)
    print(f"\n✅ GEOSPATIAL ALIGNMENT COMPLETE")
    print(f"Total Markets: {total}")
    print(f"Successfully Mapped: {matched} ({ (matched/total)*100 :.1f}%)")
    print(f"Output saved to: market_coords.csv")

if __name__ == "__main__":
    run_smart_geocoder()