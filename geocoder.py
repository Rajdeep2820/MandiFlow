import pandas as pd
import numpy as np
from rapidfuzz import process, utils

def run_smart_geocoder():
    print("📖 Loading datasets...")
    try:
        # Loading your master file
        master_df = pd.read_parquet("mandi_master_data.parquet", columns=['Market_ID', 'State', 'District', 'Market'])
    except Exception as e:
        print(f"❌ Error: Could not find master parquet. ({e})")
        return

    # Load the Pincode CSV
    pincode_df = pd.read_csv("indian_pincodes.csv")

    # 1. MATH FIX: Convert Latitude/Longitude to numeric (Float)
    pincode_df['latitude'] = pd.to_numeric(pincode_df['latitude'], errors='coerce')
    pincode_df['longitude'] = pd.to_numeric(pincode_df['longitude'], errors='coerce')
    
    # Drop rows that don't have coordinates
    pincode_df = pincode_df.dropna(subset=['latitude', 'longitude'])
    
    # Filter pincodes falling outside India to prevent corrupted district centroids
    pincode_df = pincode_df[
        (pincode_df['latitude'] >= 6.0) & (pincode_df['latitude'] <= 38.0) &
        (pincode_df['longitude'] >= 68.0) & (pincode_df['longitude'] <= 98.0)
    ]
    
    # Standardize district and state names
    pincode_df['district'] = pincode_df['district'].astype(str).str.upper().str.strip()
    pincode_df['statename'] = pincode_df['statename'].astype(str).str.upper().str.strip()
    
    # 2. MATH: Group by State and District to find the average (Centroid)
    district_coords = pincode_df.groupby(['statename', 'district']).agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    pincode_states = district_coords['statename'].unique().tolist()
    state_to_districts = {
        state: district_coords[district_coords['statename'] == state]['district'].tolist()
        for state in pincode_states
    }

    # 3. Process Mandi Districts
    mandi_mapping = master_df.drop_duplicates(subset=['Market_ID']).copy()
    mandi_mapping['District_Clean'] = mandi_mapping['District'].astype(str).str.upper().str.strip()
    mandi_mapping['State_Clean'] = mandi_mapping['State'].astype(str).str.upper().str.strip()

    # 4. HIERARCHICAL STATE-AWARE FUZZY MATCHING LOGIC
    def find_best_match(row):
        name = row['District_Clean']
        raw_state = row['State_Clean']
        
        if not name or name == 'NAN':
            return pd.Series([None, None])
            
        manual_map = {
            "ALLEPPEY": "ALAPPUZHA",
            "SHIMOGA": "SHIVAMOGGA",
            "FEROZPUR": "FIROZPUR",
            "WEST CHAMBARAN": "WEST CHAMPARAN",
            "HISSAR": "HISAR",
            "DEEG": "BHARATPUR",
            "DHARASHIV": "OSMANABAD",
            "CHHATRAPATI SAMBHAJINAGAR": "AURANGABAD"
        }
        
        # Apply manual mappings first
        target_district = manual_map.get(name, name)
        
        # 1. Fuzzy match the STATE first to guarantee perfect boundary isolation
        state_match = process.extractOne(raw_state, pincode_states, processor=utils.default_process)
        if not state_match or state_match[1] < 70:
            return pd.Series([None, None]) # State failed to match entirely
            
        matched_state = state_match[0]
        
        # 2. Retrieve only the districts that legally exist inside that specific matched State
        allowed_districts = state_to_districts.get(matched_state, [])
        if not allowed_districts:
            return pd.Series([None, None])
            
        # 3. Fuzzy match the district exclusively against its parent state's jurisdictions
        match = process.extractOne(target_district, allowed_districts, processor=utils.default_process)
        
        if match and match[1] > 80:
            return pd.Series([matched_state, match[0]])
        
        return pd.Series([matched_state, None])

    print(f"🧩 Matching {len(mandi_mapping['District_Clean'].unique())} districts using State Hierarchy...")
    mandi_mapping[['Matched_State', 'Matched_District']] = mandi_mapping.apply(find_best_match, axis=1)

    # 5. Merge coordinates strictly on BOTH State and District
    final_coords = pd.merge(
        mandi_mapping, 
        district_coords, 
        left_on=['Matched_State', 'Matched_District'], 
        right_on=['statename', 'district'], 
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