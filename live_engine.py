import requests
import pandas as pd
import streamlit as st
import os

# --- CONFIGURATION ---
RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"
API_KEY = "579b464db66ec23bdd000001b3f958d3bc90417a7358c09287a15f47"
URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

def fetch_agmarknet_data(commodity="Onion"):
    """
    Fetches real-time commodity data from Agmarknet API.
    Returns: (pd.DataFrame, bool) -> (Data, Is_Live_Status)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 50,
        "filters[Commodity]": commodity 
    }
    
    try:
        # 1. ATTEMPT LIVE API CALL
        # Increased timeout to 20s to handle slow government servers
        response = requests.get(URL, params=params, headers=headers, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', [])
            
            if records:
                df = pd.DataFrame(records)
                # Standardize column names to lowercase for consistency
                df.columns = [c.lower() for c in df.columns]
                
                # Success: Return Data and True (Is Live)
                return df, True
            else:
                print(f"API connected but no records found for {commodity}.")
        else:
            print(f"API Server returned status code: {response.status_code}")

    except Exception as e:
        # Catch timeouts, connection errors, etc.
        print(f"📡 API Connection Error: {e}")

    # 2. FALLBACK LOGIC (If API fails or returns no records)
    fallback_file = "mini_fallback.csv"
    
    if os.path.exists(fallback_file):
        try:
            offline_df = pd.read_csv(fallback_file)
            offline_df.columns = [c.lower() for c in offline_df.columns]
            
            # Filter fallback data for the selected commodity
            filtered_df = offline_df[offline_df['commodity'].str.lower() == commodity.lower()].copy()
            
            if not filtered_df.empty:
                # Return Offline Data and False (Not Live)
                return filtered_df.tail(50), False
        except Exception as fallback_error:
            print(f"Error reading fallback file: {fallback_error}")

    # 3. ABSOLUTE FAILSAFE: Return empty DataFrame and False
    return pd.DataFrame(), False