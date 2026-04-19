import os
import requests
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc

# --- CONFIGURATION ---t 
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = "579b464db66ec23bdd000001709f3046112f464c4cee72c06886efa6"
URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
MASTER_PARQUET = "mandi_master_data.parquet"


def fetch_from_parquet(commodity: str) -> pd.DataFrame:
    """
    Reads the latest available date's data for a commodity
    from the master Parquet store using PyArrow column filters.
    Finds the latest date that actually has data for this specific commodity.
    Returns a standardized lowercase-column DataFrame, or empty DataFrame on failure.
    """
    if not os.path.exists(MASTER_PARQUET):
        return pd.DataFrame()

    try:
        # Step 1: Find the latest Arrival_Date that has data for THIS commodity
        # Read only Arrival_Date + Commodity columns — much lighter than full scan
        date_comm_table = pq.read_table(
            MASTER_PARQUET,
            columns=["Arrival_Date", "Commodity"],
            filters=[("Commodity", "==", commodity)],
        )

        if date_comm_table.num_rows == 0:
            # Try case-insensitive — check all commodities and match in pandas
            date_comm_table = pq.read_table(MASTER_PARQUET, columns=["Arrival_Date", "Commodity"])
            date_comm_df = date_comm_table.to_pandas()
            date_comm_df = date_comm_df[date_comm_df["Commodity"].str.lower() == commodity.lower()]
            if date_comm_df.empty:
                return pd.DataFrame()
            max_date = date_comm_df["Arrival_Date"].max()
            # Get the actual commodity name as stored in parquet (might be different case)
            actual_commodity = date_comm_df["Commodity"].iloc[0]
        else:
            max_date = pc.max(date_comm_table.column("Arrival_Date")).as_py()
            actual_commodity = commodity

        COLS = ["State", "District", "Market", "Commodity", "Variety", "Grade",
                "Min_Price", "Max_Price", "Modal_Price", "Arrival_Date"]

        # Step 2: Read all columns for max_date + this commodity
        filters = [
            ("Arrival_Date", "==", max_date),
            ("Commodity", "==", actual_commodity),
        ]
        table = pq.read_table(
            MASTER_PARQUET,
            columns=COLS,
            filters=filters,
        )

        df = table.to_pandas()

        if df.empty:
            return pd.DataFrame()

        # Standardize column names to lowercase to match the rest of the app
        df.columns = [c.lower() for c in df.columns]
        df["arrival_date"] = pd.to_datetime(df["arrival_date"]).dt.strftime("%d/%m/%Y")
        return df

    except Exception as e:
        print(f"📦 Parquet read error: {e}")
        return pd.DataFrame()



def fetch_from_api(commodity: str) -> pd.DataFrame:
    """
    Fetches real-time commodity data from the new Agmarknet live-feed resource.
    The new resource (9ef84268) serves today's data with lowercase column names
    and requires lowercase filter keys.
    Returns a standardized lowercase-column DataFrame, or empty DataFrame on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    all_records = []
    offset = 0
    LIMIT = 10000

    while True:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": LIMIT,
            "offset": offset,
            "filters[commodity]": commodity,   # new resource uses lowercase filter keys
        }
        try:
            response = requests.get(URL, params=params, headers=headers, timeout=20)
            if response.status_code == 200:
                records = response.json().get("records", [])
                all_records.extend(records)
                if len(records) < LIMIT:
                    break   # last page
                offset += LIMIT
            else:
                print(f"API status {response.status_code}")
                break
        except Exception as e:
            print(f"📡 API error: {e}")
            break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # New resource already returns lowercase columns; normalize just in case
    df.columns = [c.lower() for c in df.columns]
    # arrival_date comes as 'DD/MM/YYYY' string — standardize format
    if 'arrival_date' in df.columns:
        df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True).dt.strftime('%d/%m/%Y')
    return df


def fetch_agmarknet_data(commodity: str = "Onion"):
    """
    Primary data gateway for the MandiFlow dashboard.

    Priority order:
      1. Live API        — always try for today's freshest data first
      2. Master Parquet  — fallback if API is down or returns no data
      3. mini_fallback.csv — last resort offline cache

    Returns: (pd.DataFrame, bool) -> (Data, Is_Live_Status)
    """

    # --- SOURCE 1: Live Agmarknet API (freshest — today's data) ---
    api_df = fetch_from_api(commodity)
    if not api_df.empty:
        print(f"🌐 Serving from Live API | {len(api_df)} rows for {commodity}")
        return api_df, True

    print(f"⚠️ Live API returned no data for '{commodity}', falling back to Master Parquet...")

    # --- SOURCE 2: Master Parquet (nightly cron-updated) ---
    parquet_df = fetch_from_parquet(commodity)
    if not parquet_df.empty:
        print(f"✅ Serving from Master Parquet | {len(parquet_df)} rows for {commodity}")
        return parquet_df, False

    print(f"⚠️ Parquet also returned no data. Falling back to offline cache...")

    # --- SOURCE 3: Offline mini-fallback CSV ---
    fallback_file = "mini_fallback.csv"
    if os.path.exists(fallback_file):
        try:
            offline_df = pd.read_csv(fallback_file)
            offline_df.columns = [c.lower() for c in offline_df.columns]
            filtered = offline_df[offline_df["commodity"].str.lower() == commodity.lower()]
            if not filtered.empty:
                return filtered.tail(50), False
        except Exception as e:
            print(f"Error reading fallback file: {e}")

    # --- ABSOLUTE FAILSAFE ---
    return pd.DataFrame(), False