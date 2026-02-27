import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from live_engine import fetch_agmarknet_data

# --- 1. DATA LOADING FUNCTIONS (Defined First) ---

@st.cache_data
def load_map_data():
    """Loads the static coordinate data for all mandis."""
    try:
        df = pd.read_csv("market_coords.csv")
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        # Pre-process keys for merging
        df['market_key'] = df['Market'].str.lower().str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading map coordinates: {e}")
        return pd.DataFrame()

def get_final_data(comm):
    """Handles session state to prevent infinite API refresh loops."""
    # If data doesn't exist or commodity changed, fetch new data
    if 'mandi_data' not in st.session_state or st.session_state.get('current_comm') != comm:
        with st.spinner(f"Connecting to Agmarknet for {comm}..."):
            # This expects the rewritten live_engine.py returning (df, bool)
            data, is_live = fetch_agmarknet_data(comm)
            
            if not data.empty:
                st.session_state.mandi_data = data
                st.session_state.is_live = is_live
                st.session_state.current_comm = comm
                # Capture latest update date
                if 'arrival_date' in data.columns:
                    st.session_state.last_update = data['arrival_date'].iloc[0]
            else:
                st.session_state.mandi_data = pd.DataFrame()
                st.session_state.is_live = False
                st.session_state.last_update = "N/A"

    return st.session_state.mandi_data, st.session_state.is_live

# --- 2. SETTINGS & UI STYLING ---
st.set_page_config(page_title="MandiFlow Intelligence", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4250; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Simulation Controls")
commodity = st.sidebar.selectbox("Select Commodity", ["Onion", "Potato", "Tomato", "Garlic", "Wheat"])

# Initialize variables from the engine
live_df, is_live = get_final_data(commodity)

# API Status Indicator
status_color = "#2ecc71" if is_live else "#f1c40f"
st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 5px;">
        <div style="width: 12px; height: 12px; background-color: {status_color}; border-radius: 50%; box-shadow: 0 0 8px {status_color};"></div>
        <b style="color: {status_color};">SYSTEM: {"LIVE" if is_live else "FALLBACK"}</b>
    </div>
    <p style="font-size: 0.75rem; color: #888;">Latest Update: {st.session_state.get('last_update', 'N/A')}</p>
""", unsafe_allow_html=True)

if st.sidebar.button("🔄 Refresh Data"):
    st.session_state.pop('mandi_data', None)
    st.rerun()

news_headline = st.sidebar.text_area("News/Shock Trigger", "Heavy rainfall in Mandsaur and Indore")

# --- 4. MAIN LAYOUT ---
st.title("🌾 MandiFlow: Spatio-Temporal AI Dashboard")
coords_df = load_map_data()

col1, col2 = st.columns([4, 1])

with col1:
    st.subheader(f"📍 {commodity} Market Network (Hover for Prices)")
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB dark_matter")
    marker_cluster = MarkerCluster(options={'disableClusteringAtZoom': 7}).add_to(m)

    if not live_df.empty and not coords_df.empty:
        # Prepare live data for merging
        live_df['market_key'] = live_df['market'].str.lower().str.strip()
        
        # Inner join to only show markets that have price data
        map_merge = pd.merge(coords_df, live_df[['market_key', 'modal_price']], on='market_key', how='inner')
        
        for _, row in map_merge.iterrows():
            price = row['modal_price']
            dist = str(row['District']).lower()
            
            # Shock Logic
            is_shocked = any(word.lower() in news_headline.lower() for word in dist.split())
            color = "#e74c3c" if is_shocked else "#2ecc71"
            
            # Hover Tooltip
            tooltip_text = f"<b>{row['Market']}</b><br>Price: ₹{price}/qtl"
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6, color=color, fill=True, fill_opacity=0.8,
                tooltip=tooltip_text
            ).add_to(marker_cluster)
    
    st_folium(m, width=850, height=500, key="mandi_map")

with col2:
    st.subheader("📊 Price Feed")
    
    if not live_df.empty:
        # Sorting Toggle
        sort_order = st.radio("Sort by Price:", ["Highest First", "Lowest First"], horizontal=True)
        
        temp_df = live_df.copy()
        temp_df['modal_price'] = pd.to_numeric(temp_df['modal_price'], errors='coerce')
        temp_df = temp_df.sort_values(by='modal_price', ascending=(sort_order == "Lowest First"))
        
        st.dataframe(
            temp_df[['market', 'district', 'modal_price']].head(25),
            column_config={
                "market": "Mandi",
                "district": "District",
                "modal_price": st.column_config.NumberColumn("Price", format="₹%d")
            },
            use_container_width=True, hide_index=True
        )
        
        st.metric("National Avg Price", f"₹{temp_df['modal_price'].mean():.2f}")
    else:
        st.warning("No price records available to display.")

# --- 5. LEGEND ---
st.markdown("""
    <div style="background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4250;">
        <span style="color: #2ecc71;">●</span> Stable Price &nbsp;&nbsp;
        <span style="color: #e74c3c;">●</span> News Shock Detected &nbsp;&nbsp;
        <span style="color: #f1c40f;">●</span> Cluster (Zoom to expand)
    </div>
""", unsafe_allow_html=True)