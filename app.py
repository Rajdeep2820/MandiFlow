import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from live_engine import fetch_agmarknet_data

# --- 1. DATA LOADING FUNCTIONS ---

@st.cache_data
def load_map_data():
    """Loads the static coordinate data and prepares keys for matching."""
    try:
        df = pd.read_csv("market_coords.csv")
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Filter for mandis within India's approximate geographical bounding box
        df = df[
            (df['latitude'] >= 6.0) & (df['latitude'] <= 38.0) &
            (df['longitude'] >= 68.0) & (df['longitude'] <= 98.0)
        ]
        
        # Standardize keys to UPPERCASE for robust matching with government API
        df['market_key'] = df['Market'].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading map coordinates: {e}")
        return pd.DataFrame()

def get_final_data(comm):
    """Handles session state to prevent infinite refresh loops and API flickering."""
    if 'mandi_data' not in st.session_state or st.session_state.get('last_comm') != comm:
        with st.spinner(f"Establishing Link for {comm}..."):
            data, is_live = fetch_agmarknet_data(comm)
            
            if not data.empty:
                # Standardize API keys to UPPERCASE
                data['market_key'] = data['market'].astype(str).str.upper().str.strip()
                st.session_state.mandi_data = data
                st.session_state.is_live = is_live
                st.session_state.last_comm = comm
                st.session_state.last_update = data['arrival_date'].iloc[0] if 'arrival_date' in data.columns else "N/A"
            else:
                st.session_state.mandi_data = pd.DataFrame()
                st.session_state.is_live = False
                st.session_state.last_comm = comm
                st.session_state.last_update = "N/A"

    return st.session_state.mandi_data, st.session_state.is_live

# --- 2. SETTINGS & UI STYLING ---
st.set_page_config(page_title="MandiFlow Intelligence", layout="wide", page_icon="🌾")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4250; }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    .pulse-dot {
        display: inline-block; width: 12px; height: 12px; border-radius: 50%;
        animation: pulse 2s infinite; margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ MandiFlow Controls")
commodity = st.sidebar.selectbox("Select Commodity", ["Onion", "Potato", "Tomato", "Garlic", "Wheat"])

# Trigger stable data fetch
live_df, is_live = get_final_data(commodity)

# API Status Indicator
status_color = "#2ecc71" if is_live and not live_df.empty else "#f1c40f"
status_text = "API CONNECTED" if is_live and not live_df.empty else "FALLBACK MODE"

st.sidebar.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; border: 1px solid {status_color}; background: rgba(0,0,0,0.2);">
        <div class="pulse-dot" style="background-color: {status_color}; box-shadow: 0 0 8px {status_color};"></div>
        <b style="color: {status_color};">{status_text}</b>
        <p style="font-size: 0.7rem; color: #888; margin-top: 5px;">Records Synced: {len(live_df)}</p>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("🔄 Sync System"):
    st.session_state.clear()
    st.rerun()

news_headline = st.sidebar.text_area("News/Shock Trigger", "Normal market conditions")

# --- 4. MAIN LAYOUT ---
st.title("🌾 MandiFlow: Spatio-Temporal AI Dashboard")
coords_df = load_map_data()

st.subheader(f"📍 {commodity} Network Analysis")
if not live_df.empty:
    st.metric("National Avg", f"₹{pd.to_numeric(live_df['modal_price']).mean():.2f}")
else:
    st.info("Waiting for data stream...")

m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="CartoDB dark_matter")
marker_cluster = MarkerCluster(options={'disableClusteringAtZoom': 7}).add_to(m)

if not coords_df.empty:
    # Map prices for O(1) lookup
    price_map = dict(zip(live_df['market_key'], live_df['modal_price'])) if not live_df.empty else {}

    for _, row in coords_df.iterrows():
        m_key = row['market_key']
        price = price_map.get(m_key)
        
        # Only render the mandi on the map if we have a live price
        if not price:
            continue
            
        dist = str(row['District']).lower()
        
        # Shock Logic (Red)
        is_shocked = any(word.lower() in news_headline.lower() for word in dist.split())
        color = "#e74c3c" if is_shocked else "#2ecc71"
        
        # Tooltip with HTML for clear hover reading
        hover_price = f"₹{price}/qtl" if price else "Checking Feed..."
        tooltip_html = f"""
            <div style='font-family: sans-serif; min-width: 120px;'>
                <b>{row['Market']}</b><br>
                <span style='color:{color};'>Price: {hover_price}</span><br>
                <small>District: {row['District']}</small>
            </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6, color=color, fill=True, fill_opacity=0.8,
            tooltip=folium.Tooltip(tooltip_html, sticky=True)
        ).add_to(marker_cluster)

# CRITICAL: returned_objects=[] prevents the map from causing reruns on zoom/move
st_folium(m, width="100%", height=750, returned_objects=[], key="mandi_map", use_container_width=True)

st.markdown("---")

# --- 5. DATA TABLE SEARCH ---
st.markdown("### Mandi Prices")
st.text(f"Price updated : {st.session_state.get('last_update', 'N/A')}")

# Dropdowns layout matching the image (Commodity, State, Market, Search Button)
col1, col2, col3, col4 = st.columns([3, 3, 3, 2])

states_opts = ["All States"] + (sorted(live_df['state'].unique().tolist()) if not live_df.empty and 'state' in live_df.columns else [])
sel_comm = col1.selectbox("Commodity", live_df['commodity'].unique().tolist() if not live_df.empty and 'commodity' in live_df.columns else [commodity], label_visibility="collapsed")
sel_state = col2.selectbox("State", states_opts, label_visibility="collapsed")

# Filter markets dynamically based on State
market_df = live_df if sel_state == "All States" or live_df.empty else live_df[live_df['state'] == sel_state]
market_opts = ["All Markets"] + (sorted(market_df['market'].unique().tolist()) if not market_df.empty and 'market' in market_df.columns else [])
sel_market = col3.selectbox("Market", market_opts, label_visibility="collapsed")

search_clicked = col4.button("🔍 Search", use_container_width=True, type="primary")

# Render Interactive Table
if not live_df.empty:
    display_df = live_df.copy()
    
    if sel_state != "All States":
        display_df = display_df[display_df['state'] == sel_state]
    if sel_market != "All Markets":
        display_df = display_df[display_df['market'] == sel_market]

    if not display_df.empty:
        # Mocking the mobile app column seen in the reference image
        display_df['Mobile App'] = "Get Free Alert"
        table_view = display_df[['commodity', 'arrival_date', 'variety', 'state', 'district', 'market', 'min_price', 'max_price', 'modal_price', 'Mobile App']].copy()
        
        # Format the price columns as "Rs X / Quintal" to identically match the UI image
        table_view['min_price'] = table_view['min_price'].apply(lambda x: f"Rs {x} / Quintal" if pd.notnull(x) else "N/A")
        table_view['max_price'] = table_view['max_price'].apply(lambda x: f"Rs {x} / Quintal" if pd.notnull(x) else "N/A")
        table_view['modal_price'] = table_view['modal_price'].apply(lambda x: f"Rs {x} / Quintal" if pd.notnull(x) else "N/A")

        # Map to columns specifically requested in the screenshot
        table_view.columns = ['Commodity', 'Arrival Date', 'Variety', 'State', 'District', 'Market', 'Min Price', 'Max Price', 'Avg price', 'Mobile App']
        
        st.dataframe(table_view, hide_index=True, use_container_width=True)
        
        # --- 6. SUMMARY SECTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Determine the location name
        if sel_market != "All Markets":
            loc_name = sel_market
        elif sel_state != "All States":
            loc_name = sel_state
        else:
            loc_name = "India"
            
        comm_name = sel_comm
        
        # Calculate stats for the natural language summary
        max_p = pd.to_numeric(display_df['max_price'], errors='coerce').max()
        min_p = pd.to_numeric(display_df['min_price'], errors='coerce').min()
        avg_p = pd.to_numeric(display_df['modal_price'], errors='coerce').mean()
        
        max_str = f"{int(max_p)} INR per quintal" if pd.notna(max_p) else "N/A"
        min_str = f"{int(min_p)} INR per quintal" if pd.notna(min_p) else "N/A"
        avg_str = f"{int(avg_p)} INR per quintal" if pd.notna(avg_p) else "N/A"

        # Render styled card container matching the user mockup
        with st.container(border=True):
            st.markdown(f"### {comm_name} Market Rates in {loc_name}")
            st.markdown(
                f"<p style='color: #cbd5e1; font-size: 1.05rem;'>In {loc_name}, the highest market price for {comm_name} is <b style='color: white;'>{max_str}</b>, "
                f"while the lowest rate for {comm_name} in {loc_name}, across all varieties is <b style='color: white;'>{min_str}</b>. "
                f"The average selling price for {comm_name} in {loc_name}, considering all its varieties, is <b style='color: white;'>{avg_str}</b>.</p>",
                unsafe_allow_html=True
            )
    else:
        st.info("No mandi data matches your specific search criteria.")
else:
    st.info("Waiting for data feed to populate the table...")

st.markdown("---")
st.caption(f"System status: Operational | Latest Update: {st.session_state.get('last_update', 'N/A')}")