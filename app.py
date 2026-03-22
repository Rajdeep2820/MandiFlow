import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from live_engine import fetch_agmarknet_data

# --- 1. DATA LOADING FUNCTIONS -----

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

def render_main_loading_skeleton(slot):
    slot.markdown(
        """
        <div class="mf-load-wrap">
            <div class="mf-skeleton mf-load-title" style="width: 46%;"></div>
            <div class="mf-skeleton mf-load-subtitle" style="width: 34%;"></div>
            <div class="mf-skeleton mf-load-metric" style="width: 220px;"></div>
            <div class="mf-skeleton mf-load-map"></div>
            <div class="mf-skeleton mf-load-subtitle" style="width: 22%; margin-top: 20px;"></div>
            <div class="mf-load-filters">
                <div class="mf-skeleton mf-load-filter"></div>
                <div class="mf-skeleton mf-load-filter"></div>
                <div class="mf-skeleton mf-load-filter"></div>
                <div class="mf-skeleton mf-load-filter-btn"></div>
            </div>
            <div class="mf-load-table">
                <div class="mf-load-table-head">
                    <div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div>
                    <div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div>
                </div>
                <div class="mf-load-table-row">
                    <div class="mf-skeleton td w1"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
                    <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div>
                </div>
                <div class="mf-load-table-row">
                    <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div>
                    <div class="mf-skeleton td w4"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div>
                </div>
                <div class="mf-load-table-row">
                    <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w3"></div>
                    <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
                </div>
                <div class="mf-load-table-row">
                    <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
                    <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w2"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_loading_skeleton(slot):
    slot.markdown(
        """
        <div class="mf-side-load-wrap">
            <div class="mf-skeleton mf-side-title"></div>
            <div class="mf-skeleton mf-side-control"></div>
            <div class="mf-skeleton mf-side-status"></div>
            <div class="mf-skeleton mf-side-btn"></div>
            <div class="mf-skeleton mf-side-subtitle"></div>
            <div class="mf-skeleton mf-side-textarea"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_final_data(comm, main_loading_slot=None, sidebar_loading_slot=None):
    """Handles session state to prevent infinite refresh loops and API flickering."""
    if 'mandi_data' not in st.session_state or st.session_state.get('last_comm') != comm:
        if main_loading_slot is not None:
            render_main_loading_skeleton(main_loading_slot)
        if sidebar_loading_slot is not None:
            render_sidebar_loading_skeleton(sidebar_loading_slot)

        data, is_live = fetch_agmarknet_data(comm)

        if main_loading_slot is not None:
            main_loading_slot.empty()
        if sidebar_loading_slot is not None:
            sidebar_loading_slot.empty()
        
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
    
    [data-testid="stSidebarHeader"] {
        height: 1.5rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 0 !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
    }
    
    /* Main Content Top Overrides */
    .block-container {
        padding-top: 2rem !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    .pulse-dot {
        display: inline-block; width: 12px; height: 12px; border-radius: 50%;
        animation: pulse 2s infinite; margin-right: 8px;
    }
    @keyframes skeleton-shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .mf-skeleton {
        background: linear-gradient(90deg, #1f2432 25%, #2a3142 37%, #1f2432 63%);
        background-size: 400% 100%;
        animation: skeleton-shimmer 2.8s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        border-radius: 8px;
    }
    .mf-skeleton-map {
        height: 420px;
        border: 1px solid #2e3446;
        border-radius: 12px;
        margin-top: 10px;
    }
    .mf-skeleton-row {
        height: 18px;
        margin: 10px 0;
    }
    .mf-load-wrap { margin-top: 6px; }
    .mf-load-title { height: 30px; margin-bottom: 10px; }
    .mf-load-subtitle { height: 18px; margin-bottom: 10px; }
    .mf-load-metric { height: 62px; margin-bottom: 14px; border-radius: 12px; }
    .mf-load-map {
        height: 680px;
        border-radius: 12px;
        border: 1px solid #2e3446;
        margin-bottom: 18px;
        background-color: #182133;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }
    .mf-load-filters { display: grid; grid-template-columns: 1fr 1fr 1fr 160px; gap: 10px; margin: 12px 0 10px 0; }
    .mf-load-filter { height: 42px; border-radius: 10px; }
    .mf-load-filter-btn { height: 42px; border-radius: 10px; }
    .mf-load-table { margin-top: 8px; border: 1px solid #2e3446; border-radius: 10px; padding: 10px; background: rgba(17, 22, 33, 0.7); }
    .mf-load-table-head, .mf-load-table-row { display: grid; grid-template-columns: repeat(10, minmax(72px, 1fr)); gap: 8px; }
    .mf-load-table-head { margin-bottom: 8px; }
    .mf-load-table-row { margin-bottom: 8px; }
    .mf-load-table-row:last-child { margin-bottom: 0; }
    .mf-load-table .th { height: 14px; border-radius: 6px; opacity: 0.92; }
    .mf-load-table .td { height: 12px; border-radius: 6px; opacity: 0.78; }
    .mf-load-table .w1 { width: 95%; } .mf-load-table .w2 { width: 78%; } .mf-load-table .w3 { width: 64%; } .mf-load-table .w4 { width: 88%; }
    .mf-side-load-wrap { margin: 8px 0 4px 0; }
    .mf-side-title { height: 72px; margin-bottom: 14px; border-radius: 12px; }
    .mf-side-control { height: 46px; margin-bottom: 12px; border-radius: 10px; }
    .mf-side-status { height: 120px; margin-bottom: 12px; border-radius: 12px; }
    .mf-side-btn { height: 40px; margin-bottom: 14px; border-radius: 10px; }
    .mf-side-subtitle { height: 16px; width: 58%; margin-bottom: 10px; border-radius: 8px; }
    .mf-side-textarea { height: 100px; border-radius: 10px; }
    @media (max-width: 900px) {
        .mf-load-filters { grid-template-columns: 1fr 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)

def render_loading_skeleton():
    """Display loading placeholders while live mandi data is unavailable."""
    st.markdown(
        """
        <div class="mf-skeleton mf-load-map" style="height: 420px;"></div>
        <div class="mf-load-table" style="margin-top: 10px;">
            <div class="mf-load-table-head">
                <div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div>
                <div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div>
            </div>
            <div class="mf-load-table-row">
                <div class="mf-skeleton td w1"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
                <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div>
            </div>
            <div class="mf-load-table-row">
                <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div>
                <div class="mf-skeleton td w4"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- 3. SIDEBAR CONTROLS ---
main_loading_slot = st.empty()

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1);'>
            <h1 style='margin-bottom: 5px; color: #2ecc71;'>🌾 MandiFlow</h1>
            <span style='color: #888; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;'>Network Intelligence</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("🕹️ Controls")
    
    # 3.1 Main Commodity Selector
    commodity = st.selectbox("Market Asset", ["Onion", "Potato", "Tomato", "Garlic", "Wheat"])
    sidebar_loading_slot = st.empty()

    # Trigger stable data fetch
    live_df, is_live = get_final_data(
        commodity,
        main_loading_slot=main_loading_slot,
        sidebar_loading_slot=sidebar_loading_slot
    )

    # 3.2 Network Status Widget
    st.markdown("<br>", unsafe_allow_html=True)
    status_color = "#2ecc71" if is_live and not live_df.empty else "#f1c40f"
    status_text = "API LIVE FEED" if is_live and not live_df.empty else "FALLBACK MODE"
    
    st.markdown(f"""
        <div style="padding: 15px; border-radius: 10px; border: 1px solid {status_color}; background: rgba(0,0,0,0.2); margin-bottom: 15px;">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div class="pulse-dot" style="background-color: {status_color}; box-shadow: 0 0 8px {status_color};"></div>
                <strong style="color: {status_color}; font-size: 1.05rem; letter-spacing: 0.5px;">{status_text}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #bbb; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
                <span>Active Nodes:</span>
                <span style="color: white; font-weight: bold;">{len(live_df)} synced</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Sync Network", use_container_width=True, type="secondary"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")

    # 3.3 Simulation & Shock Controls
    st.header("⚡ Simulation Rules")
    st.caption("Input real-world events or news specifically targeting districts (e.g. 'Floods in Nashik') to simulate structural shock in the GCN.")
    news_headline = st.text_area("News / Shock Trigger", "Normal market conditions", height=100)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;'>
            MandiFlow v1.0<br>Spatio-Temporal GCN Engine
        </div>
    """, unsafe_allow_html=True)

# --- 4. MAIN LAYOUT ---
st.title("🌾 MandiFlow: Spatio-Temporal AI Dashboard")
coords_df = load_map_data()

st.subheader(f"📍 {commodity} Network Analysis")
if not live_df.empty:
    st.metric("National Avg", f"₹{pd.to_numeric(live_df['modal_price']).mean():.2f}")
else:
    st.info("Waiting for data stream...")
    render_loading_skeleton()

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
st_folium(m, height=750, returned_objects=[], key="mandi_map", width="stretch")

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

search_clicked = col4.button("🔍 Search", width="stretch", type="primary")

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
        table_view.columns = ['Commodity', 'Arrival Date', 'Variety', 'State', 'District', 'Market', 'Min Price', 'Max Price', 'Modal Price', 'Mobile App']
        
        st.dataframe(table_view, hide_index=True, width="stretch")
        
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
    st.markdown(
        """
        <div class="mf-load-table" style="margin-top: 10px;">
            <div class="mf-load-table-head">
                <div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div>
                <div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div><div class="mf-skeleton th"></div>
            </div>
            <div class="mf-load-table-row">
                <div class="mf-skeleton td w1"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
                <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div>
            </div>
            <div class="mf-load-table-row">
                <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div>
                <div class="mf-skeleton td w4"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div>
            </div>
            <div class="mf-load-table-row">
                <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w3"></div>
                <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
            </div>
            <div class="mf-load-table-row">
                <div class="mf-skeleton td w2"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div>
                <div class="mf-skeleton td w3"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w2"></div><div class="mf-skeleton td w4"></div><div class="mf-skeleton td w2"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(f"System status: Operational | Latest Update: {st.session_state.get('last_update', 'N/A')}")
