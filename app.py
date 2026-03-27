import streamlit as st
import pandas as pd
import folium
import difflib
import math
import requests as _req
import pyarrow.parquet as pq
import os
import json
import secrets
import html
import urllib.error
import urllib.parse
import urllib.request
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from live_engine import fetch_agmarknet_data
# --- 1. DATA LOADING FUNCTIONS -----

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_data_cached(comm):
    """Cache live fetch briefly to keep UI responsive across reruns."""
    return fetch_agmarknet_data(comm)


def _haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


@st.cache_data(ttl=3600, show_spinner=False)
def _geocode(query: str):
    """Geocode an Indian location via Nominatim. Returns (lat, lon) or (None, None)."""
    try:
        resp = _req.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{query}, India", "format": "json", "limit": 1},
            headers={"User-Agent": "MandiFlow/1.0"},
            timeout=6,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None

# Columns that must be present in live_df for the dashboard to render correctly
REQUIRED_COLS = {'market', 'district', 'state', 'commodity', 'variety',
                 'modal_price', 'min_price', 'max_price', 'arrival_date'}

# Full ranked list of all 57 commodities (by all-time trade volume, descending)
ALL_RANKED_COMMODITIES = [
    "Paddy (Dhan)(Common)", "Wheat", "Potato", "Onion", "Tomato", "Brinjal", "Green Chilli",
    "Rice", "Banana", "Cauliflower", "Bhindi (Ladies Finger)", "Mustard",
    "Cabbage", "Maize", "Bengal Gram (Gram)(Whole)", "Cucumbar (Kheera)",
    "Bottle gourd", "Apple", "Soyabean", "Bitter gourd", "Pumpkin",
    "Carrot", "Arhar (Tur/Red Gram)(Whole)", "Cotton", "Raddish",
    "Black Gram (Urd Beans)(Whole)", "Ginger (Green)", "Bajra (Pearl Millet/Cumbu)",
    "Gur (Jaggery)", "Jowar (Sorghum)", "Garlic", "Moong (Whole)", "Groundnut",
    "Peas Wet", "Spinach", "Methi (Fenugreek)", "Lemon", "Sweet Potato",
    "Coriander (Leaves)", "Drumstick", "Field Pea", "Capsicum",
    "Grapes", "Mango", "Pomegranate", "Watermelon", "Orange",
    "Guava", "Papaya", "Jackfruit", "Coconut", "Sesamum (Sesame/Til)",
    "Sugarcane", "Turmeric", "Dry Chillies", "Coriander Seed", "Sunflower"
]

@st.cache_data(ttl=3600, show_spinner=False)
def get_active_prime_commodities():
    """
    Dynamically determines which commodities are 'active' (traded within last 7 days).
    Returns (prime_list, others_list) where prime_list has exactly 7 entries,
    all guaranteed to have recent data. Stale commodities are skipped and the next
    active one in rank order fills the slot.
    """
    try:
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
        table = pq.read_table(
            "mandi_master_data.parquet",
            columns=["Commodity"],
            filters=[("Arrival_Date", ">=", cutoff)]
        )
        active_set = set(table.column("Commodity").to_pylist())
    except Exception:
        # Parquet unavailable — fall back to full list, no filtering
        active_set = set(ALL_RANKED_COMMODITIES)

    prime  = [c for c in ALL_RANKED_COMMODITIES if c in active_set][:7]
    # Fill up to 7 if fewer than 7 active (edge case)
    if len(prime) < 7:
        prime = ALL_RANKED_COMMODITIES[:7]
    prime_set = set(prime)
    others = [c for c in ALL_RANKED_COMMODITIES if c not in prime_set]
    return prime, others

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
    # Invalidate cache if: commodity changed, no data yet, or required columns are missing
    cached_data = st.session_state.get('mandi_data', pd.DataFrame())
    has_all_cols = REQUIRED_COLS.issubset(set(cached_data.columns)) if not cached_data.empty else False
    needs_refresh = (
        'mandi_data' not in st.session_state
        or st.session_state.get('last_comm') != comm
        or not has_all_cols
    )

    if needs_refresh:
        if main_loading_slot is not None:
            render_main_loading_skeleton(main_loading_slot)
        if sidebar_loading_slot is not None:
            render_sidebar_loading_skeleton(sidebar_loading_slot)

        data, is_live = fetch_live_data_cached(comm)

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


def get_firebase_api_key():
    """Read Firebase Web API key from Streamlit secrets or env variable."""
    try:
        if "firebase" in st.secrets and "api_key" in st.secrets["firebase"]:
            return st.secrets["firebase"]["api_key"]
    except Exception:
        pass
    return os.getenv("FIREBASE_API_KEY", "").strip()


def get_google_oauth_config():
    """Read Google OAuth settings from Streamlit secrets or env variables."""
    client_id = ""
    client_secret = ""
    redirect_uri = ""
    try:
        if "google_oauth" in st.secrets:
            cfg = st.secrets["google_oauth"]
            client_id = str(cfg.get("client_id", "")).strip()
            client_secret = str(cfg.get("client_secret", "")).strip()
            redirect_uri = str(cfg.get("redirect_uri", "")).strip()
    except Exception:
        pass

    client_id = client_id or os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = client_secret or os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    redirect_uri = redirect_uri or os.getenv("GOOGLE_REDIRECT_URI", "").strip()

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }


def get_query_param(name):
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def set_query_param(name, value):
    if value is None:
        st.query_params.pop(name, None)
    else:
        st.query_params[name] = str(value)


def clear_auth_query_params():
    for key in ["rt", "code", "state", "scope", "authuser", "prompt"]:
        st.query_params.pop(key, None)


def parse_firebase_error(error_code):
    """Map Firebase auth error codes to user-friendly messages."""
    message_map = {
        "EMAIL_EXISTS": "This email is already registered. Please sign in.",
        "OPERATION_NOT_ALLOWED": "Email/password sign-in is not enabled in Firebase.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
        "EMAIL_NOT_FOUND": "No account found with this email.",
        "INVALID_PASSWORD": "Incorrect password.",
        "USER_DISABLED": "This account has been disabled by an administrator.",
        "INVALID_EMAIL": "Please enter a valid email address.",
        "WEAK_PASSWORD : Password should be at least 6 characters": "Password must be at least 6 characters long.",
        "WEAK_PASSWORD": "Password must be at least 6 characters long.",
    }
    return message_map.get(error_code, f"Authentication failed: {error_code}")


def firebase_auth_request(endpoint, payload):
    """Call Firebase Identity Toolkit endpoint."""
    api_key = get_firebase_api_key()
    if not api_key:
        return None, "Firebase API key is missing."

    url = f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={api_key}"
    payload_bytes = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as http_error:
        raw_error = http_error.read().decode("utf-8")
        try:
            data = json.loads(raw_error)
            error_code = data.get("error", {}).get("message", "UNKNOWN_ERROR")
        except ValueError:
            error_code = "UNKNOWN_ERROR"
        return None, parse_firebase_error(error_code)
    except urllib.error.URLError:
        return None, "Unable to reach Firebase. Check your internet connection."
    except ValueError:
        return None, "Firebase returned an invalid response."
    return data, None


def login_with_firebase(email, password):
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return firebase_auth_request("accounts:signInWithPassword", payload)


def signup_with_firebase(email, password):
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return firebase_auth_request("accounts:signUp", payload)


def refresh_firebase_session(refresh_token):
    api_key = get_firebase_api_key()
    if not api_key:
        return None, "Firebase API key is missing."

    url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
    post_data = urllib.parse.urlencode(
        {"grant_type": "refresh_token", "refresh_token": refresh_token}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=post_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError:
        return None, "Session expired. Please sign in again."
    except urllib.error.URLError:
        return None, "Unable to refresh your session right now."
    except ValueError:
        return None, "Invalid session response."

    return data, None


def get_google_auth_url():
    cfg = get_google_oauth_config()
    if not cfg["client_id"] or not cfg["redirect_uri"]:
        return None

    state = secrets.token_urlsafe(24)
    st.session_state.google_oauth_state = state
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
        "state": state,
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def exchange_google_code_for_token(code):
    cfg = get_google_oauth_config()
    if not cfg["client_id"] or not cfg["client_secret"] or not cfg["redirect_uri"]:
        return None, "Google OAuth is not configured."

    payload = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": cfg["redirect_uri"],
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError:
        return None, "Google sign-in failed while exchanging token."
    except urllib.error.URLError:
        return None, "Unable to reach Google sign-in service."
    except ValueError:
        return None, "Invalid token response from Google."

    id_token = data.get("id_token")
    if not id_token:
        return None, "Google did not return an ID token."
    return id_token, None


def login_with_google(id_token):
    cfg = get_google_oauth_config()
    request_uri = cfg["redirect_uri"] or "http://localhost"
    payload = {
        "postBody": f"id_token={id_token}&providerId=google.com",
        "requestUri": request_uri,
        "returnSecureToken": True,
        "returnIdpCredential": True,
    }
    return firebase_auth_request("accounts:signInWithIdp", payload)


def build_auth_user(auth_data, fallback_email=""):
    return {
        "email": auth_data.get("email", fallback_email),
        "local_id": auth_data.get("localId", auth_data.get("user_id", "")),
        "id_token": auth_data.get("idToken", auth_data.get("id_token", "")),
        "refresh_token": auth_data.get("refreshToken", auth_data.get("refresh_token", "")),
    }


def save_authenticated_user(auth_data, fallback_email=""):
    user = build_auth_user(auth_data, fallback_email=fallback_email)
    st.session_state.auth_user = user
    if user.get("refresh_token"):
        set_query_param("rt", user["refresh_token"])
    return user


def restore_auth_session_from_query():
    if st.session_state.get("auth_user"):
        return True

    refresh_token = get_query_param("rt")
    if not refresh_token:
        return False

    refreshed, error = refresh_firebase_session(refresh_token)
    if error or not refreshed:
        clear_auth_query_params()
        return False

    lookup_data, _ = firebase_auth_request(
        "accounts:lookup", {"idToken": refreshed.get("id_token", "")}
    )
    email = ""
    if lookup_data and lookup_data.get("users"):
        email = lookup_data["users"][0].get("email", "")

    save_authenticated_user(refreshed, fallback_email=email)
    return True


def logout_user():
    for key in ["auth_user", "google_oauth_state", "mandi_data", "is_live", "last_comm", "last_update"]:
        if key in st.session_state:
            del st.session_state[key]
    clear_auth_query_params()
    fetch_live_data_cached.clear()


def require_authentication():
    if restore_auth_session_from_query():
        return

    if st.session_state.get("auth_user"):
        return

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(-45deg, #01020a, #020617, #040b1d, #07122b);
            background-size: 400% 400%;
            animation: gradientMove 12s ease infinite;
            position: relative;
            overflow: hidden;
        }
        [data-testid="stHeader"],
        [data-testid="stDecoration"] {
            background: transparent !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-left: 4vw;
            padding-right: 4vw;
            box-sizing: border-box;
        }
        [data-testid="stAppViewContainer"] > .main .block-container {
            width: 50% !important;
            max-width: 860px !important;
            min-width: 320px !important;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 16px;
            padding: 28px 32px 34px 32px !important;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(34, 211, 238, 0.3);
            box-shadow: 0 0 40px rgba(0, 0, 0, 0.6);
            margin: 0 auto !important;
        }
        .auth-bg-blob {
            position: fixed;
            width: 300px;
            height: 300px;
            border-radius: 999px;
            filter: blur(120px);
            z-index: 0;
            pointer-events: none;
            animation: blobFloat 10s infinite alternate ease-in-out;
        }
        .auth-bg-blob.blob-a {
            top: -80px;
            left: -100px;
            background: rgba(30, 58, 138, 0.28);
        }
        .auth-bg-blob.blob-b {
            right: -80px;
            top: 25%;
            background: rgba(37, 99, 235, 0.18);
            animation-duration: 12s;
        }
        .auth-bg-blob.blob-c {
            bottom: -110px;
            left: 40%;
            background: rgba(15, 23, 42, 0.4);
            animation-duration: 14s;
        }
        .auth-title {
            margin: 0;
            color: #ffffff;
            font-size: 2rem;
            letter-spacing: 0.01em;
            text-align: center;
        }
        .auth-subtitle {
            margin: 2px 0 2px 0;
            color: #94a3b8;
            text-align: center;
        }
        [data-testid="stTabs"] {
            width: 50vw;
            max-width: 50vw;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-baseweb="tab-list"] {
            gap: 10px;
            margin: 8px 0 16px 0;
        }
        div[data-baseweb="tab-list"] button[role="tab"] {
            border-radius: 12px !important;
            color: #ffffff !important;
            border: 1px solid rgba(71, 85, 105, 0.55) !important;
            background: rgba(30, 41, 59, 0.92) !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="tab-list"] button[role="tab"]:hover {
            transform: scale(1.05);
            box-shadow: none !important;
            filter: none;
            background: rgba(15, 23, 42, 0.82) !important;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: rgba(30, 41, 59, 0.98) !important;
            border: 1px solid rgba(56, 189, 248, 0.5) !important;
            filter: none;
            box-shadow: 0 0 8px rgba(14, 165, 233, 0.2);
        }
        .stTextInput > label p {
            color: #cbd5e1 !important;
            font-weight: 500;
        }
        .stTextInput div[data-baseweb="input"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        .stTextInput div[data-baseweb="input"]:focus-within {
            border: none !important;
            box-shadow: none !important;
        }
        .stTextInput input {
            width: 100%;
            padding: 14px;
            border-radius: 12px !important;
            background: rgba(30, 41, 59, 0.6) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            color: #ffffff !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput input:focus {
            outline: none !important;
            border-color: #22d3ee !important;
            box-shadow: 0 0 10px #22d3ee !important;
        }
        .stForm [data-testid="stFormSubmitButton"] button,
        .stButton > button {
            background: linear-gradient(90deg, #0284c7, #0891b2) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            padding: 14px !important;
            border: none !important;
            transition: all 0.3s ease !important;
        }
        .stForm [data-testid="stFormSubmitButton"] button:hover,
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: none !important;
            background: linear-gradient(90deg, #0ea5e9, #22d3ee) !important;
        }
        .divider {
            text-align: center;
            color: #94a3b8;
            margin: 20px 0;
        }
        .google-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            width: 100%;
            background: #e5ebf3;
            color: #444444 !important;
            border-radius: 10px;
            padding: 12px 14px;
            text-decoration: none !important;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .google-btn img {
            width: 18px;
            height: 18px;
        }
        .google-btn:hover {
            background: #ffffff;
            box-shadow: none;
            transform: translateY(-1px);
        }
        .google-config-note {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes blobFloat {
            from { transform: translate3d(0, 0, 0) scale(1); }
            to { transform: translate3d(40px, -35px, 0) scale(1.12); }
        }
        @media (max-width: 900px) {
            [data-testid="stAppViewContainer"] > .main {
                padding-left: 0;
                padding-right: 0;
            }
            [data-testid="stAppViewContainer"] > .main .block-container {
                width: 90% !important;
                min-width: 0 !important;
                padding: 22px 18px 26px 18px !important;
                border-radius: 18px;
            }
            [data-testid="stTabs"] {
                width: 90%;
                max-width: 90%;
            }
            .auth-title { font-size: 1.6rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="auth-bg-blob blob-a"></div>
        <div class="auth-bg-blob blob-b"></div>
        <div class="auth-bg-blob blob-c"></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h1 class="auth-title">MandiFlow Authentication</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="auth-subtitle">Sign in or create an account to access your premium dashboard.</p>',
        unsafe_allow_html=True,
    )

    if not get_firebase_api_key():
        st.error("Firebase API key is not configured.")
        st.info("Add it in `.streamlit/secrets.toml` or environment variable `FIREBASE_API_KEY`.")
        st.code("[firebase]\napi_key = \"YOUR_FIREBASE_WEB_API_KEY\"")
        st.stop()

    auth_code = get_query_param("code")
    oauth_state = get_query_param("state")
    if auth_code:
        saved_state = st.session_state.get("google_oauth_state")
        if saved_state and oauth_state != saved_state:
            st.error("Google sign-in state mismatch. Please try again.")
            clear_auth_query_params()
        else:
            id_token, google_error = exchange_google_code_for_token(auth_code)
            if google_error:
                st.error(google_error)
                clear_auth_query_params()
            else:
                auth_data, firebase_error = login_with_google(id_token)
                if firebase_error:
                    st.error(firebase_error)
                    clear_auth_query_params()
                else:
                    save_authenticated_user(auth_data, fallback_email=auth_data.get("email", ""))
                    clear_auth_query_params()
                    set_query_param("rt", st.session_state.auth_user.get("refresh_token", ""))
                    st.success("Signed in with Google.")
                    st.rerun()

    google_cfg = get_google_oauth_config()
    google_ready = bool(
        google_cfg["client_id"] and google_cfg["client_secret"] and google_cfg["redirect_uri"]
    )

    login_tab, signup_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Sign In", width="stretch")

        if login_submit:
            if not login_email or not login_password:
                st.warning("Please enter both email and password.")
            else:
                auth_data, error = login_with_firebase(login_email.strip(), login_password)
                if error:
                    st.error(error)
                else:
                    save_authenticated_user(auth_data, fallback_email=login_email.strip())
                    st.success("Login successful.")
                    st.rerun()

        st.markdown('<div class="divider">Or</div>', unsafe_allow_html=True)
        if google_ready:
            google_url = get_google_auth_url()
            if google_url:
                escaped_url = html.escape(google_url, quote=True)
                st.markdown(
                    f"""
                    <a class="google-btn" href="{escaped_url}" target="_self">
                        <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" alt="Google logo" />
                        Continue with Google
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <p class="google-config-note">
                    Google sign-in is not configured yet. Add
                    <code>google_oauth.client_id</code>,
                    <code>google_oauth.client_secret</code>, and
                    <code>google_oauth.redirect_uri</code> in Streamlit secrets.
                </p>
                """,
                unsafe_allow_html=True,
            )

    with signup_tab:
        with st.form("signup_form"):
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            signup_submit = st.form_submit_button("Create Account", use_container_width=True)

        if signup_submit:
            if not signup_email or not signup_password or not signup_confirm_password:
                st.warning("Please fill all fields.")
            elif signup_password != signup_confirm_password:
                st.error("Passwords do not match.")
            else:
                auth_data, error = signup_with_firebase(signup_email.strip(), signup_password)
                if error:
                    st.error(error)
                else:
                    save_authenticated_user(auth_data, fallback_email=signup_email.strip())
                    st.success("Account created successfully.")
                    st.rerun()

    st.stop()

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

require_authentication()

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
main_loading_slot = None
sidebar_loading_slot = None
sidebar_status_slot = None

with st.sidebar:
    auth_user = st.session_state.get("auth_user", {})
    st.caption(f"Signed in as: {auth_user.get('email', 'unknown')}")
    if st.button("Logout", use_container_width=True, type="secondary"):
        logout_user()
        st.rerun()
    st.markdown("---")

    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1);'>
            <h1 style='margin-bottom: 5px; color: #2ecc71;'>🌾 MandiFlow</h1>
            <span style='color: #888; font-size: 0.9rem; letter-spacing: 1px; text-transform: uppercase;'>Network Intelligence</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.header("🕹️ Controls")
    
    # 3.1 Main Commodity Selector — dynamically filtered by recency
    # prime = top 7 with trades in last 7 days; stale ones are automatically replaced
    _prime, _others = get_active_prime_commodities()
    prime_display = [f"⭐ {c}" for c in _prime]
    all_options   = prime_display + _others

    # Default to ⭐ Onion if active, otherwise first prime
    _onion_display = "⭐ Onion"
    default_idx = all_options.index(_onion_display) if _onion_display in all_options else 0

    selected_display = st.selectbox(
        "Market Asset",
        options=all_options,
        index=default_idx,
        help="⭐ = Active prime (traded in last 7 days). Stale commodities are auto-demoted."
    )
    # Strip the ⭐ prefix before using the value anywhere
    commodity = selected_display.replace("⭐ ", "").strip()
    sidebar_loading_slot = st.empty()

    # 3.2 Network Status Widget
    st.markdown("<br>", unsafe_allow_html=True)
    sidebar_status_slot = st.empty()
    sidebar_status_slot.markdown(
        """
        <div style="padding: 15px; border-radius: 10px; border: 1px solid #3e4250; background: rgba(0,0,0,0.2); margin-bottom: 15px;">
            <div class="mf-skeleton" style="height: 20px; width: 62%; margin-bottom: 12px;"></div>
            <div class="mf-skeleton" style="height: 14px; width: 100%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if st.button("🔄 Sync Network", use_container_width=True, type="secondary"):
        fetch_live_data_cached.clear()
        for key in ["mandi_data", "is_live", "last_comm", "last_update"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("---")

    # 3.3 Simulation & Shock Controls
    st.header("⚡ Phase II Forecast Engine")
    st.caption("Enter news or upload policy docs to simulate structural shock in the GCN-LSTM.")
    news_headline = st.text_area("News / Shock Trigger", "Normal market conditions", height=80)
    uploaded_doc = st.file_uploader("Upload Policy Doc (PDF/TXT)", type=["pdf", "txt", "docx"])
    
    predict_btn = st.button("🚀 Predict Impact (1-7 Days)", use_container_width=True, type="primary")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;'>
            MandiFlow v1.0<br>Spatio-Temporal GCN Engine
        </div>
    """, unsafe_allow_html=True)

# --- 4. MAIN LAYOUT ---
st.title("🌾 MandiFlow: Spatio-Temporal AI Dashboard")
coords_df = load_map_data()

# --- STATE CENTRE COORDINATES for zoom-to-state search ---
STATE_CENTRES = {
    "andhra pradesh":     (15.9129, 79.7400, 7),
    "arunachal pradesh":  (27.1004, 93.6166, 7),
    "assam":              (26.2006, 92.9376, 7),
    "bihar":              (25.0961, 85.3131, 7),
    "chhattisgarh":       (21.2787, 81.8661, 7),
    "goa":                (15.2993, 74.1240, 9),
    "gujarat":            (22.2587, 71.1924, 7),
    "haryana":            (29.0588, 76.0856, 7),
    "himachal pradesh":   (31.1048, 77.1734, 7),
    "jharkhand":          (23.6102, 85.2799, 7),
    "karnataka":          (15.3173, 75.7139, 7),
    "kerala":             (10.8505, 76.2711, 7),
    "madhya pradesh":     (22.9734, 78.6569, 7),
    "maharashtra":        (19.7515, 75.7139, 7),
    "manipur":            (24.6637, 93.9063, 8),
    "meghalaya":          (25.4670, 91.3662, 8),
    "mizoram":            (23.1645, 92.9376, 8),
    "nagaland":           (26.1584, 94.5624, 8),
    "odisha":             (20.9517, 85.0985, 7),
    "punjab":             (31.1471, 75.3412, 7),
    "rajasthan":          (27.0238, 74.2179, 6),
    "sikkim":             (27.5330, 88.5122, 9),
    "tamil nadu":         (11.1271, 78.6569, 7),
    "telangana":          (18.1124, 79.0193, 7),
    "tripura":            (23.9408, 91.9882, 8),
    "uttar pradesh":      (26.8467, 80.9462, 6),
    "uttarakhand":        (30.0668, 79.0193, 7),
    "west bengal":        (22.9868, 87.8550, 7),
    "delhi":              (28.7041, 77.1025, 10),
    "jammu and kashmir":  (33.7782, 76.5762, 7),
    "ladakh":             (34.1526, 77.5770, 7),
}

st.subheader(f"📍 {commodity} Network Analysis")
main_loading_slot = st.empty()
live_df, is_live = get_final_data(
    commodity,
    main_loading_slot=main_loading_slot,
    sidebar_loading_slot=sidebar_loading_slot
)

status_color = "#2ecc71" if is_live and not live_df.empty else "#f1c40f"
status_text = "API LIVE FEED" if is_live and not live_df.empty else "FALLBACK MODE"
if sidebar_status_slot is not None:
    sidebar_status_slot.markdown(f"""
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

if not live_df.empty:
    st.metric("National Avg", f"₹{pd.to_numeric(live_df['modal_price']).mean():.2f}")
else:
    st.info("Waiting for data stream...")
    render_loading_skeleton()

# --- MAP SEARCH BAR ---
# Counter-key pattern: changing the key forces Streamlit to create a new empty widget
if "search_counter" not in st.session_state:
    st.session_state.search_counter = 0

search_col, clear_col = st.columns([5, 1])
map_search = search_col.text_input(
    "🔍 Search map",
    placeholder="Type a state (e.g. Maharashtra) or mandi name (e.g. Nashik)...",
    label_visibility="collapsed",
    key=f"map_search_input_{st.session_state.search_counter}"
)
if clear_col.button("✕ Clear", use_container_width=True):
    st.session_state.search_counter += 1
    st.rerun()

# --- RESOLVE SEARCH ---
map_center   = [22.9734, 78.6569]  # Default: India centre
map_zoom     = 5
flagged_row  = None   # The mandi row to pin a red flag on
search_msg   = ""

if map_search.strip():
    query = map_search.strip().lower()
    mandi_found = False

    if not coords_df.empty:
        # 1. Strict Substring Match (e.g. "jaipur" explicitly finds "Jaipur (F&V)")
        mandi_names = coords_df['Market'].str.lower().tolist()
        import re
        exact_word_matches = [m for m in mandi_names if re.search(rf"\b{re.escape(query)}\b", m)]
        exact_matches = [m for m in mandi_names if query in m]
        
        if exact_word_matches or exact_matches:
            # Prefer word boundaries first, then favor strings that *start* with the query
            pool = exact_word_matches if exact_word_matches else exact_matches
            best_match = min(pool, key=lambda x: (not x.startswith(query), len(x)))
            flagged_row = coords_df[coords_df['Market'].str.lower() == best_match].iloc[0]
            map_center  = [flagged_row['latitude'], flagged_row['longitude']]
            map_zoom    = 10
            search_msg  = f"🚩 Found mandi: **{flagged_row['Market']}**, {flagged_row['District']}"
            mandi_found = True
        else:
            # 2. Strict spelling-mistake fallback directly on Mandis (high similarity)
            mandi_match = difflib.get_close_matches(query, mandi_names, n=1, cutoff=0.8)
            if mandi_match:
                flagged_row = coords_df[coords_df['Market'].str.lower() == mandi_match[0]].iloc[0]
                map_center  = [flagged_row['latitude'], flagged_row['longitude']]
                map_zoom    = 10
                search_msg  = f"🚩 Found mandi: **{flagged_row['Market']}**, {flagged_row['District']}"
                mandi_found = True

    if not mandi_found:
        # 3. Geocode fallback — locate the city/town geographically, then explicitly flag the nearest Mandi
        geo_lat, geo_lon = _geocode(map_search.strip())
        if geo_lat is not None and not coords_df.empty:
            tmp = coords_df.copy()
            tmp['_dist_km'] = tmp.apply(
                lambda r: _haversine(geo_lat, geo_lon, r['latitude'], r['longitude']), axis=1
            )
            nearest = tmp.nsmallest(3, '_dist_km')
            flagged_row = nearest.iloc[0]
            map_center  = [flagged_row['latitude'], flagged_row['longitude']]
            map_zoom    = 10
            near_list   = ", ".join(f"{r['Market']} (~{r['_dist_km']:.0f} km)" for _, r in nearest.iterrows())
            search_msg = (
                f"📍 No mandi perfectly matched **'{map_search}'**. "
                f"**{flagged_row['Market']}** is the closest to your search! ({near_list})"
            )
        else:
            search_msg = f"❌ Geographic location not found for **'{map_search}'** — try a different region."

if search_msg:
    st.markdown(search_msg)

m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="CartoDB dark_matter")
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

# --- RED FLAG MARKER for searched mandi (outside cluster so always visible) ---
if flagged_row is not None:
    price_val = None
    matched_market_name = flagged_row['Market']
    if not live_df.empty:
        flag_key = str(flagged_row['Market']).upper().strip()
        # Stage 1: Exact market_key match
        price_val = price_map.get(flag_key)
        # Stage 2: Partial match — e.g. 'SANWER' matches 'SANWER APMC'
        if price_val is None:
            for k, v in price_map.items():
                if flag_key in k or k in flag_key:
                    price_val = v
                    matched_market_name = k.title()
                    break
        # Stage 3: District fallback — any market in same district
        if price_val is None and 'district' in live_df.columns:
            dist_df = live_df[live_df['district'].str.upper() == str(flagged_row['District']).upper()]
            if not dist_df.empty:
                price_val = dist_df['modal_price'].iloc[0]
                matched_market_name = dist_df['market'].iloc[0].title() + " (nearby)"
    flag_price = f"₹{price_val}/qtl" if price_val else "No price data for this commodity"
    flag_icon = folium.DivIcon(
        html=f"""
            <div style="
                font-size: 28px;
                line-height: 1;
                filter: drop-shadow(0 0 6px #e74c3c);
                animation: flagPulse 1s ease-in-out infinite alternate;
            ">🚩</div>
            <style>
                @keyframes flagPulse {{
                    from {{ transform: scale(1);   filter: drop-shadow(0 0 4px #e74c3c); }}
                    to   {{ transform: scale(1.3); filter: drop-shadow(0 0 12px #e74c3c); }}
                }}
            </style>
        """,
        icon_size=(35, 35),
        icon_anchor=(4, 34),
    )
    folium.Marker(
        location=[flagged_row['latitude'], flagged_row['longitude']],
        icon=flag_icon,
        tooltip=folium.Tooltip(
            f"<b>🚩 {flagged_row['Market']}</b><br>"
            f"Matched: {matched_market_name}<br>"
            f"District: {flagged_row['District']}<br>"
            f"Price: {flag_price}",
            sticky=True
        ),
        popup=folium.Popup(
            f"<b>{flagged_row['Market']}</b><br>{flagged_row['District']}<br>{flag_price}",
            max_width=200
        )
    ).add_to(m)

# CRITICAL: returned_objects=[] prevents the map from causing reruns on zoom/move
st_folium(m, height=750, returned_objects=[], key=f"mandi_map_{map_search}", width="stretch")

st.markdown("---")

if 'predict_btn' in locals() and predict_btn:
    st.markdown("### 📈 AI Price Forecasts (Zero-Shot NLP + GCN-LSTM)")
    with st.spinner("Extracting shock features & running spatio-temporal simulation math..."):
        from simulator import simulate_shock, NewsAnalyzer
        from document_processor import DocumentProcessor
        import altair as alt

        # Dynamically inject the API key into the analyzer before simulation
        analyzer = NewsAnalyzer(api_key=st.session_state.get('GEMINI_API_KEY', ''))
        
        doc_text = ""
        if uploaded_doc is not None:
            processor = DocumentProcessor()
            doc_chunks = processor.process_document(uploaded_doc, is_pdf=uploaded_doc.name.endswith('.pdf'))
            if isinstance(doc_chunks, list): doc_text = " ".join(doc_chunks)
            
        result = simulate_shock(news_headline, doc_text, commodity=commodity)
        
        with st.expander("🔍 Extracted Shock Features (JSON)"):
            st.json(result["features"])
        
        if result.get("resolution_error"):
            st.error(f"Node not found in historical data: {result['resolution_error']}")
            st.stop()
        
        st.success("Simulation Complete")
        cols = st.columns(2)
        
        # 1. Target node chart
        with cols[0]:
            st.markdown(f"**Origin Impact:** {result['origin_name']}")
            days = [1, 3, 5, 7]
            df_origin = pd.DataFrame({"Days": days, "Predicted Price (₹/q)": result['origin_forecast']})
            chart_origin = alt.Chart(df_origin).mark_line(point=True, color="#ff4b4b").encode(
                x=alt.X("Days:O", title="Days Ahead"),
                y=alt.Y("Predicted Price (₹/q):Q", scale=alt.Scale(zero=False), title="Modal Price")
            ).properties(height=280)
            st.altair_chart(chart_origin, use_container_width=True)
            
        # 2. Ripples chart
        with cols[1]:
            if len(result['served_areas']) > 0:
                # Combine all served area forecasts into one dataframe for a multi-line chart
                frames = []
                days = [1, 3, 5, 7]
                for served in result['served_areas']:
                    df_temp = pd.DataFrame({
                        "Days": days, 
                        "Predicted Price (₹/q)": served['forecast'],
                        "Mandi": served['mandi']
                    })
                    frames.append(df_temp)
                
                df_served = pd.concat(frames)
                st.markdown(f"**Ripple Effect (Served Areas):** {len(frames)} connected nodes")
                
                chart_served = alt.Chart(df_served).mark_line(point=True).encode(
                    x=alt.X("Days:O", title="Days Ahead"),
                    y=alt.Y("Predicted Price (₹/q):Q", scale=alt.Scale(zero=False), title="Modal Price"),
                    color=alt.Color("Mandi:N", legend=alt.Legend(title="Served Area", orient="bottom"))
                ).properties(height=280)
                
                # 🚨 THE FIX: Streamlit's new responsive width syntax
                st.altair_chart(chart_served, width="stretch")
            else:
                st.info("No highly correlated 'Served Areas' found in the dataset for this origin.")
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
