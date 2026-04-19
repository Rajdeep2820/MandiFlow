"""
news_analyzer.py  —  MandiFlow v2.0
======================================
Extracts structured shock parameters from unstructured news text.

Two modes:
  1. Gemini API  — if GEMINI_API_KEY is set in environment or Streamlit secrets
  2. Heuristic   — keyword-based fallback, no API required

Key fixes from v1:
  - Heuristic multiplier was hardcoded at 1.3 for all climatic/logistics
    events. Now uses severity keywords (e.g. "severe", "catastrophic")
    to scale the multiplier appropriately.
  - Always returned ["Onion"] as affected commodity regardless of input.
    Now scans the text for all known commodity names.
  - Schema was missing origin_district, which simulator.py needs for
    the 4-tier node resolution.
  - Gemini prompt now explicitly requests origin_district in the JSON.
  - Added PDF text extraction support for uploaded policy documents.
"""

import json
import os
import re


# ---------------------------------------------------------------------------
# COMMODITY VOCABULARY
# All Tier 1 + Tier 2 commodities MandiFlow handles
# ---------------------------------------------------------------------------

KNOWN_COMMODITIES = {
    "onion":    "Onion",
    "tomato":   "Tomato",
    "potato":   "Potato",
    "wheat":    "Wheat",
    "garlic":   "Garlic",
    "maize":    "Maize",
    "paddy":    "Paddy",
    "rice":     "Rice",
    "mustard":  "Mustard",
    "chilli":   "Chilli",
    "soybean":  "Soybean",
    "cotton":   "Cotton",
    "sugarcane": "Sugarcane",
    "groundnut": "Groundnut",
}

# ---------------------------------------------------------------------------
# SEVERITY MODIFIERS
# ---------------------------------------------------------------------------

# Upward severity keywords and their multiplier boost on top of base
SEVERITY_BOOST = {
    "catastrophic": 0.30,
    "devastating":  0.25,
    "severe":       0.20,
    "major":        0.15,
    "significant":  0.10,
    "slight":      -0.10,
    "minor":       -0.10,
    "partial":     -0.05,
}

# Base multipliers per shock type
BASE_MULTIPLIERS = {
    "climatic":   1.20,   # supply destruction → price up
    "logistics":  1.15,   # transport disruption → price up
    "policy_up":  1.20,   # MSP hike, procurement drive → price up
    "policy_down": 0.80,  # export ban, import duty reduction → price down
    "demand":     1.05,   # festival season, stockpiling
}

# ---------------------------------------------------------------------------
# GEOGRAPHIC VOCABULARY
# Extended alias map used by the heuristic extractor
# ---------------------------------------------------------------------------

MANDI_ALIASES = [
    # (keyword_in_text,    mandi_name_in_index,    district_name)
    ("lasalgaon (niphad)", "LASALGAON (NIPHAD)",   "Nashik"),
    ("lasalgaon (vinchur)","LASALGAON (VINCHUR)",   "Nashik"),
    ("lasalgaon",          "LASALGAON",             "Nashik"),
    ("niphad",             "LASALGAON (NIPHAD)",    "Nashik"),
    ("vinchur",            "LASALGAON (VINCHUR)",   "Nashik"),
    ("nashik",             "NASIK",                 "Nashik"),
    ("nasik",              "NASIK",                 "Nashik"),
    ("azadpur",            "AZADPUR",               "Delhi"),
    ("delhi",              "AZADPUR",               "Delhi"),
    ("new delhi",          "AZADPUR",               "Delhi"),
    ("bangalore",          "BANGALORE",             "Bengaluru Urban"),
    ("bengaluru",          "BANGALORE",             "Bengaluru Urban"),
    ("bengalore",          "BANGALORE",             "Bengaluru Urban"),
    ("gurgaon",            "GURGAON",               "Gurugram"),
    ("gurugram",           "GURGAON",               "Gurugram"),
    ("mandsaur",           "MANDSAUR",              "Mandsaur"),
    ("indore",             "INDORE",                "Indore"),
    ("nagpur",             "NAGPUR",                "Nagpur"),
    ("pune",               "PUNE",                  "Pune"),
    ("mumbai",             "MUMBAI",                "Mumbai"),
    ("agra",               "AGRA",                  "Agra"),
    ("lucknow",            "LUCKNOW",               "Lucknow"),
    ("kanpur",             "KANPUR (GRAIN)",         "Kanpur"),
    ("surat",              "SURAT",                 "Surat"),
    ("ahmedabad",          "AHMEDABAD",             "Ahmedabad"),
    ("kolkata",            "BARA BAZAR (POSTA BAZAR)", "Kolkata"),
    ("howrah",             "BARA BAZAR (POSTA BAZAR)", "Kolkata"),
    ("hyderabad",          "HYDERABAD (F&V)",        "Hyderabad"),
    ("kolar",              "KOLAR",                 "Kolar"),
    ("harda",              "HARDA",                 "Harda"),
    ("sikar",              "SIKAR",                 "Sikar"),
    ("alwar",              "ALWAR (F&V)",            "Alwar"),
    ("jaipur",             "JAIPUR (F&V)",           "Jaipur"),
    ("jodhpur",            "JODHPUR (F&V)",          "Jodhpur"),
    ("bikaner",            "BIKANER (F&V)",          "Bikaner"),
    ("fatehabad",          "FATEHABAD",             "Fatehabad"),
    ("karnal",             "NEW GRAIN MARKET (MAIN), KARNAL", "Karnal"),
    ("bathinda",           "BATHINDA",              "Bathinda"),
    ("ludhiana",           "LUDHIANA",              "Ludhiana"),
    ("amritsar",           "AMRITSAR (AMRITSAR MEWA MANDI)", "Amritsar"),
    ("jalandhar",          "JALANDHAR CITY",        "Jalandhar"),
    ("patiala",            "PATIALA",               "Patiala"),
    ("sangrur",            "SANGRUR",               "Sangrur"),
    ("moga",               "MOGA",                  "Moga"),
    ("firozpur",           "FEROZEPUR CANTT.",       "Firozpur"),
    ("amravati",           "AMARAWATI",             "Amravati"),
    ("solapur",            "SOLAPUR",               "Solapur"),
    ("kolhapur",           "KOLHAPUR",              "Kolhapur"),
    ("aurangabad",         "CHATTRAPATI SAMBHAJINAGAR", "Aurangabad"),
    ("sambhajinagar",      "CHATTRAPATI SAMBHAJINAGAR", "Aurangabad"),
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _detect_commodities(text: str) -> list:
    """Scans text for any known commodity names. Returns list of found ones."""
    text_lower = text.lower()
    found = []
    for keyword, canonical in KNOWN_COMMODITIES.items():
        if keyword in text_lower and canonical not in found:
            found.append(canonical)
    return found if found else ["Onion"]   # default if none detected


def _detect_severity(text: str) -> float:
    """Returns total severity boost from keywords in text."""
    text_lower = text.lower()
    boost = 0.0
    for keyword, value in SEVERITY_BOOST.items():
        if keyword in text_lower:
            boost += value
    return boost


def _detect_origin(text: str) -> tuple:
    """
    Scans text for known mandi/city names.
    Returns (mandi_name, district_name) or ("Unknown", "Unknown").
    """
    text_lower = text.lower()
    for alias, mandi_name, district_name in MANDI_ALIASES:
        if alias in text_lower:
            return mandi_name, district_name

    # Fallback: look for Title Case proper nouns as location candidates
    title_case = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text)
    if title_case:
        # Prefer longer matches (more specific location)
        candidate = max(title_case, key=len).strip()
        # Sanity check: reject common non-location title-case words
        non_locations = {
            "India", "Government", "Minister", "Prime", "Chief", "National",
            "State", "District", "Market", "Mandi", "Festival", "Monday",
            "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        }
        if candidate not in non_locations:
            return candidate.upper(), candidate.upper()

    return "Unknown", "Unknown"


def _compute_multiplier(shock_type: str, text: str) -> float:
    """
    Computes impact multiplier from base rate + severity keywords.
    Clamps to [0.50, 2.50] — extreme bounds for agricultural markets.
    """
    text_lower = text.lower()

    # Determine direction
    if shock_type == "policy":
        # Policy can go either way
        if any(w in text_lower for w in ["ban", "restrict", "limit", "cap", "ceiling"]):
            base = BASE_MULTIPLIERS["policy_down"]
        else:
            base = BASE_MULTIPLIERS["policy_up"]
    elif shock_type in BASE_MULTIPLIERS:
        base = BASE_MULTIPLIERS[shock_type]
    else:
        base = 1.0

    severity = _detect_severity(text)
    multiplier = base + severity

    # Clamp to realistic range
    return round(max(0.50, min(2.50, multiplier)), 3)


# ---------------------------------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------------------------------

class NewsAnalyzer:
    """
    Extracts structured shock features from agricultural news text.

    Uses Gemini API when GEMINI_API_KEY is available, otherwise falls
    back to a keyword-based heuristic model.
    """

    # Gemini prompt — structured to return consistent JSON
    SYSTEM_PROMPT = """You are an Indian agricultural commodity market intelligence AI.
Analyze the following news or policy document.
Extract shock impact parameters using this EXACT JSON schema — raw JSON only, no markdown:

{
    "commodities_affected": ["Onion", "Tomato"],
    "origin_mandi": "Lasalgaon",
    "origin_district": "Nashik",
    "shock_type": "climatic",
    "impact_multiplier": 1.35
}

Rules:
1. commodities_affected: list only commodities explicitly mentioned or strongly implied.
2. origin_mandi: the specific mandi, city, or district where the event occurs.
   MUST be a real geographic name. Never extract adjectives like "Heavy" or "Sudden".
3. origin_district: the district that contains origin_mandi. Same as origin_mandi if unsure.
4. shock_type: one of "climatic", "logistics", "policy", "demand".
5. impact_multiplier: expected price change ratio. 1.0 = no change. 1.3 = 30% price rise.
   Logistics/climatic supply disruptions → > 1.0.
   Export bans / import duty cuts → < 1.0.
   Scale with severity: "catastrophic flood" > "light rain".
6. Respond ONLY with raw JSON. No explanation. No markdown backticks."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

        # Try loading Streamlit secrets if not in env
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                pass

        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print("✅ NewsAnalyzer: Gemini API client initialised.")
            except Exception as e:
                print(f"⚠️  NewsAnalyzer: Could not init Gemini client: {e}. "
                      f"Using heuristic fallback.")

    # -------------------------------------------------------------------------

    def _heuristic_response(self, text: str) -> dict:
        """
        Keyword-based fallback. Handles all Tier 1/2 commodities,
        severity-adjusted multipliers, and extended location vocabulary.
        """
        text_lower = text.lower()

        # Shock type
        if any(w in text_lower for w in [
            "rain", "flood", "drought", "heatwave", "weather",
            "hailstorm", "crop damage", "frost", "cyclone",
        ]):
            shock_type = "climatic"
        elif any(w in text_lower for w in [
            "strike", "transport", "highway", "truck", "blockade",
            "jam", "protest", "bandh", "agitation", "road block",
        ]):
            shock_type = "logistics"
        elif any(w in text_lower for w in [
            "ban", "policy", "tax", "export", "import", "duty",
            "msp", "procurement", "subsidy", "regulation", "order",
        ]):
            shock_type = "policy"
        else:
            shock_type = "demand"

        commodities = _detect_commodities(text)
        origin_mandi, origin_district = _detect_origin(text)
        multiplier = _compute_multiplier(shock_type, text)

        return {
            "commodities_affected": commodities,
            "origin_mandi":         origin_mandi,
            "origin_district":      origin_district,
            "shock_type":           shock_type,
            "impact_multiplier":    multiplier,
            "source":               "heuristic",
        }

    # -------------------------------------------------------------------------

    def extract_shock_features(self, combined_text: str) -> dict:
        """
        Primary entry point. Returns a structured dict with shock parameters.

        Falls back to heuristic if Gemini is unavailable or returns bad JSON.
        """
        if not self.client:
            return self._heuristic_response(combined_text)

        try:
            prompt = f"{self.SYSTEM_PROMPT}\n\nInput: {combined_text}"

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            raw = response.text.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()

            result = json.loads(raw)

            # Validate required fields — if any missing, fall back
            required = {"commodities_affected", "origin_mandi", "shock_type",
                        "impact_multiplier"}
            if not required.issubset(result.keys()):
                raise ValueError(f"Missing fields in Gemini response: {result}")

            # Ensure origin_district is present (may be absent in older responses)
            if "origin_district" not in result:
                result["origin_district"] = result.get("origin_mandi", "Unknown")

            result["source"] = "gemini"
            return result

        except Exception as e:
            print(f"⚠️  Gemini extraction failed: {e}. Using heuristic fallback.")
            return self._heuristic_response(combined_text)

    # -------------------------------------------------------------------------

    def extract_from_pdf_text(self, pdf_text: str) -> dict:
        """
        Convenience wrapper for policy PDF uploads.
        Truncates to first 4000 chars (Gemini context limit for free tier).
        """
        truncated = pdf_text[:4000].strip()
        return self.extract_shock_features(truncated)


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = NewsAnalyzer()

    tests = [
        "Catastrophic floods hit Nashik district, destroying 40% of onion crop",
        "Truckers call nationwide strike on NH-48, onion and tomato supplies disrupted",
        "Government imposes ban on onion exports with immediate effect",
        "Minor drought conditions reported in Mandsaur garlic belt, slight damage",
        "Heavy rainfall in Harda district affects wheat and soybean harvest",
        "MSP for wheat increased by 8% for Rabi season 2024-25",
    ]

    print("\nNewsAnalyzer v2.0 — Test Run")
    print("=" * 60)
    for text in tests:
        result = analyzer.extract_shock_features(text)
        print(f"\nInput:      {text[:70]}")
        print(f"Commodities: {result['commodities_affected']}")
        print(f"Origin:      {result['origin_mandi']} / {result['origin_district']}")
        print(f"Shock:       {result['shock_type']}  ×{result['impact_multiplier']}")
        print(f"Source:      {result.get('source', 'unknown')}")