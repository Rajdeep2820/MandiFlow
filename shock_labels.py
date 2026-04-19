"""
shock_labels.py  —  MandiFlow v3.0
=====================================
Single source of truth for all shock event labeling.

Two-source approach:
  1. KNOWN_EVENTS: 15 hardcoded historical Indian agricultural crises
     with verified dates, affected regions, shock types, and commodities.
  2. Auto-detector: scans the actual price data within ±30 days of each
     known event to find the real spike days (when the shock hit mandis).
     Also finds unknown shocks in quiet periods via price deviation.

Shock type encoding (6 features per node):
  [is_epicenter, climatic, logistics, policy_up, policy_down, severity]

  policy_up:   MSP hike, procurement drive, export restriction that
               raises domestic prices (supply reduced for export)
  policy_down: Export ban, import duty cut — lowers prices at origin

Severity scale: 0.0–1.0
  0.0  = no shock (dead zone: price movement < 15%)
  0.3  = moderate (15–30% movement)
  0.6  = significant (30–60% movement)
  1.0  = extreme (>60% movement, e.g. 2019 onion 6× spike)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# SHOCK TYPE CONSTANTS
# ---------------------------------------------------------------------------

SHOCK_NONE        = 0
SHOCK_CLIMATIC    = 1
SHOCK_LOGISTICS   = 2
SHOCK_POLICY_UP   = 3
SHOCK_POLICY_DOWN = 4

SHOCK_NAMES = {
    SHOCK_NONE:        "none",
    SHOCK_CLIMATIC:    "climatic",
    SHOCK_LOGISTICS:   "logistics",
    SHOCK_POLICY_UP:   "policy_up",
    SHOCK_POLICY_DOWN: "policy_down",
}

# Dead zone: movements below this threshold are noise, not shocks
# Onion prices naturally vary ±20% on normal days — dead zone must be above that
SEVERITY_DEAD_ZONE = 0.25   # 25% daily movement minimum to register as a shock


# ---------------------------------------------------------------------------
# KNOWN HISTORICAL EVENTS
# ---------------------------------------------------------------------------

@dataclass
class ShockEvent:
    name:            str
    start_date:      str           # YYYY-MM-DD
    end_date:        str           # YYYY-MM-DD — when prices stabilized
    shock_type:      int
    commodities:     List[str]     # which commodities affected
    epicenter_hints: List[str]     # mandi/city names near the epicenter
    direction:       int           # +1 = prices up, -1 = prices down
    severity_hint:   float         # 0.0–1.0 expected severity
    notes:           str = ""


KNOWN_EVENTS = [

    # -----------------------------------------------------------------------
    # CLIMATIC
    # -----------------------------------------------------------------------
    ShockEvent(
        name            = "2002 Nationwide Drought",
        start_date      = "2002-06-01",
        end_date        = "2002-11-30",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["WHEAT", "MAIZE", "PADDY", "ONION", "POTATO"],
        epicenter_hints = ["Rajasthan", "Gujarat", "Madhya Pradesh",
                           "Jodhpur", "Ahmedabad", "Indore"],
        direction       = +1,
        severity_hint   = 0.7,
        notes           = "Worst kharif failure in 15 years. 18 states affected.",
    ),
    ShockEvent(
        name            = "2004 Bihar-UP Floods",
        start_date      = "2004-08-01",
        end_date        = "2004-10-15",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["POTATO", "ONION", "TOMATO"],
        epicenter_hints = ["Bihar", "Patna", "Muzaffarpur", "Gorakhpur",
                           "Barabanki", "Lucknow"],
        direction       = +1,
        severity_hint   = 0.5,
        notes           = "Vegetable supply chain disrupted across North India.",
    ),
    ShockEvent(
        name            = "2007 Bihar Floods — Severe",
        start_date      = "2007-08-15",
        end_date        = "2007-10-31",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["POTATO", "ONION", "TOMATO"],
        epicenter_hints = ["Patna", "Muzaffarpur", "Darbhanga", "Hajipur",
                           "Samastipur"],
        direction       = +1,
        severity_hint   = 0.65,
        notes           = "Kosi river flood. North Indian vegetable prices spiked.",
    ),
    ShockEvent(
        name            = "2009 Nationwide Drought — Worst in 37 Years",
        start_date      = "2009-06-01",
        end_date        = "2009-12-31",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["WHEAT", "MAIZE", "PADDY", "ONION", "POTATO",
                           "TOMATO"],
        epicenter_hints = ["Rajasthan", "UP", "Bihar", "Maharashtra",
                           "Jodhpur", "Lucknow", "Nashik", "Indore"],
        direction       = +1,
        severity_hint   = 0.85,
        notes           = "Southwest monsoon 23% below normal. National crisis.",
    ),
    ShockEvent(
        name            = "2010-11 Onion Price Shock",
        start_date      = "2010-11-01",
        end_date        = "2011-01-31",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["ONION"],
        epicenter_hints = ["Lasalgaon", "Nasik", "Nashik", "Pune",
                           "Mandsaur"],
        direction       = +1,
        severity_hint   = 0.90,
        notes           = "Consecutive monsoon failures in Maharashtra. "
                          "Onion ₹15→₹85/kg retail.",
    ),
    ShockEvent(
        name            = "2014-15 Drought — El Niño",
        start_date      = "2014-06-01",
        end_date        = "2015-03-31",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["WHEAT", "ONION", "POTATO", "TOMATO", "GARLIC"],
        epicenter_hints = ["Rajasthan", "MP", "Maharashtra", "Karnataka",
                           "Jodhpur", "Nashik", "Indore", "Bangalore"],
        direction       = +1,
        severity_hint   = 0.60,
        notes           = "Rabi sowing down 9%. Multiple consecutive dry years.",
    ),
    ShockEvent(
        name            = "2019 Maharashtra Floods — Onion Crisis",
        start_date      = "2019-09-01",
        end_date        = "2020-01-31",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["ONION"],
        epicenter_hints = ["Lasalgaon", "Nasik", "Nashik",
                           "Lasalgaon (Niphad)", "Lasalgaon (Vinchur)",
                           "Pune", "Ahmednagar"],
        direction       = +1,
        severity_hint   = 1.0,
        notes           = "Worst onion shock in recorded history. "
                          "Nashik ₹800→₹4800/qtl. Government airlifted onions.",
    ),
    ShockEvent(
        name            = "2022 March Heatwave",
        start_date      = "2022-03-01",
        end_date        = "2022-06-30",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["WHEAT", "TOMATO"],
        epicenter_hints = ["Punjab", "Haryana", "UP", "Ludhiana",
                           "Karnal", "Lucknow", "Kanpur"],
        direction       = +1,
        severity_hint   = 0.55,
        notes           = "Earliest heatwave since 1901. Wheat yield -10%.",
    ),
    ShockEvent(
        name            = "2023 El Niño — Tomato and Onion Spike",
        start_date      = "2023-06-01",
        end_date        = "2023-10-31",
        shock_type      = SHOCK_CLIMATIC,
        commodities     = ["TOMATO", "ONION"],
        epicenter_hints = ["Kolar", "Nashik", "Lasalgaon", "Nasik",
                           "Bangalore", "Pune"],
        direction       = +1,
        severity_hint   = 0.80,
        notes           = "Tomato ₹200–250/kg. Onion export ban triggered.",
    ),

    # -----------------------------------------------------------------------
    # LOGISTICS
    # -----------------------------------------------------------------------
    ShockEvent(
        name            = "2000 National Truckers Strike",
        start_date      = "2000-07-10",
        end_date        = "2000-07-20",
        shock_type      = SHOCK_LOGISTICS,
        commodities     = ["ONION", "POTATO", "TOMATO", "WHEAT"],
        epicenter_hints = ["Azadpur", "Delhi", "Mumbai", "Bangalore",
                           "Kolkata", "Hyderabad"],
        direction       = +1,
        severity_hint   = 0.50,
        notes           = "5-day all-India truckers strike. Metro mandis worst hit.",
    ),
    ShockEvent(
        name            = "2004 Truckers Strike",
        start_date      = "2004-05-01",
        end_date        = "2004-05-10",
        shock_type      = SHOCK_LOGISTICS,
        commodities     = ["ONION", "POTATO", "TOMATO"],
        epicenter_hints = ["Azadpur", "Delhi", "Mumbai", "Nasik"],
        direction       = +1,
        severity_hint   = 0.40,
        notes           = "Perishables spike in metro mandis within 48 hours.",
    ),
    ShockEvent(
        name            = "2020 COVID-19 Lockdown",
        start_date      = "2020-03-25",
        end_date        = "2020-06-30",
        shock_type      = SHOCK_LOGISTICS,
        commodities     = ["ONION", "POTATO", "TOMATO", "WHEAT", "GARLIC"],
        epicenter_hints = ["Azadpur", "Delhi", "Mumbai", "Nashik",
                           "Lasalgaon", "Bangalore", "Kolkata",
                           "Hyderabad", "Ahmedabad", "Pune"],
        direction       = +1,
        severity_hint   = 0.75,
        notes           = "Complete mandi shutdown Week 1. Transport freeze. "
                          "Farmers forced to dump perishables.",
    ),

    # -----------------------------------------------------------------------
    # POLICY — UP (raises domestic prices)
    # -----------------------------------------------------------------------
    ShockEvent(
        name            = "2013 Onion Export Restriction",
        start_date      = "2013-08-01",
        end_date        = "2013-10-31",
        shock_type      = SHOCK_POLICY_UP,
        commodities     = ["ONION"],
        epicenter_hints = ["Lasalgaon", "Nasik", "Nashik", "Mandsaur",
                           "Indore"],
        direction       = +1,
        severity_hint   = 0.70,
        notes           = "Minimum export price raised to $650/tonne. "
                          "Domestic prices surged as export supply reduced.",
    ),

    # -----------------------------------------------------------------------
    # POLICY — DOWN (lowers domestic prices at origin)
    # -----------------------------------------------------------------------
    ShockEvent(
        name            = "2022 Wheat Export Ban",
        start_date      = "2022-05-13",
        end_date        = "2022-08-31",
        shock_type      = SHOCK_POLICY_DOWN,
        commodities     = ["WHEAT"],
        epicenter_hints = ["Punjab", "Haryana", "Ludhiana", "Karnal",
                           "Bathinda", "Amritsar"],
        direction       = -1,
        severity_hint   = 0.45,
        notes           = "Sudden export ban post-Russia-Ukraine war. "
                          "Domestic prices fell as export demand removed.",
    ),
    ShockEvent(
        name            = "2023 Onion Export Ban",
        start_date      = "2023-12-08",
        end_date        = "2024-03-31",
        shock_type      = SHOCK_POLICY_DOWN,
        commodities     = ["ONION"],
        epicenter_hints = ["Lasalgaon", "Nasik", "Nashik",
                           "Lasalgaon (Niphad)", "Lasalgaon (Vinchur)",
                           "Pune", "Ahmednagar", "Mandsaur"],
        direction       = -1,
        severity_hint   = 0.65,
        notes           = "Complete export ban before 2024 elections. "
                          "Nashik prices fell 40% in two weeks.",
    ),
]


# ---------------------------------------------------------------------------
# SEVERITY COMPUTATION
# ---------------------------------------------------------------------------

def compute_severity(ratio: float) -> float:
    """
    Converts a raw price ratio to a 0.0–1.0 severity score.

    Dead zone below 15%: returns 0.0 (noise, not a shock).
    This prevents ghost shocks from minor daily fluctuations.

    Args:
        ratio: Price_t / Price_{t-1}. Values > 1.0 = price up.

    Returns:
        float in [0.0, 1.0]
    """
    deviation = abs(ratio - 1.0)

    if deviation < SEVERITY_DEAD_ZONE:
        return 0.0

    # Scale: 15% movement → 0.0, 100%+ movement → 1.0
    scaled = (deviation - SEVERITY_DEAD_ZONE) / (1.0 - SEVERITY_DEAD_ZONE)
    return float(min(scaled, 1.0))


# ---------------------------------------------------------------------------
# SHOCK FEATURE VECTOR
# ---------------------------------------------------------------------------

def make_shock_vector(
    shock_type:    int,
    is_epicenter:  bool,
    severity:      float,
) -> np.ndarray:
    """
    Builds the 6-feature shock context vector for one node.

    Features:
      [0] is_epicenter     — 1.0 if this node is the shock origin
      [1] climatic         — 1.0 if flood/drought/heatwave
      [2] logistics        — 1.0 if strike/blockade/lockdown
      [3] policy_up        — 1.0 if restriction raises domestic prices
      [4] policy_down      — 1.0 if ban/cut lowers domestic prices
      [5] severity         — 0.0–1.0 normalized magnitude

    Returns:
        np.ndarray shape (6,)
    """
    vec = np.zeros(6, dtype=np.float32)
    vec[0] = 1.0 if is_epicenter else 0.0

    if shock_type == SHOCK_CLIMATIC:
        vec[1] = 1.0
    elif shock_type == SHOCK_LOGISTICS:
        vec[2] = 1.0
    elif shock_type == SHOCK_POLICY_UP:
        vec[3] = 1.0
    elif shock_type == SHOCK_POLICY_DOWN:
        vec[4] = 1.0
    # SHOCK_NONE: all zeros

    vec[5] = float(severity)
    return vec


ZERO_SHOCK_VECTOR = np.zeros(6, dtype=np.float32)


# ---------------------------------------------------------------------------
# AUTO-DETECTOR
# ---------------------------------------------------------------------------

def detect_shock_days(
    ratio_matrix:  np.ndarray,
    anchor_mask:   np.ndarray,
    dates:         pd.DatetimeIndex,
    threshold:     float = 0.35,
    min_fraction:  float = 0.10,  # at least 10% of anchor mandis must spike
) -> np.ndarray:
    """
    Finds genuine shock days — requires BOTH:
      1. At least one anchor mandi moved > threshold (35%)
      2. At least min_fraction of anchor mandis moved > 0.15 on the same day

    Condition 2 filters out single-mandi data artifacts (one mandi
    reporting a bad price) and keeps only real market-wide events
    where multiple mandis move together.
    """
    anchor_ratios   = ratio_matrix[:, anchor_mask]              # (T, M)
    deviation       = np.abs(anchor_ratios - 1.0)               # (T, M)

    # Condition 1: any anchor mandi had a large move
    any_large       = deviation.max(axis=1) > threshold         # (T,)

    # Condition 2: enough mandis moved meaningfully (market-wide signal)
    frac_moved      = (deviation > 0.15).mean(axis=1)           # (T,)
    enough_mandis   = frac_moved >= min_fraction                 # (T,)

    return any_large & enough_mandis


def label_training_data(
    ratio_matrix:  np.ndarray,     # (T, N)
    anchor_mask:   np.ndarray,     # (N,) bool
    market_names:  list,           # length N
    dates:         pd.DatetimeIndex,
    commodity:     str,
) -> dict:
    """
    Produces per-day shock labels for the full training matrix.

    Returns a dict with:
      shock_types:    (T,) int array   — SHOCK_* constant per day
      directions:     (T,) int array   — +1 up / -1 down / 0 neutral
      severities:     (T,) float array — 0.0–1.0
      epicenter_mask: (T, N) bool      — True where node is epicenter
      is_shock_day:   (T,) bool        — True on any shock day

    Logic:
      1. Auto-detect shock days from price spikes
      2. For days within ±30 days of a known event that matches this
         commodity, assign the known event's shock_type and direction
      3. For detected spike days outside known events, assign SHOCK_CLIMATIC
         with direction inferred from dominant price movement
      4. Everything else: SHOCK_NONE
    """
    commodity = commodity.upper()
    T, N      = ratio_matrix.shape

    shock_types    = np.zeros(T, dtype=np.int8)
    directions     = np.zeros(T, dtype=np.int8)
    severities     = np.zeros(T, dtype=np.float32)
    epicenter_mask = np.zeros((T, N), dtype=bool)

    market_upper = [m.upper() for m in market_names]

    # ---- Step 1: Mark known events ----------------------------------------
    for event in KNOWN_EVENTS:
        if commodity not in [c.upper() for c in event.commodities]:
            continue

        event_start = pd.Timestamp(event.start_date)
        event_end   = pd.Timestamp(event.end_date)

        # Expand window by ±7 days to catch lagged mandi reporting
        window_start = event_start - pd.Timedelta(days=7)
        window_end   = event_end   + pd.Timedelta(days=7)

        # Find date indices in this window
        in_window = (dates >= window_start) & (dates <= window_end)
        t_indices = np.where(in_window)[0]

        if len(t_indices) == 0:
            continue

        # Identify epicenter nodes from hints
        epicenter_indices = []
        for hint in event.epicenter_hints:
            hint_upper = hint.upper()
            for i, m in enumerate(market_upper):
                if hint_upper in m or m in hint_upper:
                    epicenter_indices.append(i)

        for t in t_indices:
            shock_types[t] = event.shock_type
            directions[t]  = event.direction

            # Severity: use hint during the event, taper at edges
            days_from_start = abs((dates[t] - event_start).days)
            days_from_end   = abs((dates[t] - event_end).days)
            proximity       = 1.0 - min(days_from_start, days_from_end) / 37.0
            severities[t]   = max(severities[t],
                                  event.severity_hint * max(0.3, proximity))

            for ei in epicenter_indices:
                epicenter_mask[t, ei] = True

    # ---- Step 2: Auto-detect unlabeled spike days -------------------------
    spike_days = detect_shock_days(ratio_matrix, anchor_mask, dates)

    for t in np.where(spike_days)[0]:
        if shock_types[t] != SHOCK_NONE:
            # Already labeled by a known event — just update severity
            day_severity = compute_severity(
                float(ratio_matrix[t, anchor_mask].max())
            )
            severities[t] = max(severities[t], day_severity)
            continue

        # Unknown spike — infer type and direction from price movement
        anchor_ratios = ratio_matrix[t, anchor_mask]
        median_ratio  = float(np.median(anchor_ratios))
        direction      = +1 if median_ratio > 1.0 else -1

        # Unknown spikes default to CLIMATIC (most common cause)
        shock_types[t] = SHOCK_CLIMATIC
        directions[t]  = direction
        severities[t]  = compute_severity(median_ratio)

        # Epicenter: nodes with highest deviation from 1.0
        deviations = np.abs(ratio_matrix[t] - 1.0)
        top_n      = min(10, N)
        top_idx    = np.argsort(deviations)[::-1][:top_n]
        epicenter_mask[t, top_idx] = True

    is_shock_day = shock_types != SHOCK_NONE

    # Stats
    n_shock = is_shock_day.sum()
    n_known = 0
    for event in KNOWN_EVENTS:
        if commodity in [c.upper() for c in event.commodities]:
            in_window = (dates >= pd.Timestamp(event.start_date)) & \
                        (dates <= pd.Timestamp(event.end_date))
            n_known += in_window.sum()

    print(f"\n📊 Shock Labeling Report — {commodity}:")
    print(f"   Total training days:   {T}")
    print(f"   Shock days (total):    {n_shock}  "
          f"({100*n_shock/T:.1f}%)")
    print(f"   From known events:     ~{min(n_known, n_shock)}")
    print(f"   Auto-detected:         ~{max(0, n_shock - n_known)}")
    print(f"   Normal days:           {T - n_shock}  "
          f"({100*(T-n_shock)/T:.1f}%)  ← target ≥70%")

    type_counts = {}
    for stype, sname in SHOCK_NAMES.items():
        count = (shock_types == stype).sum()
        if count > 0:
            type_counts[sname] = count
    for sname, count in type_counts.items():
        print(f"   {sname:<20} {count:>5} days")

    return {
        "shock_types":    shock_types,
        "directions":     directions,
        "severities":     severities,
        "epicenter_mask": epicenter_mask,
        "is_shock_day":   is_shock_day,
    }


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Print event summary
    print(f"\n{'='*60}")
    print(f"  MandiFlow v3.0 — Known Shock Events ({len(KNOWN_EVENTS)} total)")
    print(f"{'='*60}")
    for e in KNOWN_EVENTS:
        print(f"\n  {e.name}")
        print(f"    Type:        {SHOCK_NAMES[e.shock_type]}")
        print(f"    Period:      {e.start_date} → {e.end_date}")
        print(f"    Commodities: {', '.join(e.commodities)}")
        print(f"    Direction:   {'↑ prices UP' if e.direction > 0 else '↓ prices DOWN'}")
        print(f"    Severity:    {e.severity_hint:.0%}")

    # Test severity function
    print(f"\n{'='*60}")
    print("  Severity function test:")
    for ratio in [1.00, 1.10, 1.15, 1.25, 1.50, 2.00, 3.00,
                  0.90, 0.85, 0.70, 0.50]:
        sev = compute_severity(ratio)
        print(f"    ratio={ratio:.2f}  severity={sev:.3f}")