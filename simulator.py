import difflib
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import scipy.sparse as sparse
import torch

from model import MandiFlowNet
from news_analyzer import NewsAnalyzer

# 1. SETUP ENVIRONMENT
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Multi-commodity Resource Cache
_RESOURCE_CACHE = {}

LOCATION_ALIASES = {
    "NASHIK": ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)", "NASIK"],
    "NASHIK DISTRICT": ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)", "NASIK"],
    "NASIK": ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)", "NASIK"],
    "LASALGAON": ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)"],
    "DELHI": ["AZADPUR"],
    "NEW DELHI": ["AZADPUR"],
    "BENGALURU": ["BANGALORE"],
    "BANGALORE": ["BANGALORE"],
    "GURUGRAM": ["GURGAON"],
    "MUMBAI": ["MUMBAI"],
}

analyzer = NewsAnalyzer()


def get_resources(commodity):
    commodity = commodity.upper()
    if commodity in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[commodity]

    adj_path = f"mandi_adjacency_{commodity.lower()}.npz"
    idx_path = f"mandi_adjacency_index_{commodity.lower()}.txt"
    model_path = f"mandiflow_gcn_lstm_{commodity.lower()}.pth"

    # 1. Load Adjacency
    if os.path.exists(adj_path):
        adj = sparse.load_npz(adj_path).tocsr()
        row, col = adj.nonzero()
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long).to(device)
        edge_weight = torch.tensor(adj.data, dtype=torch.float32).to(device)
    else:
        print(f"⚠️ Adjacency for {commodity} not found. Fallback to dummy.")
        adj = sparse.csr_matrix((10, 10))
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_weight = torch.empty((0,), dtype=torch.float32).to(device)

    # 2. Load Index
    market_names = []
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            market_names = [line.strip() for line in f]
    market_to_id = {name.strip().upper(): idx for idx, name in enumerate(market_names)}

    # 3. Load Model
    model = MandiFlowNet(node_features=7, hidden_dim=64, output_dim=4).to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"✅ Loaded trained weights for {commodity}")
        except Exception as e:
            print(f"❌ Failed to load {commodity} model: {e}")

    model.eval()

    resources = {
        "adj": adj,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "market_names": market_names,
        "market_to_id": market_to_id,
        "model": model,
    }
    _RESOURCE_CACHE[commodity] = resources
    return resources


def _normalize_market_text(value):
    value = (value or "").upper().strip()
    value = value.replace("&", "AND")
    value = re.sub(r"[^A-Z0-9() ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def resolve_market_name(raw_origin, market_to_id, raw_district=""):
    """Resolve an extracted location to the closest mandi name in the graph index."""
    normalized_origin = _normalize_market_text(raw_origin)
    normalized_district = _normalize_market_text(raw_district)
    if not normalized_origin and not normalized_district:
        return None, None, "Empty origin received from analyzer"

    candidate_names = []
    candidate_names.extend(LOCATION_ALIASES.get(normalized_origin, []))
    candidate_names.extend(LOCATION_ALIASES.get(normalized_district, []))
    if normalized_origin.endswith(" DISTRICT"):
        candidate_names.append(normalized_origin.replace(" DISTRICT", ""))
    if normalized_district.endswith(" DISTRICT"):
        candidate_names.append(normalized_district.replace(" DISTRICT", ""))
    if normalized_origin:
        candidate_names.append(normalized_origin)
    if normalized_district and normalized_district != normalized_origin:
        candidate_names.append(normalized_district)

    seen = set()
    deduped_candidates = []
    for candidate in candidate_names:
        candidate = _normalize_market_text(candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate in market_to_id:
            return candidate, market_to_id[candidate], "exact"

    for candidate in deduped_candidates:
        for market_key, idx in market_to_id.items():
            normalized_market_key = _normalize_market_text(market_key)
            if candidate in normalized_market_key or normalized_market_key in candidate:
                return market_key, idx, "substring"

    normalized_market_lookup = {
        _normalize_market_text(market_key): (market_key, idx)
        for market_key, idx in market_to_id.items()
    }
    close_match = difflib.get_close_matches(
        normalized_origin,
        list(normalized_market_lookup.keys()),
        n=1,
        cutoff=0.72,
    )
    if close_match:
        market_key, idx = normalized_market_lookup[close_match[0]]
        return market_key, idx, "fuzzy"

    return None, None, f"No mandi match found for '{raw_origin}'"


def fetch_trailing_history(origin_name, commodity):
    """Dynamically queries the massive Parquet file for the actual trailing 7 days of the requested Mandi."""
    try:
        dataset = ds.dataset("mandi_master_data.parquet", format="parquet")
        df = dataset.to_table(
            columns=["Arrival_Date", "Market", "Commodity", "Modal_Price"],
            filter=(ds.field("Market") == origin_name.upper()) | (ds.field("Market") == origin_name.title()),
        ).to_pandas()

        df = df[df["Commodity"].str.upper() == commodity.upper()]
        if df.empty:
            return None

        df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
        daily = df.groupby("Arrival_Date")["Modal_Price"].mean().reset_index()
        daily = daily.sort_values("Arrival_Date").tail(7)

        if len(daily) > 0:
            prices = daily["Modal_Price"].tolist()
            while len(prices) < 7:
                prices.insert(0, prices[0])
            return prices
    except Exception as e:
        print(f"Error fetching dynamic history: {e}")
    return None


def simulate_shock(news_text, doc_text="", commodity="ONION"):
    combined = f"{news_text} {doc_text}".strip()

    # 1. Get resources for the specific commodity
    res = get_resources(commodity)
    model = res["model"]
    edge_index = res["edge_index"]
    edge_weight = res["edge_weight"]
    adj = res["adj"]
    market_names = res["market_names"]
    market_to_id = res["market_to_id"]

    # 2. Convert Unstructured Text to Structured Subgraph Shock Multiplier
    features_json = analyzer.extract_shock_features(combined)
    impact_factor = features_json.get("impact_multiplier", 1.0)
    extracted_origin = features_json.get("origin_mandi", "Unknown").strip()
    extracted_district = features_json.get("origin_district", "").strip()
    resolved_origin_name, target_idx, match_strategy = resolve_market_name(
        extracted_origin,
        market_to_id,
        raw_district=extracted_district,
    )

    # 3. Create a Graph Snapshot with DYNAMIC History
    history_lookup_name = resolved_origin_name or _normalize_market_text(extracted_origin)
    trailing_prices = fetch_trailing_history(history_lookup_name, commodity)

    if trailing_prices is not None:
        base_price = trailing_prices[-1]
        print(f"✅ Injected dynamic 7-day history for {history_lookup_name} ({commodity}): {trailing_prices}")
    else:
        base_price = 800.0
        trailing_prices = [base_price] * 7
        print(f"⚠️ No history found for {history_lookup_name} ({commodity}). Using flat baseline: {base_price}")

    if target_idx is None:
        print(f"🚨 STRING MATCH ERROR: Could not resolve extracted origin '{extracted_origin}' to any mandi in the graph index.")
        return {
            "features": features_json,
            "origin_name": extracted_origin or "Unknown",
            "origin_forecast": trailing_prices[-4:],
            "served_areas": [],
            "resolution_error": f"Unable to map extracted origin '{extracted_origin}' / district '{extracted_district or 'Unknown'}' to a known mandi for {commodity}.",
        }

    num_nodes = adj.shape[0] if adj.shape[0] > 0 else 10

    # Provide actual 7 trailing days of features across the graph
    x = torch.zeros((num_nodes, 7), dtype=torch.float32)
    for i in range(7):
        x[:, i] = trailing_prices[i]

    print(f"✅ Resolved extracted origin '{extracted_origin}' to '{resolved_origin_name}' via {match_strategy} match")
    x[target_idx, 6] = base_price * impact_factor
    x = x.to(device)

    # 4. RUN Forward Pass across GCN-LSTM architecture
    with torch.no_grad():
        predictions = model(x, edge_index, edge_weight)

    # 5. Extract Results
    pred_cpu = predictions.cpu().numpy()

    # 6. Apply Curve Normalization
    origin_curve = pred_cpu[target_idx] / (pred_cpu[target_idx][0] + 1e-5)
    origin_forecast = (base_price * impact_factor) * origin_curve

    result = {
        "features": features_json,
        "origin_name": market_names[target_idx] if target_idx < len(market_names) else resolved_origin_name,
        "origin_forecast": origin_forecast.tolist(),
        "served_areas": [],
    }

    # Locate strongly correlated neighbors
    # Locate strongly correlated neighbors
    if adj.shape[0] > target_idx:
        row = adj[target_idx]
        
        # Zip the connected node IDs (.indices) with their correlation strengths (.data)
        connections = list(zip(row.indices, row.data))
        
        # Sort them by strength (weight) in descending order (highest correlation first)
        connections.sort(key=lambda x: x[1], reverse=True)
        
        # Grab the top 5 node IDs with the strongest mathematical connection
        top_neighbors = [idx for idx, weight in connections[:5]]
        
        for neighbor_id in top_neighbors:
            neighbor_curve = pred_cpu[neighbor_id] / (pred_cpu[neighbor_id][0] + 1e-5)
            neighbor_forecast = base_price * neighbor_curve

            if impact_factor != 1.0:
                ripple_effect = (impact_factor - 1.0) * 0.5
                neighbor_forecast = neighbor_forecast * (1.0 + ripple_effect)

            result["served_areas"].append(
                {
                    "mandi": market_names[neighbor_id],
                    "forecast": neighbor_forecast.tolist(),
                }
            )

    return result


if __name__ == "__main__":
    print(simulate_shock("Truckers strike on Delhi-Mumbai highway halts onion transport", commodity="ONION"))
