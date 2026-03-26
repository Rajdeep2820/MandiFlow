import torch
import scipy.sparse as sparse
import numpy as np
import json
import os
from model import MandiFlowNet
from news_analyzer import NewsAnalyzer

# 1. SETUP ENVIRONMENT
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Multi-commodity Resource Cache
_RESOURCE_CACHE = {}

def get_resources(commodity):
    commodity = commodity.upper()
    if commodity in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[commodity]
    
    adj_path = f"mandi_adjacency_{commodity.lower()}.npz"
    idx_path = f"mandi_adjacency_index_{commodity.lower()}.txt"
    model_path = f"mandiflow_gcn_lstm_{commodity.lower()}.pth"
    
    # 1. Load Adjacency
    if os.path.exists(adj_path):
        adj = sparse.load_npz(adj_path)
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
    market_to_id = {name: idx for idx, name in enumerate(market_names)}
    
    # 3. Load Model
    # Note: Architecture remains the same for now (7 features -> 4 predictions)
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
        "model": model
    }
    _RESOURCE_CACHE[commodity] = resources
    return resources

analyzer = NewsAnalyzer()

import pyarrow.dataset as ds
import pandas as pd

def fetch_trailing_history(origin_name, commodity):
    """Dynamically queries the massive Parquet file for the actual trailing 7 days of the requested Mandi."""
    try:
        dataset = ds.dataset("mandi_master_data.parquet", format="parquet")
        # To make it fast, we only pull the specific market
        df = dataset.to_table(
            columns=["Arrival_Date", "Market", "Commodity", "Modal_Price"],
            filter=(ds.field("Market") == origin_name.upper()) | (ds.field("Market") == origin_name.title())
        ).to_pandas()
        
        df = df[df["Commodity"].str.upper() == commodity.upper()]
        if df.empty:
            return None
            
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
        daily = df.groupby("Arrival_Date")["Modal_Price"].mean().reset_index()
        daily = daily.sort_values("Arrival_Date").tail(7)
        
        if len(daily) > 0:
            prices = daily["Modal_Price"].tolist()
            # Pad to 7 days if history is very sparse
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
    origin_name = features_json.get("origin_mandi", "Unknown").upper()
    
    # 3. Create a Graph Snapshot with DYNAMIC History
    trailing_prices = fetch_trailing_history(origin_name, commodity)
    
    if trailing_prices is not None:
        base_price = trailing_prices[-1] # The real price observed TODAY
        print(f"✅ Injected dynamic 7-day history for {origin_name} ({commodity}): {trailing_prices}")
    else:
        base_price = 800.0 # Fallback
        trailing_prices = [base_price] * 7
        print(f"⚠️ No history found for {origin_name} ({commodity}). Using flat baseline: {base_price}")
        
    num_nodes = adj.shape[0] if adj.shape[0] > 0 else 10
    
    # Provide actual 7 trailing days of features across the graph
    x = torch.zeros((num_nodes, 7), dtype=torch.float32)
    for i in range(7):
        x[:, i] = trailing_prices[i]
    
    target_idx = market_to_id.get(origin_name)
    if target_idx is not None:
        # Spike the current day (t=0, index 6) for the origin
        x[target_idx, 6] = base_price * impact_factor
    else:
        # Apply systemic impact to current day for all nodes
        x[:, 6] = base_price * impact_factor
        target_idx = 0 
        
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
        "origin_name": market_names[target_idx] if target_idx < len(market_names) else origin_name,
        "origin_forecast": origin_forecast.tolist(),
        "served_areas": []
    }
    
    # Locate strongly correlated neighbors
    if adj.shape[0] > target_idx:
        neighbors = adj[target_idx].indices
        for i, neighbor_id in enumerate(neighbors[:3]):
            neighbor_curve = pred_cpu[neighbor_id] / (pred_cpu[neighbor_id][0] + 1e-5)
            neighbor_forecast = base_price * neighbor_curve
            
            if impact_factor != 1.0:
                ripple_effect = (impact_factor - 1.0) * 0.5
                neighbor_forecast = neighbor_forecast * (1.0 + ripple_effect)

            result["served_areas"].append({
                "mandi": market_names[neighbor_id],
                "forecast": neighbor_forecast.tolist()
            })
            
    return result

if __name__ == "__main__":
    print(simulate_shock("Truckers strike on Delhi-Mumbai highway halts onion transport", commodity="ONION"))

if __name__ == "__main__":
    print(simulate_shock("Truckers strike on Delhi-Mumbai highway halts onion transport"))