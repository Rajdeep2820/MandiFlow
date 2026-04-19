"""
simulator.py  —  MandiFlow v3.0
=================================
Shock simulation engine with direction-aware output.

User input:  news text describing a shock event
Output:      Rs/quintal forecasts with ↑/↓ arrows for 4 days

Key improvements from v2:
  - Injects correct shock type vector (not just a multiplier)
  - Direction arrows from model's classification head
  - Per-node denormalization using saved anchor prices
  - Propagation is asymmetric: climatic shocks propagate outward from
    epicenter, logistics shocks affect transport corridors, policy shocks
    have opposite effects at origin vs destination mandis
"""

import difflib
import os
import re

import numpy as np
import scipy.sparse as sparse
import torch

from economic_engine import apply_economic_constraints

from model import MandiFlowNet, NODE_FEATURES
from news_analyzer import NewsAnalyzer
from shock_labels import (
    make_shock_vector,
    compute_severity,
    ZERO_SHOCK_VECTOR,
    SHOCK_NONE, SHOCK_CLIMATIC, SHOCK_LOGISTICS,
    SHOCK_POLICY_UP, SHOCK_POLICY_DOWN,
    SHOCK_NAMES,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RATIO_CLIP = (0.30, 3.00)
_RESOURCE_CACHE: dict = {}
analyzer = NewsAnalyzer()

# Shock type string → constant mapping (from NewsAnalyzer output)
SHOCK_TYPE_MAP = {
    "climatic":   SHOCK_CLIMATIC,
    "logistics":  SHOCK_LOGISTICS,
    "policy_up":  SHOCK_POLICY_UP,
    "policy_down": SHOCK_POLICY_DOWN,
    "policy":     SHOCK_POLICY_UP,   # default policy direction = up
    "demand":     SHOCK_CLIMATIC,    # treat demand shocks like climatic
    "none":       SHOCK_NONE,
}

LOCATION_ALIASES = {
    "NASHIK":    ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)", "NASIK"],
    "NASIK":     ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)", "NASIK"],
    "LASALGAON": ["LASALGAON", "LASALGAON (NIPHAD)", "LASALGAON (VINCHUR)"],
    "DELHI":     ["AZADPUR"],
    "NEW DELHI": ["AZADPUR"],
    "BENGALURU": ["BANGALORE"],
    "BANGALORE": ["BANGALORE"],
    "GURUGRAM":  ["GURGAON"],
    "MUMBAI":    ["MUMBAI"],
    "MANDSAUR":  ["MANDSAUR", "MANDSAUR (F&V)"],
    "INDORE":    ["INDORE", "INDORE (F&V)"],
    "PUNE":      ["PUNE", "PUNE (MOSHI)", "PUNE (PIMPRI)", "PUNE (MANJRI)"],
    "KOLKATA":   ["BARA BAZAR (POSTA BAZAR)"],
    "HYDERABAD": ["HYDERABAD (F&V)"],
    "KOLAR":     ["KOLAR"],
    "AHMEDABAD": ["AHMEDABAD", "AHMEDABAD (CHIMANBHAI PATAL MARKET VASANA)"],
}

GLOBAL_SHOCK_KEYWORDS = {
    "truckers strike":  ("logistics", 1.12),
    "transport strike": ("logistics", 1.12),
    "farmers protest":  ("logistics", 1.08),
    "highway blocked":  ("logistics", 1.10),
    "nationwide strike":("logistics", 1.15),
    "drought":          ("climatic",  1.18),
    "heatwave":         ("climatic",  1.12),
    "monsoon failure":  ("climatic",  1.22),
    "export ban":       ("policy_down", 0.82),
    "import duty cut":  ("policy_down", 0.88),
    "msp hike":         ("policy_up",  1.10),
}


# ---------------------------------------------------------------------------
# RESOURCE LOADER
# ---------------------------------------------------------------------------

def get_resources(commodity: str) -> dict:
    commodity = commodity.upper()
    if commodity in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[commodity]

    print(f"📦 Loading resources for {commodity}...")

    adj_path = f"mandi_adjacency_{commodity.lower()}.npz"
    idx_path = f"mandi_adjacency_index_{commodity.lower()}.txt"

    if os.path.exists(adj_path):
        adj = sparse.load_npz(adj_path).tocsr()
        row, col = adj.nonzero()
        edge_index  = torch.tensor(np.array([row, col]), dtype=torch.long).to(DEVICE)
        edge_weight = torch.tensor(adj.data, dtype=torch.float32).to(DEVICE)
    else:
        N = 100
        adj         = sparse.eye(N, format="csr")
        edge_index  = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
        edge_weight = torch.empty((0,), dtype=torch.float32).to(DEVICE)

    N = adj.shape[0]

    market_names = []
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            market_names = [l.strip() for l in f if l.strip()]
    market_to_id = {m.strip().upper(): i for i, m in enumerate(market_names)}

    # Model — prefer finetune_best → finetune → pretrain_best → pretrain
    model = MandiFlowNet(
        node_features=NODE_FEATURES, hidden_dim=64, output_dim=4, lookback=7
    ).to(DEVICE)
    model.eval()

    for path in [
        f"mandiflow_gcn_lstm_{commodity.lower()}_finetune_best.pth",
        f"mandiflow_gcn_lstm_{commodity.lower()}_finetune.pth",
        f"mandiflow_gcn_lstm_{commodity.lower()}_pretrain_best.pth",
        f"mandiflow_gcn_lstm_{commodity.lower()}_pretrain.pth",
    ]:
        if os.path.exists(path):
            try:
                model.load_state_dict(
                    torch.load(path, map_location=DEVICE, weights_only=True)
                )
                print(f"   ✅ Model: {path}")
                break
            except Exception as e:
                print(f"   ⚠️  Could not load {path}: {e}")

    # Anchor prices — most recent day's per-node Rs values
    anchor_prices = None
    for window in ("finetune", "pretrain"):
        ap = f"{commodity.lower()}_{window}_anchors.npy"
        if os.path.exists(ap):
            anchor_prices = np.load(ap)[-1, :]   # (N,)
            print(f"   Anchors: {ap}  [₹{anchor_prices.min():.0f}–₹{anchor_prices.max():.0f}]")
            break

    if anchor_prices is None:
        anchor_prices = np.full(N, 1500.0, dtype=np.float32)
        print("   ⚠️  No anchor prices — using ₹1500 fallback.")

    if len(anchor_prices) != N:
        if len(anchor_prices) > N:
            anchor_prices = anchor_prices[:N]
        else:
            anchor_prices = np.concatenate([
                anchor_prices,
                np.full(N - len(anchor_prices), np.median(anchor_prices))
            ])

    _RESOURCE_CACHE[commodity] = {
        "adj": adj, "edge_index": edge_index, "edge_weight": edge_weight,
        "market_names": market_names, "market_to_id": market_to_id,
        "model": model, "anchor_prices": anchor_prices, "N": N,
    }
    return _RESOURCE_CACHE[commodity]


# ---------------------------------------------------------------------------
# NODE RESOLUTION
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = (text or "").upper().strip().replace("&", "AND")
    text = re.sub(r"[^A-Z0-9() ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_market(raw: str, market_to_id: dict, district: str = "") -> tuple:
    """4-tier: exact → alias → substring → fuzzy. Returns (idx, name, strategy)."""
    candidates = []
    for key in [_norm(raw), _norm(district)]:
        candidates.extend(LOCATION_ALIASES.get(key, []))
        if key.endswith(" DISTRICT"):
            candidates.append(key.replace(" DISTRICT", ""))
        candidates.append(key)

    seen, deduped = set(), []
    for c in candidates:
        c = _norm(c)
        if c and c not in seen:
            seen.add(c); deduped.append(c)

    for c in deduped:
        if c in market_to_id:
            return market_to_id[c], c, "exact"

    for c in deduped:
        for k, idx in market_to_id.items():
            if c in k or k in c:
                return idx, k, "substring"

    if _norm(raw):
        norm_lookup = {_norm(k): (k, v) for k, v in market_to_id.items()}
        matches = difflib.get_close_matches(_norm(raw), list(norm_lookup), n=1, cutoff=0.72)
        if matches:
            orig_k, idx = norm_lookup[matches[0]]
            return idx, orig_k, "fuzzy"

    return None, None, f"No match for '{raw}'"


# ---------------------------------------------------------------------------
# SHOCK INJECTION
# ---------------------------------------------------------------------------

def build_input_tensor(
    N:             int,
    target_idx,    # int or list[int]
    shock_type:    int,
    severity:      float,
    anchor_prices: np.ndarray,
    is_global:     bool = False,
) -> torch.Tensor:
    """
    Builds the (N, 7, NODE_FEATURES) input tensor for inference.

    Price features: set to 1.0 for all nodes (no-change prior for days 0–6)
    Shock context:  injected at timestep 6 (the "today" the model sees)

    For policy_down shocks: non-epicenter nodes get policy_up context
    because consumer mandis will see price increases when origin supply
    drops due to the ban. The model learns this asymmetry from training.
    """
    x = np.ones((N, 7, NODE_FEATURES), dtype=np.float32)
    # Features 1–6 default to zero (no shock)
    x[:, :, 1:] = 0.0

    indices = target_idx if isinstance(target_idx, list) else [target_idx]

    for n in range(N):
        is_epicenter = n in indices

        # Determine what shock context this node gets
        if is_epicenter:
            node_shock_type = shock_type
            node_severity   = severity
        else:
            if shock_type == SHOCK_POLICY_DOWN:
                # Consumer/destination nodes: they see supply reduction
                # which raises their prices (opposite of origin)
                node_shock_type = SHOCK_POLICY_UP
                node_severity   = severity * 0.4
            elif shock_type in (SHOCK_CLIMATIC, SHOCK_LOGISTICS):
                # Propagating shock — neighbors get partial severity
                node_shock_type = shock_type
                node_severity   = severity * 0.3
            else:
                node_shock_type = SHOCK_NONE
                node_severity   = 0.0

        if node_shock_type != SHOCK_NONE:
            vec = make_shock_vector(
                shock_type   = node_shock_type,
                is_epicenter = is_epicenter,
                severity     = node_severity,
            )
            x[n, 6, 1] = vec[0]   # is_epicenter
            x[n, 6, 2:] = vec[1:] # shock type + severity

        # Apply impact to last price timestep for epicenter
        if is_epicenter:
            if shock_type in (SHOCK_CLIMATIC, SHOCK_LOGISTICS, SHOCK_POLICY_UP):
                x[n, 6, 0] = 1.0 + severity    # price spiked up
            elif shock_type == SHOCK_POLICY_DOWN:
                x[n, 6, 0] = max(0.3, 1.0 - severity * 0.5)  # price fell

    return torch.tensor(x, dtype=torch.float32).to(DEVICE)


# ---------------------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------------------

def simulate_shock(
    news_text:  str,
    doc_text:   str = "",
    commodity:  str = "ONION",
    explicit_origin: str = None
) -> dict:
    combined = f"{news_text} {doc_text}".strip()
    res      = get_resources(commodity)

    model         = res["model"]
    edge_index    = res["edge_index"]
    edge_weight   = res["edge_weight"]
    adj           = res["adj"]
    market_names  = res["market_names"]
    market_to_id  = res["market_to_id"]
    anchor_prices = res["anchor_prices"]
    N             = res["N"]

    # 1. ALWAYS extract features via NewsAnalyzer first so we don't lose the focused origin
    features_json     = analyzer.extract_shock_features(combined)
    raw_shock_type    = features_json.get("shock_type", "climatic")
    impact_multiplier = float(features_json.get("impact_multiplier", 1.2))
    
    # Use explicit UI dropdown origin if provided, completely skipping the unreliable LLM round-trip
    if explicit_origin:
        extracted_origin = explicit_origin.strip()
    else:
        extracted_origin = str(features_json.get("origin_mandi", "")).strip()
        
    extracted_district= str(features_json.get("origin_district", "")).strip()

    # Resolve target explicitly
    target_idx = None
    resolved_name = "Unknown"
    resolution_error = None
    if extracted_origin.upper() != "INDIA" and extracted_origin.upper() != "GLOBAL" and extracted_origin != "":
        target_idx, resolved_name, strategy = resolve_market(
            extracted_origin, market_to_id, extracted_district
        )
        if target_idx is None:
            resolution_error = strategy
            print(f"   ⚠️  {resolution_error}.")
        else:
            print(f"   ✅ Focus Node: {extracted_origin} → {resolved_name} ({strategy})")

    # 2. Check global keywords to see if the entire graph should be shocked
    combined_lower = combined.lower()
    global_type, global_mult = None, 1.0
    for kw, (stype, mult) in GLOBAL_SHOCK_KEYWORDS.items():
        if kw in combined_lower:
            global_type = stype; global_mult = mult; break

    if global_type:
        shock_type_int = SHOCK_TYPE_MAP.get(global_type, SHOCK_CLIMATIC)
        severity       = compute_severity(global_mult)
        target_indices = list(range(N))  # Shock ALL nodes in the graph
        
        # If user selected a specific mandi, keep focus on it. Otherwise default to GLOBAL median.
        origin_name = resolved_name if target_idx is not None else "GLOBAL"
    else:
        # Standard Local Shock
        shock_type_int = SHOCK_TYPE_MAP.get(raw_shock_type.lower(), SHOCK_CLIMATIC)
        severity       = compute_severity(impact_multiplier)
        
        if target_idx is None:
            # Fallback to mostly connected hub if no origin found
            degrees = np.array(adj.sum(axis=1)).flatten()
            target_idx = int(np.argmax(degrees))
            resolved_name = market_names[target_idx] if market_names else "Unknown"
            resolution_error = "Origin fallback to hub"
            
        target_indices = [target_idx]
        origin_name    = resolved_name

    # 3. Build input tensor with shock context
    x = build_input_tensor(
        N=N, target_idx=target_indices,
        shock_type=shock_type_int, severity=severity,
        anchor_prices=anchor_prices,
        is_global=(global_type is not None),
    )

    # 4. Forward pass
    pred_mag, pred_dir_logits = model(x, edge_index, edge_weight)
    pred_mag_np  = np.clip(pred_mag.detach().cpu().numpy(), *RATIO_CLIP)    # (N, 4)
    pred_dir_np  = torch.sigmoid(pred_dir_logits).detach().cpu().numpy()    # (N, 4) prob up

    # 5. Denormalize — per-node anchor prices
    pred_abs = pred_mag_np * anchor_prices[:, np.newaxis]   # (N, 4) Rs

    # 6. Build result
    def format_forecast(node_idx):
        prices   = pred_abs[node_idx].tolist()
        dirs     = pred_dir_np[node_idx].tolist()
        base     = float(anchor_prices[node_idx])

        # Apply absolute economic guardrails so ML doesn't hallucinate
        is_epi = (node_idx in target_indices)
        prices, dirs = apply_economic_constraints(prices, dirs, base, shock_type_int, is_epi)

        return {
            "prices":     [round(p, 0) for p in prices],
            "directions": ["↑" if d > 0.5 else "↓" for d in dirs],
            "base_price": round(base, 0),
        }

    if origin_name == "GLOBAL":
        origin_data = {
            "prices":     [round(p, 0) for p in pred_abs.mean(axis=0).tolist()],
            "directions": ["↑" if d > 0.5 else "↓"
                           for d in pred_dir_np.mean(axis=0).tolist()],
            "base_price": round(float(np.median(anchor_prices)), 0),
        }
    else:
        origin_data = format_forecast(target_idx)

    result = {
        "features":        features_json,
        "origin_name":     origin_name,
        "origin_base":     origin_data["base_price"],
        "origin_forecast": origin_data["prices"],
        "origin_direction":origin_data["directions"],
        "served_areas":    [],
    }
    if resolution_error:
        result["resolution_error"] = resolution_error

    # 7. Neighbor propagation
    if target_idx is not None and adj.shape[0] > target_idx:
        row  = adj[target_idx]
        conns = sorted(zip(row.indices, row.data), key=lambda x: x[1], reverse=True)
        for neighbor_id, _ in conns[:5]:
            ndata = format_forecast(neighbor_id)
            nname = market_names[neighbor_id] if neighbor_id < len(market_names) \
                    else f"Node_{neighbor_id}"
            result["served_areas"].append({
                "mandi":      nname,
                "base_price": ndata["base_price"],
                "forecast":   ndata["prices"],
                "direction":  ndata["directions"],
            })

    return result


# ---------------------------------------------------------------------------
# CLI TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Crop destroyed in Nashik due to floods, onion prices expected to rise", "ONION"),
        ("Government bans onion export with immediate effect", "ONION"),
        ("Truckers call nationwide strike, all highway transport halted", "ONION"),
    ]

    for news, comm in tests:
        print(f"\n{'─'*65}")
        print(f"NEWS: {news}")
        r = simulate_shock(news, commodity=comm)
        print(f"Origin: {r['origin_name']}  (base ₹{r['origin_base']:.0f})")
        dirs = r['origin_direction']
        prs  = r['origin_forecast']
        for d_idx, (p, d) in enumerate(zip(prs, dirs)):
            print(f"  Day {d_idx+1}: {d} ₹{p:.0f}")
        for sa in r["served_areas"]:
            print(f"  → {sa['mandi']:<35}", end="")
            for p, d in zip(sa["forecast"], sa["direction"]):
                print(f"  {d}₹{p:.0f}", end="")
            print()
        if "resolution_error" in r:
            print(f"  ⚠️  {r['resolution_error']}")