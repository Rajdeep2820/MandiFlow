import torch
import scipy.sparse as sparse
import numpy as np
from model import MandiFlowNet
from news_encoder import NewsEncoder

# 1. SETUP THE ENVIRONMENT
adj = sparse.load_npz("mandi_adjacency.npz")
row, col = adj.nonzero()
edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)

# Initialize Brain & News Encoder
model = MandiFlowNet(node_features=3, hidden_dim=64, output_dim=1)
news_enc = NewsEncoder()

def simulate_shock(target_mandi_id, news_headline, base_price=2000.0):
    model.eval()
    
    # 2. Convert Headline to a "Shock Multiplier"
    # Math: We simplify the 768 BERT vector into a single scalar impact factor
    news_vector = news_enc.get_shock_vector(news_headline)
    # If headline contains 'rain' or 'strike', we simulate a price hike factor
    impact_factor = 1.2 if "rain" in news_headline.lower() or "strike" in news_headline.lower() else 1.0
    
    # 3. Create a Graph Snapshot
    # All nodes start with 'base_price', but 'target_mandi_id' gets the shock
    x = torch.full((adj.shape[0], 3), base_price) # [Nodes, Features]
    x[target_mandi_id, 0] = base_price * impact_factor
    
    # 4. RUN THE PROPAGATION (The Forward Pass)
    with torch.no_grad():
        predicted_prices = model(x, edge_index)
    
    # 5. Result: How much did the neighbors' price rise?
    print(f"--- Simulation for: '{news_headline}' ---")
    print(f"Target Mandi ({target_mandi_id}) New Price: {predicted_prices[target_mandi_id].item():.2f}")
    
    # Find neighbors using the Adjacency Matrix
    neighbors = adj[target_mandi_id].indices
    if len(neighbors) > 0:
        neighbor_price = predicted_prices[neighbors[0]].item()
        print(f"Ripple Effect: Neighboring Mandi {neighbors[0]} rose to: {neighbor_price:.2f}")

# Example Run: Mandsaur (Let's assume ID 42)
simulate_shock(42, "Heavy rainfall in Mandsaur disrupts garlic supply")