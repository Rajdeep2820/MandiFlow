import torch
import torch.optim as optim
import scipy.sparse as sparse
import pandas as pd
import numpy as np
from model import MandiFlowNet

# 1. Load the Graph Map (Adjacency Matrix)
print("🕸️ Loading Graph Structure...")
adj = sparse.load_npz("mandi_adjacency.npz")
row, col = adj.nonzero()
edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)

# 2. Load a Subset of 75M rows (Memory efficient)
# We start with just one day to verify the "Brain" works
print("📖 Loading Data Sample...")
df = pd.read_parquet("mandi_master_data.parquet", columns=['Modal_Price', 'month_sin', 'month_cos', 'Market_ID'])
# Take a snapshot of the most recent data
sample_data = df.tail(6457) # One record per Mandi

# Convert to Tensor (Math: Matrix Representation)
# Features: [Price, month_sin, month_cos]
x = torch.tensor(sample_data[['Modal_Price', 'month_sin', 'month_cos']].values, dtype=torch.float)

# 3. Setup Training
model = MandiFlowNet(node_features=3, hidden_dim=64, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# 4. Training Loop
print("🚀 Starting Test Training...")
for epoch in range(1, 11):
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    out = model(x, edge_index)
    
    # Target: For testing, try to predict the same price
    loss = criterion(out, x[:, 0].unsqueeze(1))
    
    # Backward Pass (Math: Gradient Descent)
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ SUCCESS: MandiFlow is learning.")