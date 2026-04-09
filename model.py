import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MandiFlowNet(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim):
        super(MandiFlowNet, self).__init__()
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Spatial Brain: Graph Convolutional Layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Stability Layers
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Temporal Brain: LSTM sequence layer
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Regressor for forecasting (Outputting Ratios)
        self.regressor = nn.Linear(hidden_dim, output_dim)
        
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        # 0. Device Mapping
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
            
        # 1. Spatial Pass (Graph Context)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.leaky_relu(x, negative_slope=0.01) # Prevents "Dead Neuron" collapse
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.layer_norm(x) # Stabilizes training math
        
        # 2. Temporal Pass (Sequential context)
        # Reshape to [nodes, sequence_len, features]
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        
        # Use only the last sequence state
        final_state = lstm_out[:, -1, :]
        
        # 3. Output Projection (The Fix for Negative Values)
        # We apply ReLU to ensure the ratio is always positive.
        # Adding 1e-4 ensures we never divide by or return a pure zero.
        out = self.regressor(final_state)
        return torch.relu(out) + 1e-4

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"✅ MandiFlowNet v1.2 (Constraint Enabled) loaded on {device}")