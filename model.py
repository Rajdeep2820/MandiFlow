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
        
        # Temporal Brain: LSTM sequence layer
        # batch_first=True expects input shape [batch/nodes, seq_len, features]
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Regressor for forecasting 1, 3, 5, 7 days (output_dim = 4)
        self.regressor = nn.Linear(hidden_dim, output_dim)
        
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        # Ensure tensors are explicitly mapped to Apple Silicon MPS
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
            
        # 1. Spatial Pass over graph structure
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # 2. Temporal Pass
        # Add a dummy sequence dimension [nodes, 1, hidden_dim]
        # In full training loop, x would be [nodes, seq_len, features]
        x = x.unsqueeze(1)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        final_state = lstm_out[:, -1, :]
        
        # 3. Final Prediction (4 time steps: 1, 3, 5, 7 days)
        return self.regressor(final_state)

if __name__ == "__main__":
    print("MandiFlowNet architecture with GCN-LSTM loaded successfully on", torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))