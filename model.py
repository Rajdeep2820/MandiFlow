import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # The Spatial Brain

class MandiFlowNet(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim):
        super(MandiFlowNet, self).__init__()
        
        # Math: Graph Convolutional Layer
        # It aggregates features from a node's neighbors
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Regression head to predict the price
        self.regressor = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x: [Number of Mandis, 3 Features]
        # edge_index: [2, Number of Connections]
        
        # 1. First Spatial Pass + Activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2. Second Spatial Pass (Captures 2-hop neighbors)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 3. Final Prediction
        return self.regressor(x)

if __name__ == "__main__":
    print("MandiFlowNet architecture loaded successfully.")