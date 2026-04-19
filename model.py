"""
model.py  —  MandiFlow v3.0
============================
Shock-aware Spatio-Temporal GNN.

Input per node per timestep:  7 features
  [0]     price ratio          (normalized price change)
  [1]     is_epicenter         (1.0 if shock origin node)
  [2]     shock_climatic       (1.0 if flood/drought/heatwave)
  [3]     shock_logistics      (1.0 if strike/blockade/lockdown)
  [4]     shock_policy_up      (1.0 if restriction raises prices)
  [5]     shock_policy_down    (1.0 if ban/cut lowers prices)
  [6]     severity             (0.0–1.0 shock magnitude)

On normal days features [1]–[6] are all 0.0. The model learns
"zeros = no shock" vs each shock type pattern.

Output per node:
  magnitude:  (N, 4) — predicted price ratios for days +1 to +4
  direction:  (N, 4) — logits for P(price goes UP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

NODE_FEATURES = 7   # 1 price + 6 shock context


class MandiFlowNet(nn.Module):

    def __init__(
        self,
        node_features: int   = NODE_FEATURES,
        hidden_dim:    int   = 64,
        output_dim:    int   = 4,
        lookback:      int   = 7,
        dropout:       float = 0.2,
    ):
        super().__init__()

        self.lookback      = lookback
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.node_features = node_features

        # Spatial layers — weight-shared across all 7 timesteps
        self.gcn1       = GCNConv(node_features, hidden_dim)
        self.gcn2       = GCNConv(hidden_dim,    hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Temporal layer
        self.lstm = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
        )

        # Magnitude head — predicts price ratios (regression)
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Direction head — predicts P(price up) as logits (classification)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x:           torch.Tensor,          # (N, lookback, node_features)
        edge_index:  torch.Tensor,          # (2, E)
        edge_weight: torch.Tensor = None,   # (E,)
    ) -> tuple:
        """
        Returns:
            magnitude:  (N, output_dim) — predicted ratios, always > 0
            direction:  (N, output_dim) — logits for P(price up)
        """
        N, T, n_feat = x.shape
        assert T == self.lookback,      f"Expected {self.lookback} timesteps, got {T}"
        assert n_feat == self.node_features, f"Expected {self.node_features} features, got {n_feat}"

        # 1. Spatial pass — GCN at each of the T timesteps
        gcn_outputs = []
        for t in range(T):
            h = x[:, t, :]                              # (N, node_features)
            h = self.gcn1(h, edge_index, edge_weight)
            h = F.leaky_relu(h, negative_slope=0.01)
            h = self.dropout(h)
            h = self.gcn2(h, edge_index, edge_weight)
            h = F.leaky_relu(h, negative_slope=0.01)
            h = self.layer_norm(h)
            gcn_outputs.append(h)                       # (N, hidden_dim)

        # 2. Temporal pass — LSTM over 7-step spatial embedding sequence
        x_seq       = torch.stack(gcn_outputs, dim=1)  # (N, T, hidden_dim)
        lstm_out, _ = self.lstm(x_seq)
        final_state = lstm_out[:, -1, :]                # (N, hidden_dim)

        # 3. Dual output
        magnitude = torch.relu(self.magnitude_head(final_state)) + 1e-4
        direction = self.direction_head(final_state)

        return magnitude, direction

    def predict(self, x, edge_index, edge_weight=None):
        """Inference wrapper — returns (magnitude, direction_probability)."""
        with torch.no_grad():
            mag, dir_logits = self.forward(x, edge_index, edge_weight)
            return mag, torch.sigmoid(dir_logits)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    N, T, F, E = 100, 7, NODE_FEATURES, 300
    model = MandiFlowNet().to(device)

    x           = torch.zeros(N, T, F).to(device)
    x[:, :, 0]  = 1.0  # price ratio = 1.0
    edge_index  = torch.randint(0, N, (2, E)).to(device)
    edge_weight = torch.rand(E).to(device)

    mag, dir_prob = model.predict(x, edge_index, edge_weight)
    print(f"Magnitude shape: {mag.shape}   — expected ({N}, 4)")
    print(f"Direction shape: {dir_prob.shape} — expected ({N}, 4)")
    print(f"Magnitude range: [{mag.min():.4f}, {mag.max():.4f}]  — should be > 0")
    print(f"Direction range: [{dir_prob.min():.4f}, {dir_prob.max():.4f}] — should be 0–1")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ MandiFlow v3.0 OK")