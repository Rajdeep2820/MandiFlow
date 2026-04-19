"""
data_loader.py  —  MandiFlow v3.0
===================================
Shock-aware dataset. Each sample yields:

  x:              (N, 7, 7)  — N nodes, 7 timesteps, 7 features
                               [price_ratio, is_epicenter, climatic,
                                logistics, policy_up, policy_down, severity]
  y_magnitude:    (N, 4)     — target price ratios days +1 to +4
  y_direction:    (N, 4)     — 1.0 if price went up, 0.0 if down
  anchor:         (N,)       — Rs prices for denormalization
  anchor_mask:    (N,)       — bool, high-coverage nodes
  is_shock:       bool
  shock_type:     int

Stratified sampling: 70% normal days, 30% shock days per epoch.
Quiet period:     Year 2018 held out — never used in training.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from shock_labels import (
    label_training_data,
    make_shock_vector,
    compute_severity,
    SHOCK_NONE,
)
from model import NODE_FEATURES

QUIET_YEAR       = 2018
SHOCK_BATCH_FRAC = 0.30
LOOKBACK         = 7
HORIZON          = 4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MandiParquetDataset(IterableDataset):

    def __init__(self, commodity: str = "ONION"):
        super().__init__()
        self.commodity = commodity.upper()

        # Adjacency
        adj_path = f"mandi_adjacency_{self.commodity.lower()}.npz"
        if os.path.exists(adj_path):
            adj = scipy.sparse.load_npz(adj_path)
        else:
            idx_path = f"mandi_adjacency_index_{self.commodity.lower()}.txt"
            N = sum(1 for _ in open(idx_path)) if os.path.exists(idx_path) else 100
            adj = scipy.sparse.eye(N, format="csr")
            print(f"⚠️  No adjacency found — using identity for {N} nodes.")

        coo = adj.tocoo()
        self.edge_index  = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        self.edge_weight = torch.tensor(coo.data, dtype=torch.float32)
        self.N           = adj.shape[0]

        # Market names
        idx_path = f"mandi_adjacency_index_{self.commodity.lower()}.txt"
        self.market_names = []
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                self.market_names = [l.strip() for l in f if l.strip()]

        # Anchor mask
        mask_path = f"{self.commodity.lower()}_anchor_mask.npy"
        if os.path.exists(mask_path):
            self.anchor_mask = torch.tensor(np.load(mask_path), dtype=torch.bool)
        else:
            self.anchor_mask = torch.ones(self.N, dtype=torch.bool)

        # Load and label windows
        self.windows = []
        for window_name in ("pretrain", "finetune"):
            matrix_path = f"{self.commodity.lower()}_{window_name}_matrix.npy"
            anchor_path = f"{self.commodity.lower()}_{window_name}_anchors.npy"
            dates_path  = f"{self.commodity.lower()}_{window_name}_dates.npy"

            if not os.path.exists(matrix_path):
                print(f"ℹ️  {window_name} matrix not found — skipping.")
                continue

            matrix  = np.load(matrix_path)   # (T, N)
            anchors = np.load(anchor_path)   # (T, N)
            T       = matrix.shape[0]

            # Date index
            if os.path.exists(dates_path):
                dates = pd.DatetimeIndex(np.load(dates_path, allow_pickle=True))
            else:
                start = "2010-01-02" if window_name == "pretrain" else "2021-01-02"
                dates = pd.date_range(start=start, periods=T, freq="D")

            # Label shock events
            print(f"\n🏷️  Labeling {window_name} window...")
            labels = label_training_data(
                ratio_matrix = matrix,
                anchor_mask  = self.anchor_mask.numpy(),
                market_names = self.market_names,
                dates        = dates,
                commodity    = self.commodity,
            )

            # Split: quiet year | normal training | shock training
            quiet_mask = np.array([d.year == QUIET_YEAR for d in dates])
            valid      = (np.arange(T) >= LOOKBACK) & (np.arange(T) <= T - HORIZON - 1)

            shock_idx  = np.where(labels["is_shock_day"] & ~quiet_mask & valid)[0]
            normal_idx = np.where(~labels["is_shock_day"] & ~quiet_mask & valid)[0]
            quiet_idx  = np.where(quiet_mask & valid)[0]

            print(f"   Normal: {len(normal_idx)} | Shock: {len(shock_idx)} | "
                  f"Quiet ({QUIET_YEAR}): {len(quiet_idx)}")

            self.windows.append({
                "name":       window_name,
                "matrix":     matrix,
                "anchors":    anchors,
                "dates":      dates,
                "labels":     labels,
                "shock_idx":  shock_idx,
                "normal_idx": normal_idx,
                "quiet_idx":  quiet_idx,
            })

    # -------------------------------------------------------------------------

    def _build_sample(self, window: dict, t: int) -> Data:
        matrix  = window["matrix"]
        anchors = window["anchors"]
        labels  = window["labels"]
        N       = self.N

        price_window  = matrix[t - LOOKBACK : t, :]    # (7, N)
        x = np.zeros((N, LOOKBACK, NODE_FEATURES), dtype=np.float32)
        x[:, :, 0] = price_window.T                    # price ratios

        shock_type   = int(labels["shock_types"][t])
        event_sev    = float(labels["severities"][t])
        epicenter_row = labels["epicenter_mask"][t]    # (N,) bool

        if shock_type != SHOCK_NONE:
            for n in range(N):
                node_ratio = float(price_window[-1, n])
                node_sev   = compute_severity(node_ratio)
                if epicenter_row[n]:
                    final_sev = max(event_sev, node_sev)
                else:
                    final_sev = min(node_sev, event_sev * 0.7)

                vec = make_shock_vector(
                    shock_type   = shock_type,
                    is_epicenter = bool(epicenter_row[n]),
                    severity     = final_sev,
                )
                x[n, -1, 1]  = vec[0]    # is_epicenter  → slot 1
                x[n, -1, 2:] = vec[1:]   # 5 shock features → slots 2-6

        y_ratio     = matrix[t : t + HORIZON, :].T          # (N, 4)
        y_direction = (y_ratio > 1.0).astype(np.float32)
        anchor      = anchors[t - 1, :]

        return Data(
            x           = torch.tensor(x,           dtype=torch.float32),
            y_magnitude = torch.tensor(y_ratio,     dtype=torch.float32),
            y_direction = torch.tensor(y_direction, dtype=torch.float32),
            anchor      = torch.tensor(anchor,      dtype=torch.float32),
            anchor_mask = self.anchor_mask,
            edge_index  = self.edge_index,
            edge_weight = self.edge_weight,
            is_shock    = (shock_type != SHOCK_NONE),
            shock_type  = shock_type,
        )

    # -------------------------------------------------------------------------

    def __iter__(self):
        rng = np.random.default_rng(seed=42)
        for window in self.windows:
            print(f"\n   📂 Streaming {window['name']} (70% normal / 30% shock)...")
            shock_idx  = window["shock_idx"].copy()
            normal_idx = window["normal_idx"].copy()
            rng.shuffle(shock_idx)
            rng.shuffle(normal_idx)

            si = 0
            ni = 0
            while ni < len(normal_idx):
                # 7 normal
                for _ in range(7):
                    if ni >= len(normal_idx):
                        break
                    yield self._build_sample(window, normal_idx[ni]).to(DEVICE)
                    ni += 1
                # 3 shock (cycle if pool exhausted)
                for _ in range(3):
                    if len(shock_idx) == 0:
                        break
                    yield self._build_sample(window, shock_idx[si % len(shock_idx)]).to(DEVICE)
                    si += 1

    def iter_quiet_period(self):
        """Yields 2018 samples with no shock context for ghost-shock validation."""
        for window in self.windows:
            for t in window["quiet_idx"]:
                matrix  = window["matrix"]
                anchors = window["anchors"]
                x = np.zeros((self.N, LOOKBACK, NODE_FEATURES), dtype=np.float32)
                x[:, :, 0] = matrix[t - LOOKBACK : t, :].T
                y_ratio     = matrix[t : t + HORIZON, :].T
                y_direction = (y_ratio > 1.0).astype(np.float32)
                yield Data(
                    x           = torch.tensor(x,           dtype=torch.float32),
                    y_magnitude = torch.tensor(y_ratio,     dtype=torch.float32),
                    y_direction = torch.tensor(y_direction, dtype=torch.float32),
                    anchor      = torch.tensor(anchors[t-1,:], dtype=torch.float32),
                    anchor_mask = self.anchor_mask,
                    edge_index  = self.edge_index,
                    edge_weight = self.edge_weight,
                    is_shock    = False,
                    shock_type  = 0,
                ).to(DEVICE)


if __name__ == "__main__":
    import sys
    commodity = sys.argv[1].upper() if len(sys.argv) > 1 else "ONION"
    ds = MandiParquetDataset(commodity=commodity)
    n, s = 0, 0
    for i, b in enumerate(ds):
        if b.is_shock: s += 1
        else: n += 1
        if i == 0:
            print(f"x: {b.x.shape}  y_mag: {b.y_magnitude.shape}  y_dir: {b.y_direction.shape}")
        if i >= 99: break
    print(f"Normal: {n} ({100*n/(n+s):.0f}%)  Shock: {s} ({100*s/(n+s):.0f}%)")
    print("✅ DataLoader v3.0 OK")