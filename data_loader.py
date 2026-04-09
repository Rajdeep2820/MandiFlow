import torch
from torch.utils.data import IterableDataset
import numpy as np
from torch_geometric.data import Data
import scipy.sparse
import os

class MandiParquetDataset(IterableDataset):
    def __init__(self, parquet_path, commodity="ONION"):
        super().__init__()
        self.commodity = commodity.upper()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.matrix_file = f"{self.commodity.lower()}_training_matrix.npy"
        adj_path = f"mandi_adjacency_{self.commodity.lower()}.npz"
        
        # Load Adjacency
        if os.path.exists(adj_path):
            self.adj = scipy.sparse.load_npz(adj_path)
        else:
            self.adj = scipy.sparse.eye(1088, format='csr')

        coo = self.adj.tocoo()
        self.edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        self.edge_weight = torch.tensor(coo.data, dtype=torch.float32)
        
    def __iter__(self):
        if not os.path.exists(self.matrix_file):
            print(f"❌ Loader Error: {self.matrix_file} not found!")
            return

        prices_array = np.load(self.matrix_file)
        T, N = prices_array.shape
        
        # Parameters must match your model output_dim
        lookback = 7
        max_lookahead = 4 
        
        # DEBUG: Print this once to confirm the loop starts
        print(f"📊 Dataset Info: {T} days found. Starting window slide...")

        for t in range(lookback, T - max_lookahead):
            # X: [Nodes, 7] | Y: [Nodes, 4]
            x_features = prices_array[t-lookback : t, :].T 
            
            y_targets = np.column_stack((
                prices_array[t + 1, :],
                prices_array[t + 2, :],
                prices_array[t + 3, :],
                prices_array[t + 4, :]
            ))
            
            # 🟢 RELAXED LEAKAGE CHECK (Prevents the 1-second finish)
            # We only skip if the data is purely 100% identical zeros or NaNs
            if np.isnan(x_features).any() or np.isnan(y_targets).any():
                continue

            yield Data(
                x=torch.tensor(x_features, dtype=torch.float32),
                edge_index=self.edge_index,
                edge_weight=self.edge_weight,
                y=torch.tensor(y_targets, dtype=torch.float32)
            ).to(self.device)

if __name__ == "__main__":
    # Test script to verify the loader works independently
    MASTER_FILE = "mandi_master_data.parquet"
    # Change commodity here to test different files
    test_dataset = MandiParquetDataset(MASTER_FILE, commodity="ONION")
    
    print(f"🚀 Initializing test stream for {test_dataset.commodity}...")
    found = False
    for i, batch in enumerate(test_dataset):
        print(f"✅ Batch {i} Loaded: X={batch.x.shape}, Y={batch.y.shape}")
        found = True
        if i >= 2: break
        
    if not found:
        print("Empty stream: Check if .npy file exists and contains valid data.")