import torch
from torch.utils.data import IterableDataset
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import scipy.sparse
import os

class MandiParquetDataset(IterableDataset):
    """
    Memory-safe IterableDataset to stream 808 MB Parquet file lazily.
    Extracts trailing 7-day prices (X) and 1,3,5,7-day lookahead targets (Y).
    """
    def __init__(self, parquet_path, commodity="ONION"):
        super().__init__()
        self.parquet_path = parquet_path
        self.commodity = commodity.upper()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        adjacency_path = f"mandi_adjacency_{self.commodity.lower()}.npz"
        index_path = f"mandi_adjacency_index_{self.commodity.lower()}.txt"
        
        # Load Adjacency matrix for graph structure
        if os.path.exists(adjacency_path):
            self.adj = scipy.sparse.load_npz(adjacency_path)
            with open(index_path, "r") as f:
                self.market_names = [line.strip() for line in f]
        else:
            print(f"⚠️ Warning: Adjacency file {adjacency_path} not found. Using identity.")
            self.adj = scipy.sparse.eye(10, format='csr')
            self.market_names = [f"Market_{i}" for i in range(10)]

        coo = self.adj.tocoo()
        self.edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        self.edge_weight = torch.tensor(coo.data, dtype=torch.float32)
        
    def __iter__(self):
        dataset = ds.dataset(self.parquet_path, format="parquet")
        
        # Load in chunks of 1 Million rows (roughly several months/years depending on volume)
        for batch in dataset.to_batches(batch_size=1000000):
            df = batch.to_pandas()
            
            # Filter strictly for the requested commodity to match the adjacency logic
            if 'Commodity' in df.columns:
                df = df[df['Commodity'].str.upper() == self.commodity]
            
            if df.empty: continue
            
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
            df['Market'] = df['Market'].astype(str).str.upper().str.strip()
            
            # Aggregate to daily modal prices
            daily = df.groupby(["Arrival_Date", "Market"])["Modal_Price"].mean().reset_index()
            
            # Pivot to Time-Series Matrix: [Days, Markets]
            pivot = daily.pivot(index="Arrival_Date", columns="Market", values="Modal_Price")
            
            # Align exactly to the Adjacency Matrix Market Names
            pivot = pivot.reindex(columns=self.market_names)
            
            # Continuous daily resampling & filling
            pivot = pivot.resample('D').mean().ffill(limit=7).fillna(1500.0)
            
            prices_array = pivot.values.astype(np.float32) # [T, N]
            T, N = prices_array.shape
            
            lookback = 7
            max_lookahead = 7
            
            if T <= lookback + max_lookahead:
                continue
                
            # Yield daily sliding window graphs for this chunk
            for t in range(lookback, T - max_lookahead):
                # X: trailing 7 days [N, 7]
                x_features = prices_array[t-lookback : t, :].T 
                
                # Y: target future prices at +1, +3, +5, +7 days [N, 4]
                y_targets = np.column_stack((
                    prices_array[t + 1, :],
                    prices_array[t + 3, :],
                    prices_array[t + 5, :],
                    prices_array[t + 7, :]
                ))
                
                graph_data = Data(
                    x=torch.tensor(x_features, dtype=torch.float32),
                    edge_index=self.edge_index,
                    edge_weight=self.edge_weight,
                    y=torch.tensor(y_targets, dtype=torch.float32)
                )
                
                yield graph_data.to(self.device)

if __name__ == "__main__":
    dataset = MandiParquetDataset("mandi_master_data.parquet", "mandi_adjacency_correlated.npz", "mandi_adjacency_index.txt")
    print("Dataset initialized. Yielding first graph:")
    for data in dataset:
        print(f"X shape: {data.x.shape}, Y shape: {data.y.shape}, Edges: {data.edge_index.shape[1]}")
        break
