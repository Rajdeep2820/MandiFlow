import numpy as np
import scipy.sparse as sparse

# Check Matrix vs Index alignment
adj = sparse.load_npz("mandi_adjacency_onion.npz")
with open("mandi_adjacency_index_onion.txt", "r") as f:
    indices = f.readlines()

if adj.shape[0] == len(indices):
    print(f"✅ ALIGNED: {len(indices)} markets matched to {adj.shape[0]} matrix rows.")
else:
    print(f"❌ MISMATCH: Index has {len(indices)} but Matrix has {adj.shape[0]}. Fix build_graph.py first!")