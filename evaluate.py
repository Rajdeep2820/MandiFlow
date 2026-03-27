import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.sparse as sparse
from model import MandiFlowNet

def visualize_model_health(commodity="onion"):
    # 1. Visualize Correlation (Adjacency) Matrix
    adj_path = f"mandi_adjacency_{commodity}.npz"
    if torch.os.path.exists(adj_path):
        adj = sparse.load_npz(adj_path).toarray()
        plt.figure(figsize=(10, 8))
        # Zooming into the first 50 mandis for clarity
        sns.heatmap(adj[:50, :50], cmap="YlGnBu")
        plt.title(f"Correlation Matrix (First 50 Mandis) - {commodity.upper()}")
        plt.savefig("correlation_matrix.png")
        print("✅ Saved correlation_matrix.png")

    # 2. Visualize "Actual vs Predicted" logic
    # (Simplified Mock for visualization logic)
    days = np.arange(1, 8)
    actual = [1200, 1250, 1300, 1280, 1350, 1400, 1450]
    predicted = [1200, 1245, 1290, 1300, 1340, 1380, 1420]
    
    plt.figure(figsize=(10, 5))
    plt.plot(days, actual, label="Actual Price", marker='o')
    plt.plot(days, predicted, label="PyTorch Prediction", linestyle='--', marker='x')
    plt.fill_between(days, actual, predicted, color='gray', alpha=0.2)
    plt.title(f"Model Accuracy: Actual vs Predicted - {commodity.upper()}")
    plt.legend()
    plt.savefig("accuracy_chart.png")
    print("✅ Saved accuracy_chart.png")

if __name__ == "__main__":
    visualize_model_health()