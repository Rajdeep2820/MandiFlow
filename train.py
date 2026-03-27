import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import MandiParquetDataset
from model import MandiFlowNet
import time
import argparse
import os

def train(commodity="ONION", limit=None, epochs=50, lr=0.001):
    # 1. Pipeline Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Training for {commodity} on Hardware: {device}")
    
    # Ensure dataset handles internal scaling/normalization
    dataset = MandiParquetDataset("mandi_master_data.parquet", commodity=commodity)
    
    # Model (7 trailing days -> 4 future days)
    # node_features=7 (Daily price history)
    model = MandiFlowNet(node_features=7, hidden_dim=64, output_dim=4).to(device)

    # --- ADD THESE 3 LINES HERE ---
    weights_path = f"mandiflow_gcn_lstm_{commodity.lower()}.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"🔄 Resuming training from existing brain state: {weights_path}")
# ------------------------------
    
    # Optimizer with Weight Decay to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # HubberLoss is more robust to "outliers" (random price spikes) than MSE
    criterion = nn.HuberLoss() 
    
    print(f"Initiating historical data stream for {commodity}...")
    
    best_loss = float('inf')

    # 2. Training Loop
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            count = 0
            start_time = time.time()
            
            for batch_graph in dataset:
                optimizer.zero_grad()
                
                # Move to GPU (MPS)
                batch_graph = batch_graph.to(device) 
                
                # Normalize input locally (Scale by last day's price)
                # This helps the model learn 'Percentage shifts' instead of raw numbers
                base_prices = batch_graph.x[:, -1].unsqueeze(1) + 1e-5
                x_norm = batch_graph.x / base_prices
                y_norm = batch_graph.y / base_prices
                
                # Forward pass
                predictions = model(x_norm, batch_graph.edge_index, batch_graph.edge_weight)
                
                # Calculate loss based on relative shifts
                loss = criterion(predictions, y_norm)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
                
                if count % 100 == 0:
                    avg_loss = epoch_loss / count
                    elapsed = time.time() - start_time
                    print(f"Epoch [{epoch}/{epochs}] | Batch {count} | Avg Loss: {avg_loss:.4f} | {100/(time.time()-start_time + 1e-5):.1f} b/s")
                    start_time = time.time()

                if limit and count >= limit: break

            # Save "Best" version
            current_avg = epoch_loss / count
            if current_avg < best_loss:
                best_loss = current_avg
                torch.save(model.state_dict(), f"mandiflow_gcn_lstm_{commodity.lower()}_best.pth")

    except KeyboardInterrupt:
        print("\n🛑 Manual interrupt. Saving checkpoint...")

    # 3. Final Save
    weights_path = f"mandiflow_gcn_lstm_{commodity.lower()}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Training Complete. Model saved to {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", type=str, default="ONION")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.commodity, epochs=args.epochs, lr=args.lr)