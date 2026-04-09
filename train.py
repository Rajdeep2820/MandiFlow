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
    
    dataset = MandiParquetDataset("mandi_master_data.parquet", commodity=commodity)
    model = MandiFlowNet(node_features=7, hidden_dim=64, output_dim=4).to(device)

    weights_path = f"mandiflow_gcn_lstm_{commodity.lower()}.pth"
    if os.path.exists(weights_path):
        # Use weights_only=True for security and compatibility
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"🔄 Resuming training from: {weights_path}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss() 
    
    print(f"Initiating historical data stream for {commodity}...")
    best_loss = float('inf')

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            count = 0
            start_time = time.time()
            
            for batch_graph in dataset:
                # 1. MOVE .to(device) UP
                batch_graph = batch_graph.to(device) 

                # 2. REMOVE THE allclose CHECK. 
                # Even if prices are the same, the model needs to learn that "No Change" is a valid prediction.

                optimizer.zero_grad()
                
                # 3. ROBUST BASELINE
                # We use the last known price to normalize the scale
                base_prices = batch_graph.x[:, -1].unsqueeze(1).detach().clamp(min=1.0)
                
                # Normalize data (Scale 0 to 2 usually)
                x_norm = batch_graph.x / base_prices
                y_norm = batch_graph.y / base_prices
                
                # 4. FORWARD PASS
                predictions = model(x_norm, batch_graph.edge_index, batch_graph.edge_weight)
                
                # 5. LOSS + GRADIENT CLIPPING
                loss = criterion(predictions, y_norm)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # Tighter clipping
                optimizer.step()
                
                epoch_loss += loss.item()
                count += 1
                
                # print updates more frequently to see it working
                if count % 10 == 0: 
                    print(f"Epoch [{epoch}/{epochs}] | Batch {count} | Loss: {loss.item():.6f}")

            # Save Checkpoints
            if count > 0 and (epoch_loss / count) < best_loss:
                best_loss = epoch_loss / count
                torch.save(model.state_dict(), f"mandiflow_gcn_lstm_{commodity.lower()}_best.pth")

    except KeyboardInterrupt:
        print("\n🛑 Manual interrupt. Saving checkpoint...")

    torch.save(model.state_dict(), weights_path)
    print(f"✅ Training Complete. Model saved to {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", type=str, default="ONION")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args.commodity, epochs=args.epochs, lr=args.lr)