import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import MandiParquetDataset
from model import MandiFlowNet
import time
import argparse

def train(commodity="ONION", limit=None, epochs=10):
    # 1. Pipeline Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Training for {commodity} on Hardware: {device}")
    
    dataset = MandiParquetDataset("mandi_master_data.parquet", commodity=commodity)
    
    # Model (7 trailing days -> 4 future days)
    model = MandiFlowNet(node_features=7, hidden_dim=64, output_dim=4).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Initiating historical data stream for {commodity}...")
    
    # 2. Training Loop with Epochs
    try:
        for epoch in range(1, epochs + 1):
            print(f"\n--- Starting Epoch {epoch}/{epochs} ---")
            model.train()
            total_loss = 0.0
            count = 0
            start_time = time.time()
            
            for batch_graph in dataset:
                optimizer.zero_grad()
                
                # Explicitly move pure CPU data from data_loader to Apple Silicon GPU
                batch_graph = batch_graph.to(device) 
                
                # Forward pass
                predictions = model(batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight)
                
                # Calculate error
                loss = criterion(predictions, batch_graph.y)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                
                if count % 100 == 0:
                    avg_loss = total_loss / 100
                    elapsed = time.time() - start_time
                    print(f"[Epoch {epoch} | {count} Batches Processed] MSE Loss: {avg_loss:.2f} | Speed: {100/elapsed:.1f} batches/sec")
                    total_loss = 0.0
                    start_time = time.time()
                    
                # Optional limit for testing/debugging
                if limit and count >= limit:
                    print(f"Reached training limit of {limit} days for this epoch.")
                    break
                    
    except KeyboardInterrupt:
        print("\n🛑 Manual interrupt detected. Stopping early and saving current brain state...")

    # 3. Save the Brain
    weights_path = f"mandiflow_gcn_lstm_{commodity.lower()}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Model weights saved to {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commodity", type=str, default="ONION", help="Commodity to train for")
    parser.add_argument("--limit", type=int, default=None, help="Number of days to train per epoch (None for all)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of full passes over the dataset")
    args = parser.parse_args()
    
    train(args.commodity, args.limit, args.epochs)