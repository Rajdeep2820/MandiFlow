"""
train.py  —  MandiFlow v3.0
=============================
Two-stage training with dual loss (magnitude + direction).

Loss = HuberLoss(magnitude) + 0.3 × BCEWithLogitsLoss(direction)
     computed ONLY on anchor nodes (≥60% temporal coverage).

Ghost-shock safeguard:
  After every epoch, the model is validated on the 2018 quiet period
  with zero shock context. If predicted price movement exceeds 15% on
  average across anchor nodes, a warning is printed. If it exceeds 25%,
  training stops and the best checkpoint is preserved.

Two-stage:
  Stage 1 — pretrain (2010–2020):  learns baseline spatial patterns
  Stage 2 — finetune (2021–2024):  adapts to modern market structure,
                                    loads pretrain_best weights as start

Usage:
    python train.py --commodity ONION --epochs 50
    python train.py --commodity ONION --stage finetune --epochs 30
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import MandiParquetDataset
from model import MandiFlowNet, NODE_FEATURES

GRADIENT_CLIP        = 0.5
LOG_INTERVAL         = 50
DIRECTION_LOSS_WEIGHT = 0.3
GHOST_SHOCK_WARN     = 0.15   # warn if quiet-period movement > 15%
GHOST_SHOCK_STOP     = 0.25   # stop training if > 25%
REGIME_LOSS_WEIGHT   = 0.3    # down-weight structural break years


def validate_quiet_period(model, dataset, device) -> float:
    """
    Runs the model on 2018 (quiet year) with no shock context.
    Returns mean absolute deviation from 1.0 across anchor nodes.
    A value > GHOST_SHOCK_WARN means the model is seeing ghost shocks.
    """
    model.eval()
    deviations = []
    count = 0

    with torch.no_grad():
        for batch in dataset.iter_quiet_period():
            x           = batch.x.to(device)
            edge_index  = batch.edge_index.to(device)
            edge_weight = batch.edge_weight.to(device)
            anchor_mask = batch.anchor_mask.to(device)

            mag, _ = model(x, edge_index, edge_weight)
            # Deviation from 1.0 on anchor nodes (no-shock prediction)
            dev = torch.abs(mag[anchor_mask] - 1.0).mean().item()
            deviations.append(dev)
            count += 1
            if count >= 200:   # sample 200 quiet days
                break

    return float(np.mean(deviations)) if deviations else 0.0


def train_stage(
    commodity: str,
    stage:     str,
    epochs:    int,
    lr:        float,
    resume:    bool = True,
):
    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    commodity = commodity.upper()

    print(f"\n{'='*60}")
    print(f"  MandiFlow v3.0 — {stage.upper()} | {commodity} | {device}")
    print(f"{'='*60}\n")

    # Dataset
    dataset = MandiParquetDataset(commodity=commodity)
    dataset.windows = [w for w in dataset.windows if w["name"] == stage]
    if not dataset.windows:
        print(f"❌ No '{stage}' matrix found. Run prepare_commodity.py first.")
        return

    N = dataset.N

    # Model
    model = MandiFlowNet(
        node_features = NODE_FEATURES,
        hidden_dim    = 64,
        output_dim    = 4,
        lookback      = 7,
    ).to(device)

    # Checkpoint loading
    weights_path = f"mandiflow_gcn_lstm_{commodity.lower()}_{stage}.pth"
    best_path    = f"mandiflow_gcn_lstm_{commodity.lower()}_{stage}_best.pth"

    if resume:
        if stage == "finetune" and not os.path.exists(weights_path):
            pretrain_best = f"mandiflow_gcn_lstm_{commodity.lower()}_pretrain_best.pth"
            if os.path.exists(pretrain_best):
                model.load_state_dict(
                    torch.load(pretrain_best, map_location=device, weights_only=True)
                )
                print(f"🔄 Fine-tuning from: {pretrain_best}")
            else:
                print("⚠️  No pretrain checkpoint found — starting finetune from scratch.")
        elif os.path.exists(weights_path):
            model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )
            print(f"🔄 Resuming from: {weights_path}")

    # Optimizer
    effective_lr = lr if stage == "pretrain" else lr * 0.3
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = effective_lr,
        weight_decay = 1e-4 if stage == "pretrain" else 2e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    # Loss functions
    magnitude_loss_fn = nn.HuberLoss(delta=0.1)
    direction_loss_fn = nn.BCEWithLogitsLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Nodes: {N}  |  LR: {effective_lr}  |  Params: {total_params:,}\n")

    best_loss    = float("inf")
    no_improve   = 0

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            mag_loss_total = 0.0
            dir_loss_total = 0.0
            count = 0
            shock_count = 0
            start = time.time()

            for batch in dataset:
                x           = batch.x.to(device)           # (N, 7, 7)
                y_mag       = batch.y_magnitude.to(device)  # (N, 4)
                y_dir       = batch.y_direction.to(device)  # (N, 4)
                edge_index  = batch.edge_index.to(device)
                edge_weight = batch.edge_weight.to(device)
                anchor_mask = batch.anchor_mask.to(device)  # (N,) bool
                is_shock    = batch.is_shock

                optimizer.zero_grad()

                pred_mag, pred_dir = model(x, edge_index, edge_weight)

                # Compute loss on anchor nodes only
                mag_loss = magnitude_loss_fn(
                    pred_mag[anchor_mask], y_mag[anchor_mask]
                )
                dir_loss = direction_loss_fn(
                    pred_dir[anchor_mask], y_dir[anchor_mask]
                )
                loss = mag_loss + DIRECTION_LOSS_WEIGHT * dir_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()

                epoch_loss     += loss.item()
                mag_loss_total += mag_loss.item()
                dir_loss_total += dir_loss.item()
                count          += 1
                if is_shock:
                    shock_count += 1

                if count % LOG_INTERVAL == 0:
                    print(
                        f"  Epoch [{epoch}/{epochs}] | "
                        f"Batch {count:>5} | "
                        f"Loss: {loss.item():.6f} "
                        f"(mag: {mag_loss.item():.4f} "
                        f"dir: {dir_loss.item():.4f}) | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )

            if count == 0:
                print(f"  Epoch {epoch}: no valid batches.")
                continue

            avg_loss = epoch_loss / count
            elapsed  = time.time() - start
            shock_pct = 100 * shock_count / count

            print(
                f"\n📊 Epoch {epoch}/{epochs} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"Shock batches: {shock_pct:.1f}% (target 30%) | "
                f"Time: {elapsed:.1f}s"
            )

            # Scheduler
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"  📉 LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

            # Best checkpoint
            if avg_loss < best_loss:
                best_loss  = avg_loss
                no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f"  💾 New best ({best_loss:.6f}) → {best_path}")
            else:
                no_improve += 1

            # Ghost-shock validation (quiet period)
            quiet_dev = validate_quiet_period(model, dataset, device)
            status = "✅" if quiet_dev < GHOST_SHOCK_WARN else "⚠️ "
            print(f"  {status} Quiet-period deviation: {quiet_dev:.4f} "
                  f"(warn>{GHOST_SHOCK_WARN}, stop>{GHOST_SHOCK_STOP})")

            if quiet_dev > GHOST_SHOCK_STOP:
                print(f"\n🛑 Ghost-shock threshold exceeded ({quiet_dev:.3f} > "
                      f"{GHOST_SHOCK_STOP}). Loading best checkpoint and stopping.")
                model.load_state_dict(
                    torch.load(best_path, map_location=device, weights_only=True)
                )
                break

            # Early stopping
            if no_improve >= 8:
                print(f"\n⏹️  No improvement for 8 epochs. Stopping early.")
                break

            print()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Saving checkpoint...")

    torch.save(model.state_dict(), weights_path)
    print(f"\n✅ {stage.upper()} complete.")
    print(f"   Final: {weights_path}")
    print(f"   Best:  {best_path}  (loss={best_loss:.6f})")


def main():
    parser = argparse.ArgumentParser(description="MandiFlow v3.0 Training")
    parser.add_argument("--commodity", type=str, default="ONION")
    parser.add_argument("--stage",     type=str, default="both",
                        choices=["pretrain", "finetune", "both"])
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    stages = ["pretrain", "finetune"] if args.stage == "both" else [args.stage]
    for stage in stages:
        train_stage(
            commodity = args.commodity,
            stage     = stage,
            epochs    = args.epochs,
            lr        = args.lr,
            resume    = not args.no_resume,
        )
    print(f"\n🎉 All stages complete for {args.commodity.upper()}.")


if __name__ == "__main__":
    main()