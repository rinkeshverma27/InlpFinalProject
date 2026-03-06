import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, stage='Stage0'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.stage = stage
        
        # Huber Loss delta=0.01 as per blueprint
        self.criterion = nn.HuberLoss(delta=0.01)
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.best_mae = float('inf')
        self.checkpoint_dir = Path("models/checkpoints") / stage
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_mae = 0
        
        for batch_idx, (x, y, lengths) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            # Remove lengths dependency if sorting is problematic or just let LSTM handle unpadded 
            # (In PyTorch, pack_padded_sequence needs lengths. If issues, can omit lengths packing and just feed raw pad)
            
            self.optimizer.zero_grad()
            
            # Predict
            preds = self.model(x) # Removing lengths for simplicty of batching
            
            # Loss
            loss = self.criterion(preds, y)
            mae = torch.abs(preds - y).mean()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            
        return total_loss / len(self.train_loader), total_mae / len(self.train_loader)
        
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        # Tracking for metrics
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                
                loss = self.criterion(preds, y)
                mae = torch.abs(preds - y).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())
                
        # Calculate Direction Accuracy
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        
        pred_dir = np.sign(preds_arr)
        target_dir = np.sign(targets_arr)
        
        # Handle zeros if any (neutral)
        pred_dir[pred_dir == 0] = 1
        target_dir[target_dir == 0] = 1
        
        direction_acc = np.mean(pred_dir == target_dir)
        
        # High-Vol Acc (move > 1.5%)
        # Note: target labels are continuous floats. 1.5% is 0.015
        high_vol_idx = np.abs(targets_arr) > 0.015
        if np.sum(high_vol_idx) > 0:
            high_vol_acc = np.mean(pred_dir[high_vol_idx] == target_dir[high_vol_idx])
        else:
            high_vol_acc = 0.0
            
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_mae': total_mae / len(self.val_loader),
            'val_direction_acc': direction_acc,
            'val_high_vol_acc': high_vol_acc
        }
            
        return metrics
        
    def save_checkpoint(self, ticker_or_all, metrics, epoch):
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        target_dir = self.checkpoint_dir / ticker_or_all
        target_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = target_dir / f"model_{date_str}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_mae,
        }, model_path)
        
        # Provide matching metadata JSON
        meta_path = target_dir / f"metadata_{date_str}.json"
        
        meta_data = {
            "checkpoint_hash": f"baseline-{date_str}-{epoch}",
            "created_at": datetime.now().isoformat(),
            "val_mae": metrics['val_mae'],
            "val_direction_acc": metrics['val_direction_acc'],
            "val_high_vol_acc": metrics['val_high_vol_acc'],
            "deployed": False,
            "stage": self.stage
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)
            
        print(f"Saved Checkpoint to {model_path} with MAE: {metrics['val_mae']:.4f}")

    def train(self, epochs, ticker_group_name):
        print(f"Starting Training for {epochs} epochs...")
        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_mae = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs} [{time.time()-t0:.1f}s] - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} "
                  f"| Val Loss: {val_metrics['val_loss']:.4f}, Val MAE: {val_metrics['val_mae']:.4f} "
                  f"| Dir Acc: {val_metrics['val_direction_acc']:.3f}")
                  
            if val_metrics['val_mae'] < self.best_mae:
                self.best_mae = val_metrics['val_mae']
                self.save_checkpoint(ticker_group_name, val_metrics, epoch)
                
        return self.best_mae
