import os
import argparse
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import numpy as np

# Import the shared backbone
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../isaaclab_patch/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift"))
from vision_backbone import MobileNetVisionBackbone, save_backbone_checkpoint

class ExpertDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # We don't keep the file open to avoid multiprocessing issues with DataLoader workers
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['image'].shape[0]
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Open file locally per worker
        with h5py.File(self.h5_path, 'r') as f:
            img = f['image'][idx] # [3, 64, 64], float32, range [0, 1]
            state = f['state'][idx] # [15]
            
        # Extract labels: x, y, yaw
        x = state[0]
        y = state[1]
        cos_dyaw = state[2]
        sin_dyaw = state[3]
        yaw = math.atan2(sin_dyaw, cos_dyaw)
        
        # We can normalize the labels to make training easier
        # x is typically [-4.0, 0.0]
        # y is typically [-1.0, 1.0]
        # yaw is [-pi, pi]
        
        labels = np.array([x, y, yaw], dtype=np.float32)
        
        return torch.from_numpy(img), torch.from_numpy(labels)

class PosePredictor(nn.Module):
    def __init__(self, use_imagenet_init=True):
        super().__init__()
        self.backbone = MobileNetVisionBackbone(imagenet_init=use_imagenet_init)
        
        # MobileNetV3-Small features output is 576
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # output: x, y, yaw
        )
        
    def forward(self, img):
        feat = self.backbone(img)
        pred = self.head(feat)
        return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../expert_dataset_64k.h5")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="../outputs/vision_pretrain")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))
    
    # 1. Prepare Data
    print(f"Loading dataset from {args.dataset}")
    full_dataset = ExpertDataset(args.dataset)
    
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # 2. Prepare Model
    model = PosePredictor(use_imagenet_init=True).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, labels in pbar:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            # Loss weights: x, y, yaw might have different scales. 
            # Simple MSE for now, but can be weighted if needed.
            loss = criterion(preds, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})
            
        train_loss /= train_size
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_loss_x = 0.0
        val_loss_y = 0.0
        val_loss_yaw = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for imgs, labels in pbar_val:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                
                preds = model(imgs)
                loss = criterion(preds, labels)
                
                val_loss += loss.item() * imgs.size(0)
                
                # Calculate individual MAE (Mean Absolute Error) for interpretability
                abs_err = torch.abs(preds - labels)
                val_loss_x += abs_err[:, 0].sum().item()
                val_loss_y += abs_err[:, 1].sum().item()
                val_loss_yaw += abs_err[:, 2].sum().item()
                
        val_loss /= val_size
        val_mae_x = val_loss_x / val_size
        val_mae_y = val_loss_y / val_size
        val_mae_yaw = val_loss_yaw / val_size
        
        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        print(f"         | Val MAE -> X: {val_mae_x:.4f}m, Y: {val_mae_y:.4f}m, Yaw: {val_mae_yaw*180/math.pi:.2f}°")
        
        # Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("MAE/Val_X", val_mae_x, epoch)
        writer.add_scalar("MAE/Val_Y", val_mae_y, epoch)
        writer.add_scalar("MAE/Val_Yaw_deg", val_mae_yaw * 180 / math.pi, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_backbone.pt")
            
            # We only save the backbone part, because that's what RL needs
            save_backbone_checkpoint(
                backbone=model.backbone,
                output_path=save_path,
                metadata={"epoch": epoch+1, "val_loss": val_loss, "val_mae_y": val_mae_y}
            )
            print(f"Saved new best backbone to {save_path}")

    writer.close()
    print("Pretraining finished!")

if __name__ == "__main__":
    main()
