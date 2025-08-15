#!/usr/bin/env python3
"""
train_3dunet.py - 3D U-Net training for coronary artery segmentation
"""

import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import zoom

# Change to TransUNet directory
os.chdir('/content/drive/MyDrive/TransUNet-main/')

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class DoubleConv3D(nn.Module):
    """3D Double Convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net for coronary artery segmentation"""
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature*2, feature))

        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:],
                                mode='trilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss3D(nn.Module):
    """3D Dice Loss for binary segmentation"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred_fg = pred[:, 1]  # foreground channel
        target_fg = (target == 1).float()

        intersection = (pred_fg * target_fg).sum()
        union = pred_fg.sum() + target_fg.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class FocalTverskyLoss3D(nn.Module):
    """3D Focal Tversky Loss for imbalanced segmentation"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred_fg = pred[:, 1]
        target_fg = (target == 1).float()

        tp = (pred_fg * target_fg).sum()
        fp = (pred_fg * (1 - target_fg)).sum()
        fn = ((1 - pred_fg) * target_fg).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky

# ============================================================================
# DATASET class 
# ============================================================================

class ImageCas3DDataset(Dataset):
    """3D Dataset for ImageCas volumes"""
    def __init__(self, img_dir, lbl_dir, target_size=(64, 128, 128), augment=False):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.target_size = target_size
        self.augment = augment

        # Collect image-label pairs
        self.pairs = []
        for img_file in sorted(self.img_dir.glob("*.img.nii.gz")):
            lbl_file = self.lbl_dir / img_file.name.replace(".img.nii.gz", ".label.nii.gz")
            if lbl_file.exists():
                self.pairs.append((img_file, lbl_file))

        print(f"Found {len(self.pairs)} volume pairs")

    def __len__(self):
        return len(self.pairs)

    def preprocess_volume(self, volume_array, is_label=False):
        """Preprocess volume with same normalization as TransUNet"""
        if not is_label:
            volume_array = np.clip(volume_array, 0, 4000)
            volume_array = (volume_array - volume_array.min()) / (volume_array.max() - volume_array.min() + 1e-7)

        # Resize volume to target size
        current_size = volume_array.shape
        zoom_factors = [self.target_size[i] / current_size[i] for i in range(3)]

        if is_label:
            volume_array = zoom(volume_array, zoom_factors, order=0)
        else:
            volume_array = zoom(volume_array, zoom_factors, order=1)

        return volume_array.astype(np.float32)

    def augment_volume(self, img, lbl):
        """Simple 3D augmentation"""
        if random.random() > 0.5:
            img = np.flip(img, axis=0).copy()
            lbl = np.flip(lbl, axis=0).copy()

        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=1).copy()

        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            lbl = np.flip(lbl, axis=2).copy()

        return img, lbl

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        # Load volumes
        img_sitk = sitk.ReadImage(str(img_path))
        lbl_sitk = sitk.ReadImage(str(lbl_path))

        img_array = sitk.GetArrayFromImage(img_sitk)
        lbl_array = sitk.GetArrayFromImage(lbl_sitk)

        # Preprocess
        img_array = self.preprocess_volume(img_array, is_label=False)
        lbl_array = self.preprocess_volume(lbl_array, is_label=True)

        # Augment if training
        if self.augment:
            img_array, lbl_array = self.augment_volume(img_array, lbl_array)

        # Convert to tensors
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, D, H, W)
        lbl_tensor = torch.from_numpy(lbl_array).long()  # (D, H, W)

        return {
            'image': img_tensor,
            'label': lbl_tensor,
            'case_name': img_path.stem
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_adaptive_target_size():
    """Get target size based on available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 12:
            target_size = (32, 96, 96)
            print("Using small size for limited GPU memory")
        else:
            target_size = (64, 128, 128)
            print("Using standard size")
    else:
        target_size = (16, 64, 64)
        print("No GPU - using tiny size")
    
    return target_size

def train_unet3d(config):
    """Main training function"""
    
    print("🚀 Starting 3D U-Net training...")
    
    # Setup logging
    log_dir = Path(config['snapshot_path']) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if config.get('use_logging', True):
        logging.basicConfig(
            filename=log_dir / "training.log",
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(f"Configuration: {config}")
    
    # Create dataset
    dataset = ImageCas3DDataset(
        img_dir=config['img_dir'],
        lbl_dir=config['lbl_dir'],
        target_size=config['target_size'],
        augment=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )
    
    # Model and losses
    model = UNet3D(in_channels=1, out_channels=2).to(DEVICE)
    dice_loss = DiceLoss3D().to(DEVICE)
    focal_tversky_loss = FocalTverskyLoss3D().to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['max_epochs'] * len(train_loader)
    )
    
    # TensorBoard writer
    if config.get('use_tensorboard', True):
        writer = SummaryWriter(log_dir / "tensorboard")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_dice = 0.0
    
    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute losses
            loss_dice = dice_loss(outputs, labels)
            loss_ft = focal_tversky_loss(outputs, labels)
            loss = 0.5 * loss_dice + 0.5 * loss_ft
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if config.get('use_tensorboard', True):
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                outputs = model(images)
                
                loss_dice = dice_loss(outputs, labels)
                loss_ft = focal_tversky_loss(outputs, labels)
                loss = 0.5 * loss_dice + 0.5 * loss_ft
                
                val_loss += loss.item()
                
                # Calculate Dice score
                pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                pred_fg = (pred == 1).float()
                target_fg = (labels == 1).float()
                
                intersection = (pred_fg * target_fg).sum()
                union = pred_fg.sum() + target_fg.sum()
                dice_score = (2 * intersection) / (union + 1e-5)
                val_dice_scores.append(dice_score.item())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = np.mean(val_dice_scores)
        
        # Logging
        log_message = f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, " \
                     f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_dice_score:.4f}"
        print(log_message)
        
        if config.get('use_logging', True):
            logging.info(log_message)
        
        if config.get('use_tensorboard', True):
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            writer.add_scalar('val/dice', avg_dice_score, epoch)
        
        # Save best model
        if avg_dice_score > best_val_dice:
            best_val_dice = avg_dice_score
            best_val_loss = avg_val_loss
            
            checkpoint_path = Path(config['snapshot_path']) / "best_unet3d.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dice': avg_dice_score,
                'config': config
            }, checkpoint_path)
            
            print(f"✅ New best model saved: Dice={avg_dice_score:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            checkpoint_path = Path(config['snapshot_path']) / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dice': avg_dice_score,
            }, checkpoint_path)
    
    if config.get('use_tensorboard', True):
        writer.close()
    
    print("🎉 Training completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    
    return model

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3D U-Net for coronary artery segmentation')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img-dir', type=str, default='/content/local_data/train', help='Image directory')
    parser.add_argument('--lbl-dir', type=str, default='/content/local_data/label', help='Label directory')
    parser.add_argument('--output-dir', type=str, default='/content/drive/MyDrive/TransUNet-main/model/unet3d_checkpoint', 
                       help='Output directory')
    parser.add_argument('--adaptive-size', action='store_true', help='Use adaptive target size based on GPU memory')
    
    args = parser.parse_args()
    
    # Set random seeds
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration
    config = {
        'seed': seed,
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'target_size': get_adaptive_target_size() if args.adaptive_size else (64, 128, 128),
        'img_dir': args.img_dir,
        'lbl_dir': args.lbl_dir,
        'snapshot_path': args.output_dir,
        'num_workers': 2,
        'save_interval': 10,
        'use_tensorboard': True,
        'use_logging': True
    }
    
    # Create output directory
    Path(config['snapshot_path']).mkdir(parents=True, exist_ok=True)
    
    # Train model
    model = train_unet3d(config)

if __name__ == "__main__":
    main()
