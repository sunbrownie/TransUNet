#!/usr/bin/env python3
"""train_imagecas_512_optimized_balanced_v3_tensorboard.py - Enhanced with TensorBoard Visualization

ADDITIONS in v3:
✅ Image logging: input images, ground truth labels, predictions
✅ Overlay visualization: predictions overlaid on input images
✅ Confusion visualization: TP, FP, FN regions
✅ Per-class metrics visualization
✅ Histogram logging: model weights, gradients, predictions
✅ Learning rate and gradient norm tracking

Usage:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python train_imagecas_512_optimized_balanced_v3_tensorboard.py
    
    # Monitor with TensorBoard:
    tensorboard --logdir /lambda/nfs/segmentation/TransUNet/model/vit_checkpoint/imagenet21k_512/log_512_optimized_balanced --port 6008 --bind_all
"""

import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from tqdm.auto import tqdm
from pathlib import Path
import json
from datetime import datetime
import torch.nn.functional as F

# Add TransUNet to path - UPDATE THIS PATH
sys.path.insert(0, '/lambda/nfs/segmentation/TransUNet')

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import SegmentationHead
from utils import DiceLoss, FocalTverskyLoss

# ============================================================================
# CONFIGURATION - MEMORY-OPTIMIZED BALANCED
# ============================================================================

CONFIG = {
    # === DATA PATHS === (UPDATE THESE)
    'root_path_train': '/lambda/nfs/segmentation/TransUNet/data/ImageCas/train_processed_512',
    'root_path_val': '/lambda/nfs/segmentation/TransUNet/data/ImageCas/test_vol_processed_512',
    'list_dir': '/lambda/nfs/segmentation/TransUNet/data/ImageCas/lists_512',
    
    # === SPLITS ===
    'train_split': 'train',        # Name of training list file (train.txt)
    'val_split': 'test_vol',       # Name of validation list file (test_vol.txt)
    
    # === MODEL ===
    'img_size': 512,
    'num_classes': 2,
    'n_skip': 3,
    'vit_name': 'R50-ViT-B_16',
    'vit_patches_size': 16,
    
    # === TRAINING (MEMORY-OPTIMIZED) ===
    'max_epochs': 100,
    'batch_size': 4,                # REDUCED from 16 to fit in GPU memory
    'gradient_accumulation': 4,     # Simulate batch_size=16 (4*4=16)
    'base_lr': 0.0005,
    'weight_decay': 1e-4,
    
    # === DATA SUBSAMPLING (KEY OPTIMIZATION) ===
    'train_fraction': 0.5,          # Use 30% of all training data per epoch
    'val_fraction': 1,            # Use 20% of validation data (faster validation)
    'resample_each_epoch': True,    # Resample different subset each epoch
    
    # === EARLY STOPPING ===
    'early_stopping': True,
    'patience': 20,
    'min_dice_threshold': 0.70,
    'min_epochs': 30,
    
    # === MONITORING (REDUCED FREQUENCY) ===
    'val_frequency': 2,             # Validate every 2 epochs
    'log_frequency': 200,           # Log scalars every 200 iterations
    'image_log_frequency': 400,     # Log images every 400 iterations (less frequent)
    'save_frequency': 10,
    
    # === TENSORBOARD VISUALIZATION ===
    'log_images': True,             # Enable image logging (NO PIL NEEDED!)
    'log_histograms': True,         # Enable histogram logging
    'num_vis_images': 4,            # Number of images to visualize per batch
    
    # === SPEED OPTIMIZATIONS ===
    'use_amp': True,                # Mixed precision training
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 2,
    
    # === SYSTEM ===
    'n_gpu': 1,
    'seed': 1234,
    'deterministic': False,
    
    # === PATHS === (UPDATE THESE)
    'snapshot_path': '/lambda/nfs/segmentation/TransUNet/model/vit_checkpoint/imagenet21k_512',
    'pretrained_path': '/lambda/nfs/segmentation/TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
    
    # === CLASS BALANCE ===
    'positive_ratio': 0.6,
}

# ============================================================================
# TENSORBOARD VISUALIZATION UTILITIES
# ============================================================================

def normalize_image_for_vis(img_tensor):
    """Normalize image tensor to [0, 1] for visualization"""
    # img_tensor shape: (C, H, W) or (H, W)
    img = img_tensor.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.squeeze(0)
    
    # Normalize to [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def create_overlay_visualization(image, prediction, ground_truth=None, alpha=0.4):
    """Create overlay of prediction on image with optional ground truth comparison
    
    Args:
        image: Input image (H, W) normalized to [0, 1]
        prediction: Binary prediction mask (H, W) with values 0 or 1
        ground_truth: Optional ground truth mask (H, W) with values 0 or 1
        alpha: Transparency for overlay
        
    Returns:
        RGB overlay image (3, H, W)
    """
    # Convert grayscale to RGB
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=0)  # (3, H, W)
    else:
        image_rgb = image
    
    # Create color overlays
    overlay = image_rgb.copy()
    
    # Prediction in green
    pred_mask = prediction > 0.5
    overlay[1, pred_mask] = overlay[1, pred_mask] * (1 - alpha) + alpha  # Green channel
    
    # Ground truth in red (if provided)
    if ground_truth is not None:
        gt_mask = ground_truth > 0.5
        overlay[0, gt_mask] = overlay[0, gt_mask] * (1 - alpha) + alpha  # Red channel
    
    # Clip to valid range
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def create_confusion_visualization(prediction, ground_truth):
    """Create confusion matrix visualization: TP, FP, FN, TN
    
    Returns:
        RGB image (3, H, W) where:
        - TP (True Positive): Green
        - FP (False Positive): Red
        - FN (False Negative): Blue
        - TN (True Negative): Black
    """
    pred_mask = prediction > 0.5
    gt_mask = ground_truth > 0.5
    
    h, w = pred_mask.shape
    confusion_rgb = np.zeros((3, h, w), dtype=np.float32)
    
    # TP: Green
    tp_mask = pred_mask & gt_mask
    confusion_rgb[1, tp_mask] = 1.0
    
    # FP: Red
    fp_mask = pred_mask & ~gt_mask
    confusion_rgb[0, fp_mask] = 1.0
    
    # FN: Blue
    fn_mask = ~pred_mask & gt_mask
    confusion_rgb[2, fn_mask] = 1.0
    
    # TN: Black (already zeros)
    
    return confusion_rgb


def log_images_to_tensorboard(writer, imgs, labs, outputs, iter_num, phase='train', num_images=4):
    """Log comprehensive image visualizations to TensorBoard
    
    Args:
        writer: TensorBoard SummaryWriter
        imgs: Input images (B, C, H, W)
        labs: Ground truth labels (B, H, W)
        outputs: Model outputs (B, num_classes, H, W)
        iter_num: Current iteration number
        phase: 'train' or 'val'
        num_images: Number of images to log from batch
    """
    with torch.no_grad():
        # Get predictions
        pred_softmax = torch.softmax(outputs, dim=1)
        pred_fg = pred_softmax[:, 1]  # Foreground probability
        pred_binary = (pred_fg > 0.5).float()  # Binary prediction
        
        # Limit number of images
        num_images = min(num_images, imgs.shape[0])
        
        for idx in range(num_images):
            # Get single image, label, prediction
            img = imgs[idx, 0]  # (H, W)
            lab = labs[idx]     # (H, W)
            pred = pred_fg[idx]  # (H, W)
            pred_bin = pred_binary[idx]  # (H, W)
            
            # Normalize image for visualization
            img_vis = normalize_image_for_vis(img)
            lab_vis = lab.detach().cpu().numpy()
            pred_vis = pred.detach().cpu().numpy()
            pred_bin_vis = pred_bin.detach().cpu().numpy()
            
            # 1. Original images
            writer.add_image(f'{phase}/image_{idx}/input', 
                           torch.from_numpy(img_vis).unsqueeze(0), 
                           iter_num)
            
            # 2. Ground truth
            writer.add_image(f'{phase}/image_{idx}/ground_truth', 
                           torch.from_numpy(lab_vis).unsqueeze(0), 
                           iter_num)
            
            # 3. Prediction probability
            writer.add_image(f'{phase}/image_{idx}/prediction_prob', 
                           torch.from_numpy(pred_vis).unsqueeze(0), 
                           iter_num)
            
            # 4. Binary prediction
            writer.add_image(f'{phase}/image_{idx}/prediction_binary', 
                           torch.from_numpy(pred_bin_vis).unsqueeze(0), 
                           iter_num)
            
            # 5. Overlay: Prediction (green) + Ground truth (red)
            overlay = create_overlay_visualization(img_vis, pred_vis, lab_vis)
            writer.add_image(f'{phase}/image_{idx}/overlay', 
                           torch.from_numpy(overlay), 
                           iter_num)
            
            # 6. Confusion visualization (TP, FP, FN, TN)
            confusion = create_confusion_visualization(pred_bin_vis, lab_vis)
            writer.add_image(f'{phase}/image_{idx}/confusion', 
                           torch.from_numpy(confusion), 
                           iter_num)


def log_model_histograms(writer, model, iter_num):
    """Log model weight and gradient histograms"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Log weights
            writer.add_histogram(f'weights/{name}', param.data, iter_num)
            
            # Log gradients if available
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, iter_num)


def log_prediction_statistics(writer, outputs, labs, iter_num, phase='train'):
    """Log prediction statistics"""
    with torch.no_grad():
        pred_softmax = torch.softmax(outputs, dim=1)
        pred_fg = pred_softmax[:, 1]
        
        # Log prediction distribution
        writer.add_histogram(f'{phase}/prediction_distribution', pred_fg, iter_num)
        
        # Log confidence statistics
        pred_mean = pred_fg.mean().item()
        pred_std = pred_fg.std().item()
        pred_max = pred_fg.max().item()
        pred_min = pred_fg.min().item()
        
        writer.add_scalar(f'{phase}/prediction_mean', pred_mean, iter_num)
        writer.add_scalar(f'{phase}/prediction_std', pred_std, iter_num)
        writer.add_scalar(f'{phase}/prediction_max', pred_max, iter_num)
        writer.add_scalar(f'{phase}/prediction_min', pred_min, iter_num)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    pred_softmax = torch.softmax(pred, dim=1)
    pred_fg = pred_softmax[:, 1]
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_softmax = torch.softmax(pred, dim=1)
    pred_binary = (pred_softmax[:, 1] > 0.5).float()
    target_binary = (target == 1).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
    smooth = 1e-5
    
    return {
        'dice': ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item(),
        'precision': ((tp + smooth) / (tp + fp + smooth)).item(),
        'recall': ((tp + smooth) / (tp + fn + smooth)).item(),
        'specificity': ((tn + smooth) / (tn + fp + smooth)).item(),
        'iou': ((tp + smooth) / (tp + fp + fn + smooth)).item(),
        'accuracy': ((tp + tn + smooth) / (tp + tn + fp + fn + smooth)).item()
    }


# ============================================================================
# DATASET (Keep original implementation)
# ============================================================================

from scipy.ndimage import zoom as scipy_zoom
from scipy import ndimage
import cv2


def _random_rot_flip(img, lbl):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    lbl = np.rot90(lbl, k)
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    lbl = np.flip(lbl, axis=axis).copy()
    return img, lbl


def _random_rotate(img, lbl):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=1, reshape=False)
    lbl = ndimage.rotate(lbl, angle, order=0, reshape=False)
    return img, lbl


class RandomGenerator512:
    def __init__(self, output_size=(512, 512), elastic=False, gamma=None):
        self.output_size = output_size
        self.elastic = elastic
        self.gamma = gamma
    
    def _elastic(self, img, lbl, alpha=20, sigma=4):
        h, w = img.shape
        dx = cv2.GaussianBlur((np.random.rand(h, w)*2 - 1).astype(np.float32),
                              ksize=(0, 0), sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w)*2 - 1).astype(np.float32),
                              ksize=(0, 0), sigmaX=sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        img_def = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        lbl_def = cv2.remap(lbl.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        return img_def, lbl_def.astype(lbl.dtype)
    
    def __call__(self, sample):
        img, lbl = sample['image'], sample['label']
        
        if random.random() > 0.5:
            img, lbl = _random_rot_flip(img, lbl)
        elif random.random() > 0.5:
            img, lbl = _random_rotate(img, lbl)
        
        if self.elastic and random.random() > 0.5:
            img, lbl = self._elastic(img, lbl)
        
        if self.gamma and random.random() > 0.5:
            g = random.uniform(1 - self.gamma, 1 + self.gamma)
            imin, imax = img.min(), img.max()
            img = np.power((img - imin) / (imax - imin + 1e-8), g)
            img = img * (imax - imin) + imin
        
        x, y = img.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            img = scipy_zoom(img, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            lbl = scipy_zoom(lbl, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        lbl = torch.from_numpy(lbl.astype(np.float32))
        sample = {'image': img, 'label': lbl.long()}
        return sample


class ImageCAS_512(torch.utils.data.Dataset):
    def __init__(self, base_dir, split, list_dir=None, transform=None):
        self.transform = transform
        self.split = split
        self.base_dir = Path(base_dir)
        
        if list_dir:
            self.list_dir = Path(list_dir)
        else:
            self.list_dir = self.base_dir / 'lists_512'
        
        self.sample_list = []
        list_path = self.list_dir / f'{split}.txt'
        
        if not list_path.exists():
            raise FileNotFoundError(f"Split file not found: {list_path}")
        
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"  Loaded {len(self.sample_list)} samples from {list_path}")
        
        self._is_pos = self._compute_positive_flags()
    
    def _compute_positive_flags(self):
        """Check which slices have foreground pixels - WITH CACHING FOR SPEED."""
        import pickle
        
        # Create cache file path
        cache_file = self.base_dir.parent / f"{self.base_dir.name}_positive_flags.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            print(f"  📦 Loading positive flags from cache: {cache_file.name}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache is valid (same number of samples)
                if len(cached_data.get('is_pos', [])) == len(self.sample_list):
                    n_positive = sum(cached_data['is_pos'])
                    print(f"  ✓ Loaded {n_positive}/{len(cached_data['is_pos'])} positive samples from cache ({n_positive/len(cached_data['is_pos'])*100:.1f}%)")
                    return cached_data['is_pos']
                else:
                    print(f"  ⚠ Cache size mismatch, recomputing...")
            except Exception as e:
                print(f"  ⚠ Cache load failed: {e}, recomputing...")
        
        # Compute if cache doesn't exist or is invalid
        print(f"  🔍 Checking {len(self.sample_list)} files for positive samples (this will take a few minutes)...")
        is_pos = []
        
        from tqdm import tqdm
        for case in tqdm(self.sample_list, desc="  Scanning labels", unit="file", ncols=80):
            npz_path = self.base_dir / case
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    label = data['label']
                    has_foreground = (label == 1).any()
                    is_pos.append(has_foreground)
                    data.close()  # Free memory
                except Exception:
                    is_pos.append(False)
            else:
                is_pos.append(False)
        
        # Save to cache for next time
        print(f"  💾 Saving to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'is_pos': is_pos}, f)
        except Exception as e:
            print(f"  ⚠ Cache save failed: {e}")
        
        n_positive = sum(is_pos)
        print(f"  ✓ Found {n_positive}/{len(is_pos)} positive samples ({n_positive/len(is_pos)*100:.1f}%)")
        return is_pos
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        npz_path = self.base_dir / case
        
        # Load from .npz file (contains 'image' and 'label' arrays)
        data = np.load(str(npz_path))
        img = data['image']
        lbl = data['label']
        
        sample = {'image': img, 'label': lbl}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# ============================================================================
# VALIDATION WITH ENHANCED VISUALIZATION
# ============================================================================

def validate(model, val_loader, dice_loss, ft_loss, device, writer=None, epoch=None, 
             use_amp=True, log_images=True, num_vis_images=4):
    """Enhanced validation with image logging"""
    model.eval()
    val_loss = 0.0
    metrics_sum = {'dice': 0.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 0.0, 
                   'iou': 0.0, 'accuracy': 0.0}
    n_batches = 0
    
    logged_images = False  # Only log images once per validation
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation', leave=False)):
            imgs = batch['image'].to(device)
            labs = batch['label'].to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(imgs)
                loss_d = dice_loss(outputs, labs, softmax=True)
                loss_ft = ft_loss(outputs, labs)
                loss = 0.5 * loss_d + 0.5 * loss_ft
            
            val_loss += loss.item()
            batch_metrics = compute_metrics(outputs.float(), labs)
            
            for k in metrics_sum.keys():
                metrics_sum[k] += batch_metrics[k]
            
            # Log images from first batch only
            if writer and log_images and not logged_images and batch_idx == 0:
                log_images_to_tensorboard(writer, imgs, labs, outputs, epoch, 
                                         phase='val', num_images=num_vis_images)
                log_prediction_statistics(writer, outputs, labs, epoch, phase='val')
                logged_images = True
            
            n_batches += 1
    
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_metrics['loss'] = val_loss / n_batches
    
    if writer and epoch:
        # Log all metrics
        writer.add_scalar('val/loss', avg_metrics['loss'], epoch)
        writer.add_scalar('val/dice', avg_metrics['dice'], epoch)
        writer.add_scalar('val/precision', avg_metrics['precision'], epoch)
        writer.add_scalar('val/recall', avg_metrics['recall'], epoch)
        writer.add_scalar('val/specificity', avg_metrics['specificity'], epoch)
        writer.add_scalar('val/iou', avg_metrics['iou'], epoch)
        writer.add_scalar('val/accuracy', avg_metrics['accuracy'], epoch)
    
    return avg_metrics


# ============================================================================
# UTILITIES
# ============================================================================

def get_balanced_subsample_indices(is_positive, fraction, target_pos_ratio=0.6):
    """Get balanced subsample maintaining target positive ratio"""
    pos_idx = [i for i, p in enumerate(is_positive) if p]
    neg_idx = [i for i, p in enumerate(is_positive) if not p]
    
    n_total = int(len(is_positive) * fraction)
    n_pos = int(n_total * target_pos_ratio)
    n_neg = n_total - n_pos
    
    n_pos = min(n_pos, len(pos_idx))
    n_neg = min(n_neg, len(neg_idx))
    
    selected_pos = random.sample(pos_idx, n_pos)
    selected_neg = random.sample(neg_idx, n_neg)
    
    indices = selected_pos + selected_neg
    random.shuffle(indices)
    return indices


class EarlyStopping:
    def __init__(self, patience=10, mode='max', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.stop = False
    
    def __call__(self, val_metric):
        score = val_metric if self.mode == 'max' else -val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.stop


class TrainingHistory:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'val_dice': [],
            'val_precision': [], 'val_recall': [], 'val_specificity': [], 
            'val_iou': [], 'val_accuracy': [], 'lr': []
        }
        self.best_dice = 0.0
        self.best_epoch = 0
    
    def update(self, epoch, train_loss, val_metrics, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_dice'].append(val_metrics['dice'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_specificity'].append(val_metrics['specificity'])
        self.history['val_iou'].append(val_metrics['iou'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['lr'].append(lr)
        
        if val_metrics['dice'] > self.best_dice:
            self.best_dice = val_metrics['dice']
            self.best_epoch = epoch
    
    def save(self):
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def print_status(self, current_epoch):
        if len(self.history['val_dice']) > 0:
            print(f"\n📈 Training Status (Epoch {current_epoch}):")
            print(f"   Best Dice: {self.best_dice:.4f} (epoch {self.best_epoch})")
            recent_dice = self.history['val_dice'][-1]
            print(f"   Recent Dice: {recent_dice:.4f}")


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train_imagecas_512_optimized(config):
    """Main training function with enhanced TensorBoard visualization"""
    
    # =========================================================================
    # SETUP
    # =========================================================================
    snapshot_path = Path(config['snapshot_path'])
    snapshot_path.mkdir(parents=True, exist_ok=True)
    
    log_dir = snapshot_path / 'log_512_optimized_balanced'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        filename=str(snapshot_path / 'train_512_balanced.log'),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s'
    )
    
    # Save config
    with open(snapshot_path / 'config_512_balanced.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set seed
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    if config['deterministic']:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    
    # =========================================================================
    # DATA
    # =========================================================================
    print("\n📂 Loading data...")
    db_train = ImageCAS_512(
        base_dir=config['root_path_train'],
        split=config['train_split'],
        list_dir=config['list_dir'],
        transform=RandomGenerator512(output_size=(config['img_size'], config['img_size']),
                                     elastic=True, gamma=0.2)
    )
    
    db_val = ImageCAS_512(
        base_dir=config['root_path_val'],
        split=config['val_split'],
        list_dir=config['list_dir'],
        transform=RandomGenerator512(output_size=(config['img_size'], config['img_size']))
    )
    
    # Subsample sizes
    train_subset_size = int(len(db_train) * config['train_fraction'])
    val_subset_size = int(len(db_val) * config['val_fraction'])
    
    print(f"  Training: {len(db_train)} → {train_subset_size} per epoch ({config['train_fraction']*100:.0f}%)")
    print(f"  Validation: {len(db_val)} → {val_subset_size} ({config['val_fraction']*100:.0f}%)")
    
    # =========================================================================
    # MODEL
    # =========================================================================
    print("\n🏗️  Building model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    vit_config = CONFIGS_ViT_seg[config['vit_name']]
    vit_config.n_classes = config['num_classes']
    vit_config.n_skip = config['n_skip']
    vit_config.patches.size = (config['vit_patches_size'], config['vit_patches_size'])
    
    if config['vit_name'].find('R50') != -1:
        vit_config.patches.grid = (
            int(config['img_size'] / config['vit_patches_size']),
            int(config['img_size'] / config['vit_patches_size'])
        )
    
    model = ViT_seg(vit_config, img_size=config['img_size'], num_classes=config['num_classes'])
    
    # Load pretrained weights
    if config['pretrained_path'] and Path(config['pretrained_path']).exists():
        model.load_from(weights=np.load(config['pretrained_path']))
        print(f"  ✅ Loaded pretrained: {config['pretrained_path']}")
    
    model = model.to(device)
    
    # =========================================================================
    # OPTIMIZER & SCHEDULER
    # =========================================================================
    dice_loss = DiceLoss(config['num_classes'])
    ft_loss = FocalTverskyLoss()
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['base_lr'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    max_iters = (train_subset_size // config['batch_size']) * config['max_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_iters,
        eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=config['use_amp'])
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # History tracking
    history = TrainingHistory(snapshot_path)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    # =========================================================================
    # LOAD CHECKPOINT (ADD THIS SECTION)
    # =========================================================================
    checkpoint_path = snapshot_path / 'best_512_balanced.pth'
    start_epoch = 1
    best_dice = 0.0
    
    if checkpoint_path.exists():
        print(f"\n⚡ Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally load optimizer state (for exact resumption)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('val_dice', 0.0)
        print(f"   ✅ Resuming from epoch {start_epoch}")
        print(f"   ✅ Best Dice so far: {best_dice:.4f}")
    else:
        print("\n➤ No checkpoint found – starting from scratch")
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print("\n🚀 Starting training...")
    print(f"  Total epochs: {config['max_epochs']}")
    print(f"  Batch size: {config['batch_size']} (gradient accumulation: {config['gradient_accumulation']})")
    print(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")
    print(f"  Val frequency: every {config['val_frequency']} epochs")
    print(f"  Mixed precision: {config['use_amp']}")
    print(f"  Early stopping: patience={config['patience']}")
    print(f"  Image logging: {config['log_images']}")
    print(f"\n📊 Monitor with TensorBoard:")
    print(f"   tensorboard --logdir {log_dir} --port 6008 --bind_all\n")
    
    iter_num = 0
    best_dice = 0.0
    last_val_metrics = {'loss': 0, 'dice': 0, 'precision': 0, 'recall': 0, 
                        'specificity': 0, 'iou': 0, 'accuracy': 0}
    
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        
        # =====================================================================
        # SUBSAMPLE DATA FOR THIS EPOCH
        # =====================================================================
        if config['resample_each_epoch'] and db_train._is_pos:
            # Get balanced subsample indices
            subset_indices = get_balanced_subsample_indices(
                db_train._is_pos, 
                config['train_fraction'],
                config['positive_ratio']
            )
        else:
            # Random subsample
            subset_indices = random.sample(range(len(db_train)), train_subset_size)
        
        # Create subset dataset
        train_subset = Subset(db_train, subset_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config.get('prefetch_factor', 2),
            drop_last=True
        )
        
        # Reset optimizer at start of epoch
        optimizer.zero_grad()
        
        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['max_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            imgs = batch['image'].to(device, non_blocking=True)
            labs = batch['label'].to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast(enabled=config['use_amp']):
                outputs = model(imgs)
                loss_d = dice_loss(outputs, labs, softmax=True)
                loss_ft = ft_loss(outputs, labs)
                loss = 0.5 * loss_d + 0.5 * loss_ft
                
                # Scale loss by gradient accumulation steps
                loss = loss / config['gradient_accumulation']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights every gradient_accumulation steps
            if (batch_idx + 1) % config['gradient_accumulation'] == 0:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                # Log gradient norm
                if iter_num % config['log_frequency'] == 0:
                    writer.add_scalar('train/grad_norm', grad_norm.item(), iter_num)
            
            # Track metrics (use unscaled loss for reporting)
            with torch.no_grad():
                batch_dice = compute_dice_score(outputs.float(), labs)
            
            # Accumulate with unscaled loss
            epoch_loss += (loss.item() * config['gradient_accumulation'])
            epoch_dice += batch_dice
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() * config["gradient_accumulation"]:.4f}',
                'dice': f'{batch_dice:.4f}'
            })
            
            # ================================================================
            # TENSORBOARD LOGGING
            # ================================================================
            # Scalar logging (frequent)
            if iter_num % config['log_frequency'] == 0:
                writer.add_scalar('train/loss', loss.item() * config['gradient_accumulation'], iter_num)
                writer.add_scalar('train/dice', batch_dice, iter_num)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iter_num)
                
                # Log prediction statistics
                log_prediction_statistics(writer, outputs, labs, iter_num, phase='train')
            
            # Image logging (less frequent to save disk space)
            if config['log_images'] and iter_num % config['image_log_frequency'] == 0:
                log_images_to_tensorboard(writer, imgs, labs, outputs, iter_num, 
                                         phase='train', num_images=config['num_vis_images'])
            
            # Histogram logging (even less frequent)
            if config['log_histograms'] and iter_num % (config['image_log_frequency'] * 2) == 0:
                log_model_histograms(writer, model, iter_num)
            
            iter_num += 1
        
        # Average training metrics
        avg_train_loss = epoch_loss / n_batches
        avg_train_dice = epoch_dice / n_batches
        
        # Log epoch-level training metrics
        writer.add_scalar('train_epoch/loss', avg_train_loss, epoch)
        writer.add_scalar('train_epoch/dice', avg_train_dice, epoch)
        
        # =====================================================================
        # VALIDATION (with frequency control)
        # =====================================================================
        if epoch % config['val_frequency'] == 0:
            # Subsample validation set
            val_indices = random.sample(range(len(db_val)), val_subset_size)
            val_subset = Subset(db_val, val_indices)
            val_loader = DataLoader(
                val_subset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=config['pin_memory']
            )
            
            val_metrics = validate(model, val_loader, dice_loss, ft_loss, device, 
                                   writer, epoch, config['use_amp'],
                                   log_images=config['log_images'],
                                   num_vis_images=config['num_vis_images'])
            last_val_metrics = val_metrics
            
            history.update(epoch, avg_train_loss, val_metrics, scheduler.get_last_lr()[0])
            
            log_msg = (f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                      f"train_dice={avg_train_dice:.4f}, val_dice={val_metrics['dice']:.4f}, "
                      f"val_iou={val_metrics['iou']:.4f}")
            logging.info(log_msg)
            
            # Save best model
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_metrics['dice'],
                    'val_loss': val_metrics['loss'],
                    'config': config
                }, snapshot_path / 'best_512_balanced.pth')
                logging.info(f"✅ New best model: Dice={best_dice:.4f}")
            
            # Periodic checkpoint
            if epoch % config['save_frequency'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_dice': val_metrics['dice']
                }, snapshot_path / f'epoch_{epoch}_512_balanced.pth')
            
            # Early stopping
            if config['early_stopping'] and epoch >= config['min_epochs']:
                if val_metrics['dice'] >= config['min_dice_threshold']:
                    if early_stopping(val_metrics['dice']):
                        print(f"\n🛑 Early stopping at epoch {epoch}")
                        print(f"   Best Dice: {best_dice:.4f}")
                        break
        else:
            # Log training only
            log_msg = f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, train_dice={avg_train_dice:.4f}"
            logging.info(log_msg)
        
        # Status update every 10 epochs
        if epoch % 10 == 0:
            history.print_status(epoch)
    
    # =========================================================================
    # FINALIZE
    # =========================================================================
    writer.close()
    history.save()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"  Best Dice: {best_dice:.4f} (epoch {history.best_epoch})")
    print(f"  Total epochs: {epoch}")
    print(f"  Model: {snapshot_path / 'best_512_balanced.pth'}")
    print(f"\n📊 View results in TensorBoard:")
    print(f"   tensorboard --logdir {log_dir} --port 6008 --bind_all")
    print("="*60)
    
    return model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TransUNet 512×512 MEMORY-OPTIMIZED Training")
    print("WITH ENHANCED TENSORBOARD VISUALIZATION")
    print("="*60)
    print("\n⚡ MEMORY OPTIMIZATIONS ENABLED:")
    print(f"   • Batch size: {CONFIG['batch_size']} (reduced from 16)")
    print(f"   • Gradient accumulation: {CONFIG['gradient_accumulation']} steps")
    print(f"   • Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")
    print(f"   • Data subsampling: {CONFIG['train_fraction']*100:.0f}% per epoch")
    print(f"   • Mixed precision (AMP): {CONFIG['use_amp']}")
    print(f"   • Validation frequency: every {CONFIG['val_frequency']} epochs")
    print("\n🎨 TENSORBOARD VISUALIZATION:")
    print(f"   • Image logging: {CONFIG['log_images']}")
    print(f"   • Histogram logging: {CONFIG['log_histograms']}")
    print(f"   • Images per log: {CONFIG['num_vis_images']}")
    print(f"   • Scalar log freq: every {CONFIG['log_frequency']} iterations")
    print(f"   • Image log freq: every {CONFIG['image_log_frequency']} iterations")
    print("\n💡 Before running, set environment variable:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("\n📊 Monitor with TensorBoard:")
    print(f"   tensorboard --logdir {CONFIG['snapshot_path']}/log_512_optimized_balanced --port 6008 --bind_all")
    print("\n")
    
    train_imagecas_512_optimized(CONFIG)
