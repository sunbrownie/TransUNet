#!/usr/bin/env python3
"""
unified_inference.py - Unified inference for 3D U-Net and TransUNet comparison
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom
from typing import Dict, Optional, Tuple

# Import the model architecture from training script
from train_3dunet import UNet3D

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class UnifiedInference:
    """Unified inference for both 3D U-Net and TransUNet models"""
    
    def __init__(self, 
                 unet3d_checkpoint: Optional[str] = None,
                 transunet_checkpoint: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 unet3d_target_size: Tuple[int, int, int] = (64, 128, 128)):
        """
        Initialize unified inference system
        
        Args:
            unet3d_checkpoint: Path to 3D U-Net checkpoint
            transunet_checkpoint: Path to TransUNet checkpoint
            output_dir: Directory for saving results
            unet3d_target_size: Target size for 3D U-Net inference
        """
        self.output_dir = Path(output_dir or "/content/drive/MyDrive/TransUNet-main/unified_predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.unet3d_target_size = unet3d_target_size
        
        # Create subdirectories
        for subdir in ["unet3d", "transunet", "comparisons", "overlays"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Load models
        self.unet3d_model = self._load_unet3d(unet3d_checkpoint)
        self.transunet_model = self._load_transunet(transunet_checkpoint)
        
        print("✅ Unified inference system ready!")
    
    def _load_unet3d(self, checkpoint_path: Optional[str]) -> nn.Module:
        """Load 3D U-Net model"""
        model = UNet3D(in_channels=1, out_channels=2).to(DEVICE)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            # Load checkpoint with weights_only=False to handle numpy arrays in checkpoint
            try:
                # Try with weights_only=False (needed for checkpoints with numpy arrays)
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't have weights_only parameter
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded 3D U-Net from {checkpoint_path}")
            
            # Print checkpoint info if available
            if 'val_dice' in checkpoint:
                print(f"   Model performance: Dice={checkpoint['val_dice']:.4f}")
            if 'epoch' in checkpoint:
                print(f"   Checkpoint from epoch: {checkpoint['epoch']}")
        else:
            print("⚠️ 3D U-Net checkpoint not found - using random weights")
        
        model.eval()
        return model
    
    def _load_transunet(self, checkpoint_path: Optional[str]) -> Optional[nn.Module]:
        """Load TransUNet model"""
        try:
            # Import TransUNet modules
            sys.path.append('/content/drive/MyDrive/TransUNet-main')
            from networks.vit_seg_modeling import VisionTransformer as ViT_seg
            from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg, SegmentationHead
            
            cfg = CONFIGS_ViT_seg["R50-ViT-B_16"]
            cfg.n_classes = 2
            cfg.n_skip = 3
            cfg.patches.size = (16, 16)
            cfg.patches.grid = (14, 14)
            
            model = ViT_seg(cfg, img_size=224, num_classes=2).to(DEVICE)
            model.segmentation_head = SegmentationHead(
                cfg["decoder_channels"][-1], 2, 3
            ).to(DEVICE)
            
            if checkpoint_path and Path(checkpoint_path).exists():
                try:
                    # Try with weights_only=False
                    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions
                    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
                
                model.load_state_dict(state_dict)
                print(f"✅ Loaded TransUNet from {checkpoint_path}")
            else:
                print("⚠️ TransUNet checkpoint not found")
                return None
            
            model.eval()
            return model
            
        except ImportError as e:
            print(f"⚠️ Could not import TransUNet modules: {e}")
            print("   TransUNet inference will be skipped")
            return None
    
    def preprocess_volume_for_unet3d(self, volume: np.ndarray) -> np.ndarray:
        """Preprocessing for 3D U-Net (using scipy.zoom)"""
        # Clip and normalize
        volume = np.clip(volume, 0, 4000)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-7)
        
        # Resize to target size
        zoom_factors = [self.unet3d_target_size[i] / volume.shape[i] for i in range(3)]
        volume = zoom(volume, zoom_factors, order=1)
        
        return volume.astype(np.float32)
    
    def preprocess_slice_for_transunet(self, slice_2d: np.ndarray) -> torch.Tensor:
        """Preprocess 2D slice for TransUNet - matching training preprocessing"""
        # Clip and normalize (same as training)
        slice_2d = np.clip(slice_2d, 0, 4000)
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-7)
        
        # Use SimpleITK resampling to match training
        img = sitk.GetImageFromArray(slice_2d)
        orig_size = img.GetSize()
        orig_spacing = img.GetSpacing()
        new_spacing = (
            orig_spacing[0] * orig_size[0] / 224,
            orig_spacing[1] * orig_size[1] / 224,
        )
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize((224, 224))
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        
        slice_2d = sitk.GetArrayFromImage(resampler.Execute(img))
        
        # Pad if necessary (matching training)
        if slice_2d.shape != (224, 224):
            dh = max(0, 224 - slice_2d.shape[0])
            dw = max(0, 224 - slice_2d.shape[1])
            slice_2d = np.pad(slice_2d, ((0, dh), (0, dw)), "constant", constant_values=0)
        
        # Convert to tensor
        slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return slice_tensor.to(DEVICE)
    
    def predict_unet3d(self, volume: np.ndarray) -> np.ndarray:
        """Run 3D U-Net inference"""
        original_shape = volume.shape
        
        # Preprocess using the specific 3D U-Net preprocessing
        volume_processed = self.preprocess_volume_for_unet3d(volume)
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume_processed).float()
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = self.unet3d_model(volume_tensor)
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            prediction = prediction.cpu().numpy()[0]
        
        # Resize back to original shape
        if prediction.shape != original_shape:
            zoom_factors = [original_shape[i] / prediction.shape[i] for i in range(3)]
            prediction = zoom(prediction, zoom_factors, order=0)
        
        return prediction.astype(np.uint8)
    
    def predict_transunet(self, volume: np.ndarray) -> np.ndarray:
        """Run TransUNet inference slice by slice with proper preprocessing"""
        if self.transunet_model is None:
            print("⚠️ TransUNet model not loaded, returning zeros")
            return np.zeros_like(volume, dtype=np.uint8)
        
        prediction = np.zeros_like(volume, dtype=np.uint8)
        
        with torch.no_grad():
            for z in tqdm(range(volume.shape[0]), desc="TransUNet inference", leave=False):
                slice_2d = volume[z]
                original_shape = slice_2d.shape
                
                # Use the TransUNet-specific preprocessing
                slice_tensor = self.preprocess_slice_for_transunet(slice_2d)
                
                # Predict
                outputs = self.transunet_model(slice_tensor)
                pred_slice = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                pred_slice = pred_slice.cpu().numpy()[0]
                
                # Resize back to original shape if needed
                if pred_slice.shape != original_shape:
                    # Use SimpleITK for consistency
                    pred_img = sitk.GetImageFromArray(pred_slice.astype(np.float32))
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetSize([original_shape[1], original_shape[0]])  # Note: SimpleITK uses (W, H)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest for labels
                    
                    # Calculate spacing to resize back
                    pred_spacing = pred_img.GetSpacing()
                    new_spacing = (
                        pred_spacing[0] * pred_slice.shape[1] / original_shape[1],
                        pred_spacing[1] * pred_slice.shape[0] / original_shape[0]
                    )
                    resampler.SetOutputSpacing(new_spacing)
                    resampler.SetOutputOrigin(pred_img.GetOrigin())
                    resampler.SetOutputDirection(pred_img.GetDirection())
                    
                    pred_slice = sitk.GetArrayFromImage(resampler.Execute(pred_img))
                
                prediction[z] = pred_slice.astype(np.uint8)
        
        return prediction
    
    def calculate_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Calculate Dice coefficient and other metrics"""
        pred_binary = (pred > 0).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)
        
        # Dice score
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum()
        dice = (2.0 * intersection) / (union + 1e-8)
        
        # IoU (Jaccard index)
        iou = intersection / (union - intersection + 1e-8)
        
        # Sensitivity (Recall)
        sensitivity = intersection / (gt_binary.sum() + 1e-8)
        
        # Precision
        precision = intersection / (pred_binary.sum() + 1e-8)
        
        # F1 Score (same as Dice but calculated differently for verification)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
        
        return {
            'dice': dice,
            'iou': iou,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1': f1
        }
    
    def create_visualization(self, 
                           case_name: str,
                           img: np.ndarray,
                           gt: np.ndarray,
                           pred_unet: np.ndarray,
                           pred_transunet: np.ndarray,
                           metrics_unet: Dict,
                           metrics_transunet: Dict) -> None:
        """Create and save comparison visualization"""
        
        # Find slices with ground truth
        gt_slices = np.where(gt.sum(axis=(1, 2)) > 0)[0]
        
        if len(gt_slices) > 0:
            # Select beginning, middle, and end of GT region
            selected_slices = [
                gt_slices[0],
                gt_slices[len(gt_slices) // 2],
                gt_slices[-1]
            ]
        else:
            # No ground truth, use middle slices
            selected_slices = [
                img.shape[0] // 4,
                img.shape[0] // 2,
                3 * img.shape[0] // 4
            ]
        
        # Create figure
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f'{case_name} - Model Comparison\n'
                    f'3D U-Net: Dice={metrics_unet["dice"]:.3f}, IoU={metrics_unet["iou"]:.3f}, '
                    f'Sens={metrics_unet["sensitivity"]:.3f}, Prec={metrics_unet["precision"]:.3f}\n'
                    f'TransUNet: Dice={metrics_transunet["dice"]:.3f}, IoU={metrics_transunet["iou"]:.3f}, '
                    f'Sens={metrics_transunet["sensitivity"]:.3f}, Prec={metrics_transunet["precision"]:.3f}',
                    fontsize=12)
        
        for i, slice_idx in enumerate(selected_slices):
            if slice_idx >= img.shape[0]:
                continue
            
            # Original image
            axes[i, 0].imshow(img[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Original (slice {slice_idx})')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(gt[slice_idx], cmap='hot')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # 3D U-Net prediction
            axes[i, 2].imshow(pred_unet[slice_idx], cmap='hot')
            axes[i, 2].set_title('3D U-Net')
            axes[i, 2].axis('off')
            
            # TransUNet prediction
            axes[i, 3].imshow(pred_transunet[slice_idx], cmap='hot')
            axes[i, 3].set_title('TransUNet')
            axes[i, 3].axis('off')
            
            # Combined overlay
            img_norm = (img[slice_idx] - img[slice_idx].min()) / (img[slice_idx].max() - img[slice_idx].min() + 1e-8)
            overlay = np.stack([img_norm, img_norm, img_norm], axis=-1)
            
            # Color coding: GT=green, U-Net=red, TransUNet=blue
            gt_mask = gt[slice_idx] > 0
            unet_mask = pred_unet[slice_idx] > 0
            trans_mask = pred_transunet[slice_idx] > 0
            
            # Apply colors with transparency
            if gt_mask.any():
                overlay[gt_mask] = overlay[gt_mask] * 0.5 + np.array([0, 0.5, 0])
            if unet_mask.any():
                overlay[unet_mask] = overlay[unet_mask] * 0.5 + np.array([0.5, 0, 0])
            if trans_mask.any():
                overlay[trans_mask] = overlay[trans_mask] * 0.5 + np.array([0, 0, 0.5])
            
            axes[i, 4].imshow(overlay)
            axes[i, 4].set_title('Overlay\n(G=GT, R=U-Net, B=TransUNet)')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / "overlays" / f"{case_name}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"   Visualization saved to {save_path}")
    
    def save_results(self,
                    case_name: str,
                    img: np.ndarray,
                    gt: np.ndarray,
                    pred_unet: np.ndarray,
                    pred_transunet: np.ndarray,
                    metrics_unet: Dict,
                    metrics_transunet: Dict) -> None:
        """Save prediction results as NIfTI files and metrics"""
        affine = np.eye(4)
        
        # Save predictions
        nib.save(nib.Nifti1Image(pred_unet.astype(np.uint8), affine),
                self.output_dir / "unet3d" / f"{case_name}_pred.nii.gz")
        
        nib.save(nib.Nifti1Image(pred_transunet.astype(np.uint8), affine),
                self.output_dir / "transunet" / f"{case_name}_pred.nii.gz")
        
        # Save original and GT for reference
        nib.save(nib.Nifti1Image(img.astype(np.float32), affine),
                self.output_dir / "comparisons" / f"{case_name}_img.nii.gz")
        
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), affine),
                self.output_dir / "comparisons" / f"{case_name}_gt.nii.gz")
        
        # Save metrics to text file
        metrics_file = self.output_dir / "comparisons" / f"{case_name}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Case: {case_name}\n")
            f.write("="*50 + "\n")
            f.write("\n3D U-Net Metrics:\n")
            for key, value in metrics_unet.items():
                f.write(f"  {key:12s}: {value:.4f}\n")
            f.write("\nTransUNet Metrics:\n")
            for key, value in metrics_transunet.items():
                f.write(f"  {key:12s}: {value:.4f}\n")
    
    def run_inference_on_volume(self, volume_path: str, label_path: str) -> Dict:
        """Run inference on a single volume"""
        
        # Extract case name
        case_name = Path(volume_path).stem.replace('.img', '').replace('.nii', '')
        print(f"\n📊 Processing {case_name}...")
        
        # Load volumes
        img_sitk = sitk.ReadImage(volume_path)
        lbl_sitk = sitk.ReadImage(label_path)
        
        img_volume = sitk.GetArrayFromImage(img_sitk)
        lbl_volume = sitk.GetArrayFromImage(lbl_sitk)
        
        print(f"   Volume shape: {img_volume.shape}")
        print(f"   Volume intensity range: [{img_volume.min():.1f}, {img_volume.max():.1f}]")
        
        # Run predictions
        print("   🔍 Running 3D U-Net inference...")
        pred_unet = self.predict_unet3d(img_volume)
        
        print("   🔍 Running TransUNet inference (with SimpleITK preprocessing)...")
        pred_transunet = self.predict_transunet(img_volume)
        
        # Calculate metrics
        metrics_unet = self.calculate_metrics(pred_unet, lbl_volume)
        metrics_transunet = self.calculate_metrics(pred_transunet, lbl_volume)
        
        print(f"   📈 3D U-Net:  Dice={metrics_unet['dice']:.3f}, IoU={metrics_unet['iou']:.3f}, "
              f"Sens={metrics_unet['sensitivity']:.3f}, Prec={metrics_unet['precision']:.3f}")
        print(f"   📈 TransUNet: Dice={metrics_transunet['dice']:.3f}, IoU={metrics_transunet['iou']:.3f}, "
              f"Sens={metrics_transunet['sensitivity']:.3f}, Prec={metrics_transunet['precision']:.3f}")
        
        # Save results
        self.save_results(case_name, img_volume, lbl_volume, pred_unet, pred_transunet,
                         metrics_unet, metrics_transunet)
        
        # Create visualization
        self.create_visualization(case_name, img_volume, lbl_volume, 
                                pred_unet, pred_transunet,
                                metrics_unet, metrics_transunet)
        
        return {
            'case_name': case_name,
            'unet3d': metrics_unet,
            'transunet': metrics_transunet
        }
    
    def run_batch_inference(self, img_dir: str, lbl_dir: str, max_volumes: Optional[int] = None) -> Dict:
        """Run inference on multiple volumes"""
        
        img_dir = Path(img_dir)
        lbl_dir = Path(lbl_dir)
        
        # Collect volume pairs
        img_files = sorted(img_dir.glob("*.img.nii.gz"))
        
        if max_volumes:
            img_files = img_files[:max_volumes]
        
        results = []
        
        for img_file in tqdm(img_files, desc="Processing volumes"):
            lbl_file = lbl_dir / img_file.name.replace(".img.nii.gz", ".label.nii.gz")
            
            if not lbl_file.exists():
                print(f"⚠️ Label not found for {img_file.name}")
                continue
            
            result = self.run_inference_on_volume(str(img_file), str(lbl_file))
            results.append(result)
        
        # Print summary
        self._print_summary(results)
        
        # Save summary to CSV
        self._save_summary_csv(results)
        
        return results
    
    def _save_summary_csv(self, results: list) -> None:
        """Save summary results to CSV file"""
        import csv
        
        csv_path = self.output_dir / "summary_results.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Case', 
                'UNet3D_Dice', 'UNet3D_IoU', 'UNet3D_Sensitivity', 'UNet3D_Precision',
                'TransUNet_Dice', 'TransUNet_IoU', 'TransUNet_Sensitivity', 'TransUNet_Precision'
            ])
            
            # Data rows
            for r in results:
                writer.writerow([
                    r['case_name'],
                    f"{r['unet3d']['dice']:.4f}",
                    f"{r['unet3d']['iou']:.4f}",
                    f"{r['unet3d']['sensitivity']:.4f}",
                    f"{r['unet3d']['precision']:.4f}",
                    f"{r['transunet']['dice']:.4f}",
                    f"{r['transunet']['iou']:.4f}",
                    f"{r['transunet']['sensitivity']:.4f}",
                    f"{r['transunet']['precision']:.4f}"
                ])
        
        print(f"\n📊 Summary CSV saved to {csv_path}")
    
    def _print_summary(self, results: list) -> None:
        """Print summary statistics"""
        if not results:
            print("No results to summarize")
            return
        
        # Extract metrics
        metrics = {
            'unet3d': {
                'dice': [r['unet3d']['dice'] for r in results],
                'iou': [r['unet3d']['iou'] for r in results],
                'sensitivity': [r['unet3d']['sensitivity'] for r in results],
                'precision': [r['unet3d']['precision'] for r in results]
            },
            'transunet': {
                'dice': [r['transunet']['dice'] for r in results],
                'iou': [r['transunet']['iou'] for r in results],
                'sensitivity': [r['transunet']['sensitivity'] for r in results],
                'precision': [r['transunet']['precision'] for r in results]
            }
        }
        
        print("\n" + "="*70)
        print("📊 SUMMARY RESULTS")
        print("="*70)
        print(f"Processed {len(results)} volumes\n")
        
        # Print metrics for each model
        for model_name, model_metrics in metrics.items():
            model_display = "3D U-Net" if model_name == "unet3d" else "TransUNet"
            print(f"{model_display}:")
            for metric_name, values in model_metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric_name.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")
            print()
        
        # Best and worst cases for Dice score
        unet_dice = metrics['unet3d']['dice']
        trans_dice = metrics['transunet']['dice']
        
        best_unet_idx = np.argmax(unet_dice)
        worst_unet_idx = np.argmin(unet_dice)
        best_trans_idx = np.argmax(trans_dice)
        worst_trans_idx = np.argmin(trans_dice)
        
        print("Best/Worst Cases (by Dice score):")
        print(f"  Best 3D U-Net:  {results[best_unet_idx]['case_name']:10s} (Dice: {unet_dice[best_unet_idx]:.4f})")
        print(f"  Worst 3D U-Net: {results[worst_unet_idx]['case_name']:10s} (Dice: {unet_dice[worst_unet_idx]:.4f})")
        print(f"  Best TransUNet: {results[best_trans_idx]['case_name']:10s} (Dice: {trans_dice[best_trans_idx]:.4f})")
        print(f"  Worst TransUNet: {results[worst_trans_idx]['case_name']:10s} (Dice: {trans_dice[worst_trans_idx]:.4f})")
        
        # Performance comparison
        print("\nPerformance Comparison:")
        unet_wins = sum(u > t for u, t in zip(unet_dice, trans_dice))
        trans_wins = sum(t > u for u, t in zip(unet_dice, trans_dice))
        ties = len(results) - unet_wins - trans_wins
        
        print(f"  3D U-Net performs better: {unet_wins}/{len(results)} cases")
        print(f"  TransUNet performs better: {trans_wins}/{len(results)} cases")
        print(f"  Tied performance: {ties}/{len(results)} cases")
        print("="*70)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified inference for 3D U-Net and TransUNet')
    parser.add_argument('--unet3d-checkpoint', type=str, 
                       default='/content/drive/MyDrive/TransUNet-main/model/unet3d_checkpoint/best_unet3d.pth',
                       help='Path to 3D U-Net checkpoint')
    parser.add_argument('--transunet-checkpoint', type=str,
                       default='/content/drive/MyDrive/TransUNet-main/model/vit_checkpoint/imagenet21k/best.pth',
                       help='Path to TransUNet checkpoint')
    parser.add_argument('--img-dir', type=str, default='/content/local_data/train',
                       help='Directory containing images')
    parser.add_argument('--lbl-dir', type=str, default='/content/local_data/label',
                       help='Directory containing labels')
    parser.add_argument('--output-dir', type=str,
                       default='/content/drive/MyDrive/TransUNet-main/unified_predictions',
                       help='Output directory for results')
    parser.add_argument('--volume', type=str, default=None,
                       help='Specific volume to process (e.g., "379")')
    parser.add_argument('--max-volumes', type=int, default=None,
                       help='Maximum number of volumes to process')
    parser.add_argument('--target-size', type=int, nargs=3, default=[64, 128, 128],
                       help='Target size for 3D U-Net (D H W)')
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference = UnifiedInference(
        unet3d_checkpoint=args.unet3d_checkpoint,
        transunet_checkpoint=args.transunet_checkpoint,
        output_dir=args.output_dir,
        unet3d_target_size=tuple(args.target_size)
    )
    
    # Run inference
    if args.volume:
        # Single volume
        img_path = f"{args.img_dir}/{args.volume}.img.nii.gz"
        lbl_path = f"{args.lbl_dir}/{args.volume}.label.nii.gz"
        
        if not os.path.exists(img_path):
            print(f"❌ Volume {args.volume} not found")
            return
        
        results = inference.run_inference_on_volume(img_path, lbl_path)
    else:
        # Batch inference
        results = inference.run_batch_inference(
            args.img_dir, 
            args.lbl_dir,
            max_volumes=args.max_volumes
        )
    
    print(f"\n✅ Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
