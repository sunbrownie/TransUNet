import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for small, imbalanced structures.
    alpha → weight for FN, beta → weight for FP
    gamma → focal parameter (γ=1 ⇒ plain Tversky)
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, softmax=True):
        if softmax:
            logits = torch.softmax(logits, dim=1)

        # foreground channel only (index 1)
        probs_fg  = logits[:, 1]
        target_fg = (target == 1).float()

        tp = (probs_fg * target_fg).sum()
        fp = (probs_fg * (1 - target_fg)).sum()
        fn = ((1 - probs_fg) * target_fg).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss    = (1 - tversky) ** self.gamma
        return loss
        
class DiceLoss(nn.Module):
    """
    Dice loss that matches the original call pattern **but**
    returns Dice loss for *only* the artery / foreground class (index 1).

    ─ Call exactly as before ─────────────────────────────────────────────
        loss_fn = DiceLoss(n_classes=2)
        loss    = loss_fn(logits, labels, weight=None, softmax=True)

      • logits: (B, 2, H, W[, D])  raw network outputs
      • labels: (B, H, W[, D])     integer mask 0 = background, 1 = artery
    """

    def __init__(self, n_classes: int = 2, smooth: float = 1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth    = smooth

    # --------------------------------------------------------------------- utils
    def _one_hot_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, H, W[, D]) integer mask to one-hot (B, C, …).
        Implemented with torch’s native one_hot; keeps device & dtype.
        """
        x = x.long()
        one_hot = F.one_hot(x, num_classes=self.n_classes)            # (B, …, C)
        one_hot = one_hot.permute(0, -1, *range(1, x.ndim))           # (B, C, …)
        return one_hot.float()

    def _dice_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        pred, tgt shapes: (B, …) with probs in pred and binary mask in tgt.
        """
        inter = (pred * tgt).sum()
        denom = (pred * pred).sum() + (tgt * tgt).sum()
        dice  = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice                     # 1 – Dice = loss

    # ------------------------------------------------------------------- forward
    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        weight=None,                    # kept for API compatibility (ignored)
        softmax: bool = False,
    ) -> torch.Tensor:

        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        # One-hot encode on the **same device** as inputs
        target_oh = self._one_hot_encoder(target).to(inputs.device)

        # Use only class-1 (artery) channel
        pred_fg   = inputs[:, 1]
        tgt_fg    = target_oh[:, 1]

        loss = self._dice_loss(pred_fg, tgt_fg)
        return loss


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list, prediction