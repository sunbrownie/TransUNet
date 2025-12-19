"""test_imagecas.py — volume‑level evaluation for ImageCas using **DiceLoss** class

* Evaluates the full test set (default), a random volume, or a specified prefix.
* Computes **mean Dice (foreground class only)** with `utils.DiceLoss` and
  **mean HD95** per volume.
* **Only two volumes are saved**: the best‑Dice and worst‑Dice volumes.
* Prints grand‑mean Dice & HD95 at the end for easy reporting.

Run examples
------------
```bash
python test_imagecas.py          # whole test set, save best and worst
python test_imagecas.py random   # evaluate one random volume
python test_imagecas.py extracted_601-800_774.img.nii
```
"""
from __future__ import annotations
import sys, random, logging, numpy as np, shutil
from pathlib import Path
from collections import defaultdict
from typing import List
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ───────── paths & constants ───────────────────────────────────────────────
ROOT_PATH = Path("/home/ubuntu/hist/TransUNet/data")
LIST_DIR  = Path("/home/ubuntu/hist/TransUNet/lists/lists_ImageCas")
CKPT_DIR  = Path("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k")
PRED_DIR  = CKPT_DIR / "preds"
IMG_SIZE  = 224
NUM_CLASSES = 2
SEED = 1236
ARG = (sys.argv[1].lower() if len(sys.argv) > 1 else "all")

random.seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark, cudnn.deterministic = True, False

# ───────── model init ─────────────────────────────────────────────────────-
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg, SegmentationHead
from utils import DiceLoss  # ← your Dice implementation

cfg = CONFIGS_ViT_seg["R50-ViT-B_16"]
cfg.n_classes = NUM_CLASSES; cfg.n_skip = 3
cfg.patches.size = (16,16); cfg.patches.grid = (IMG_SIZE//16, IMG_SIZE//16)
net = ViT_seg(cfg, img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
net.segmentation_head = SegmentationHead(cfg["decoder_channels"][-1], NUM_CLASSES, 3).to(DEVICE)
ckpt = CKPT_DIR / "best.pth"
if not ckpt.exists():
    raise FileNotFoundError(f"{ckpt} missing; train first.")
net.load_state_dict(torch.load(ckpt, map_location="cpu")); net.eval()
print(f"Loaded {ckpt.name}")

# instantiate DiceLoss (foreground class only)
dice_fn = DiceLoss(n_classes=NUM_CLASSES).to(DEVICE)

# ───────── dataset organisation ───────────────────────────────────────────-
from datasets.dataset_imagecas import ImageCas_dataset

ds = ImageCas_dataset(str(ROOT_PATH), str(LIST_DIR), split="test_vol", transform=None)

if ARG == "random":
    s = random.choice(ds.sample_list); VOL_FILTER = s.split("_slice")[0]
    print(f"🎲 Random volume: {VOL_FILTER}")
elif ARG in {"", "all"}:
    VOL_FILTER = ""
else:
    VOL_FILTER = ARG

vol_slices: dict[str, List[str]] = defaultdict(list)
for s in ds.sample_list:
    if VOL_FILTER and not s.startswith(VOL_FILTER):
        continue
    vol_slices[s.split("_slice")[0]].append(s)
if not vol_slices:
    raise ValueError(f"No volume matches '{VOL_FILTER}'.")
print(f"Evaluating {len(vol_slices)} volume(s)")

PRED_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(str(CKPT_DIR/"tb"/"infer_vol"))

# ───────── helpers ─────────────────────────────────────────────────────────

def overlay(img, gt, pred, alpha=0.5):
    base = (img-img.min())/(img.ptp()+1e-5)
    rgb = np.stack([base]*3,-1)
    rgb[...,1] = np.where(gt>0, alpha*1+(1-alpha)*rgb[...,1], rgb[...,1])
    rgb[...,0] = np.where(pred>0,alpha*1+(1-alpha)*rgb[...,0],rgb[...,0])
    return (rgb*255).astype(np.uint8)

from scipy.ndimage import distance_transform_edt as edt

def hd95_binary(a: np.ndarray, b: np.ndarray) -> float:
    """95th percentile Hausdorff distance between two binary masks (slice).
    Returns 0 if both masks are empty, 95 if only one is empty."""
    if a.sum()==0 and b.sum()==0: return 0.0
    if a.sum()==0 or b.sum()==0:  return 95.0
    dt_a, dt_b = edt(1-a), edt(1-b)
    return float(np.percentile(np.hstack([dt_a[b>0], dt_b[a>0]]), 95))

# ───────── evaluation loop ─────────────────────────────────────────────---
best_vol, worst_vol, best_dice, worst_dice = None, None, -1.0, 2.0
all_dices, all_hd95 = [], []

for v_idx, (vid, slices) in enumerate(vol_slices.items()):
    slices.sort(key=lambda x:int(x.split("_slice")[-1].split(".npz")[0]))
    imgs, gts, preds, slice_dice, slice_hd = [], [], [], [], []

    for s in tqdm(slices, desc=vid):
        npz = np.load(ROOT_PATH/"val_processed_224"/s)
        img_np, gt_np = npz["image"], npz["label"]

        # predict logits then dice ---------------------------------------
        t_img  = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        t_gt   = torch.from_numpy(gt_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = net(t_img)
            loss   = dice_fn(logits, t_gt, softmax=True).item()  # 1‑Dice
        dice_val = 1 - loss
        pred_np  = torch.argmax(torch.softmax(logits,1),1).cpu().numpy()[0]

        slice_dice.append(dice_val)
        slice_hd.append(hd95_binary(pred_np>0, gt_np>0))
        imgs.append(img_np); gts.append(gt_np); preds.append(pred_np)

    # volume metrics -------------------------------------------------------
    dice_vol = float(np.mean(slice_dice))
    hd95_vol = float(np.mean(slice_hd))
    all_dices.append(dice_vol); all_hd95.append(hd95_vol)
    writer.add_scalar("dice_vol", dice_vol, v_idx)
    writer.add_scalar("hd95_vol", hd95_vol, v_idx)
    print(f"{vid}: Dice {dice_vol:.4f}  HD95 {hd95_vol:.2f}")

    # track best & worst ---------------------------------------------------
    def save_volume(vol_id):
        arr_imgs, arr_gts, arr_preds = map(np.stack, (imgs, gts, preds))
        aff = np.eye(4)
        nib.save(nib.Nifti1Image(arr_preds.astype(np.uint8), aff), PRED_DIR/f"{vol_id}_pred.nii.gz")
        nib.save(nib.Nifti1Image(arr_imgs.astype(np.float32), aff),  PRED_DIR/f"{vol_id}_img.nii.gz")
        fg = np.where(arr_gts.reshape(len(arr_gts), -1).sum(1)>0)[0]
        sel = [fg[0], fg[len(fg)//2], fg[-1]] if len(fg)>=3 else list(fg)
        fig, ax = plt.subplots(1, len(sel), figsize=(4*len(sel),4))
        if len(sel)==1: ax=[ax]
        for a,idx in zip(ax,sel):
            a.imshow(overlay(arr_imgs[idx], arr_gts[idx], arr_preds[idx])); a.axis('off'); a.set_title(idx)
        plt.tight_layout(); plt.savefig(PRED_DIR/f"{vol_id}_overlay.png", dpi=150); plt.close(fig)

    if dice_vol > best_dice:
        if best_vol:
            for f in PRED_DIR.glob(f"{best_vol}_*.*"): f.unlink()
        best_vol, best_dice = vid, dice_vol
        save_volume(vid)
    if dice_vol < worst_dice:
        if worst_vol and worst_vol != best_vol:
            for f in PRED_DIR.glob(f"{worst_vol}_*.*"): f.unlink()
        worst_vol, worst_dice = vid, dice_vol
        save_volume(vid)

# ───────── summary ─────────────────────────────────────────────────────---
print("==============================")
print(f"Grand‑mean Dice  over {len(all_dices)} volumes: {np.mean(all_dices):.4f}")
print(f"Grand‑mean HD95  over {len(all_hd95)} volumes: {np.mean(all_hd95):.2f}")
print(f"Best:  {best_vol}  Dice {best_dice:.4f}")
print(f"Worst: {worst_vol} Dice {worst_dice:.4f}")
print("Saved results for best & worst volumes to", PRED_DIR)


