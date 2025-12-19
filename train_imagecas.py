# %%
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import SegmentationHead
from trainer import trainer_imagecas

# %%
# --- CLI-style args -----------------------------------------------------------
args = {
    'root_path': '../data/ImageCas/train_processed_224',   # dummy; overwritten below
    'dataset': 'ImageCas',                                 # ← choose dataset here
    'list_dir': './lists/lists_ImageCas',                  # dummy; overwritten below
    'num_classes': 2,                                      # number of ImageCas labels
    'max_iterations': 30000,
    'max_epochs': 150,
    'batch_size': 24,
    'n_gpu': 1,
    'deterministic': 1,
    'base_lr': 0.01,
    'img_size': 224,
    'seed': 1234,
    'n_skip': 3,
    'vit_name': 'R50-ViT-B_16',
    'vit_patches_size': 16,
}

# %%
if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #  1.  Reproducibility
    # -------------------------------------------------------------------------#
    cudnn.benchmark   = not args['deterministic']
    cudnn.deterministic = bool(args['deterministic'])

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    # -------------------------------------------------------------------------#
    #  2.  Dataset-specific parameters
    # -------------------------------------------------------------------------#
    print('Loading data\n')
    dataset_config = {
        'ImageCas': {
            'root_path': "/home/ubuntu/hist/TransUNet/data",
            'list_dir':  "/home/ubuntu/hist/TransUNet/lists/lists_ImageCas",
            'num_classes': 2,
        },
    }

    dataset_name      = args['dataset']
    args['num_classes'] = dataset_config[dataset_name]['num_classes']
    args['root_path']   = dataset_config[dataset_name]['root_path']
    args['list_dir']    = dataset_config[dataset_name]['list_dir']
    args['is_pretrain'] = True
    args['exp']         = 'TU_' + dataset_name + str(args['img_size'])
    args['max_epochs']  = 150

    # -------------------------------------------------------------------------#
    #  3.  Snapshot path assembly  (identical logic to your block)
    # -------------------------------------------------------------------------#
    print('Saving snapshot\n')
    snapshot_path  = "/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # -------------------------------------------------------------------------#
    #  4.  Model
    # -------------------------------------------------------------------------#
    print('Loading model\n')
    config_vit = CONFIGS_ViT_seg[args['vit_name']]
    config_vit.n_classes = 2                   
    config_vit.n_skip    = args['n_skip']
    config_vit.pretrained_path = os.path.join(
        snapshot_path, "R50+ViT-B_16.npz"
    )

    if 'R50' in args['vit_name']:
        config_vit.patches.grid = (
            int(args['img_size'] / args['vit_patches_size']),
            int(args['img_size'] / args['vit_patches_size'])
        )

    net = ViT_seg(config_vit,
                  img_size=args['img_size'],
                  num_classes=9).cuda()

    # optional: load a previous checkpoint
    # net.load_state_dict(torch.load(os.path.join(snapshot_path, "epoch_59.pth")))

    # ---------------------------------------------------------
    # try to resume from the newest "imagecas_epoch_*.pth"
    # ---------------------------------------------------------
    print('Loading epoch if anything is saved\n')
    from pathlib import Path
    ckpt_dir = snapshot_path
    ckpts = sorted(Path(ckpt_dir).glob("best.pth"))
    start_epoch = 0
    if ckpts:
        last_ckpt = ckpts[-1]
        print(f"⚡ Resuming from {last_ckpt.name}")
        net.load_state_dict(torch.load(last_ckpt, map_location="cpu"), strict=False)
        start_epoch = 1#int(last_ckpt.stem.split('_')[-1]) + 1
    else:
        print("➤ No checkpoint found – starting from scratch")


    print('changing segmentation\n')
    # replace the segmentation head to match the new num_classes
    net.segmentation_head = SegmentationHead(
        in_channels=config_vit['decoder_channels'][-1],
        out_channels=args['num_classes'],
        kernel_size=3
    ).cuda()

    # -------------------------------------------------------------------------#
    #  5.  Trainer dispatch
    # -------------------------------------------------------------------------#
    print('Starting training\n')
    trainer = {
        'ImageCas': trainer_imagecas,
    }
    trainer[dataset_name](args, net, snapshot_path)


