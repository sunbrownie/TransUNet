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
from trainer import trainer_penguin

# %%
# Manually setting what previously were command-line arguments
args = {
    'root_path': '../data/Penguin/train_processed_224',
    'dataset': 'Penguin',
    'list_dir': './lists/lists_Penguin',
    'num_classes': 30,
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
     'base_lr': 0.01
}

# Use the arguments
# For example, if you had a line like this in your original script:
# print(args.root_path)
# Replace it with this in the modified script:
# print(args['root_path'])

if __name__ == "__main__":
    if not args['deterministic']:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    dataset_name = args['dataset']
    dataset_config = {
        'Penguin': {
            'root_path': "/home/ubuntu/files/project_TransUNet/data/Penguin/train_processed_224",
            'list_dir': "/home/ubuntu/files/project_TransUNet/TransUNet/lists/lists_Penguin",
            'num_classes': 30,
        },
    }
    args['num_classes'] = dataset_config[dataset_name]['num_classes']
    args['root_path'] = dataset_config[dataset_name]['root_path']
    args['list_dir'] = dataset_config[dataset_name]['list_dir']
    args['is_pretrain'] = True
    args['exp'] = 'TU_' + dataset_name + str(args['img_size'])
    args['max_epochs'] = 150

    snapshot_path = "../model/{}/{}".format(args['exp'], 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args['is_pretrain'] else snapshot_path
    snapshot_path += '_' + args['vit_name']
    snapshot_path = snapshot_path + '_skip' + str(args['n_skip'])
    snapshot_path = snapshot_path + '_vitpatch' + str(args['vit_patches_size']) if args['vit_patches_size'] != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args['max_iterations'])[0:2] + 'k' if args['max_iterations'] != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args['max_epochs']) if args['max_epochs'] != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args['batch_size'])
    snapshot_path = snapshot_path + '_lr' + str(args['base_lr']) if args['base_lr'] != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args['img_size'])
    snapshot_path = snapshot_path + '_s' + str(args['seed']) if args['seed'] != 1234 else snapshot_path

    snapshot_path = "/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args['vit_name']]
    config_vit.n_classes = 9
    config_vit.n_skip = args['n_skip']
    config_vit.pretrained_path = "/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    if args['vit_name'].find('R50') != -1:
        config_vit.patches.grid = (int(args['img_size'] / args['vit_patches_size']), int(args['img_size'] / args['vit_patches_size']))
    net = ViT_seg(config_vit, img_size=args['img_size'], num_classes=9).cuda()
    
    #net.load_from(weights=np.load(config_vit.pretrained_path))
    net.load_state_dict(torch.load("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/epoch_59.pth"))

    net.segmentation_head = SegmentationHead(
        in_channels=config_vit['decoder_channels'][-1],
        out_channels= args['num_classes'], 
        kernel_size=3
    ).cuda()

    #net = ViT_seg(config_vit, img_size=args['img_size'], num_classes=30).cuda()
    #net.load_state_dict(torch.load("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/epoch_3.pth"))

    trainer = {'Penguin': trainer_penguin}
    trainer[dataset_name](args, net, snapshot_path)
