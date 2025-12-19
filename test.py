import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tensorboardX import SummaryWriter


# %%
# Manually setting what previously were command-line arguments
args = {
    'dataset': 'Synapse',
    'list_dir': "/home/ubuntu/files/project_TransUNet/TransUNet/lists/lists_Synapse",
    'num_classes': 9,
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
    'base_lr': 0.01,
    'is_savenii': 'store_true',
    'volume_path': '../data/Synapse/test_vol_h5',
    'test_save_dir': '../predictions',
}

# Use the arguments
# For example, if you had a line like this in your original script:
# print(args.root_path)
# Replace it with this in the modified script:
# print(args['root_path'])


def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args['volume_path'], split="test_vol", list_dir=args['list_dir'])
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    writer = SummaryWriter("/home/ubuntu/files/project_TransUNet/TransUNet/test_log/test_log_TU_Synapse224/")
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image_display = image[:, 3*image.shape[1]//4, :, :]
        label_display = label[:, 3*label.shape[1]//4, :, :]
        image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())
        writer.add_image('Test_image', image_display, i_batch)
        writer.add_image('Ground_Truth', label_display*50, i_batch)
        metric_i, prediction = test_single_volume(image, label, model, classes=args['num_classes'], patch_size=[args['img_size'], args['img_size']],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args['z_spacing'])
        prediction_tensor = torch.from_numpy(prediction)  # Convert numpy array to tensor
        prediction_display = prediction_tensor[3*prediction_tensor.shape[0]//4, :, :].unsqueeze(0)
        writer.add_image('Prediction', prediction_display*50, i_batch)                              
        metric_list += np.array(metric_i)
        mean_dice = np.mean(metric_i, axis=0)[0]
        mean_hd95 = np.mean(metric_i, axis=0)[1]
        writer.add_text('Metrics', f'idx {i_batch} case {case_name} mean_dice {mean_dice:.4f} mean_hd95 {mean_hd95:.4f}')
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args['num_classes']):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '/home/ubuntu/files/project_TransUNet/data/Synapse/test_vol_h5',
            'list_dir': "/home/ubuntu/files/project_TransUNet/TransUNet/lists/lists_Synapse",
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args['dataset']
    args['num_classes'] = dataset_config[dataset_name]['num_classes']
    args['volume_path'] = dataset_config[dataset_name]['volume_path']
    args['Dataset'] = dataset_config[dataset_name]['Dataset']
    args['list_dir'] = dataset_config[dataset_name]['list_dir']
    args['z_spacing'] = dataset_config[dataset_name]['z_spacing']
    args['is_pretrain'] = True

    # name the same snapshot defined in train script!
    args['exp'] = 'TU_' + dataset_name + str(args['img_size'])
    snapshot_path = "../model/{}/{}".format(args['exp'], 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args['is_pretrain'] else snapshot_path
    snapshot_path += '_' + args['vit_name']
    snapshot_path = snapshot_path + '_skip' + str(args['n_skip'])
    snapshot_path = snapshot_path + '_vitpatch' + str(args['vit_patches_size']) if args['vit_patches_size']!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args['max_epochs']) if args['max_epochs'] != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args['max_iterations'])[0:2] + 'k' if args['max_iterations'] != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args['batch_size'])
    snapshot_path = snapshot_path + '_lr' + str(args['base_lr']) if args['base_lr'] != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args['img_size'])
    snapshot_path = snapshot_path + '_s'+str(args['seed']) if args['seed']!=1234 else snapshot_path

    snapshot_path = "/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/"
    config_vit = CONFIGS_ViT_seg[args['vit_name']]
    config_vit.n_classes = args['num_classes']
    config_vit.n_skip = args['n_skip']
    config_vit.patches.size = (args['vit_patches_size'], args['vit_patches_size'])
    if args['vit_name'].find('R50') !=-1:
        config_vit.patches.grid = (int(args['img_size']/args['vit_patches_size']), int(args['img_size']/args['vit_patches_size']))
    net = ViT_seg(config_vit, img_size=args['img_size'], num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args['max_epochs']-1))
    snapshot = os.path.join(snapshot_path, 'epoch_59.pth')
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args['exp']
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)
    writer = SummaryWriter(log_folder)

    if args['is_savenii']:
        args['test_save_dir'] = '../predictions'
        test_save_path = "/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/predictions"
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


