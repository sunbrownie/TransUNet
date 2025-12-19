
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, FocalTverskyLoss  
from torchvision import transforms
import cv2
from PIL import Image


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args['base_lr']
    num_classes = args['num_classes']
    batch_size = args['batch_size'] * args['n_gpu']
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args['root_path'], list_dir=args['list_dir'], split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args['img_size'], args['img_size']])]))
    print("The length of train set is: {}".format(len(db_train)))
    
    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args['max_epochs']
    max_iterations = args['max_epochs'] * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 30  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 4) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break


def process_batch(images, labels, model, ce_loss, dice_loss):
    outputs = model(images)
    loss_ce = ce_loss(outputs, labels[:].squeeze().long())
    loss_dice = dice_loss(outputs, labels.squeeze(), softmax=True)
    loss = 0.2 * loss_ce + 0.8 * loss_dice
    return loss, outputs


import os
import sys
import random
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data            import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard     import SummaryWriter
from torchvision                 import transforms
from tqdm.auto                   import tqdm

from datasets.dataset_imagecas   import ImageCas_dataset, RandomGenerator


def trainer_imagecas(args, model, snapshot_path):
    """
    Train TransUNet (or any torch model) on the pre-processed ImageCas slices.
    """

    # ───────────────────────── imports
    import os, random, logging, sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import transforms
    from torch.nn import CrossEntropyLoss
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.auto import tqdm

    from datasets.dataset_imagecas import ImageCas_dataset, RandomGenerator

    # ───────────────────────── quick validation helper (5 pos + 5 neg)
    def _quick_val(model, loader, ce_loss, dice_loss, ft_loss,
                   pos_target=5, neg_target=5):
        model.eval()
        pos_left, neg_left = pos_target, neg_target
        val_loss, seen = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                imgs, lbls = batch['image'].cuda(), batch['label'].cuda()
                outs = model(imgs)
                is_pos = (lbls.view(lbls.size(0), -1).sum(dim=1) > 0)
                for i in range(imgs.size(0)):
                    if is_pos[i] and pos_left == 0:  continue
                    if (not is_pos[i]) and neg_left == 0:  continue
                    ce   = ce_loss(outs[i:i+1], lbls[i:i+1].long())
                    dice = dice_loss(outs[i:i+1], lbls[i:i+1], softmax=True)
                    ft   = ft_loss  (outs[i:i+1], lbls[i:i+1])
                    #val_loss += 0.5 * ce + 0.5 * dice
                    val_loss += 0.5 * dice + 0.5 * ft    
                    seen += 1
                    if is_pos[i]:  pos_left -= 1
                    else:          neg_left -= 1
                    if pos_left == 0 and neg_left == 0:  break
                if pos_left == 0 and neg_left == 0:  break
        model.train()
        if pos_left or neg_left:
            logging.warning(
                "quick_val: ran out of data — %d pos, %d neg still missing",
                pos_left, neg_left
            )
        return (val_loss / seen).item() if seen else float("nan")
    # ───────────────────────────────────────────────────────────────

    # ---------- Logging -------------------------------------------
    log_dir = os.path.join(snapshot_path, "log_imagecas")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, "log.txt"),
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # ---------- Data ----------------------------------------------
    base_lr   = args['base_lr']
    n_cls     = args['num_classes']
    batch_sz  = args['batch_size'] * args['n_gpu']
 
    logging.info("Starting with data")
    tr_transform = transforms.Compose([
        RandomGenerator(
            output_size=[args['img_size'], args['img_size']],
            elastic=True,
            intensity_sigma=0.1,
            gamma=0.5
            )
    ])
    db_train = ImageCas_dataset(
        base_dir=args['root_path'],
        list_dir=args['list_dir'],
        split="train",
        transform=tr_transform,
        positive_ratio=0.6
    )
    logging.info("Train set size: %d", len(db_train))

    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)

    pos_mask = torch.tensor(db_train._is_pos, dtype=torch.bool)
    n_pos, n_neg = pos_mask.sum().item(), (~pos_mask).sum().item()
    r = db_train.positive_ratio
    w_pos = r   / max(1, n_pos)
    w_neg = (1-r) / max(1, n_neg)
    weights = torch.where(pos_mask,
                          torch.full_like(pos_mask, w_pos, dtype=torch.float),
                          torch.full_like(pos_mask, w_neg, dtype=torch.float))

    sampler = WeightedRandomSampler(weights, num_samples=len(db_train),
                                    replacement=False)# replacememnt True

    trainloader = DataLoader(db_train,
                             batch_size=batch_sz,
                             sampler=sampler,
                             num_workers=8,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    db_val = ImageCas_dataset(
        base_dir=args['root_path'],
        list_dir=args['list_dir'],
        split="val",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args['img_size'],
                                          args['img_size']])]
        ),
    )
    val_loader = DataLoader(
        db_val,
        batch_size=batch_sz,
        shuffle=False,            # NEW → random sampling for quick-val & visuals
        num_workers=4,
        pin_memory=True
    )

    val_vis_loader = DataLoader(
        db_val,
        batch_size=batch_sz,
        shuffle=True,           
        num_workers=4,
        pin_memory=True
    )
    val_vis_iter = iter(val_vis_loader)


    # ---------- Model / loss / opt --------------------------------
    if args['n_gpu'] > 1:
        model = nn.DataParallel(model.cuda())
    model.train()

    ce_loss   = CrossEntropyLoss()
    dice_loss = DiceLoss(n_cls)
    ft_loss   = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    #optimizer = optim.SGD(model.parameters(),
    #                      lr=base_lr,
    #                      momentum=0.9,
    #                      weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)  # NEW
    max_epoch      = args['max_epochs']
    max_iterations = 5 * len(trainloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)

    writer = SummaryWriter(log_dir)

    # baseline val before first gradient step
    init_val = _quick_val(model, val_loader, ce_loss, dice_loss, ft_loss, pos_target=200, neg_target=50)
    writer.add_scalar('val_loss', init_val, 0)
    logging.info("Initial val_loss: %.5f", init_val)

    max_epoch      = args['max_epochs']
    max_iterations = max_epoch * len(trainloader)
    logging.info("%d iters / epoch, %d max iters", len(trainloader),
                 max_iterations)

    iter_num, best_val = 0, float("inf")

    # ---------- Training loop -------------------------------------
    for epoch in tqdm(range(max_epoch), ncols=70):

        for batch in trainloader:
            imgs, labs = batch['image'].cuda(), batch['label'].cuda()

            outs       = model(imgs)
            #loss_ce    = ce_loss(outs, labs.long())
            #loss_dice  = dice_loss(outs, labs, softmax=True)
            #loss       = 0.5 * loss_ce + 0.5 * loss_dice
            loss_dice  = dice_loss(outs, labs, softmax=True)
            loss_ft    = ft_loss (outs, labs)          # already includes soft-max
            loss       = 0.5 * loss_dice + 0.5 * loss_ft   # CE is removed


            optimizer.zero_grad()
            loss.backward()

            # LR schedule
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            #for pg in optimizer.param_groups:  pg['lr'] = lr_
            # optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); 
            scheduler.step()
            lr_ = scheduler.get_last_lr()[0]

            # -------- scalar logs every 10 iters --------------------
            if iter_num % 10 == 0:
                writer.add_scalar('lr',         lr_,    iter_num)
                writer.add_scalar('loss_total', loss,   iter_num)
                #writer.add_scalar('loss_ce',    loss_ce,iter_num)

            # -------- visuals + quick-val every 1000 iters ----------
            if iter_num % 1000 == 0:
                # ── train overlay
                idx = _pick_pos(imgs, labs);  idx = idx[0] if isinstance(idx,tuple) else idx
                WL, WW = 50, 2000
                lower  = WL - WW/2
                base   = torch.clamp(imgs[idx,0]*4000 - lower, 0, WW) / WW
                gt     = (labs[idx] > 0).float()
                pred   = torch.argmax(torch.softmax(outs,1),1)[idx].float()

                writer.add_image('train/overlay',     make_overlay_hu(base, gt, pred), iter_num)
                writer.add_image('train/image',       base.unsqueeze(0),               iter_num)
                writer.add_image('train/prediction',  pred.unsqueeze(0)*50,            iter_num)
                writer.add_image('train/label',       gt.unsqueeze(0)*50,              iter_num)

                # ── quick validation loss
                quick_val = _quick_val(model, val_loader, ce_loss, dice_loss, ft_loss, pos_target=200, neg_target=50)
                val_loss = quick_val
                writer.add_scalar('val_loss', quick_val, iter_num)

                # ── validation visuals (random batch because val_loader shuffle=True)  

                with torch.no_grad():
                    try:
                        v_batch = next(val_vis_iter)
                    except StopIteration:               # reached end → reshuffle & restart
                        val_vis_iter = iter(val_vis_loader)
                        v_batch      = next(val_vis_iter)

                    v_img = v_batch['image'].cuda()
                    v_lbl = v_batch['label'].cuda()
                    v_out = model(v_img)    

                v_idx = _pick_pos(v_img, v_lbl);  v_idx = v_idx[0] if isinstance(v_idx,tuple) else v_idx
                base_v = torch.clamp(v_img[v_idx,0]*4000 - lower, 0, WW) / WW
                gt_v   = (v_lbl[v_idx] > 0).float()
                pred_v = torch.argmax(torch.softmax(v_out,1),1)[v_idx].float()

                writer.add_image('val/overlay',    make_overlay_hu(base_v, gt_v, pred_v), iter_num)
                writer.add_image('val/image',      base_v.unsqueeze(0),                   iter_num)
                writer.add_image('val/prediction', pred_v.unsqueeze(0)*50,                iter_num)
                writer.add_image('val/label',      gt_v.unsqueeze(0)*50,                  iter_num)

                if val_loss < best_val:
                    best_val = val_loss
                    ckpt = os.path.join(snapshot_path, f"best.pth")
                    torch.save(model.state_dict(), ckpt)
                    logging.info("✅  val_loss improved to %.5f — saved %s", best_val, ckpt)
                # ----------------------------------------------------

            iter_num += 1

        # ---------- full validation (unchanged) --------------------
        model.eval()
        val_loss, overlay_done = 0.0, False
        with torch.no_grad():
            for v_batch in val_loader:
                v_img = v_batch['image'].cuda()
                v_lbl = v_batch['label'].cuda()
                v_out = model(v_img)

                v_ce   = ce_loss(v_out, v_lbl.long())
                v_dice = dice_loss(v_out, v_lbl, softmax=True)
                v_ft   = ft_loss  (v_out, v_lbl)
                val_loss += 0.5 * v_dice + 0.5 * v_ft
                #val_loss += 0.5 * v_ce + 0.5 * v_dice
        val_loss /= len(val_loader)
        writer.add_scalar('val_loss', val_loss, epoch)
        model.train()

        # checkpoint if improved
        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(snapshot_path, f"imagecas_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            logging.info("✅  val_loss improved to %.5f — saved %s", best_val, ckpt)

    writer.close()
    return "ImageCas training finished!"

# ------------------------------------------------------------------
# Helper: pick the first slice that contains any foreground (= artery)
# ------------------------------------------------------------------
def _pick_pos(imgs, labs):
    """
    imgs: (B, 1, H, W) float  |  labs: (B, H, W) long
    returns index in batch (int) you can use for visualisation.
    """
    pos = (labs > 0).view(labs.size(0), -1).any(-1)   # (B,) bool
    idx = int(pos.nonzero(as_tuple=True)[0][0]) if pos.any() else 0
    return idx

# ------------------------------------------------------------------
# helper lives near the top of trainer_imagecas.py
# ------------------------------------------------------------------
def make_overlay_hu(img_1ch,            # (H,W) tensor in 0‥1 after HU-window
                    gt_mask,            # (H,W) 0/1
                    pred_mask,          # (H,W) 0/1
                    alpha=0.6):
    """
    Colour-codes a windowed CT slice:
        GT  → green
        Pred → red
        Overlap → yellow
    Returns (3,H,W) RGB tensor in 0‥1.
    """
    rgb = torch.stack([img_1ch, img_1ch, img_1ch], dim=0)
    rgb[1] = torch.where(gt_mask>0, alpha*1. + (1-alpha)*rgb[1], rgb[1])  # green
    rgb[0] = torch.where(pred_mask>0, alpha*1. + (1-alpha)*rgb[0], rgb[0])# red
    return rgb.clamp(0,1)


def trainer_penguin(args, model, snapshot_path):
    
    from datasets.dataset_penguin import Penguin_dataset, RandomGenerator
    logging.basicConfig(filename="/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/log_penguin/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args['base_lr']
    num_classes = args['num_classes']
    batch_size = args['batch_size'] * args['n_gpu']
    # max_iterations = args.max_iterations
    db_train = Penguin_dataset(base_dir=args['root_path'], list_dir=args['list_dir'], split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args['img_size'], args['img_size']])]))
    print("The length of train set is: {}".format(len(db_train)))
    
    
    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter("/home/ubuntu/files/project_TransUNet/model/vit_checkpoint/imagenet21k/log_penguin/")
    iter_num = 0
    max_epoch = args['max_epochs']
    max_iterations = args['max_epochs'] * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    class_weights = torch.tensor([1.97288597e-07, 3.96445289e-05, 3.17975748e-04, 7.10543115e-04,
                              1.02018914e-03, 1.03861384e-03, 1.00895855e-03, 1.18726043e-03,
                              1.10602507e-03, 9.15400192e-04, 1.02374106e-03, 2.69684280e-05,
                              3.53018914e-04, 6.14111846e-04, 1.98149530e-03, 2.06013121e-03,
                              4.09716955e-03, 2.97664854e-03, 1.91125018e-03, 2.64694398e-03,
                              2.73462262e-03, 2.71537110e-05, 3.93861040e-04, 6.55661204e-04,
                              7.85597368e-03, 1.92659288e-01, 1.92659288e-01, 1.92659288e-01,
                              1.92659288e-01, 1.92659288e-01]).cuda()

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, weight=class_weights, softmax=True)
            loss = 0.2 * loss_ce + 0.8 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
    
        model.eval()
        data_directory = '/home/ubuntu/files/project_TransUNet/data/Penguin/val_224'
        val_loss = 0
        with torch.no_grad():
            images_list = []
            labels_list = []
            kol = 0
            for file in os.listdir(data_directory):
                full_path = os.path.join(data_directory, file)
                data = np.load(full_path)
                image, label = data['image'], data['label']

                # Convert numpy arrays to tensors and add batch dimension if missing
                image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
                label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float().cuda()

                images_list.append(image_tensor)
                labels_list.append(label_tensor)
                
                # Check if we have gathered a full batch
                if len(images_list) == batch_size:
                    batch_images = torch.cat(images_list)
                    batch_labels = torch.cat(labels_list)
                    loss, outputs = process_batch(batch_images, batch_labels, model, ce_loss, dice_loss)
                    images_list = []
                    labels_list = []
                    kol += 1
                    val_loss += loss.item()
                    if kol % 100 == 0:
                        image = batch_images[1, 0:1, :, :]
                        image = (image - image.min()) / (image.max() - image.min())
                        writer.add_image('val/Image', image, iter_num+kol)
                        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                        writer.add_image('val/Prediction', outputs[1, ...] * 50, iter_num+kol)
                        labs = batch_labels[1, ...] * 50
                        writer.add_image('val/GroundTruth', labs, iter_num+kol)    

            # Process the last batch if it has fewer than batch_size images
            if images_list:
                batch_images = torch.cat(images_list)
                batch_labels = torch.cat(labels_list)
                loss, outputs = process_batch(batch_images, batch_labels, model, ce_loss, dice_loss)
                val_loss += loss.item()
                kol += 1
            # Compute average validation loss
            if kol > 0:
                average_val_loss = val_loss / kol
                writer.add_scalar('info/val_loss', average_val_loss, iter_num)
            else:
                print("No batches processed.")    
          
        save_interval = 3  # int(max_epoch/6)
        if epoch_num  % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break       

    writer.close()
    return "Training Finished!"