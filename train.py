import os
import sys
import argparse
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from unet import UNet
from utils import blue
from dataset import BasicDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--learning_rate', type=int, default=0.1)
parser.add_argument('--val_percent', type=int, default=0.2)
parser.add_argument('--save_cp', action='store_false')
parser.add_argument('--dir_patch', type=str, default='patch/')
parser.add_argument('--dir_checkpoint', type=str, default='checkpoints/')
parser.add_argument('--load_cp', type=str,
                    default='checkpoints/CP_epoch100.pth')  # checkpoints/CP_epoch51.pth

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#using tensorboard: tensorboard --logdir=runs

if __name__ == "__main__":
    opt = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=3, n_classes=2)
    if opt.load_cp != '':
        net.load_state_dict(torch.load(opt.load_cp, map_location=device))
    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    dataset = BasicDataset(opt.dir_patch)
    n_val = int(len(dataset) * opt.val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=opt.batch_size,
                              shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    val_loader = DataLoader(val, batch_size=opt.batch_size,
                            shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    writer = SummaryWriter(
        comment=f'_lr{opt.learning_rate}_bs{opt.batch_size}')
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(opt.nepoch):
        train_loss, train_count = 0, 0
        net.train()
        for batch in train_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            pred_masks = net(imgs)
            loss = criterion(pred_masks, true_masks.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_count += 1
        writer.add_scalar('Loss/train', train_loss/train_count, epoch+1)
        # writer.add_scalar('Acc/train', acc, train_step)
        logging.info(
            f'[Epoch {epoch + 1}/{opt.nepoch}] Train loss:{(train_loss/train_count):f}')

        val_loss, val_count = 0, 0
        net.eval()
        for batch in val_loader:
            imgs = batch['image']
            true_masks = batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            pred_masks = net(imgs)
            loss = criterion(pred_masks, true_masks.squeeze(1))
            val_loss += loss.item()
            val_count += 1
        writer.add_scalar('Loss/validate', val_loss/val_count, epoch+1)
        # writer.add_scalar('Acc/test', acc, validate_step)
        logging.info(
            blue(f'[Epoch {epoch + 1}/{opt.nepoch}] Valid loss:{(val_loss/val_count):f}'))
        scheduler.step(val_loss/val_count)
        writer.add_scalar(
            'learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
        # writer.add_images('images', imgs, epoch+1)

        if opt.save_cp:
            try:
                os.mkdir(opt.dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       opt.dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            # logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
