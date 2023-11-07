import time
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import math

from torchvision import transforms
from PIL import Image, ImageDraw
import os
import torch
from utils import CustomDataset
from helper import *
from model import eca_resnet50


class args():
    gpu = None
    distributed = False
    seed = 42
    pretrained = False
    arch = 'eca_resnet50'
    ksize=None
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    evaluate = False
    action = ''
    epochs = 100
    start_epoch = 0
    print_freq = 100
    
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice_loss = AverageMeter()
    losses_batch = {}
    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        
        target = data['annotations']
        input_image = data['image']
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        #target = target.cuda(args.gpu, non_blocking=True)
        target = target.cpu()
        
        output = model(input_image)
        
        output_probs = torch.sigmoid(output)
        
        output = output.squeeze(1)  
        output_binary = (output_probs > 0.5).float()

        target = (target > 0).float()
        target = target.squeeze(1)  
        
        loss = criterion(output, target)

        dice = dice_coeff(output_binary, target).item()
        #print('---'*10,'dice', dice,'---'*10)
        
        losses.update(loss.item(), input_image.size(0))
        dice_loss.update(dice, input_image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f"Epoch {epoch}, Loss {loss}, avg. Dice {dice_loss.avg} curr. Dice {dice}")
            print('---'*50)

          
    return losses.avg, dice_loss.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.arch + '_' + args.action)
    
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def dice_coeff(pred, target):
    """
    Calculate the Dice coefficient for batch of predictions and targets.

    Args:
        pred: Predicted tensor of shape (N, C, H, W), where N is the batch size.
        target: Ground truth tensor of shape (N, C, H, W), with the same dimensions as pred.

    Returns:
        dice_score: Computed Dice coefficient.
    """
    smooth = 1.0  # Add smooth to avoid divide by zero error
    
    # Flatten the tensors to make the computation easier
    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(1)  # Sum over the spatial dimensions
    
    #print('pred_flat', pred_flat.sum(1))
    #print('target_flat', target_flat.sum(1))
    
    dice_score = (2. * intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
    
    # We take the mean over the batch
    dice_score = dice_score.mean()
    
    return dice_score


def data_save(root, file):
    if not os.path.exists(root):
        os.mknod(root)
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()