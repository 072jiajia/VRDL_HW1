import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.transforms import *
import numpy as np
from datasets import RandomDataset, BatchDataset, BalancedBatchSampler
import torch.nn.functional as F

# custom module
from model import API_Net
from utils import accuracy, AverageMeter, save_checkpoint
import utils

'''
Define data transformers for training and validation
Use train_tfms to do some data augmentation in traing phase
    and use val_tfms to obtain the data for validation
'''
SIZE = 320
train_tfms = Compose([Resize(SIZE + SIZE // 32),
                      RandomCrop([SIZE, SIZE]),
                      RandomApply([RandomRotation((-15, 15))]),
                      RandomHorizontalFlip(),
                      RandomGrayscale(p=0.1),
                      ToTensor(),
                      utils.RandomMaskOut,
                      Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

val_tfms = Compose([Resize(SIZE + SIZE // 32),
                    CenterCrop([SIZE, SIZE]),
                    ToTensor(),
                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


def train(args):
    # set seed
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)

    # create model
    model = API_Net(args.device)
    model = model.to(args.device)
    model.conv = nn.DataParallel(model.conv)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # resume the previous state of training if possible
    if args.resume:
        if os.path.isfile(args.resume):
            print('loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('loaded checkpoint {}(epoch {})'.format(
                args.resume, checkpoint['epoch']))
            print('best acc:', args.best_prec1)
        else:
            print('no checkpoint found at {}'.format(args.resume))

    cudnn.benchmark = True

    # Load Data
    train_dataset = BatchDataset(
        args.KFolder, args.nFolder, transform=train_tfms)
    train_sampler = BalancedBatchSampler(
        train_dataset, args.n_classes, args.n_samples)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=15)

    val_dataset = RandomDataset(args.KFolder, args.nFolder, transform=val_tfms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=15)

    # Initialize Schedular
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)
    for _ in range(args.start_epoch):
        scheduler.step()

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(train_loader, model, optimizer,
                        epoch, val_loader, args)
        scheduler.step()


def train_one_epoch(train_loader, model, optimizer, epoch, val_loader, args):
    # Define criterions
    criterion = nn.CrossEntropyLoss().to(args.device)
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    softmax_layer = nn.Softmax(dim=1).to(args.device)

    # Initialize result recorder for batches
    softmax_losses = AverageMeter()
    rank_losses = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for i, (data, target) in enumerate(train_loader):
        model.train()

        input_var = data.to(args.device)
        target_var = target.to(args.device).squeeze()

        # compute output
        logit1_self, logit1_other, logit2_self, logit2_other, labels1, \
            labels2 = model(input_var, target_var, flag='train')

        # Handle output
        batch_size = logit1_self.shape[0]
        labels1 = labels1.to(args.device)
        labels2 = labels2.to(args.device)

        self_logits = torch.zeros(2 * batch_size, 196).to(args.device)
        other_logits = torch.zeros(2 * batch_size, 196).to(args.device)

        self_logits[:batch_size] = logit1_self
        self_logits[batch_size:] = logit2_self
        other_logits[:batch_size] = logit1_other
        other_logits[batch_size:] = logit2_other

        # compute loss
        logits = torch.cat([self_logits, other_logits], dim=0)
        targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
        # cross entropy
        softmax_loss = criterion(logits, targets)

        # Obtain the softmax data and compute the MarginRankingLoss
        obj_idx = torch.arange(2 * batch_size).to(args.device).long()
        label_idx = torch.cat([labels1, labels2], dim=0)

        self_scores = softmax_layer(self_logits)[obj_idx, label_idx]
        other_scores = softmax_layer(other_logits)[obj_idx, label_idx]

        flag = torch.ones([2*batch_size, ]).to(args.device)
        rank_loss = rank_criterion(self_scores, other_scores, flag)

        # compute total loss
        loss = softmax_loss + rank_loss

        # measure accuracy and record loss
        prec1 = accuracy(logits, targets, 1)
        losses.update(loss.item(), 2 * batch_size)
        softmax_losses.update(softmax_loss.item(), 4 * batch_size)
        rank_losses.update(rank_loss.item(), 2 * batch_size)
        acc.update(prec1, 4 * batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: [{0}][{1}/{2}]  '
              'Loss {loss.val:.4f} ({loss.avg:.4f})  '
              'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})  '
              'RankLoss {rank_loss.val:.4f} ({rank_loss.avg:.4f})  '
              'Acc {acc.val:.3f} ({acc.avg:.3f})               '
              .format(epoch, i + 1, len(train_loader), loss=losses,
                      softmax_loss=softmax_losses,
                      rank_loss=rank_losses, acc=acc), end='\r')

    print(' ' * 100, end='\r')
    # save the result of the training phase of this epoch
    args.io.cprint('Epoch: {0}  Loss {loss.avg:.4f}  '
                   'SoftmaxLoss {softmax_loss.avg:.4f}  '
                   'RankLoss {rank_loss.avg:.4f}  '
                   'Acc {acc.avg:.3f}'
                   .format(epoch, loss=losses, softmax_loss=softmax_losses,
                           rank_loss=rank_losses, acc=acc))

    # Do validation
    prec1 = validate(val_loader, model, args)

    # remember best acc and save checkpoint
    is_best = prec1 > args.best_prec1
    args.best_prec1 = max(prec1, args.best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': args.best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, args)

    return


def validate(val_loader, model, args):
    ''' Do validation and record it '''
    acc = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            input_var = data.to(args.device)
            target_var = target.to(args.device).squeeze()

            logits = model(input_var, targets=None, flag='val')
            prec1 = accuracy(logits, target_var, 1)
            acc.update(prec1, logits.size(0))
        args.io.cprint(' * Acc {acc.avg:.3f}'.format(acc=acc))
    return acc.avg
