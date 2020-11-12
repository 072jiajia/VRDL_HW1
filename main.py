import os
import torch
import argparse

# custom module
from utils import IOStream
from train import train

# ID of GPUs gonna be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
device = torch.device("cuda")


def _init_(args):
    # initialize parameters
    path = 'results/' + args.exp_name + str(args.nFold)
    if not os.path.exists(path):
        os.mkdir(path)
    args.resume = path + '/checkpoint.pth.tar'
    args.best = path + '/bestmodel.pth.tar'
    args.io = IOStream(path + '/run.log')
    args.start_epoch = 0
    args.best_prec1 = 0
    args.device = device


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VRDL HW1')
    parser.add_argument('--exp_name', default='APINet_KFold', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--n_classes', default=30, type=int,
                        help='number of classes inone training batch')
    parser.add_argument('--n_samples', default=4, type=int,
                        help='number of samples in each class')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='validation batch size')
    parser.add_argument('--KFold', default=10, type=int,
                        help='train K models for K-Fold')
    args = parser.parse_args()

    # prepare data
    if not os.path.exists('data'):
        os.system('python3 prepare.py')

    # start training
    for nFold in range(args.KFold):
        args.nFold = nFold
        _init_(args)
        train(args)

    # generate prediction.csv
    os.system('python3 get_answer.py ' + args.exp_name)
