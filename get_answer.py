import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import *
import numpy as np
from PIL import Image

from model import API_Net

# ID of GPUs gonna be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


# Define data transformers for obtaining testing data
SIZE = 320
test_tfms = Compose([Resize(SIZE + SIZE // 32),
                     CenterCrop([SIZE, SIZE]),
                     ToTensor(),
                     Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


class TestData(Dataset):
    ''' A Dataset of Test Data '''

    def __init__(self, x):
        self.x = x
        return

    def __getitem__(self, item):
        if type(self.x[item]) is str:
            img = Image.open(self.x[item]).convert('RGB')
            self.x[item] = test_tfms(img)

        return self.x[item]

    def __len__(self):
        return len(self.x)


def load_testdata():
    ''' load test data '''
    TEST = []
    INDEX = []
    for filename in glob.glob("data/testing_data/*.jpg"):
        TEST.append(filename)
        INDEX.append(filename[-10:-4])

    return DataLoader(TestData(TEST), batch_size=100), INDEX


def load_model(File):
    ''' Load the model '''
    model = API_Net(device)
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)

    print('loading model {}'.format(File))
    checkpoint = torch.load(File)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def compute_logits(model, test_loader):
    ''' Compute the logits '''
    model.eval()
    with torch.no_grad():
        logitslist = []
        for i, input in enumerate(test_loader):
            print(i, end='\r')
            input_var = input.to(device)
            logits = model(input_var, targets=None, flag='val')
            logitslist.append(logits.cpu().detach().numpy())

    return np.concatenate(logitslist)


def load_logits(path, test_loader):
    ''' load the logits computed by a specific model. If we haven't
      computed it yet, call compute_logits to Obtain it
    '''
    npfilepath = path + "/logits.npy"
    if not os.path.exists(npfilepath):
        model = load_model(path + "/bestmodel.pth.tar")
        logit = compute_logits(model, test_loader)
        np.save(npfilepath, logit)
    return np.load(npfilepath)

if __name__ == '__main__':
    test_loader, INDEX = load_testdata()
    exp_name = 'APINet_KFold'
    KFold = 10
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    if len(sys.argv) > 2:
        KFold = sys.argv[2]

    # prepare data
    if not os.path.exists('data'):
        os.system('python3 prepare.py')

    # compute the logits by all the models you have train
    # until now in this experiment
    # To reproduce my submission, you may delete the code below
    logitsum = 0
    logits = [None] * KFold
    for i in range(KFold):
        path = 'results/' + exp_name + str(i)
        logits[i] = load_logits(path, test_loader)
        logitsum = logitsum + logits[i]

    '''
    # To reproduce my submission, you may use the code below
    logitsum = 0
    logits = [None] * 10
    for i in [0, 1, 3, 5, 7]:
        path = 'results/' + exp_name + str(i)
        logits[i] = load_logits(path, test_loader)
        logitsum = logitsum + logits[i]
    '''

    pred = logitsum.argmax(axis=1)

    # write down my prediction
    with open('data/label.txt', 'r') as LabelFile:
        labels = LabelFile.readlines()
    with open(exp_name + '_pred.csv', 'w') as WriteFile:
        WriteFile.write('id,label\n')
        for j in range(5000):
            WriteFile.write(INDEX[j] + ',' + labels[pred[j]])
