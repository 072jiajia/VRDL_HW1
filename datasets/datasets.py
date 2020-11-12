import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import Image
import numpy as np


def get_data(KFold, nFold):
    '''get the training/testing data of the nth Fold
    '''
    y = np.load('data/training_data/y_train.npy')
    l = len(y)
    x = ['data/training_data/' + str(i) + '.jpg' for i in range(l)]

    st = l * nFold // KFold
    ed = l * (nFold + 1) // KFold

    x_test = x[st: ed]
    y_test = y[st: ed]

    x_train1 = x[: st]
    y_train1 = y[: st]
    x_train2 = x[ed:]
    y_train2 = y[ed:]

    x_train = x_train1 + x_train2
    y_train = np.concatenate([y_train1, y_train2])

    return x_train, y_train, x_test, y_test


class RandomDataset(Dataset):
    ''' Dataset for Testing and Validation '''

    def __init__(self, KFold, nFold, transform=None):
        self.transform = transform
        _, _, x_test, y_test = get_data(KFold, nFold)
        self.labels = y_test
        self.imglist = x_test

    def __getitem__(self, index):
        if type(self.imglist[index]) is str:
            path = self.imglist[index].strip()
            self.imglist[index] = Image.open(path).convert('RGB')

        img = self.transform(self.imglist[index])
        label = torch.LongTensor([self.labels[index]])

        return [img, label]

    def __len__(self):
        return len(self.imglist)


class BatchDataset(Dataset):
    ''' Dataset for Training '''

    def __init__(self, KFold, nFold, transform=None):
        self.transform = transform
        x_train, y_train, _, _ = get_data(KFold, nFold)
        self.labels = y_train
        self.imglist = x_train
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        if type(self.imglist[index]) is str:
            path = self.imglist[index].strip()
            self.imglist[index] = Image.open(path).convert('RGB')

        img = self.transform(self.imglist[index])
        label = torch.LongTensor([self.labels[index]])

        return [img, label]

    def __len__(self):
        return len(self.imglist)


class BalancedBatchSampler(BatchSampler):
    ''' Batch Sampler for Training

    - n_classes : n_classes categories of objects in a batch
    - n_samples : n_samples of object for each class in a batch
    # batch_size is n_classes * n_samples

    '''

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        np_labels = self.labels.numpy()
        self.labels_set = list(set(np_labels))
        self.label_to_indices = {label: np.where(np_labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        ''' Randomly Generate Batches of Training Data '''
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                st = self.used_label_indices_count[class_]
                ed = st + self.n_samples
                indices.extend(self.label_to_indices[class_][st: ed])
                self.used_label_indices_count[class_] = ed
                if ed + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
