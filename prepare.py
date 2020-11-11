import os
import shutil
import glob
import csv

import numpy as np


def prepare():
    print('Preparing Image Data...')
    if not os.path.exists('data/training_data'):
        os.mkdir('data/training_data')

    with open('training_labels.csv') as csvfile:
        rows = csv.reader(csvfile)
        Data = []
        for idx, label in rows:
            Data.append((idx, label))
        Data = Data[1:]

    LabelDict = {}
    with open('data/label.txt', 'w') as LabelFile:
        for idx, label in Data:
            if label not in LabelDict:
                LabelDict[label] = len(LabelDict)
                LabelFile.write(label + '\n')

    Images = []
    Labels = []
    for idx, label in Data[1:]:
        Images.append(idx)

        label = LabelDict[label]
        Labels.append(label)

    randidx = np.random.permutation(len(Images))
    Images = np.array(Images)[randidx]
    Labels = np.array(Labels)[randidx]

    for i, idx in enumerate(Images):
        print(idx + '.jpg', end='\r')
        From = 'training_data/training_data/' + idx + '.jpg'
        To = 'data/training_data/' + str(i) + '.jpg'
        shutil.move(From, To)

    np.save('data/training_data/y_train.npy', Labels)

    print('Image Preparation Done')


if __name__ == '__main__':
    os.system('unzip cs-t0828-2020-hw1.zip')
    if not os.path.exists('data'):
        os.mkdir('data')
    os.system('mv testing_data/testing_data data/testing_data')

    prepare()
    os.system('rm training_labels.csv')
    os.system('rm -r testing_data')
    os.system('rm -r training_data')
    if not os.path.exists('results'):
        os.mkdir('results')
