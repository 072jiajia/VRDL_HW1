import torch
import numpy as np


def rand_bbox(size, lam):
    ''' Obtain 4 values left, top, right, bottom,
    which are used to randomly mask out a part of image
    '''
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    left = np.clip(cx - cut_w // 2, 0, W)
    top = np.clip(cy - cut_h // 2, 0, H)
    right = np.clip(cx + cut_w // 2, 0, W)
    bottom = np.clip(cy + cut_h // 2, 0, H)

    return left, top, right, bottom


def RandomMaskOut(data):
    ''' Randomly Mask Out a part of image to do data augmentation
    - lam : a random value between 0. and 0.25, which indicates
        the ratio of the mask square to the image

    I use torch.rand rather than set them to 0
    because I found that it has better performance
    '''
    lam = np.random.uniform(0., 0.25)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    shape = (3, bbx2 - bbx1, bby2 - bby1)
    data[:, bbx1:bbx2, bby1:bby2] = torch.rand(shape)
    return data
