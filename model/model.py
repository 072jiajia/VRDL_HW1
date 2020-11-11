import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


def pdist(vectors):
    ''' Compute squares of Pair-Wise Distances

    Input:
        - vectors : (batch_size, num_feature)

    Output:
        - distance_matrix : (batch_size, batch_size)

    As we can compute the distance in euclidean space by
        (vx - vy) ** 2 = vx * vx + vy * vy - 2 * vx * vy

    Compute  vx * vx  and  vy * vy  by element-wise multiplication
            vectors.pow(2).sum(dim=1)
    And compute  2 * vx * vy  by matrix multiplication
            2 * vectors.mm(torch.t(vectors))
    '''
    v2 = vectors.pow(2).sum(dim=1)
    xy2 = 2 * vectors.mm(torch.t(vectors))
    distance_matrix = v2.view(1, -1) - xy2 + v2.view(-1, 1)
    return distance_matrix


class API_Net(nn.Module):
    ''' Module of API_Net

    *** Using resnet101 as its backbone ***
    You may change the backbone to any module you'd like to use
    ex:
        backbone = models.densenet201(pretrained=True)
        num_feature = backbone.classifier.in_features
        layers = list(backbone.children())[:-1]

    *** Parameters ***
    - device     : the GPU device you're going to use
    - conv       : CNN backbone of this module
    - map1, map2 : mlp to compute contrastive clues
    - fc         : fully-connected layer (last layer)
    '''

    def __init__(self, device):
        super(API_Net, self).__init__()
        self.device = device

        backbone = models.resnet101(pretrained=True)
        num_feature = backbone.fc.in_features
        layers = list(backbone.children())[:-2]

        self.conv = nn.Sequential(*layers)
        self.map1 = nn.Linear(num_feature * 2, 512)
        self.map2 = nn.Linear(512, num_feature)
        self.fc = nn.Linear(num_feature, 196)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, targets=None, flag='train'):
        ''' Forward Propageting Function
        - images : Input image
        - flag   : to train or to validate
        - target : if it's training now, alse feed in the labels of images
                to do discriminative feature learning
        '''
        conv_out = self.conv(images)
        pool_out = F.adaptive_avg_pool2d(conv_out, (1, 1)).squeeze()

        if flag == 'val':
            return self.fc(pool_out)
        elif flag == 'train':
            intra_pairs, inter_pairs, intra_labels, \
                inter_labels = self.get_pairs(pool_out, targets)

            # labels of objects in pairs
            # return it to compute MarginRankingLoss
            labels1 = torch.cat([targets, targets], dim=0)
            labels2 = torch.cat([intra_labels, inter_labels], dim=0)

            # Obtain the mutual vectors and generate the gate vectors
            features1 = torch.cat([pool_out, pool_out], dim=0)
            features2 = torch.cat([pool_out[intra_pairs],
                                   pool_out[inter_pairs]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)

            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)
            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            # Obtain attentive features via residual attention
            # * self  : feature vector activated by its own gate
            # * other : feature vector activated by the gate of paired image
            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1
            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            # Use the four vetors to Obtain the logits
            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))

            return (logit1_self, logit1_other, logit2_self, logit2_other,
                    labels1, labels2)

    def get_pairs(self, embeddings, labels):
        ''' Compute the Intra Pairs and the Inter Pairs

        * intra : The most similar object which is in the same catagory
        * inter : The most similar object which is not in the same catagory

        Use euclidean distance in the feature space
            to determine the similarity between objects

        '''
        distance_matrix = pdist(embeddings).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().reshape(-1, 1)

        # lb_eqs is a batch_size * batch_size boolean matrix about
        # whether object_n and object_m is in the same category
        lb_eqs = (labels == labels.T)

        # If we find the most similar object directly, We might
        # just find out that itself it's the most similar object
        # So use np.diag_indices to easily handle this problem
        dia_inds = np.diag_indices(labels.shape[0])

        # Compute the intra / inter pairs
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs is False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs is True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        # Generate the return data
        intra_pairs = intra_idxs
        inter_pairs = inter_idxs
        intra_labels = labels[intra_idxs]
        inter_labels = labels[inter_idxs]

        intra_labels = torch.from_numpy(intra_labels)
        intra_labels = intra_labels.long().to(self.device).view(-1)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(self.device)
        inter_labels = torch.from_numpy(inter_labels)
        inter_labels = inter_labels.long().to(self.device).view(-1)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(self.device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels
