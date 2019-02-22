import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, t, positive, negative):
        dist = torch.sum(
            torch.pow((anchor - positive), 2) - torch.pow((anchor - negative), 2),
            dim=1) + self.margin
        dist = F.relu(dist)
        loss = torch.mean(dist)
        y_0 = anchor[t==0]
        if len(y_0) > 0:
            loss += torch.mean(y_0**2)
        return loss


class L2SoftmaxLoss(nn.Module):
    def __init__(self, fc_layers):
        super(L2SoftmaxLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.fc_layers = fc_layers

    def forward(self, labels, anc_metric, pos_metric, neg_metric):
        anc_classes = self.fc_layers(F.normalize(anc_metric))
        pos_classes = self.fc_layers(F.normalize(pos_metric))
        neg_classes = self.fc_layers(F.normalize(neg_metric))
        
        ce_loss = nn.CrossEntropyLoss()

        loss = None
        for classes in [anc_classes, pos_classes, neg_classes]:
            if loss is None:
                loss = ce_loss(classes, labels)
            else:
                loss += ce_loss(classes, labels)

        return loss


class TripletAngularLoss(nn.Module):
    # Angular Loss for Triplet Sampling
    def __init__(self, alpha=45, in_degree=True, size_average=False):
        # y=dnn(x), must be L2 Normalized.
        super(TripletAngularLoss, self).__init__()
        if in_degree:
            alpha = np.deg2rad(alpha)
        self.tan_alpha = np.tan(alpha) ** 2
        self.size_average = size_average

    def forward(self, a, p, n):
        c = (a + p) / 2
        loss = F.relu(F.normalize(a - p).pow(2) - 4 * self.tan_alpha * F.normalize(n - c).pow(2))
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(inputs)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()