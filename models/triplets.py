import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from params import args


class TripletResNet(nn.Module):
    def __init__(self, metric_dim):
        super(TripletResNet, self).__init__()
        resnet = torchvision.models.__dict__['resnet18'](pretrained=True)
        for params in resnet.parameters():
            params.requires_grad = False

        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, metric_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        metric = self.fc(x)
        return metric