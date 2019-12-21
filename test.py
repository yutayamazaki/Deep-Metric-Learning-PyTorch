import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from datasets import TripletSampler, load_datasets, load_test
from models import TripletResNet
from losses import TripletLoss, TripletAngularLoss
from params import args


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(args.img_size),
                                    transforms.CenterCrop(args.img_size),
                                    transforms.ToTensor()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = load_test(args.test_json, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = TripletResNet(args.output_dim)
    model = model.to(device)
    model.load_state_dict(torch.load('./weights/' + args.experiment_name + '_0.37773.pth'))
    model.eval()

    pred_metric = []
    y_test = []
    for i, (anchors, _, _, labels) in enumerate(test_loader):
        anchors = anchors.to(device)
        metric = model(anchors)
        pred_metric.append(metric.detach().cpu().numpy())
        y_test.append(labels.detach().numpy())

    pred_metric = np.concatenate(pred_metric, 0)
    y_test = np.concatenate(y_test, 0)

    y_reduced = TSNE(n_components=2, random_state=0).fit_transform(pred_metric)
    plt.scatter(y_reduced[:, 0], y_reduced[:, 1], c=y_test)
    plt.colorbar()
    plt.savefig(args.experiment_name + '_tSNE.png')
    plt.show()
