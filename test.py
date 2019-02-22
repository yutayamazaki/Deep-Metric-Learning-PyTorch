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

from datasets import TripletSamplerCifar, load_dataset
from models import TripletResNet
from losses import TripletLoss, TripletAngularLoss
from params import args


if __name__ == '__main__':
    _, X_test, _, y_test = load_dataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test_dataset = TripletSamplerCifar(X_test, y_test)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = TripletResNet(args.output_dim, args.n_classes)
    model = model.to(device)
    model.load_state_dict(torch.load('./weights/' + args.experiment_name + '_0.14452.pth'))
    model.eval()

    pred_metric, pred_classes = model(X_test.cuda())

    pred_metric = pred_metric.detach().cpu().numpy()
    pred_classes = pred_classes.detach().cpu().numpy()
    pred_labels = pred_classes.argmax(1)
    for i in pred_labels:
        print(i)
    acc = accuracy_score(pred_labels, y_test)
    f1 = f1_score(pred_labels, y_test)
    conf_mat = confusion_matrix(pred_labels, y_test)
    print(f'ACCURACY: {acc}')
    print(f'F1_SCORE: {f1}')
    print(conf_mat)

    Y_reduced = TSNE(n_components=2, random_state=0).fit_transform(pred_metric)
    plt.scatter(Y_reduced[:, 0], Y_reduced[:, 1], c=y_test)
    plt.colorbar()
    plt.savefig(args.experiment_name + '_tSNE.png')