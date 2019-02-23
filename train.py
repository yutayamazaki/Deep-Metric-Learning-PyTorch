import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from datasets    import load_datasets
from models import TripletResNet
from losses import TripletLoss, TripletAngularLoss
from params import args


def calculate_loss(model, data_loader, optimizer=None):
    if optimizer is None:
        model.eval()
        training = False
    else:
        model.train()
        optimizer.zero_grad()
        training = True

    epoch_loss = 0
    for i, (anchors, positives, negatives) in enumerate(data_loader):
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        anc_metric = model(anchors)
        pos_metric = model(positives)
        neg_metric = model(negatives)

        loss = criterion(anc_metric, pos_metric, neg_metric)

        if training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


if not os.path.exists(args.weight_dir):
    os.mkdir(args.weight_dir)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize(args.img_size),
                                    transforms.CenterCrop(args.img_size),
                                    transforms.ToTensor()])

    train_dataset, valid_dataset, test_dataset = \
        load_datasets(args.train_json, args.test_json, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TripletResNet(args.output_dim)
    model = model.to(device)
    # criterion = TripletAngularLoss()
    criterion = TripletLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_loss = 1e8
    for epoch in range(1, args.n_epochs+1):
        train_loss = calculate_loss(model, train_loader, optimizer=optimizer)
        valid_loss = calculate_loss(model, valid_loader)

        print(f'EPOCH: [{epoch}/{args.n_epochs}], train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}')

        if valid_loss < best_loss:
            weights_name = args.weight_dir + args.experiment_name + f'_{valid_loss:.5f}' + '.pth'
            best_loss = valid_loss
            best_param = model.state_dict()
            torch.save(best_param, weights_name)
            print(f'save wieghts to {weights_name}')