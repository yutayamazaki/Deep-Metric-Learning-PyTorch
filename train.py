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

from datasets    import TripletSamplerCifar, load_dataset
from models import TripletResNet
from losses import TripletLoss, TripletAngularLoss
from params import args


def flow_data(model, data_loader, optimizer=None):
    if optimizer is None:
        model.eval()
        training = False
    else:
        model.train()
        optimizer.zero_grad()
        training = True

    epoch_loss = 0
    for i, (anchors, labels, positives, negatives) in enumerate(data_loader):
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        out_anc, _ = model(anchors)
        out_pos, _ = model(positives)
        out_neg, _ = model(negatives)

        loss = criterion(out_anc, out_pos, out_neg)

        if training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


if not os.path.exists(args.weight_dir):
    os.mkdir(args.weight_dir)


if __name__ == '__main__':
    X, X_test, y, y_test = load_dataset()

    X_train, X_valid, y_train, y_valid = train_test_split(X.numpy(), y.numpy(), shuffle=True, random_state=27)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = TripletSamplerCifar(X_train, y_train)
    valid_dataset = TripletSamplerCifar(X_valid, y_valid)
    test_dataset = TripletSamplerCifar(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = TripletResNet(args.output_dim, args.n_classes)
    model = model.to(device)
    # criterion = TripletAngularLoss()
    criterion = TripletLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_loss = 1e8
    lr = args.lr
    for epoch in range(1, args.n_epochs+1):
        train_loss = flow_data(model, train_loader, optimizer=optimizer)
        valid_loss = flow_data(model, valid_loader)

        print(f'EPOCH: [{epoch}/{args.n_epochs}], train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}')

        if valid_loss < best_loss:
            weights_name = args.weight_dir + args.experiment_name + f'_{valid_loss:.5f}' + '.pth'
            best_loss = valid_loss
            best_param = model.state_dict()
            torch.save(best_param, weights_name)
            print(f'save wieghts to {weights_name}')

        if epoch%20 == 0:
            print(f'Learning Rate: {lr} -> {lr/5}')
            lr = lr / 5
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)