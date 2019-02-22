import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms, datasets

from params import args


def load_dataset():
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=transform)
    X_train, y_train = load_filter_dataset(trainset)
    X_test, y_test = load_filter_dataset(testset)

    return X_train, X_test, y_train, y_test


def load_filter_dataset(dataset):
    images, labels = [], []
    cat_counter = 0
    for train in dataset:
        X, y = train[0], train[1]
        if y == 5: # dog
            images.append(X)
            labels.append(y)
        elif y == 3: # cat
            if cat_counter == 500:
                continue
            images.append(X)
            labels.append(y)
            cat_counter += 1
    return torch.stack(images), torch.Tensor(labels)


class TripletSamplerCifar(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        x = self.images[idx]
        t = self.labels[idx]
        xp_idx = np.random.choice(np.where(self.labels == t)[0])
        xn_idx = np.random.choice(np.where(self.labels != t)[0])
        xp = self.images[xp_idx]
        xn = self.images[xn_idx]
        return x, t, xp, xn


class NPairSamplerCifar(Dataset):
    def __init__(self, images, labels, n):
        self.images = images
        self.labels = labels
        self.n = n

    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        x = self.images[idx]
        t = self.labels[idx]
        xp_idx = np.random.choice(np.where(self.labels == t)[0])
        xn_idx = np.random.choice(np.where(self.labels != t)[0], size=self.n)
        xp = self.images[xp_idx]
        xn = self.images[xn_idx]
        return x, t, xp, xn