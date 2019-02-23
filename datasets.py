import json

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

from params import args


# 0: 猫, 1: 犬
def load_fname(json_path):
    with open(json_path) as f:
        d = f.read()
    data_dict = json.loads(d)

    image_fname_list = []
    for animal in data_dict:
        for seed in data_dict[animal]:
            image_fname_list += data_dict[animal][seed]['images']

    return image_fname_list


def load_images(image_fname_list, transform):
    ret_images = []
    ret_labels = []
    for fname in image_fname_list:
        path = './data/' + fname + '.jpg'
        img = Image.open(path).convert('RGB')
        img = transform(img)
        ret_images.append(img)

        label = 1 if fname[0].islower() else 0
        ret_labels.append(label)
    
    ret_images = torch.stack(ret_images)
    ret_labels = torch.Tensor(ret_labels)

    return ret_images, ret_labels


def load_datasets(train_json, test_json, transform):
    fnames = load_fname(train_json)
    test_fnames = load_fname(test_json)

    size = int(len(fnames) * 0.3)
    valid_fnames = np.random.choice(fnames, size)
    train_fnames = []
    for fname in fnames:
        if fname not in valid_fnames:
            train_fnames.append(fname)

    X_train, y_train = load_images(train_fnames, transform)
    X_valid, y_valid = load_images(valid_fnames, transform)
    X_test, y_test = load_images(test_fnames, transform)
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=27)

    train_dataset = TripletSampler(X_train, y_train)
    valid_dataset = TripletSampler(X_valid, y_valid)
    test_dataset = TripletSampler(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset



class TripletSampler(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)
 
    def __getitem__(self, idx):
        x = self.inputs[idx]
        t = self.targets[idx]
        xp_idx = np.random.choice(np.where(self.targets == t)[0])
        xn_idx = np.random.choice(np.where(self.targets != t)[0])
        xp = self.inputs[xp_idx]
        xn = self.inputs[xn_idx]
        return x, xp, xn