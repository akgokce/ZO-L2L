from pickletools import optimize
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, Subset

import os
import numpy as np
import random

import optimizee

class CIFAR10Model(optimizee.mnist.MnistModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CIFAR10Attack(optimizee.mnist.MnistAttack):
    def dataset_loader(self, data_dir, batch_size, test_batch_size, train_num=100, test_num=100, index_path="data/cifar_correct/label_correct_index.npy"):
        label_correct_indices = list(np.load(index_path))
        random.seed(1234)
        random.shuffle(label_correct_indices)
        train_indices = label_correct_indices[:train_num]
        test_indices = label_correct_indices[5000:5000 + test_num]

        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10("data", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(28), transforms.Grayscale(),
                transforms.Normalize(self.mean, self.std)
            ])), batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(train_indices), drop_last=True, num_workers=32)

        test_loader = torch.utils.data.DataLoader(Subset(datasets.CIFAR10("data", train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(28), transforms.Grayscale(),
                transforms.Normalize(self.mean, self.std)
            ])), test_indices), batch_size=test_batch_size, shuffle=False, num_workers=32)

        return train_loader, test_loader