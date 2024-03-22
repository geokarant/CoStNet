#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:19:02 2022

@author: karantai
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils
# defining the Dataset class


class data_set(Dataset):
    def __init__(self):
        numbers = list(range(0, 100, 1))
        self.data = numbers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


dataset = data_set()

data = Dataset
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

transforms = utils.build_transforms(second_stage=(stage == 'second'))
train_supcon_dataset = create_supcon_dataset('cifar10',
                                             data_dir='data/cifar10',
                                             train=True,
                                             transform=TwoCropTransform(
                                                 transforms['train_transforms']),
                                             second_stage=False)
