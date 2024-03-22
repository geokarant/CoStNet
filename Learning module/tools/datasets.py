import torchvision
import torch
import cv2
import numpy as np
import glob
import os
from PIL import Image
from torch.utils.data import Dataset  # NEW
from torchvision import transforms, utils  # NRE
from skimage import io, transform  # NEW


class SupConDatasetCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, data_dir, train, transform, second_stage):
        super().__init__(root=data_dir, train=train, download=True, transform=transform)

        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)
        return image, label


class SupConDatasetCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, data_dir, train, transform, second_stage):
        super().__init__(root=data_dir, train=train, download=True, transform=transform)

        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]
        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)
        print(image, label)
        return image, label


class SupconDatasetMine(Dataset):
    def __init__(self, data_dir, train, transf, second_stage):
        # def __init__(self, data_dir):
        self.train_data = os.path.join(data_dir, 'train')
        self.test_data = os.path.join(data_dir, 'test')
        self.second_stage = second_stage
        self.transform = transf
        self.train = train

        if self.train:
            labels = []
            data_path_cgg = glob.glob(
                os.path.join(self.train_data, 'CGG', '*'))
            data_path_real = glob.glob(
                os.path.join(self.train_data, 'Real', '*'))
            data_all = data_path_cgg + data_path_real
        else:
            labels = []
            data_path_cgg = glob.glob(
                os.path.join(self.test_data, 'CGG', '*'))
            data_path_real = glob.glob(
                os.path.join(self.test_data, 'Real', '*'))
            data_all = data_path_cgg + data_path_real

        self.data_all_len = len(data_all)
        for cl in data_all:
            category = cl.split('/')[-2]
            if category == 'Real':
                labels.append(1)
            else:
                labels.append('0')
        # data_all = np.asarray(data_all)  # new
        labels = np.asarray(labels)  # new
        self.path_arr = data_all
        self.labels = [int(i) for i in labels]

    def __len__(self):     # new
        return self.data_all_len  # new

    def __getitem__(self, idx):
       # image, label = self.data[idx], self.labels[idx]

        source_img_name = self.path_arr[idx]  # NEW
        label = self.labels[idx]
        img = cv2.imread(source_img_name)
        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            transformed = self.transform(image=img)
            image = transformed['image']
        else:
            image = self.transform(img)

        return image, label


DATASETS = {'cifar10': SupConDatasetCifar10,
            'photos': SupconDatasetMine,
            'cifar100': SupConDatasetCifar100}


# , csv, second_stage):
def create_supcon_dataset(dataset_name, data_dir, train, transform, second_stage):
    try:
        # , csv, second_stage)
        # obj = DATASETS[dataset_name](data_dir, train, transform, second_stage)
        # obj.grab_data()
        # obj.preprocessing()
        return DATASETS[dataset_name](data_dir, train, transform, second_stage)
    except KeyError:
        Exception(
            'Can\'t find such a dataset. Either use cifar10 or cifar100, or write your own one in tools.datasets')
