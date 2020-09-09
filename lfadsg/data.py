# -*- coding: utf-8 -*-
"""
@Time    : 9/6/20 11:34 AM
@Author  : Lucius
@FileName: data.py
@Software: PyCharm
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

DATASET_PATH = 'dataset/chaotic_rnn_inputs_g2p5_dataset_N50_S50'


def get_dataset(path=DATASET_PATH):
    with h5py.File(path, 'r') as data:
        train_data = torch.from_numpy(np.array(data['train_data']))
        valid_data = torch.from_numpy(np.array(data['valid_data']))
        train_dataset = TensorDataset(train_data)
        valid_dataset = TensorDataset(valid_data)
    return train_dataset, valid_dataset


def get_train_data_loader(batch_size=256, shuffle=True):
    train_dataset, valid_dataset = get_dataset()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return train_data_loader


def get_valid_data_loader(batch_size=256):
    train_dataset, valid_dataset = get_dataset()
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return valid_data_loader

