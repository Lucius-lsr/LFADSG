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
from torch.utils.data import *


# synthetic data
# DATASET = 'syn'
# DATASET_PATH = './datasetrnn_synth_data_v1.0/thits_data_dataset_N50_S50_C10_nrep20'
# with_rates = True

# allen data
# DATASET = 'allen'
# DATASET_PATH = './dataset/allen_data/502666254.npy'

# neuropixel data
# DATASET = 'neu'
# DATASET_PATH = './dataset/neuropixel_data/spike_data_drifting_gratings_75_repeats_VISp.npy'

# svoboda data
DATASET = 'svo'
DATASET_PATH = './dataset/svoboda_data/svobodaSpikes_N100_T100_Size1000_255.npy'

LOAD_RATES = False
if DATASET=='syn' and with_rates:
    LOAD_RATES = True


class SpikeRateDataset(Dataset):
    def __init__(self, spikes, rates):
        super().__init__()
        self.spikes = spikes
        self.with_rates = True if rates is not None else False
        if self.with_rates:
            self.rates = rates
        else:
            self.rates = torch.zeros_like(spikes)

    def __getitem__(self, idx: int):
        spike = self.spikes[idx].float()
        rate = self.rates[idx].float()
        return spike, rate

    def __len__(self) -> int:
        return self.spikes.shape[0]


def get_dataset(dataset=DATASET, path=DATASET_PATH, load_rates=LOAD_RATES):
    if dataset=='syn':
        with h5py.File(path, 'r') as data:
            train_spikes = torch.from_numpy(np.array(data['train_data']))
            valid_spikes = torch.from_numpy(np.array(data['valid_data']))
            train_rates = torch.from_numpy(np.array(data['train_truth']))
            valid_rates = torch.from_numpy(np.array(data['valid_truth']))
            graph = torch.from_numpy(np.array(data['graph_truth']))

            train_dataset = SpikeRateDataset(train_spikes, train_rates)
            valid_dataset = SpikeRateDataset(valid_spikes, valid_rates)
        return train_dataset, valid_dataset, graph
    elif dataset=='svo':
        data_all = np.load(path)
        np.random.shuffle(data_all)
        num_data = data_all.shape[0]
        train_spikes = data_all[:int(0.9*num_data)]
        valid_spikes = data_all[int(0.9*num_data):]
        train_spikes = torch.from_numpy(train_spikes)
        valid_spikes = torch.from_numpy(valid_spikes)

        train_dataset = SpikeRateDataset(train_spikes, None)
        valid_dataset = SpikeRateDataset(valid_spikes, None)

        graph = np.load(path.replace('Spikes', 'Graph'))
        graph = torch.from_numpy(graph)

        return train_dataset, valid_dataset, graph
    else:
        data_all = np.load(path)
        np.random.shuffle(data_all)
        num_data = data_all.shape[0]
        train_spikes = data_all[:int(0.9*num_data)]
        valid_spikes = data_all[int(0.9*num_data):]
        train_spikes = torch.from_numpy(train_spikes)
        valid_spikes = torch.from_numpy(valid_spikes)

        train_dataset = SpikeRateDataset(train_spikes, None)
        valid_dataset = SpikeRateDataset(valid_spikes, None)

        return train_dataset, valid_dataset, None


def get_data_loader(train_batch_size=256, valid_batch_size=256, train_shuffle=True):
    train_dataset, valid_dataset, _ = get_dataset()
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=2)
    valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=2)
    return train_data_loader, valid_data_loader

def get_graph_truth():
    _, _, graph = get_dataset()
    return graph