# -*- coding: utf-8 -*-
"""
@Time    : 9/7/20 10:36 AM
@Author  : Lucius
@FileName: train.py
@Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data import get_train_data_loader, get_valid_data_loader
from model import LFADSG


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
    def train_epoch(model, train_loader, optimizer, scheduler, criterion):
        model.train(True)
        total_loss = 0.0
        for spikes in train_loader:
            spikes = spikes[0]
            spikes = spikes.to(device)
            optimizer.zero_grad()
            rates = model(spikes)
            loss = criterion(rates, spikes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * spikes.size(0)
        scheduler.step()

        epoch_loss = total_loss / len(train_loader.dataset)
        return epoch_loss

    def valid_epoch(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        for spikes in valid_loader:
            spikes = spikes[0]
            spikes = spikes.to(device)
            rates = model(spikes)
            loss = criterion(rates, spikes)
            total_loss += loss.item() * spikes.size(0)
        epoch_loss = total_loss / len(valid_loader.dataset)
        return epoch_loss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        print("training loss: {:.4f}".format(train_loss))

        valid_loss = valid_epoch(model, valid_loader, criterion)
        print("validation loss: {:.4f}".format(valid_loss))


def train_manager():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_epochs = 300
    lr = 0.000001

    model = LFADSG(time=100, neurons=50, dim_e=16, dim_c=24, dim_d=32, gcn_hidden=[8])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.PoissonNLLLoss(log_input=True)

    train_loader = get_train_data_loader(batch_size=128)
    valid_loader = get_valid_data_loader(batch_size=128)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # lr scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)

    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)


if __name__ == "__main__":
    train_manager()
