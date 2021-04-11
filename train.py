# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tensorboardX import SummaryWriter
from data import get_data_loader, get_graph_truth
from model import LFADSG
from config import parse_args
from utils import Poisson


def train_model(model, model_name, train_loader, valid_loader, optimizer, scheduler, num_epochs):
    def train_epoch(model, train_loader, optimizer, scheduler):
        model.train()
        total_loss = 0.0
        for spike, _ in train_loader:
            spike = spike.to(device)
            optimizer.zero_grad()
            loss = model(spike)
            loss.backward()
            # for name, parms in model.named_parameters():
            #     if parms.grad is not None:
            #         print('-->name:', name, ' -->grad_value:', torch.mean(torch.abs(parms.grad)).item())
            #     else:
            #         print('-->name:', name, ' -->grad_value: None')
            # torch.set_printoptions(8)
            # for name, parms in model.named_parameters():
            #     if name=='prior_graph_0.random_matrix':
            #         print(parms.grad[0])
            # print(model.prior_graph_0.A_matrix[0])
            # print(model.prior_graph_0()[0][0])
            # print()
            optimizer.step()
            total_loss += loss.item() * spike.size(0)
        scheduler.step()

        epoch_loss = total_loss / len(train_loader.dataset)
        return epoch_loss

    def valid_epoch(model, valid_loader):
        model.eval()
        total_loss = 0.0
        for spike, _ in valid_loader:
            spike = spike.to(device)
            loss = model(spike)
            total_loss += loss.item() * spike.size(0)
        epoch_loss = total_loss / len(valid_loader.dataset)
        return epoch_loss

    def rates_epoch(model, train_loader, valid_loader):
        model.eval()

        train_loss = 0.0
        criterion = nn.MSELoss()
        for spike, rate in train_loader:
            spike, rate = spike.to(device), rate.to(device)
            rate_pred = model(spike, return_rates = True)
            loss = criterion(rate.float(), rate_pred.float())
            train_loss += loss.item() * spike.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        valid_loss = 0.0
        criterion = nn.MSELoss()
        for spike, rate in valid_loader:
            spike, rate = spike.to(device), rate.to(device)
            rate_pred = model(spike, return_rates = True)
            loss = criterion(rate.float(), rate_pred.float())
            valid_loss += loss.item() * spike.size(0)
        valid_loss = valid_loss / len(valid_loader.dataset)

        return train_loss, valid_loss

    def graph_epoch(model, train_loader, valid_loader):
        model.eval()
        graph = get_graph_truth().to(device)
        graph_pred = model.get_prior_graph()
        # print(graph_pred)

        criterion = nn.MSELoss()
        loss = criterion(graph.float(), torch.squeeze(graph_pred.float()))
        return loss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter('log/tensorboard_{}'.format(arg.model_name))
    model_save_path = 'log/models_{}'.format(arg.model_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    best_valid_loss = np.inf

    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)

        # train loss
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print("training loss: {:.4f}".format(train_loss))
        writer.add_scalar('training loss', train_loss, epoch)

        # valid loss
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
            print("validation loss: {:.4f}".format(valid_loss))
            writer.add_scalar('validation loss', valid_loss, epoch)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'log/models_{}/best_model.pt'.format(arg.model_name))

            # # train and valid rate MSE loss
            # mse_train_loss, mse_valid_loss = rates_epoch(model, train_loader, valid_loader)
            # print("rates MSE train loss: {:.4f}".format(mse_train_loss))
            # print("rates MSE valid loss: {:.4f}".format(mse_valid_loss))
            # writer.add_scalar('rates MSE train loss', mse_train_loss, epoch)
            # writer.add_scalar('rates MSE valid loss', mse_valid_loss, epoch)

            # train and valid graph MSE loss
            mse_graph_loss = graph_epoch(model, train_loader, valid_loader)
            print("graph MSE loss: {:.4f}".format(mse_graph_loss))
            writer.add_scalar('graph MSE loss', mse_graph_loss, epoch)


def train_manager(arg):
    print(arg)
    torch.cuda.set_device(arg.cuda)
    num_epochs = arg.num_epochs
    lr = arg.lr

    model = LFADSG(arg)

    if arg.model_path is not None:
        print('load model from', arg.model_path)
        model.load_state_dict(torch.load(arg.model_path))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loader, valid_loader = get_data_loader(train_batch_size=arg.batch_size, valid_batch_size=arg.batch_size)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=arg.momentum)
    # lr scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=arg.gamma)

    train_model(model, arg.model_name, train_loader, valid_loader, optimizer, scheduler, num_epochs)


if __name__ == "__main__":
    arg = parse_args()
    train_manager(arg)