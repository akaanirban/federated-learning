#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 7/13/20 6:57 PM 2020

@author: Anirban Das
"""

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from sklearn import metrics
from compressor.randomk import RandomKCompressor


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, sparsity=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.num_samples = len(DatasetSplit(dataset, idxs))
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.compressor = RandomKCompressor(sparsity)

    def compress(self, net):
        if isinstance(net, nn.Module):  # a nn.Module has been passed inside, extract the named parameters
            net_inside = copy.deepcopy(net)
            dict_params = dict(net_inside.named_parameters())
        else:
            dict_params = copy.deepcopy(net)
        with torch.no_grad():
            for k in dict_params.keys():
                res = self.compressor.compress(dict_params[k].data)
                dict_params[k].set_(self.compressor.decompress(res[0], res[1]))
        print("this is called")
        if isinstance(net, nn.Module):  # a nn.Module has been passed inside, extract the named parameters
            return net_inside
        else:
            return dict_params

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.compress(net.state_dict()), sum(epoch_loss) / len(epoch_loss)

