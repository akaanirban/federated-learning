#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 7/13/20 7:32 PM 2020

@author: Anirban Das
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:51:10 2020

@author: Anirban Das
"""

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from typing import List
from torch.utils.data import DataLoader, TensorDataset, Dataset

np.random.seed(42)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientWorker(object):
    """
    Class for the clients which will do the training
    """

    def __init__(self, client_id, args, dataset, idxs, compressor_object, net):
        """
        Initiates a client object with its data
        :param compressor_object:
        :param criterion:
        :param optimizer:
        :param client_id:
        :param args:
        :param dataset:
        :param idxs:
        :param sparsity:
        """
        self.client_id = client_id
        self.args = args
        self.selected_clients = []
        self.net = copy.deepcopy(net)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.num_samples = len(DatasetSplit(dataset, idxs))
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.compressor = compressor_object

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
        if isinstance(net, nn.Module):  # a nn.Module has been passed inside, extract the named parameters
            return net_inside
        else:
            return dict_params

    def train(self):
        self.net.to(self.args.device)
        # train and update
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # zero out the grad of the optimizer
                self.optimizer.zero_grad()

                # Use CPU or GPU
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                #  Run the network on each data instance in the minibatch
                #  and then compute the object function value
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)

                #  Back-propagate the gradient through the network using the
                #  implicitly defined backward function
                loss.backward()
                #  Complete the mini-batch by actually updating the parameters.
                self.optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            # Epoch loss calculate
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.compress(self.net.state_dict()), sum(epoch_loss) / len(epoch_loss)

    def set_model(self, weights, optimizer=None):
        """
        Replace the local model with the global model and reset the optimizer with the parameters of the new model

        Args:
            global_model: The global model as nn.Module object

        Returns: None

        """
        self.net.load_state_dict(weights)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # self.net.train()