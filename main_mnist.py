#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 7/13/20 6:57 PM 2020

@author: Anirban Das
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from node.client_worker import ClientWorker
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvgOld
from models.test import test_img
import importlib
#from compressor.randomk import RandomKCompressor
import random
from tqdm import tqdm
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


compressor_modules = {
    "TopKCompressor" : "compressor.topk",
    "RandomKCompressor" : "compressor.randomk"
}


def create_client_data_indices(arguments, x_train):
    # Create and sample users
    if arguments.iid:
        user_dict = mnist_iid(x_train, arguments.num_users)
    else:
        user_dict = mnist_noniid(x_train, arguments.num_users)
    return user_dict


def initiate_network(arguments, dataset, user_dict, model, compressor):
    internet = {}
    for user_idx in user_dict.keys():
        internet[user_idx] = ClientWorker(client_id=user_idx,
                                          args=arguments,
                                          dataset=dataset,
                                          idxs=user_dict[user_idx],
                                          compressor_object=compressor,
                                          net=copy.deepcopy(model)
                                          )
    return internet


def select_clients(round_idx, total_network, num_clients=20):
    """Selects num_clients clients randomly from possible_clients.
    """
    # np.random.choice(range(args.num_users), m, replace=False)
    num_clients = min(num_clients, len(total_network))
    user_ids = list(total_network.keys())
    np.random.seed(round_idx)
    selected_clients = np.random.choice(user_ids, num_clients, replace=False)

    return [(client_id, total_network[client_id].num_samples) for client_id in selected_clients]


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

    # Create CLients
    dict_users = create_client_data_indices(args, dataset_train)
    img_size = dataset_train[0][0].shape

    # Create the server model structure
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # Initialize other things
    comp = getattr(importlib.import_module(compressor_modules[args.compressor_class]), args.compressor_class)
    compressor = comp(args.sparsity)
    print(compressor)

    # Initialize network
    network = initiate_network(arguments=args,
                               dataset=dataset_train,
                               user_dict=dict_users,
                               model=copy.deepcopy(net_glob),
                               compressor=compressor)

    # training
    loss_train_distributed = []
    acc_train_global = []
    acc_test_global = []
    loss_test_global = []
    loss_train_global = []

    for global_iteration in range(args.epochs):
        w_locals, loss_locals = [], []
        num_selected = max(int(args.frac * args.num_users), 1)
        selected_users = select_clients(global_iteration, network, num_selected)

        for idx, num_samples in tqdm(selected_users):
            client = network[idx]
            client.set_model(copy.deepcopy(w_glob))
            w, loss = client.train()
            w_locals.append((copy.deepcopy(w), num_samples))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # # Set weights on All clients
        # for idx in network.keys():
        #     client = network[idx]
        #     client.set_model(copy.deepcopy(w_glob))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(global_iteration, loss_avg))
        loss_train_distributed.append(loss_avg)

        # testing
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        loss_test_global.append(loss_test)
        loss_train_global.append(loss_train)
        acc_train_global.append(acc_train)
        acc_test_global.append(acc_test)
        print(acc_train_global, acc_train_global, loss_train_global)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train_distributed)), loss_train_distributed)
    # plt.ylabel('train_loss')
    # plt.savefig(
    #     f"./save/fed_{args.dataset}_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}_Sp{args.sparsity}.png")
    with open(f"./save/fed_loss_dist_train{args.dataset}_{args.model}_LE{args.local_ep}_GE{args.epochs}_C{args.frac}_iid{args.iid}_Sp{args.sparsity}_Comp{args.compressor_class}.pkl",
              "wb") as f:
        pickle.dump(loss_train_distributed, f)
    with open(f"./save/fed_loss_glob_train{args.dataset}_{args.model}_LE{args.local_ep}_GE{args.epochs}_C{args.frac}_iid{args.iid}_Sp{args.sparsity}_Comp{args.compressor_class}.pkl",
              "wb") as f:
        pickle.dump(loss_train_global, f)
    with open(f"./save/fed_loss_glob_test{args.dataset}_{args.model}_LE{args.local_ep}_GE{args.epochs}_C{args.frac}_iid{args.iid}_Sp{args.sparsity}_Comp{args.compressor_class}.pkl",
              "wb") as f:
        pickle.dump(loss_test_global, f)
    with open(f"./save/fed_acc_glob_train{args.dataset}_{args.model}_LE{args.local_ep}_GE{args.epochs}_C{args.frac}_iid{args.iid}_Sp{args.sparsity}_Comp{args.compressor_class}.pkl",
              "wb") as f:
        pickle.dump(acc_train_global, f)
    with open(f"./save/fed_acc_glob_test{args.dataset}_{args.model}_LE{args.local_ep}_GE{args.epochs}_C{args.frac}_iid{args.iid}_Sp{args.sparsity}_Comp{args.compressor_class}.pkl",
              "wb") as f:
        pickle.dump(acc_test_global, f)

