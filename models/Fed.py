#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 7/13/20 6:57 PM 2020

@author: Anirban Das
"""

import copy
import torch
from typing import List, Dict
from torch import nn


def FedAvg(w: List):
    first_sample = w[0]
    w_avg = copy.deepcopy(first_sample[0])
    total_samples = sum([i[1] for i in w])
    with torch.no_grad():
        for key in w_avg:
            w_avg[key] = w_avg[key] * first_sample[1]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][0][k] * w[i][1]
        w_avg[k] = torch.div(w_avg[k], total_samples)
    return w_avg


def FedAvgOld(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg