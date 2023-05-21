# -*- coding: utf-8 -*-

import torch

from ..feature_enhancer_base import EnhanceBase
from ...registry import ENHANCERS
from ...metric import KNN

from typing import Dict
import math
from tqdm import tqdm
import copy

@ENHANCERS.register
class DBA(EnhanceBase):
    """
    Every feature in the database is replaced with a weighted sum of the point ’s own value and those of its top k nearest neighbors (k-NN).
    c.f. https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

    Hyper-Params:
        enhance_k (int): number of the nearest points to be calculated.
    """
    default_hyper_params = {
        "enhance_k": 10,
    }

    def __init__(self, hps: Dict or None = None):
        """
        Args:
            hps (dict): default hyper parameters in a dict (keys, values).
        """
        super(DBA, self).__init__(hps)
        knn_hps = {
            "top_k": self._hyper_params["enhance_k"] + 1,
        }
        self.knn = KNN(knn_hps)

    def __call__(self, feature: torch.tensor) -> torch.tensor:
        _, sorted_idx = self.knn(feature, feature)
        sorted_idx = sorted_idx[:, 1:].reshape(-1)

        arg_fea = feature[sorted_idx].view(feature.shape[0], -1, feature.shape[1]).sum(dim=1)
        feature = feature + arg_fea

        # ebay
        # feature_copy = copy.deepcopy(feature)
        # print("Performing DBA ...")
        # print(sorted_idx.size())
        # sorted_idx = sorted_idx[:, 1:]
        # for i in tqdm(range(math.ceil(sorted_idx.size(0)/400))):
        #     MIN_INDEX, MAX_INDEX = i*400, min((i+1)*400,sorted_idx.size(0))
        #     temp_idx = sorted_idx[MIN_INDEX:MAX_INDEX].reshape(-1).long()

        #     arg_fea = feature_copy[temp_idx].view(MAX_INDEX-MIN_INDEX, -1, feature.shape[1]).sum(dim=1)
        #     feature[MIN_INDEX:MAX_INDEX]+=arg_fea

        feature = feature / torch.norm(feature, dim=1, keepdim=True)   # 干什么？

        del sorted_idx

        return feature
