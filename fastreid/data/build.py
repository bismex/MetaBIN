# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import torch
import sys
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader
from fastreid.utils import comm

from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_reid_train_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)

    train_set_all = []
    train_items = list()
    cnt = 0
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        if cfg.META.DATA.NAMES == 'DG':
            if len(dataset.train[0]) < 4: # add domain label
                for i, x in enumerate(dataset.train):
                    add_info = {}  # dictionary
                    add_info['domains'] = int(cnt)
                    dataset.train[i] = list(dataset.train[i])
                    dataset.train[i].append(add_info)
                    dataset.train[i] = tuple(dataset.train[i])
            cnt += 1
            train_set_all.append(dataset.train)
        train_items.extend(dataset.train)

    train_set = CommDataset(train_items, train_transforms, relabel=True)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('*'*100)
        print('Hmm, Big Debugger is watching me')
        print('*'*100)
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    train_loader = make_sampler(train_set = train_set,
                                num_batch = cfg.SOLVER.IMS_PER_BATCH,
                                num_instance = cfg.DATALOADER.NUM_INSTANCE,
                                num_workers = num_workers,
                                mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
                                drop_last = True,
                                flag1 = cfg.DATALOADER.PK_SAMPLER,
                                flag2 = cfg.DATALOADER.NAIVE_WAY)

    train_loader_add = {}
    if not cfg.META.DATA.NAMES is "": # order of direction (0-> clockwise, 1,7,5,6,0,3,2,4)

        if 'keypoint' in cfg.META.DATA.NAMES:
            cfg, train_set_all = make_keypoint_data(cfg = cfg,
                                                    data_name = cfg.META.DATA.NAMES,
                                                    train_items = train_items)

        for i, x in enumerate(train_set_all):
            train_set_all[i] = CommDataset(x, train_transforms, relabel=cfg.META.DATA.RELABEL)
            if not cfg.META.DATA.RELABEL:
                train_set_all[i].relabel = True
                train_set_all[i].pid_dict = train_set.pid_dict

        # Check number of data
        cnt_data = 0
        for x in train_set_all:
            cnt_data += len(x.img_items)
        if cnt_data != len(train_set.img_items):
            print("data loading error, check build.py")


        if len(train_set_all) > 0:
            train_loader_add['init'] = []
            train_loader_add['mtrain'] = []
            train_loader_add['mtest'] = []
            for i, x in enumerate(train_set_all):
                train_loader_add['init'].append(make_sampler(train_set=x,
                                                             num_batch=cfg.META.SOLVER.INIT.IMS_PER_BATCH,
                                                             num_instance=cfg.META.SOLVER.INIT.NUM_INSTANCE,
                                                             num_workers=num_workers,
                                                             mini_batch_size=cfg.META.SOLVER.INIT.IMS_PER_BATCH
                                                                             // comm.get_world_size(),
                                                             drop_last=cfg.META.DATA.DROP_LAST,
                                                             flag1=cfg.DATALOADER.PK_SAMPLER,
                                                             flag2=cfg.DATALOADER.NAIVE_WAY))
                train_loader_add['mtrain'].append(make_sampler(train_set=x,
                                                               num_batch=cfg.META.SOLVER.MTRAIN.IMS_PER_BATCH,
                                                               num_instance=cfg.META.SOLVER.MTRAIN.NUM_INSTANCE,
                                                               num_workers=num_workers,
                                                               mini_batch_size=cfg.META.SOLVER.MTRAIN.IMS_PER_BATCH
                                                                               // comm.get_world_size(),
                                                               drop_last=cfg.META.DATA.DROP_LAST,
                                                               flag1=cfg.DATALOADER.PK_SAMPLER,
                                                               flag2=cfg.DATALOADER.NAIVE_WAY))
                train_loader_add['mtest'].append(make_sampler(train_set=x,
                                                              num_batch=cfg.META.SOLVER.MTEST.IMS_PER_BATCH,
                                                              num_instance=cfg.META.SOLVER.MTEST.NUM_INSTANCE,
                                                              num_workers=num_workers,
                                                              mini_batch_size=cfg.META.SOLVER.MTEST.IMS_PER_BATCH
                                                                              // comm.get_world_size(),
                                                              drop_last=cfg.META.DATA.DROP_LAST,
                                                              flag1=cfg.DATALOADER.PK_SAMPLER,
                                                              flag2=cfg.DATALOADER.NAIVE_WAY))


    return train_loader, train_loader_add, cfg


def build_reid_test_loader(cfg, dataset_name, opt=None):
    test_transforms = build_transforms(cfg, is_train=False)

    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            dataset.show_test()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    test_items = dataset.query + dataset.gallery

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs


def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True):
    if flag1:
        if flag2:
            data_sampler = samplers.NaiveIdentitySampler(train_set.img_items,
                                                         num_batch, num_instance)

        else:
            data_sampler = samplers.BalancedIdentitySampler(train_set.img_items,
                                                            num_batch, num_instance)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )

    return train_loader

def make_keypoint_data(cfg, data_name, train_items):


    cluster_view = []
    if data_name == "VeRi_keypoint_each_2":  # 2 (7560/3241)
        cluster_view = [[7, 5, 6, 0], [3, 2, 4, 1]]
    elif data_name == "VeRi_keypoint_each_4":  # 4 (75/60/32/41)
        cluster_view = [[7, 5], [6, 0], [3, 2], [4, 1]]
    elif data_name == "VeRi_keypoint_each_8":  # 8
        cluster_view = [[7], [5], [6], [0], [3], [2], [4], [1]]
    else:
        print("error_dataset_names")

    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()
    cfg.META.DATA.CLUSTER_VIEW = cluster_view
    if frozen: cfg.freeze()

    train_set_all = []
    for i, x in enumerate(cluster_view):
        train_items_all = train_items.copy()
        len_data = len(train_items_all)
        for j, y in enumerate(reversed(train_items_all)):
            if not y[3]['domains'] in cluster_view[i]:
                del train_items_all[len_data - j - 1]
        train_set_all.append(train_items_all)

    return cfg, train_set_all