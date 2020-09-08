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

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
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

    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if cfg.DATALOADER.PK_SAMPLER:
        if cfg.DATALOADER.NAIVE_WAY:
            data_sampler = samplers.NaiveIdentitySampler(train_set.img_items,
                                                         cfg.SOLVER.IMS_PER_BATCH, num_instance)

        else:
            data_sampler = samplers.BalancedIdentitySampler(train_set.img_items,
                                                            cfg.SOLVER.IMS_PER_BATCH, num_instance)
    else:
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )




    #### Add view dataloader

    train_loader_add = {}
    try:
        if not cfg.META.DATA.NAMES is "": # order of direction (0-> clockwise, 1,7,5,6,0,3,2,4)

            # parameters
            # cfg.META.DATA.RELABEL = False
            # cfg.META.DATA.DROP_LAST = True
            # cfg.META.DATA.IMS_PER_BATCH = 64
            # cfg.META.DATA.NUM_INSTANCE = 8
            # cfg.META.DATA.NAMES = "VeRi_keypoint_each_4"

            train_set_all = []
            cluster_view = []
            if cfg.META.DATA.NAMES == "VeRi_keypoint_each_2": #2 (7560/3241)
                cluster_view = [[7,5,6,0],[3,2,4,1]]
            elif cfg.META.DATA.NAMES == "VeRi_keypoint_each_4": #4 (75/60/32/41)
                cluster_view = [[7,5],[6,0],[3,2],[4,1]]
            elif cfg.META.DATA.NAMES == "VeRi_keypoint_each_8": #8
                cluster_view = [[7],[5],[6],[0],[3],[2],[4],[1]]
            else:
                print("error_dataset_names")

            cfg = cfg.clone()
            frozen = cfg.is_frozen()
            cfg.defrost()
            cfg.META.DATA.CLUSTER_VIEW = cluster_view
            if frozen: cfg.freeze()


            for i, x in enumerate(cluster_view):
                train_items_all = train_items.copy()
                len_data = len(train_items_all)
                for j, y in enumerate(reversed(train_items_all)):
                    if not y[3]['dir'] in cluster_view[i]:
                        del train_items_all[len_data-j-1]
                train_set_all.append(train_items_all)

            for i, x in enumerate(train_set_all):
                train_set_all[i] = CommDataset(x, train_transforms, relabel=cfg.META.DATA.RELABEL)
                if not cfg.META.DATA.RELABEL:
                    train_set_all[i].relabel = True
                    train_set_all[i].pid_dict = train_set.pid_dict


            cnt_data = 0
            for x in train_set_all:
                cnt_data += len(x.img_items)

            if cnt_data != len(train_set.img_items):
                print("data loading error, check build.py")


            if len(train_set_all) > 0:
                train_loader_init = []
                mini_batch_size_init = cfg.META.DATA.IMS_PER_BATCH // comm.get_world_size()
                for i, x in enumerate(train_set_all):
                    data_sampler_init = samplers.NaiveIdentitySampler(x.img_items,
                                                                     cfg.META.DATA.IMS_PER_BATCH,
                                                                     cfg.META.DATA.NUM_INSTANCE)
                    batch_sampler_init = torch.utils.data.sampler.BatchSampler(
                        data_sampler_init, mini_batch_size_init, cfg.META.DATA.DROP_LAST)
                    train_loader_init.append(torch.utils.data.DataLoader(
                        x, num_workers=num_workers, batch_sampler=batch_sampler_init,
                        collate_fn=fast_batch_collator,
                    ))
                train_loader_add['init'] = train_loader_init


                train_loader_mtrain = []
                mini_batch_size_mtrain = cfg.META.SOLVER.MTRAIN.IMS_PER_BATCH // comm.get_world_size()
                for i, x in enumerate(train_set_all):
                    data_sampler_mtrain = samplers.NaiveIdentitySampler(x.img_items,
                                                                     cfg.META.SOLVER.MTRAIN.IMS_PER_BATCH,
                                                                     cfg.META.SOLVER.MTRAIN.NUM_INSTANCE)
                    batch_sampler_mtrain = torch.utils.data.sampler.BatchSampler(
                        data_sampler_mtrain, mini_batch_size_mtrain, cfg.META.DATA.DROP_LAST)
                    train_loader_mtrain.append(torch.utils.data.DataLoader(
                        x, num_workers=num_workers, batch_sampler=batch_sampler_mtrain,
                        collate_fn=fast_batch_collator,
                    ))
                train_loader_add['mtrain'] = train_loader_mtrain


                train_loader_mtest = []
                mini_batch_size_mtest = cfg.META.SOLVER.MTEST.IMS_PER_BATCH // comm.get_world_size()
                for i, x in enumerate(train_set_all):
                    data_sampler_mtest = samplers.NaiveIdentitySampler(x.img_items,
                                                                     cfg.META.SOLVER.MTEST.IMS_PER_BATCH,
                                                                     cfg.META.SOLVER.MTEST.NUM_INSTANCE)
                    batch_sampler_mtest = torch.utils.data.sampler.BatchSampler(
                        data_sampler_mtest, mini_batch_size_mtest, cfg.META.DATA.DROP_LAST)
                    train_loader_mtest.append(torch.utils.data.DataLoader(
                        x, num_workers=num_workers, batch_sampler=batch_sampler_mtest,
                        collate_fn=fast_batch_collator,
                    ))
                train_loader_add['mtest'] = train_loader_mtest
    except:
        print("error in meta_data loader")


    return train_loader, train_loader_add, cfg


def build_reid_test_loader(cfg, dataset_name):
    test_transforms = build_transforms(cfg, is_train=False)

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query + dataset.gallery

    test_set = CommDataset(test_items, test_transforms, relabel=False)

    batch_size = cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=4,  # save some memory
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
