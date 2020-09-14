#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
import re

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import hooks
from fastreid.evaluation import ReidEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return ReidEvaluator(cfg, num_query)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # automatic OUTPUT dir
    cfg.merge_from_file(args.config_file)
    config_file_name = args.config_file.split('/')
    for i, x in enumerate(config_file_name):
        if x == 'configs':
            config_file_name[i] = 'logs'
        if '.yml' in x:
            config_file_name[i] = config_file_name[i][:-4]
    cfg.OUTPUT_DIR = '/'.join(config_file_name)

    # automatic resume file
    # if args.resume and os.path.isdir(cfg.OUTPUT_DIR):
    #     max_iter = 0
    #     find_file = ''
    #     for file in os.listdir(cfg.OUTPUT_DIR):
    #         if file.endswith(".pth"):
    #             str_iter = re.findall(r'\d+', file)
    #             num_iter = int(str_iter[-1])
    #             if num_iter > max_iter:
    #                 find_file = file
    #     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, find_file)
    #     print("cfg.MODEL.WEIGHTS:", cfg.MODEL.WEIGHTS)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        logger = logging.getLogger("fastreid.trainer")
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = Trainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(model):
            prebn_cfg = cfg.clone()
            prebn_cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
            prebn_cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN
            logger.info("Prepare precise BN dataset")
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                model,
                # Build a new data loader to not affect training
                Trainer.build_train_loader(prebn_cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ).update_stats()
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    if cfg.META.SOLVER.TRAIN_ORIGINAL:
        trainer.resume_or_load(resume=args.resume)
    return trainer.train() # train_loop.py -> train


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
