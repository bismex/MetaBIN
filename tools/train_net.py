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

        cfg.MODEL.WEIGHTS = "./logs/Visualize/u01/model_final.pth"
        # Trainer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model


        res = Trainer.visualize(cfg, model)
        # res = Trainer.test(cfg, model)

        return res

    trainer = Trainer(cfg)
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

    # sdsdf
