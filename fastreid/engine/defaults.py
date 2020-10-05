# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.
The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict
import numpy
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from fastreid.data import build_reid_test_loader, build_reid_train_loader
from fastreid.evaluation import (DatasetEvaluator, ReidEvaluator,
                                 inference_on_dataset, print_csv_format)
from fastreid.modeling.meta_arch import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.utils import comm
from fastreid.utils.env import seed_all_rng
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.collect_env import collect_env_info
from fastreid.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from fastreid.utils.file_io import PathManager
from fastreid.utils.logger import setup_logger
import torch.optim as optim
from . import hooks
from .train_loop import SimpleTrainer
# import logging
# logger = logging.getLogger(__name__)

__all__ = ["default_argument_parser", "default_setup", "DefaultPredictor", "DefaultTrainer"]
def default_argument_parser():
    """
    Create a parser with some common arguments used by fastreid users.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="fastreid Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:
    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """


    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng()

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config.
    The predictor takes an BGR image, resizes it to the specified resolution,
    runs the model and produces a dict of predictions.
    This predictor takes care of model loading and input preprocessing for you.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
    Examples:
    .. code-block:: python
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

    def __call__(self, image):
        """
        Args:
            image (torch.tensor): an image tensor of shape (B, C, H, W).
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        inputs = {"images": image}
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            predictions = self.model(inputs)
            # Normalize feature to compute cosine distance
            features = F.normalize(predictions)
            features = F.normalize(features).cpu().data
            return features
class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg):
        logger = logging.getLogger("fastreid")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for fastreid
            setup_logger()

        data_loader, data_loader_add, cfg = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader)
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model) # params, lr, momentum, ..

        torch.cuda.empty_cache()
        meta_param = dict()
        if cfg.META.DATA.NAMES != "":
            meta_param['num_domain'] = cfg.META.DATA.NUM_DOMAINS
            meta_param['whole'] = cfg.META.DATA.WHOLE

            meta_param['meta_compute_layer'] = cfg.META.MODEL.META_COMPUTE_LAYER
            meta_param['meta_update_layer'] = cfg.META.MODEL.META_UPDATE_LAYER

            meta_param['iter_init_inner'] = cfg.META.SOLVER.INIT.INNER_LOOP
            meta_param['iter_init_outer'] = cfg.META.SOLVER.INIT.OUTER_LOOP

            meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.META
            # meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.GATE_CYCLIC_RATIO
            # meta_param['update_ratio'] = cfg.META.SOLVER.LR_FACTOR.GATE_CYCLIC_PERIOD_PER_EPOCH
            meta_param['update_cyclic_ratio'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_RATIO
            meta_param['update_cyclic_period'] = cfg.META.SOLVER.LR_FACTOR.META_CYCLIC_PERIOD_PER_EPOCH
            meta_param['iters_per_epoch'] = cfg.SOLVER.ITERS_PER_EPOCH


            meta_param['iter_mtrain'] = cfg.META.SOLVER.MTRAIN.INNER_LOOP
            meta_param['shuffle_domain'] = cfg.META.SOLVER.MTRAIN.SHUFFLE_DOMAIN
            meta_param['use_second_order'] = cfg.META.SOLVER.MTRAIN.SECOND_ORDER
            meta_param['num_mtrain'] = cfg.META.SOLVER.MTRAIN.NUM_DOMAIN
            meta_param['freeze_gradient_meta'] = cfg.META.SOLVER.MTRAIN.FREEZE_GRAD_META
            meta_param['allow_unused'] = cfg.META.SOLVER.MTRAIN.ALLOW_UNUSED
            meta_param['zero_grad'] = cfg.META.SOLVER.MTRAIN.BEFORE_ZERO_GRAD
            meta_param['type_running_stats_init'] = cfg.META.SOLVER.INIT.TYPE_RUNNING_STATS
            meta_param['type_running_stats_mtrain'] = cfg.META.SOLVER.MTRAIN.TYPE_RUNNING_STATS
            meta_param['type_running_stats_mtest'] = cfg.META.SOLVER.MTEST.TYPE_RUNNING_STATS
            meta_param['auto_grad_outside'] = cfg.META.SOLVER.AUTO_GRAD_OUTSIDE

            if cfg.META.SOLVER.MTEST.ONLY_ONE_DOMAIN:
                meta_param['num_mtest'] = 1
            else:
                meta_param['num_mtest'] = meta_param['num_domain']\
                                                       - meta_param['num_mtrain']

            meta_param['sync'] = cfg.META.SOLVER.SYNC
            meta_param['detail_mode'] = cfg.META.SOLVER.DETAIL_MODE
            meta_param['stop_gradient'] = cfg.META.SOLVER.STOP_GRADIENT
            meta_param['flag_manual_zero_grad'] = cfg.META.SOLVER.MANUAL_ZERO_GRAD
            meta_param['flag_manual_memory_empty'] = cfg.META.SOLVER.MANUAL_MEMORY_EMPTY

            meta_param['loss_combined'] = cfg.META.LOSS.COMBINED
            meta_param['loss_weight'] = cfg.META.LOSS.WEIGHT
            meta_param['loss_name_mtrain'] = cfg.META.LOSS.MTRAIN_NAME
            meta_param['loss_name_mtest'] = cfg.META.LOSS.MTEST_NAME

            logger.info('-' * 30)
            logger.info('Meta-learning paramters')
            logger.info('-' * 30)
            for name, val in meta_param.items():
                logger.info('[M_param] {}: {}'.format(name, val))
            logger.info('-' * 30)


        if comm.get_world_size() > 1:
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        super().__init__(cfg, model, data_loader, data_loader_add, optimizer, meta_param)

        self.scheduler = self.build_lr_scheduler(
            optimizer = optimizer,
            scheduler_method = cfg.SOLVER.SCHED,
            warmup_factor = cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            milestones=cfg.SOLVER.STEPS,
            gamma=cfg.SOLVER.GAMMA,
            max_iters=cfg.SOLVER.MAX_ITER,
            delay_iters=cfg.SOLVER.DELAY_ITERS,
            eta_min_lr=cfg.SOLVER.ETA_MIN_LR,
        )

        self.checkpointer = Checkpointer(
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=comm.is_main_process(),
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        if cfg.SOLVER.SWA.ENABLED:
            self.max_iter = cfg.SOLVER.MAX_ITER + cfg.SOLVER.SWA.ITER
        else:
            self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.
        Otherwise, load a model specified by the config.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

        # Reinitialize dataloader iter because when we update dataset person identity dict
        # to resume training, DataLoader won't update this dictionary when using multiprocess
        # because of the function scope.
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        logger = logging.getLogger(__name__)
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN
        cfg.DATASETS.NAMES = tuple([cfg.TEST.PRECISE_BN.DATASET])  # set dataset name for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]

        if cfg.SOLVER.SWA.ENABLED:
            ret.append(
                hooks.SWA(
                    cfg.SOLVER.MAX_ITER,
                    cfg.SOLVER.SWA.PERIOD,
                    cfg.SOLVER.SWA.LR_FACTOR,
                    cfg.SOLVER.SWA.ETA_MIN_LR,
                    cfg.SOLVER.SWA.LR_SCHED,
                )
            )

        if cfg.TEST.PRECISE_BN.ENABLED and hooks.get_bn_modules(self.model):
            logger.info("Prepare precise BN dataset")
            ret.append(hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ))

        if cfg.MODEL.FREEZE_LAYERS != [''] and cfg.SOLVER.FREEZE_ITERS > 0:
            freeze_layers = ",".join(cfg.MODEL.FREEZE_LAYERS)
            logger.info(f'Freeze layer group "{freeze_layers}" training for {cfg.SOLVER.FREEZE_ITERS:d} iterations')
            ret.append(hooks.FreezeLayer(
                self.model,
                self.optimizer,
                cfg.MODEL.FREEZE_LAYERS,
                cfg.SOLVER.FREEZE_ITERS,
            ))
        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            if comm.is_main_process():
                self._last_eval_results = self.test(self.cfg, self.model)
                return self._last_eval_results
            else:
                return None

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), cfg.SOLVER.WRITE_PERIOD))

        return ret

        # IterationTimer: compute processing time each epoch
        # LRScheduler: step LR scheduler and summarize the LR
        # PeriodicCheckpointer: fastreid/utils/checkpoint.py, save checkpoint
        # EvalHook
        # PeriodicWriter: engine/defaults.py -> build_writers, fastreid/uitls/events.py
    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        .. code-block:: python
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]
    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        return 0
        if comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            # verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        # logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model))
        return model
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`fastreid.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)
    @classmethod
    def build_lr_scheduler(cls, optimizer, scheduler_method, warmup_factor,
                           warmup_iters, warmup_method, milestones,
                           gamma, max_iters, delay_iters, eta_min_lr):
        """
        It now calls :func:`fastreid.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(optimizer,
                                  scheduler_method,
                                  warmup_factor,
                                  warmup_iters,
                                  warmup_method,
                                  milestones,
                                  gamma,
                                  max_iters,
                                  delay_iters,
                                  eta_min_lr)
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        logger = logging.getLogger(__name__)
        logger.info("Prepare training set")
        return build_reid_train_loader(cfg)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name, opt=None):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_reid_test_loader(cfg, dataset_name, opt)
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_dir=None):
        return ReidEvaluator(cfg, num_query, output_dir)
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.
        Returns:
            dict: a dict of result metrics
        """

        gettrace = getattr(sys, 'gettrace', None)
        if gettrace():
            print('*' * 100)
            print('Hmm, Big Debugger is watching me')
            print('*' * 100)


        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]

        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TESTS):
            logger.info("Prepare testing set")
            if 'ALL' in dataset_name:
                report_all = cfg.TEST.REPORT_ALL
                results_local = OrderedDict()

                if 'VIPER' in dataset_name:
                    dataset_name_local = 'DG_VIPeR'
                    if 'only' in dataset_name:
                        sub_set = 'only_a'
                    else:
                        sub_set = 'all'
                    try:
                        num_test = int(dataset_name.split('_')[-1])
                    except:
                        num_test = 10
                    sub_type = ['a','b','c','d']
                    sub_name = [["split_" + str(i+1) + x for i in range(num_test)] for j, x in enumerate(sub_type)]
                    if sub_set == 'only_a':
                        sub_name = sub_name[0]
                    elif sub_set == 'all':
                        sub_name2 = sub_name
                        sub_name = []
                        for i in range(len(sub_name2)):
                            sub_name.extend(sub_name2[i])
                elif 'PRID' in dataset_name:
                    dataset_name_local = 'DG_PRID'
                    sub_name = [x for x in range(10)]
                elif 'GRID' in dataset_name:
                    dataset_name_local = 'DG_GRID'
                    sub_name = [x for x in range(10)]
                elif 'iLIDS' in dataset_name:
                    dataset_name_local = 'DG_iLIDS'
                    sub_name = [x for x in range(10)]


                for x in sub_name:
                    logger.info("Subset: {}".format(x))
                    data_loader, num_query = cls.build_test_loader(cfg, dataset_name_local, opt = x)
                    evaluator = cls.build_evaluator(cfg, num_query)
                    results_i = inference_on_dataset(model, data_loader, evaluator, opt=report_all)
                    if isinstance(x, int):
                        x = str(x)
                    if report_all:
                        results[dataset_name+'_'+x] = results_i
                    results_local[dataset_name+'_'+x] = results_i
                results_local_average = OrderedDict()
                results_local_std = OrderedDict()
                for name_global, val_global in results_local.items():
                    if len(results_local_average) == 0:
                        for name, val in results_local[name_global].items():
                            results_local_average[name] = val
                            results_local_std[name] = val
                    else:
                        for name, val in results_local[name_global].items():
                            results_local_average[name] += val
                            results_local_std[name] = \
                                numpy.hstack([results_local_std[name], val])
                for name, val in results_local_std.items():
                        results_local_std[name] = numpy.std(val)

                for name, val in results_local_average.items():
                    results_local_average[name] /= float(len(results_local))
                results[dataset_name+'_average'] = results_local_average
                results[dataset_name+'_std'] = results_local_std

            else:
                data_loader, num_query = cls.build_test_loader(cfg, dataset_name)
                # When evaluators are passed in as arguments,
                # implicitly assume that evaluators can be created before data_loader.
                if evaluators is not None:
                    evaluator = evaluators[idx]
                else:
                    try:
                        evaluator = cls.build_evaluator(cfg, num_query)
                    except NotImplementedError:
                        logger.warn(
                            "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                            "or implement its `build_evaluator` method."
                        )
                        results[dataset_name] = {}
                        continue
                results_i = inference_on_dataset(model, data_loader, evaluator)
                results[dataset_name] = results_i

        results_all_average = OrderedDict()
        cnt_average = 0
        for name_global, val_global in results.items():
            if 'average' in name_global:
                cnt_average += 1
                if len(results_all_average) == 0:
                    for name, val in results[name_global].items():
                        results_all_average[name] = val
                else:
                    for name, val in results[name_global].items():
                        results_all_average[name] += val

        for name, val in results_all_average.items():
            results_all_average[name] /= float(cnt_average)

        results['** all_average **'] = results_all_average


        if comm.is_main_process():
            assert isinstance(
                results, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results
            )
            print_csv_format(results)

        if len(results) == 1: results = list(results.values())[0]

        return results
    @staticmethod
    def auto_scale_hyperparams(cfg, data_loader):
        r"""
        This is used for auto-computation actual training iterations,
        because some hyper-param, such as MAX_ITER, means training epochs rather than iters,
        so we need to convert specific hyper-param to training iterations.
        """

        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        if isinstance(data_loader, list):
            num_images = max([x.batch_sampler.sampler.total_images for x in data_loader])
            num_classes = len(data_loader[0].dataset.pid_dict)
        else:
            num_images = data_loader.batch_sampler.sampler.total_images
            num_classes = data_loader.dataset.num_classes

        if cfg.META.DATA.NAMES != "": # meta-learning
            if cfg.META.SOLVER.INIT.INNER_LOOP == 0:
                iters_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
            else:
                iters_per_epoch = num_images // (cfg.SOLVER.IMS_PER_BATCH * cfg.META.SOLVER.INIT.INNER_LOOP)
        else:
            iters_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.ITERS_PER_EPOCH = iters_per_epoch
        cfg.MODEL.HEADS.NUM_CLASSES = num_classes
        cfg.SOLVER.MAX_ITER *= iters_per_epoch
        cfg.SOLVER.WARMUP_ITERS *= iters_per_epoch
        cfg.SOLVER.FREEZE_ITERS *= iters_per_epoch
        cfg.SOLVER.DELAY_ITERS *= iters_per_epoch
        for i in range(len(cfg.SOLVER.STEPS)):
            cfg.SOLVER.STEPS[i] *= iters_per_epoch
        cfg.SOLVER.SWA.ITER *= iters_per_epoch
        cfg.SOLVER.SWA.PERIOD *= iters_per_epoch
        cfg.SOLVER.CHECKPOINT_PERIOD *= iters_per_epoch


        # Evaluation period must be divided by 200 for writing into tensorboard.
        num_mod = (cfg.SOLVER.WRITE_PERIOD - cfg.TEST.EVAL_PERIOD * iters_per_epoch) % cfg.SOLVER.WRITE_PERIOD
        cfg.TEST.EVAL_PERIOD = cfg.TEST.EVAL_PERIOD * iters_per_epoch + num_mod
        # cfg.TEST.EVAL_PERIOD = 1

        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to num_classes={cfg.MODEL.HEADS.NUM_CLASSES}, "
            f"max_Iter={cfg.SOLVER.MAX_ITER}, wamrup_Iter={cfg.SOLVER.WARMUP_ITERS}, "
            f"freeze_Iter={cfg.SOLVER.FREEZE_ITERS}, delay_Iter={cfg.SOLVER.DELAY_ITERS}, "
            f"step_Iter={cfg.SOLVER.STEPS}, ckpt_Iter={cfg.SOLVER.CHECKPOINT_PERIOD}, "
            f"eval_Iter={cfg.TEST.EVAL_PERIOD}."
        )

        if frozen: cfg.freeze()

        return cfg
