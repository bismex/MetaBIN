# encoding: utf-8
"""
credit:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/train_loop.py
"""

import logging
import time
import weakref
import os

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import fastreid.utils.comm as comm
from fastreid.utils.events import EventStorage
from fastreid.utils.file_io import PathManager
logger = logging.getLogger(__name__)
import copy

from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    .. code-block:: python
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()
    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).
        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.
           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.
    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.
    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.
    Attributes:
        iter(int): the current iteration.
        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.
        max_iter(int): The iteration to end training.
        storage(EventStorage): An EventStorage that's opened during the course of training.
    """
    def __init__(self):
        self._hooks = []
    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
        # try:
            if self.meta_learning_parameter['meta_learning']:
                self.meta_learning()
            if self.meta_learning_parameter['load_parameter']:
                self.load_meta_param()
            if self.meta_learning_parameter['meta_learning'] or self.meta_learning_parameter['load_parameter']:
                self.update_meta_param()
            if self.meta_learning_parameter['ori_iter']:
                self.before_train() # check hooks.py, engine/defaults.py
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
        # except Exception:
        #     logger.exception("Exception during training:")
        # finally:
                self.after_train()
    def before_train(self):
        for h in self._hooks:
            h.before_train()
    def after_train(self):
        for h in self._hooks:
            h.after_train()
    def before_step(self):
        for h in self._hooks:
            h.before_step()
    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()
    def run_step(self):
        raise NotImplementedError

class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.
    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, cfg, model, model_meta, data_loader, optimizer, meta_learning_parameter):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of heads.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """

        self.model = model
        self.model_meta = model_meta
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.meta_learning_parameter = meta_learning_parameter
        if cfg.SOLVER.AMP:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """
        opt = {}
        opt['ds_flag'] = False
        opt['param_update'] = False
        opt['original_learning'] = True
        opt['loss'] = self.cfg['MODEL']['LOSSES']['NAME']

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model


        bin_gates = [p for p in model.parameters() if getattr(p, 'bin_gate', False)]

        for name, param in model.named_parameters():
            param.grad = None
            if 'reg' in name:
                param.requires_grad = False

        if self.cfg['META']['GRL']['DO_IT']:
            p = float(max(0, self.iter - self.cfg['SOLVER']['WARMUP_ITERS']) / (self.cfg['SOLVER']['MAX_ITER']-self.cfg['SOLVER']['WARMUP_ITERS']))
            constant = 2. / (1. + np.exp(-self.cfg['META']['GRL']['GAMMA'] * p)) - 1
            opt['GRL_constant'] = constant

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outs = model(data, opt)
            loss_dict = model.losses(outs, opt)
            losses = sum(loss_dict.values())

        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.scaler is None:
            losses.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        for p in bin_gates:
            p.data.clamp_(min=0, max=1)
            # print(p)

        if self.iter % (self.cfg.SOLVER.WRITE_PERIOD_PARAM * self.cfg.SOLVER.WRITE_PERIOD) == 0:
            self.logger_parameter_info(self.model)

    def basic_forward(self, opt, data):
        model = self.model_meta.module if isinstance(self.model_meta, DistributedDataParallel) else self.model_meta

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outs = model(data, opt)
            loss_dict = model.losses(outs, opt)
            losses = sum(loss_dict.values())

        self._detect_anomaly(losses, loss_dict)
        return losses, loss_dict
    def find_selected_optimizer(self, find_group, optimizer):

        # find parameter, lr, required_grad, shape
        logger.info('Storage parameter, lr, requires_grad, shape! in {}'.format(find_group))
        idx_group = []
        # name_group = []
        dict_group = dict()
        for j in range(len(find_group)):
            idx_local = []
            # name_local = []
            for i, x in enumerate(optimizer.param_groups):
                if find_group[j] in x['name']:
                    dict_group[x['name']] = i
                    idx_local.append(i)
            if len(idx_local) > 0:
                logger.info('Find {} in {}'.format(find_group[j], optimizer.param_groups[idx_local[0]]['name']))
                idx_group.append(idx_local[0])
            else:
                logger.info('error in find_group')
        return idx_group, dict_group
    def print_selected_optimizer(self, txt, idx_group, optimizer, detail_mode):

        if detail_mode:
            num_float = 8
            only_reg = True

            for x in idx_group:
                t_name = optimizer.param_groups[x]['name']
                if only_reg and not 'reg' in t_name:
                    continue
                t_param = optimizer.param_groups[x]['params'][0].view(-1)[0]
                t_lr = optimizer.param_groups[x]['lr']
                t_grad = optimizer.param_groups[x]['params'][0].requires_grad
                # t_shape = optimizer.param_groups[x]['params'][0].shape
                for name, param in self.model_meta.named_parameters():
                    if name == t_name:
                        m_param = param.view(-1)[0]
                        m_grad = param.requires_grad
                        val = torch.sum(param - optimizer.param_groups[x]['params'][0])
                        break
                if float(val) != 0:
                    logger.info('*****')
                    logger.info('<=={}==>[optimizer] --> [{}], w:{}, grad:{}, lr:{}'.format(txt, t_name, round(float(t_param), num_float), t_grad, t_lr))
                    logger.info('<=={}==>[self.model_meta] --> [{}], w:{}, grad:{}'.format(txt, t_name, round(float(m_param), num_float), m_grad))
                    logger.info('*****')

                else:
                    logger.info('[**{}**] --> [{}], w:{}, grad:{}, lr:{}'.format(txt, t_name, round(float(t_param), num_float), t_grad, t_lr))
    def load_meta_param(self):
        logger.info("******* Load meta paramter *******")
        data = torch.load(self.meta_learning_parameter['load_parameter_dir'])
        self.model_meta = data['model']
        # self.model_meta.load_state_dict(data['model'])
        logger.info("**********************************")
    def update_meta_param(self):

        logger.info("******* Update meta paramter *******")
        try:
            for name, param in self.model.state_dict().items():
                if 'reg' in name: # load reg
                    logger.info('[{}] model (before): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                    self.model.state_dict()[name].copy_(self.model_meta.state_dict()[name])
                    logger.info('Update reg parameters!!!!!!!!')
                    logger.info('[{}] model (after): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                if not self.meta_learning_parameter['initialize_conv'] and 'backbone' in name\
                        and (('weight' in name) or ('bias' in name)): # without initialize -> should load model_meta
                    logger.info('[{}] model (before): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                    self.model.state_dict()[name].copy_(self.model_meta.state_dict()[name])
                    logger.info('[{}] model (after): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                if not self.meta_learning_parameter['initialize_fc'] and 'heads' in name\
                        and (('weight' in name) or ('bias' in name)): # without initialize -> should load model_meta
                    logger.info('[{}] model (before): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                    self.model.state_dict()[name].copy_(self.model_meta.state_dict()[name])
                    logger.info('[{}] model (after): {}'.format(name, torch.sum(self.model.state_dict()[name])))
        except:
            for name, param in self.model.state_dict().items():
                if 'reg' in name: # load reg
                    logger.info('[{}] model (before): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                    self.model.state_dict()[name].copy_(self.model_meta[name])
                    logger.info('Update reg parameters!!!!!!!!')
                    logger.info('[{}] model (after): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                if not self.meta_learning_parameter['initialize_conv'] and 'backbone' in name\
                        and (('weight' in name) or ('bias' in name)): # without initialize -> should load model_meta
                    logger.info('[{}] model (before): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                    self.model.state_dict()[name].copy_(self.model_meta[name])
                    logger.info('[{}] model (after): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                if not self.meta_learning_parameter['initialize_fc'] and 'heads' in name\
                        and (('weight' in name) or ('bias' in name)): # without initialize -> should load model_meta
                    logger.info('[{}] model (before): {}'.format(name, torch.sum(self.model.state_dict()[name])))
                    self.model.state_dict()[name].copy_(self.model_meta[name])
                    logger.info('[{}] model (after): {}'.format(name, torch.sum(self.model.state_dict()[name])))





        logger.info("************************************")
    def logger_parameter_info(self, model):
        with torch.no_grad():
            write_dict = dict()
            round_num = 4
            name_num = 20
            for name, param in model.named_parameters():  # only update regularizer
                if 'reg' in name:
                    name = '_'.join([x[:name_num] for x in name.split('.')[1:]])
                    name = name + '+'
                    write_dict[name] = round(float(torch.sum(param.data.view(-1) > 0)) / len(param.data.view(-1)),
                                             round_num)

            for name, param in model.named_parameters():
                if ('meta' in name) and ('fc' in name) and ('weight' in name) and (not 'view' in name):
                    name = '_'.join([x[:name_num] for x in name.split('.')[1:]])
                    # name_std = name + '_std'
                    # write_dict[name_std] = round(float(torch.std(param.data.view(-1))), round_num)
                    # name_mean = name + '_mean'
                    # write_dict[name_mean] = round(float(torch.mean(param.data.view(-1))), round_num)
                    name_std10 = name + '_std10'
                    ratio = 0.1
                    write_dict[name_std10] = round(
                        float(torch.sum((param.data.view(-1) > - ratio * float(torch.std(param.data.view(-1)))) * (
                                param.data.view(-1) < ratio * float(torch.std(param.data.view(-1)))))) / len(
                            param.data.view(-1)), round_num)

            for name, param in model.named_parameters():
                if ('gate' in name) and (not 'view' in name):
                    name = '_'.join([x[:name_num] for x in name.split('.')[1:]])
                    name_mean = name + '_mean'
                    write_dict[name_mean] = round(float(torch.mean(param.data.view(-1))), round_num)
            logger.info(write_dict)
    def meta_learning(self):

        # 1. Initial parameter setting (shared layer + domain specific layers)
        # Dataloader: each domain dataloader
        # Parameter compute and update: shared layers, domain specific layers
        # Regularizer: None
        # Loss: CE + TRIP

        t_init = time.time()

        bin_gates = [p for p in self.model_meta.parameters() if getattr(p, 'bin_gate', False)]

        detail_mode = self.meta_learning_parameter['detail_mode']
        use_second_order = True
        find_group = ['conv', 'meta', 'reg']
        write_period = self.meta_learning_parameter['write_period']
        loss_combined = self.meta_learning_parameter['loss_combined']
        iter_local = self.meta_learning_parameter['iter_local'] # META.SOLVER.INIT.NUM_EPOCH
        iteration_all = self.meta_learning_parameter['iteration_all'] # META.SOLVER.FINAL.NUM_EPOCH
        iter_init_inner = self.meta_learning_parameter['iter_init_inner'] # META.SOLVER.INIT.INNER_LOOP
        iter_init_outer = self.meta_learning_parameter['iter_init_outer'] # META.SOLVER.INIT.OUTER_LOOP
        iter_mtrain = self.meta_learning_parameter['iter_mtrain'] # META.SOLVER.MTRAIN.INNER_LOOP
        num_mtrain = self.meta_learning_parameter['num_mtrain'] # META.SOLVER.MTEST.NUM_DOMAIN
        num_mtest = self.meta_learning_parameter['num_mtest'] # META.SOLVER.MTEST.NUM_DOMAIN
        inner_loop_type = self.meta_learning_parameter['inner_loop_type'] # META.SOLVER.MTRAIN.INNER_LOOP_TYPE
        dataloader_init = self.meta_learning_parameter['dataloader_init']
        dataloader_mtrain = self.meta_learning_parameter['dataloader_mtrain']
        dataloader_mtest = self.meta_learning_parameter['dataloader_mtest']
        loss_name_init = self.meta_learning_parameter['loss_name_init']
        loss_name_mtrain = self.meta_learning_parameter['loss_name_mtrain']
        loss_name_mtest = self.meta_learning_parameter['loss_name_mtest']
        num_view = len(dataloader_init)
        if num_mtrain < 1:
            logger.info('error in num_mtrain')
        dataloader_init_iter = []
        for x in dataloader_init:
            dataloader_init_iter.append(iter(x))
        dataloader_mtrain_iter = []
        for x in dataloader_mtrain:
            dataloader_mtrain_iter.append(iter(x))
        dataloader_mtest_iter = []
        for x in dataloader_mtest:
            dataloader_mtest_iter.append(iter(x))
        optimizer_init = self.meta_learning_parameter['optimizer_init']
        scheduler_init = self.meta_learning_parameter['scheduler_init']
        optimizer_final = self.meta_learning_parameter['optimizer_final']
        scheduler_final = self.meta_learning_parameter['scheduler_final']
        idx_group, dict_group = self.find_selected_optimizer(find_group, optimizer_final)
        # meta_idx = idx_group[1]
        # reg_idx = idx_group[2]

        opt = {}
        opt['original_learning'] = False
        opt['ds_flag'] = True # forward domain specific layers
        opt['param_update'] = False # apply updated_parameter
        opt['loss'] = loss_name_init

        cnt = 0
        t0 = time.time()
        # iter_local = 0
        while(cnt < iter_local):
            cnt += 1
            scheduler_init.step()
            for i in range(num_view):
                data = next(dataloader_init_iter[i])
                opt['view_idx'] = int(i)
                losses, loss_dict = self.basic_forward(opt, data)
                optimizer_init.zero_grad()
                if self.scaler is None:
                    losses.backward()
                    optimizer_init.step()
                else:
                    self.scaler.scale(losses).backward()
                    self.scaler.step(optimizer_init)
                    self.scaler.update()
            t1 = time.time()
            remaining_time = (iter_local - cnt)/cnt*(t1-t0)
            if cnt % write_period == 0:
                logger.info('1) [{}/{}] Meta-Initialization ({} views), eta:{}h:{}m:{}s, loss:{}'.format(
                    cnt, iter_local, num_view, int(remaining_time // 3600), int((remaining_time // 60) % 60),
                    int((remaining_time) % 60), ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict.items()]))
            self.print_selected_optimizer('Init', idx_group, optimizer_init, detail_mode)
        t_final = time.time()
        processing_time = t_final - t_init
        logger.info('Meta-initialization is finished (processing time:{}h:{}m:{}s)'.format(
            int(processing_time // 3600),
            int((processing_time // 60) % 60),
            int((processing_time) % 60)))

        # 2. Meta-learning for regularizer
        initial_requires_grad = dict()
        for name, param in self.model_meta.named_parameters():
            initial_requires_grad[name] = param.requires_grad

        cnt_global = 0
        t0 = time.time()
        # iteration_all = 50
        save_iter = list()
        if self.meta_learning_parameter['save_meta_param']:
            save_iter.append(int(iteration_all // 3))
            save_iter.append(int(2 * iteration_all // 3))
            save_iter.append(int(iteration_all))
        while(cnt_global < iteration_all):
            cnt_global += 1
            scheduler_final.step()

            # 2.1. Learning domain specific layer
            # Dataloader: each domain dataloader (continue) -> batch N
            # Parameter compute and update: shared layers, domain specific layers (continue)
            # Regularizer: None (continue)
            # Loss: CE/TRIP (continue)
            opt = {}
            opt['original_learning'] = False
            opt['ds_flag'] = True
            opt['param_update'] = False
            opt['loss'] = loss_name_init

            for name, param in self.model_meta.named_parameters():
                if 'reg' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = initial_requires_grad[name]

            cnt_local = 0
            while(cnt_local < iter_init_inner):
                cnt_local += 1
                for i in range(num_view):
                    data = next(dataloader_init_iter[i])
                    opt['view_idx'] = int(i)
                    losses, loss_dict_minit = self.basic_forward(opt, data)
                    optimizer_final.zero_grad()
                    if self.scaler is None:
                        losses.backward()
                        for name, param in self.model_meta.named_parameters(): # don't update reg
                            if ('reg' in name) and (param.grad is not None):
                                param.grad = None
                        optimizer_final.step()
                    else:
                        self.scaler.scale(losses).backward()
                        for name, param in self.model_meta.named_parameters(): # don't update reg
                            if ('reg' in name) and (param.grad is not None):
                                param.grad = None
                        self.scaler.step(optimizer_final)
                        self.scaler.update()

            for p in bin_gates:
                p.data.clamp_(min=0, max=1)
            self.print_selected_optimizer('after meta-init', idx_group, optimizer_final, detail_mode)

            if self.meta_learning_parameter['sync']:
                torch.cuda.synchronize()

            # 2.2. Meta-training
            # Dataloader: each domain dataloader (meta-training) -> unroll batch K x #(meta-train dataset)
            # Parameter compute: domain specific layers (unroll) -> new step paramater (actually same as lr)
            # Freeze layers: conv
            # Regularizer: Yes
            # Loss: CE/TRIP + Reg
            for name, param in self.model_meta.named_parameters():
                param.requires_grad = initial_requires_grad[name]

            # list_mtrain = [0]
            # list_mtest = [1]
            cnt_outer = 0
            while(cnt_outer < iter_init_outer):
                cnt_outer += 1
                list_all = np.random.permutation(num_view)
                list_mtrain = list(list_all[0:num_mtrain])
                list_mtest = list(list_all[num_mtrain:num_mtrain+num_mtest])

                # freeze layers (required grad = False)
                for name, param in self.model_meta.named_parameters():
                    if 'backbone' in name:
                        param.requires_grad = False # self.model_meta.backbone -> all false
                    else:
                        param.requires_grad = initial_requires_grad[name]
                for idx_mtrain in list_mtrain:
                    opt['ds_flag'] = True
                    opt['param_update'] = False
                    opt['loss'] = loss_name_mtrain
                    opt['use_second_order'] = use_second_order
                    meta_train_losses = []

                    cnt_local = 0
                    opt['new_param'] = dict()
                    while(cnt_local < iter_mtrain):
                        cnt_local += 1
                        if cnt_local == 1:
                            data = next(dataloader_mtrain_iter[idx_mtrain])
                        elif inner_loop_type == 'diff':
                            data = next(dataloader_mtrain_iter[idx_mtrain])
                        opt['view_idx'] = int(idx_mtrain)

                        losses, loss_dict_mtrain = self.basic_forward(opt, data)

                        # optimizer_final.zero_grad()
                        if not opt['param_update']: # first inner-loop
                            for name, param in self.model_meta.named_parameters(): # parameter grad_zero
                                if param.grad is not None:
                                    param.grad = None
                            for name, param in self.model_meta.named_parameters(): # grad update
                                if 'view{}'.format(opt['view_idx']) in name:
                                    lr = optimizer_final.param_groups[dict_group[name]]['lr']
                                    grads = torch.autograd.grad(losses, param,
                                                                create_graph=opt['use_second_order'],
                                                                allow_unused=True)[0]
                                    if not grads == None:
                                        grads = param - lr * grads
                                        opt['new_param'][name] = grads
                        else: # after first inner-loop
                            for name, param in self.model_meta.named_parameters(): # parameter grad_zero
                                if param.grad is not None:
                                    param.grad = None
                            for name, param in opt['new_param'].items(): # new parameter grad_zero
                                if param.requires_grad == True:
                                    if param.grad is not None:
                                        param.grad = None
                            old_param = opt['new_param']
                            opt['new_param'] = dict()
                            for name, param in old_param.items(): # grad update
                                if 'view{}'.format(opt['view_idx']) in name:
                                    grads = torch.autograd.grad(losses, param,
                                                                create_graph=opt['use_second_order'],
                                                                allow_unused=True)[0]
                                    if not grads == None:
                                        grads = param - lr * grads
                                        opt['new_param'][name] = grads
                        opt['param_update'] = True
                        self.print_selected_optimizer('after meta train (iter, after backward)', idx_group, optimizer_final, detail_mode)
                        if loss_combined:
                            meta_train_losses.append(losses)

                    # 2.3. Meta-testing
                    # Dataloader: each domain dataloader (meta-testing) -> batch K' x #(meta-test dataset)
                    # [maybe N = K*(D_train) + K'*(D_test), where D = D_train + D_test]
                    # Load parameter: domain specific layers
                    # Parameter compute and update: regularizer -> new step paramater (actually same as lr)
                    # Freeze layers: conv
                    # Regularizer: Yes
                    # Loss: CE/TRIP + (Reg)
                    if self.meta_learning_parameter['sync']:
                        torch.cuda.synchronize()
                    opt['ds_flag'] = True
                    opt['param_update'] = True
                    opt['loss'] = loss_name_mtest
                    opt['view_idx'] = int(idx_mtrain)
                    meta_test_losses = []
                    for j, idx_mtest in enumerate(list_mtest):
                        if j == 0:
                            data = next(dataloader_mtest_iter[idx_mtest])
                        else:
                            new_data = next(dataloader_mtest_iter[idx_mtest])
                            data['images'] = torch.cat((data['images'], new_data['images']), 0)
                            data['targets'] = torch.cat((data['targets'], new_data['targets']), 0)
                            data['camid'] = torch.cat((data['camid'], new_data['camid']), 0)
                            data['img_path'].extend(new_data['img_path'])
                            data['others']['dir'] = torch.cat((data['others']['dir'], new_data['others']['dir']), 0)

                    final_losses, loss_dict_mtest = self.basic_forward(opt, data)
                    # meta_test_losses.append(losses)
                    # final_losses = torch.sum(torch.stack(meta_test_losses))

                    self.print_selected_optimizer('after meta test (before backward)', idx_group, optimizer_final, detail_mode)
                    if loss_combined:
                        final_losses += torch.sum(torch.stack(meta_train_losses))

                    optimizer_final.zero_grad()
                    if self.scaler is None:
                        final_losses.backward()
                        for name, param in self.model_meta.named_parameters(): # only update regularizer
                            if (param.grad is not None) and ('reg' not in name):
                                param.grad = None
                        optimizer_final.step()
                    else:
                        self.scaler.scale(final_losses).backward()
                        for name, param in self.model_meta.named_parameters(): # only update regularizer
                            if (param.grad is not None) and ('reg' not in name):
                                param.grad = None
                        self.scaler.step(optimizer_final)
                        self.scaler.update()

                    for p in bin_gates:
                        # print(p)
                        p.data.clamp_(min=0, max=1)
                    self.print_selected_optimizer('after meta test (final)', idx_group, optimizer_final, detail_mode)

                    for name, param in self.model_meta.named_parameters():
                        if param.grad is not None:
                            param.grad = None

                    if self.meta_learning_parameter['sync']:
                        torch.cuda.synchronize()

            t1 = time.time()
            remaining_time = (iteration_all - cnt_global)/cnt_global*(t1-t0)

            if cnt_global % write_period == 0:

                logger.info('2) [{}/{}] Meta-Optimization ({} views), eta:{}h:{}m:{}s, [test]{}, [reg_sum]:{} //// [train]{}, [init]{}, '.format(
                    cnt_global, iteration_all, num_view, int(remaining_time // 3600), int((remaining_time // 60) % 60),
                    int((remaining_time) % 60),
                    ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict_mtest.items()],
                    [round(float(torch.sum(torch.abs(param))), 6) for name, param in self.model_meta.named_parameters() if 'reg' in name],
                    ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict_mtrain.items()],
                    ['{}:{}'.format(name, round(float(val), 4)) for name, val in loss_dict_minit.items()],
                ))

            if cnt_global % (self.meta_learning_parameter['write_period_param']*write_period) == 0:
                self.logger_parameter_info(self.model_meta)

            if cnt_global in save_iter:
                idx_save = [i for i, x in enumerate(save_iter) if x == cnt_global][0] + 1
                save_name = 'meta{}.pth'.format(idx_save)
                logger.info('=*=* [save_dir, iter:{}/{}] *=*= [dir:{}/name:{}]'.format(
                    cnt_global, iteration_all, self.meta_learning_parameter['output_dir'], save_name))
                save_dir = os.path.join(self.meta_learning_parameter['output_dir'], save_name)
                data = {}
                data["model"] = self.model_meta.state_dict()
                with PathManager.open(save_dir, "wb") as f:
                    torch.save(data, f)


                # self.meta_learning_parameter['output_dir']
                # checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)

        t_final = time.time()
        processing_time = t_final - t_init
        logger.info('Total meta-optimization is finished (processing time:{}h:{}m:{}s)'.format(
            int(processing_time // 3600),
            int((processing_time // 60) % 60),
            int((processing_time) % 60)))

        for name, param in self.model_meta.named_parameters():
            param.grad = None
            param.requires_grad = initial_requires_grad[name]
    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )
    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in fastreid.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
