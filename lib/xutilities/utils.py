"""A collection of tools to build various parts of a pytorch project."""

import sys
import os
from os import path as osp
import json
import fcntl
from tempfile import TemporaryFile
import time
import subprocess
import inspect
import logging
import argparse
import warnings
from datetime import datetime
from contextlib import contextmanager
from timeit import default_timer
import importlib

import matplotlib.pyplot as plt
import random
import uuid
import numpy as np


# ------------------------------- #
# Multi-GPU Handling
# ------------------------------- #

# GPU_LOCK_FD holds a lock file descriptor for lock_gpu and unlock_gpu
GPU_LOCK_FD = None
GPU_USAGE_FD = None

def lock_gpu(lockname='.gpu.lock'):  # noqa: E302
    """Locks a file to prevent simultaneous query of GPU information. Note that
    this lock only affects programs that recognizes the specified lock file and
    does not block other programs' usage of GPU.
    """
    global GPU_LOCK_FD
    GPU_LOCK_FD = open(lockname, 'w')
    fcntl.lockf(GPU_LOCK_FD, fcntl.LOCK_EX)


def unlock_gpu(lockname='.gpu.lock'):
    """Release the lock applied by lock_gpu().
    This function can be called when GPU RAM usage has been stablized
    (e.g. after first backward of the whole model).
    """
    global GPU_LOCK_FD
    if GPU_LOCK_FD is not None:
        fcntl.flock(GPU_LOCK_FD, fcntl.LOCK_UN)
        GPU_LOCK_FD.close()
        GPU_LOCK_FD = None


def obtain_gpu(lockname='.gpu.usage.json'):
    global GPU_USAGE_FD
    gpu_usage = json.load(open(lockname, 'r'))
    GPU_USAGE_FD = open(lockname, 'w')
    fcntl.lockf(GPU_USAGE_FD, fcntl.LOCK_EX)
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    me = os.getpid()
    for gid in gpu_ids:
        if gid not in gpu_usage: gpu_usage[gid] = []
        gpu_usage[gid].append(me)

    json.dump(gpu_usage, GPU_USAGE_FD)
    fcntl.flock(GPU_USAGE_FD, fcntl.LOCK_UN)
    GPU_USAGE_FD.close()


def release_gpu(lockname='.gpu.usage.json'):
    global GPU_USAGE_FD
    gpu_usage = json.load(open(lockname, 'r'))
    GPU_USAGE_FD = open(lockname, 'w')
    fcntl.lockf(GPU_USAGE_FD, fcntl.LOCK_EX)
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    me = os.getpid()
    for gid in gpu_ids:
        gpu_usage[gid].remove(me)
    json.dump(gpu_usage, GPU_USAGE_FD)
    fcntl.flock(GPU_USAGE_FD, fcntl.LOCK_UN)
    GPU_USAGE_FD.close()


def get_usable_gpu(threshold=2048, req_cards=1, gpu_id_map=None, allowed_collisions=2):
    """Find a usable gpu

    Args:
        threshold :int: required GPU free memory.
        req_card :int: requested number of cards.
        gpu_id_remap :[int]: in cases where GPU IDs mess up, use a remap.
        self_collision :[int]: allow this number of jobs to be allocated to the
            same card. 

    Returns:
        GPU ID :int:, or
        :None: if no GPU is found
    """
    try:
        gpu_info = json.loads(subprocess.check_output(['gpustat', '--json']))
    except Exception as e:
        print('No gpustat detected, using nvidia-smi.')
        free_mem = list(map(int, subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'
        ]).decode().split()))

        free_id_mem = [(i, m) for i, m in enumerate(free_mem)]
        id_mem_sorted = sorted(free_id_mem, key=lambda x: x[1], reverse=True)

        bests = id_mem_sorted[:req_cards]
        if bests[-1][1] < threshold:
            gpu_id = None
        else:
            gpu_id = [i for i, _ in bests]
            if gpu_id_map:
                gpu_id = [gpu_id_map[i] for i in gpu_id]
            gpu_id = [str(i) for i in gpu_id]
        return gpu_id

    current_user = os.environ['USER']
    id_info = []
    for c in gpu_info['gpus']:
        current_index = c['index']
    
        current_free_mem = c['memory.total'] - c['memory.used']
        current_util = c['utilization.gpu']
        current_temp = c['temperature.gpu']
        
        collision = 0
        try:
            gpu_usage = json.load(open('.gpu.usage.json', 'r'))
            if str(c['index']) not in gpu_usage:
                collision = 0
            else:
                collision = len(gpu_usage[str(c['index'])])
        except Exception:
            print('No gpu usage found, try to determine with gpustat.')
            for p in c['processes']:
                if p['username'] == current_user: collision += 1

        if current_free_mem < threshold:
            continue
        if collision + 1 > allowed_collisions:
            continue
        id_info.append((current_index, current_free_mem, current_temp, current_util, collision))
    
    if len(id_info) < req_cards:
        gpu_id = None
    else:
        # criterion:  (1) temperature, (2) utilization, (3) -collision, (4) remaining memory
        id_info_sorted = sorted(id_info, key=lambda x: (100-x[2], 100-x[3], -x[4], x[1]), reverse=True)
        gpu_id = [i[0] for i in id_info_sorted[:req_cards]]
        if gpu_id_map:
            gpu_id = [gpu_id_map[i] for i in gpu_id]
        gpu_id = [str(i) for i in gpu_id]

    return gpu_id


def wait_gpu(req_mem=8000, req_cards=1, id_map=None, allowed_collisions=2):
    wait_time = int(random.random() * 5)
    time.sleep(wait_time)
    lock_gpu()
    while True:
        gpu_id = get_usable_gpu(req_mem, req_cards=req_cards, gpu_id_map=id_map, allowed_collisions=allowed_collisions)
        if gpu_id is not None:
            break
        time.sleep(5)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)


# ------------------------------- #
# Debugging
# ------------------------------- #
def myself():
    return inspect.stack()[1][3]


def collect_layer_grad(named_parameters):
    n_ave_grads = {}
    n_max_grads = {}
    for n, p in named_parameters:
        if p.requires_grad:
            if p.grad is not None:
                n_ave_grads.update({n: p.grad.abs().mean()})
                n_max_grads.update({n: p.grad.abs().max()})
            else:
                n_ave_grads.update({n: 0.})
                n_max_grads.update({n: 0.})
    return n_ave_grads, n_max_grads
    

def plot_grad_flow(model, plot=False):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(model)" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    n_ave_grads, n_max_grads = collect_layer_grad(model.named_parameters())

    for n, p in model.named_parameters():
        if p.requires_grad:
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(n_ave_grads[n])
                max_grads.append(n_max_grads[n])
                print(f'layer: {n}, shape: {list(p.shape)}, ave grad: {ave_grads[-1]:.8f}, max grad: {max_grads[-1]:.8f}, grad norm: {p.grad.norm():.8f}.')
            else:
                print(f'layer: {n} has no grad.')
                ave_grads.append(0.)
                max_grads.append(0.)
    if plot:
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="b")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="g")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.savefig('grad.local.png', bbox_inches='tight')


def global_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return float(total_norm)


# ------------------------------- #
# Benchmark
# ------------------------------- #
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start  # noqa: E731
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start    # noqa: E731


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------- #
# Checkpoint Handling
# ------------------------------- #
def load_parallel_state_dict(state_dict):
    """Remove the module.xxx in the keys for models trained
        using data_parallel.

    Returns:
        new_state_dict
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def save_checkpoint(path, **kwargs):
    import torch
    torch.save(kwargs, path)


def load_checkpoint(path, state_dict_to_load=None, from_parallel=False, map_location=None):
    """Load checkpoint from path

    Args:
        path :str: path of checkpoint file.
        state_dict_to_load :[]: keys of states to load. Set it to None if checkpoint has only

    Returns:
        checkpoint :dict of state_dicts:
    """
    import torch
    checkpoint = torch.load(path, map_location=map_location)
    if from_parallel:
        checkpoint['model'] = load_parallel_state_dict(checkpoint['model'])
    if state_dict_to_load is None:
        return checkpoint
    if set(state_dict_to_load) != set(list(checkpoint.keys())):
        logging.warning(f'Checkpoint key mismatch. '
                        f'Requested {set(state_dict_to_load)}, found {set(list(checkpoint.keys()))}.')

    return checkpoint


def prepare_train(model, optimizer, lr_scheduler, scalar, args, **kwargs):
    """Do the dirty job of loading model/model weights, optimizer and
        lr_scheduler states from saved state-dicts.
    
    If `args.from_model` is set, the states will be fully recovered.
    If `args.load_weight_from` is set instead, only model weight will be loaded.
        Optimizer and lr_scheduler will not be loaded.
    
    Args:
        model, optimizer, lr_scheduler
        args: argument returned by init()
        If args.finetune is set:
            kwargs['finetune_old_head'] :str: name of the head the model that is to be replaced
            kwargs['finetune_new_head'] :torch.nn.Module: new head that will be appended to model

    Returns:
        model, optimizer, lr_scheduler
    
    """
    import torch
    if args.from_model:
        state_dict = load_checkpoint(args.from_model)

        if 'checkpoint_epoch' in state_dict.keys():
            args.start_epoch = state_dict['checkpoint_epoch'] + 1
        
        if scalar is not None and 'grad_scaler' in state_dict.keys():
            scalar.load_state_dict(state_dict['grad_scaler'])

        if 'model' in state_dict.keys():
            if not args.parallel:
                model.load_my_state_dict(state_dict['model'])
            else:
                model.load_my_state_dict(
                    load_parallel_state_dict(state_dict['model']))
        else:  # backward compatibility
            if not args.parallel:
                model.load_my_state_dict(state_dict)
            else:
                model.load_my_state_dict(load_parallel_state_dict(state_dict))

        # if --finetune is set, the head is reset to a new 1x1 conv layer
        if args.finetune:
            setattr(model, kwargs['finetune_old_head'], kwargs['finetune_new_head'])
        
        if optimizer is not None:
            if 'optimizer' in state_dict.keys():
                optimizer.load_state_dict(state_dict['optimizer'])
            if 'initial_lr' in state_dict.keys():
                optimizer.param_groups[0]['initial_lr'] = state_dict['initial_lr']
            else:
                optimizer.param_groups[0]['initial_lr'] = args.lr
        
        if lr_scheduler is not None:
            if 'lr_scheduler' in state_dict.keys():
                lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        
        if 'amp' in state_dict.keys():
            import apex
            apex.amp.load_state_dict(state_dict['amp'])

    if args.load_weight_from and not args.from_model:
        state_dict = load_checkpoint(args.load_weight_from)
        if 'model' in state_dict.keys():
            if not args.parallel:
                model.load_my_state_dict(state_dict['model'])
            else:
                model.load_my_state_dict(
                    load_parallel_state_dict(state_dict['model']))
        else:
            if not args.parallel:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(load_parallel_state_dict(state_dict))

    if args.parallel:
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(args.device)
    
    return model, optimizer, lr_scheduler


# ------------------------------- #
# Data Handling
# ------------------------------- #
def longtensor_to_one_hot(labels, num_classes):
    """convert int encoded label to one-hot encoding

    Args:
        labels :[batch_size, 1]:
        num_classes :int:

    Returns:
        one-hot encoded label :[batch_size, num_classes]:
    """
    import torch
    return torch.zeros(labels.shape[0], num_classes).scatter_(1, labels, 1)


def label_segmented_pooling(x):
    """Pool input tensor into segments along temporal dimension, based on predicted label.

    Args:
        - x: Tensor of shape CxT, where C is the number of classes and T is the temporal dimension
    
    Returns:
        - Tensor of shape CxS, where S is the number of segments determined by input x
    """
    import torch
    from torch_scatter import segment_csr
    predictions = x.argmax(0)
    segments = (predictions[1:] - predictions[:-1]).nonzero().squeeze(1) + 1
    segments = torch.cat([
        torch.Tensor([0]).to(x.device), 
        segments, 
        torch.Tensor([len(predictions)]).to(x.device)
        ], dim=0).long().unsqueeze(0)
    
    output = segment_csr(x, segments, reduce='mean')
    return output, segments


# ------------------------------- #
# Training Control
# ------------------------------- #
class RotateCheckpoint:
    def __init__(self, length, base_name='checkpoint'):
        self.length = length
        self.idx = -1
        self.checkpoint_names = [base_name + f'_{i}.state' for i in range(length)]

    def step(self):
        self.idx = (self.idx + 1) % self.length
        return self.checkpoint_names[self.idx]


class EarlyStop:
    def __init__(self, patience: int, verbose: bool = True):
        self.patience = patience
        self.init_patience = patience
        self.verbose = verbose
        self.lowest_loss = 9999999.999
        self.highest_acc = -0.1
    
    def state_dict(self):
        return {
            'patience': self.patience,
            'init_patience': self.init_patience,
            'verbose': self.verbose,
            'lowest_loss': self.lowest_loss,
            'highest_acc': self.highest_acc,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.init_patience = state_dict['init_patience']
        self.verbose = state_dict['verbose']
        self.lowest_loss = state_dict['lowest_loss']
        self.highest_acc = state_dict['highest_acc']

    def step(self, loss=None, acc=None, criterion=lambda x1, x2: x1 or x2):
        if loss is None:
            loss = self.lowest_loss
            better_loss = True
        else:
            better_loss = (loss < self.lowest_loss) and ((self.lowest_loss-loss)/self.lowest_loss > 0.01)
        if acc is None:
            acc = self.highest_acc
            better_acc = True
        else:
            better_acc = acc > self.highest_acc
        
        if better_loss:
            self.lowest_loss = loss
        if better_acc:
            self.highest_acc = acc

        if criterion(better_loss, better_acc):
            self.patience = self.init_patience
            if self.verbose:
                logging.getLogger(myself()).debug(
                    'Remaining patience: {}'.format(self.patience))
            return False
        else:
            self.patience -= 1
            if self.verbose:
                logging.getLogger(myself()).debug(
                    'Remaining patience: {}'.format(self.patience))
            if self.patience < 0:
                if self.verbose:
                    logging.getLogger(myself()).warning('Ran out of patience.')
                return True


class ShouldSaveModel:
    """
    Args:
        init_step :int: start_epoch - 1
    """
    def __init__(self, init_step=-1, highest_acc=-0.1, lowest_loss=999999999999.):
        self.lowest_loss = lowest_loss
        self.highest_acc = highest_acc
        self.current_step = init_step
        self.best_step = init_step
    
    def state_dict(self):
        return {
            'lowest_loss': self.lowest_loss,
            'highest_acc': self.highest_acc,
            'current_step': self.current_step,
            'best_step': self.best_step,
        }
    
    def load_state_dict(self, state_dict):
        self.lowest_loss = state_dict['lowest_loss']
        self.highest_acc = state_dict['highest_acc']
        self.current_step = state_dict['current_step']
        self.best_step = state_dict['best_step']

    def step(self, current_step=None, loss=None, acc=None, criterion=lambda x1, x2: x1 or x2):
        """
        Decides whether a model should be saved, based on the criterion.

        Args:
            loss :float: loss after current epoch.
            acc :float: acc after current epoch.
            criterion :callable:
                A function that takes two params and returns a bool.
                The first parameter is loss, the second acc.

        Returns:
            :bool: Whether this model should be saved.
        """
        if current_step is None:
            self.current_step += 1
        else:
            self.current_step = current_step
        if loss is None:
            loss = self.lowest_loss
            better_loss = True
        else:
            better_loss = (loss < self.lowest_loss) and \
                            ((self.lowest_loss-loss)/self.lowest_loss > 0.005)
        if acc is None:
            acc = self.highest_acc
            better_acc = True
        else:
            better_acc = acc > self.highest_acc

        if better_loss:
            self.lowest_loss = loss
        if better_acc:
            self.highest_acc = acc
        if criterion(better_loss, better_acc):
            logging.getLogger(myself()).info(
                f'New model: epoch: {self.current_step}, '
                f'highest performance: {acc:.4}, lowest loss: {loss:.4}.')
            self.best_step = self.current_step
            return True
        else:
            return False


class RunningAverage:
    def __init__(self, window_size, initial_step=0):
        self.data = np.zeros([window_size, 1])
        self.window_size = window_size
        self.step = initial_step
        self.idx = -1
    
    def value(self):
        try:
            return self.data[:self.step].sum() / min(self.step, self.window_size)
        except ZeroDivisionError:
            return 0

    def add(self, d):
        self.idx = (self.idx + 1) % self.window_size
        self.data[self.idx] = d
        self.step += 1
        return self.value()


def apply_loss_weights(losses, lsw):
    if set(losses.keys()) != set(lsw.keys()) != 0:
        all_keys = set(losses.keys()).union(set(lsw.keys()))
        problems = all_keys - set(losses.keys()).intersection(set(lsw.keys()))
        warnings.warn(
            'Loss keys inconsistent. '
            f'Losses contains {list(losses.keys())}, '
            f'whereas provided weights contains {list(lsw.keys())}. '
            f'Check these keys: {problems}')

    pop_keys = []
    for k in losses.keys():
        w = lsw.get(k, 0.0)
        if w == 0.0 or type(losses[k]) is float:
            pop_keys.append(k)
            continue
        losses[k] *= w
    for ki in pop_keys:
        losses.pop(ki)
    
    loss = 0
    for _, v in losses.items():
        loss += v
    
    return loss, losses


# ------------------------------- #
# Initialization
# ------------------------------- #
def get_callable(spec_str):
    tmp = spec_str.split('.')
    package_name = '.'.join(tmp[:-1])
    callable_name = tmp[-1]
    return getattr(importlib.import_module(package_name), callable_name)


def config_logger(log_file=None):
    if log_file is not None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=log_file,
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)-8.8s] (%(name)-8.8s %(filename)15.15s:%(lineno)5.5d) %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger('').critical("Uncaught exception",
                                       exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def append_test_args(p):
    """Append arguments for model testing

    Args:
        p :argparse.ArgumentParser object:

    Returns
        parameters :argparse.ArgumentParser object: with appended arguments
    """
    p.add_argument('--from_model', '--from-model',
                   nargs='+', type=str, required=True)
    p.add_argument('--cuda', action='store_true', default=False)
    p.add_argument('--parallel', action='store_true', default=False)
    p.add_argument('--num_workers', '--num-workers', type=int, default=2)
    p.add_argument('--batch_size', '--batch-size', type=int, default=32)
    p.add_argument('--dataset', choices=['mitstates', 'ut-zap50k'], required=True,
                   help='Dataset for training and testing.')

    return p


def create_parser(user_param=None):
    """Create the basic argument parser for environment setup.

    Args:
        user_param : callable. A function that takes and returns an ArgumentParser.
                    Can be used to add user parameters.

    Return:
        p :argparse.ArgumentParser:
    """
    p = argparse.ArgumentParser(
        description='input arguments.',
        fromfile_prefix_chars='@')

    # p.add_argument('--no-pbar', action='store_true',
    #                default=False, help='Subpress progress bar.')
    p.add_argument('--pbar', type=int, default=0, help='Whether to enable progress bar.')
    p.add_argument('--log_dir', default=None)
    p.add_argument('--debug_mode', '--debug-mode', action='store_true', default=False)
    p.add_argument('--summary_to', type=str, default=None)
    p.add_argument('--uuid', default=None,
                   help='UUID of the model. Will be generated automatically if unspecified.')
    p.add_argument('--cuda', action='store_true', default=False,
                   help='Flag for cuda. Will be automatically determined if unspecified.')
    p.add_argument('--device', default='cuda',
                   help='Option for cuda. Will be automatically determined if unspecified.')
    p.add_argument('--device_list', nargs='+', default=[0],
                   help='List of cuda devices, only functional when parallel flag is set.')
    p.add_argument('--parallel', action='store_true', default=False,
                   help='Flag for parallel.')
    p.add_argument('--start_epoch', type=int, default=0)
    p.add_argument('--max_epoch', type=int, default=100)
    p.add_argument('--from_model', '--from-model', type=str, default=None,
                        help='Load model, optimizer, lr_scheduler state from path.')
    p.add_argument('--finetune', action='store_true', default=False)
    p.add_argument('--load_weight_from', '--load-weight-from', type=str, nargs='?',
                    default=None, const=None,
                        help='Load model state from path. This will invalidate --finetune flag.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test_only', '--test-only', action='store_true', default=False,
                    help='Disable training. Model will only be tested.')
    p.add_argument('--save_model_to', type=str, default='./snapshots/')
    p.add_argument('--patience', type=int, default=10,
                   help='Number of epochs to continue when test acc stagnants.')
    
    p.add_argument('--ignored_names', type=str, default=[], nargs='*',
                   help='List of names to be ignored when loading weights.')
    p.add_argument('--ignored_filter_names', type=str, default=[], nargs='*',
                   help='List of names to filter out names when loading weights.')
    p.add_argument('--freeze_names', type=str, default=[], nargs='*',
                   help='List of names to be frozen after weights are loaded.')
    p.add_argument('--freeze_filter_names', type=str, default=[], nargs='*',
                   help='List of names to filter the weights to be frozen.')
    
    p.add_argument('--comment', nargs='+')

    if user_param:
        p = user_param(p)

    if type(p) != argparse.ArgumentParser:
        raise ValueError(
            f'user_param must return an ArgumentParser object, found {type(p)} instead.')

    return p


def create_parser_yacs():
    """Create the basic argument parser for environment setup for YACS.

    Args:

    Return:
        p :argparse.ArgumentParser:
    """
    p = argparse.ArgumentParser(
        description='input arguments.',
        fromfile_prefix_chars='@')
    
    p.add_argument('-f', '--file', type=str, default=None,
                    help='Override with files.')
    p.add_argument('-c', '--cli', type=str, default=[], nargs='*',
                    help='Override with command line.')
    return p


def worker_init_fn_seed(args=None, provided_seed=None):
    import torch

    def worker_init_fn(x):
        if provided_seed is not None:
            seed = x + provided_seed
        else:
            seed = x + args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        return
    return worker_init_fn


def set_randomness(seed, pytorch_deterministic=True):
    import torch
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(pytorch_deterministic)
    except Exception:
        torch.set_deterministic(pytorch_deterministic)


def add_weight_decay(net, l2_value, include_filter=()):
    """Add weight decay to a subset of named parameters based on a filter.
    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        for include_filter_name in include_filter:
            if include_filter_name in name:
                decay.append(param)
                break
        else: no_decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def init(parse_list=None, user_param=None, user_args_modifier=None, pytorch_deterministic=True):
    """Parse and return arguments, prepare the environment. Set up logging.
    Args:
        user_param :callable: append user parameters to the argument parser.
        user_args_modifier :callable: override parsed arguments.
    Returns:
        args :Namespace: of arguments
    """
    import torch
    # Input arguments
    # For backward compatibility: YACS is disabled when DISABLE_YACS=1
    if os.getenv('DISABLE_YACS', '0') == '1':
        parser = create_parser(user_param)
        args = parser.parse_args(parse_list)
    else:
        from config.defaults import get_cfg_defaults
        print('YACS Enabled. Set DISABLE_YACS=1 to disable.')
        overrides = create_parser_yacs().parse_args()
        args = get_cfg_defaults()
        if overrides.file is not None:
            args.merge_from_file(overrides.file)
            args.config_path = overrides.file
        args.merge_from_list(overrides.cli)

    # Device options
    # args.cuda is deprecated
    args.cuda = torch.cuda.is_available() if args.cuda is None else args.cuda

    # hostname and gpu id
    hostname = subprocess.check_output(
        "hostname", shell=True).decode("utf-8")[:-1]
    git_branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('utf-8')[:-1]
    git_commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
    git_msg = subprocess.check_output(
        ['git', 'log', '--format=%B', '-n1', 'HEAD']).decode('utf-8')[:-2]
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES") if args.cuda else 'null'
    # date_time = subprocess.check_output(['date', '-Iminutes']).decode('utf-8')[:-7]
    date_time = datetime.today().strftime('%Y%m%d%H%M')

    args.git_commit = git_branch + '_' + git_commit
    if len(args.comment) == 0:
        args.comment = git_msg

    # Randomness control
    set_randomness(args.seed, pytorch_deterministic=pytorch_deterministic)

    # Model ID
    # if debug_mode flag is set, all logs will be saved to {save_model_to}/model_id folder,
    # otherwise will be saved to {save_model_to}/model_id folder
    if not args.debug_mode:
        if args.uuid is None: args.uuid = str(uuid.uuid4().hex)[:8]
        args.model_id = f'{args.uuid}_{git_branch}_{git_commit}_{hostname}_{gpu_id}_{os.getpid()}_{date_time}'
    else:
        args.model_id = f'debug_{git_branch}_{git_commit}_{hostname}_{gpu_id}_{os.getpid()}_{date_time}'

    # create model save path
    if not args.test_only:
        # create logger
        if args.log_dir is None:
            args.log_dir = osp.join(args.save_model_to, args.model_id)
        if args.summary_to is None:
            args.summary_to = osp.join(args.save_model_to, args.model_id, 'run')
        os.makedirs(args.log_dir, exist_ok=True)
        config_logger(osp.join(args.log_dir, args.model_id + '.log'))

        # log meta information
        logging.getLogger(myself()).info(
            f'Model {args.model_id}, running in {sys.argv[0]}, code revision {git_commit}')
        logging.getLogger(myself()).info(
            f"Assigned GPU {os.environ.get('CUDA_VISIBLE_DEVICES', None)}: {torch.cuda.get_device_name()}.")
        for arg in args.keys():
            logging.getLogger(myself()).debug(
                f'{arg:<30s} = {str(getattr(args, arg)):<30s}')
    else:
        print(f'Model {args.model_id}, executable {sys.argv[0]}')
        for arg in args.keys():
            print(f'{arg:<30s} = {str(getattr(args, arg)):<30s}')

    if user_args_modifier:
        args = user_args_modifier(args)

    if not args:
        raise ValueError('user_args_modifier must return arguments.')

    return args


# ---------- Debug use only ---------- #
if __name__ == '__main__':
    args = init(user_param=params)
