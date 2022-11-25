import json
import logging
import os
import os.path as osp
import resource
import shutil
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from distutils.dir_util import copy_tree
from typing import Dict
from datetime import timedelta, datetime

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import numpy as np
import torch
from lib.xutilities.eval import confusion_matrix, plot_confusion_matrix
from torch.cuda.amp import GradScaler, autocast
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .nn import Module
from . import utils
from .utils import (EarlyStop, RunningAverage, ShouldSaveModel,
                    apply_loss_weights, collect_layer_grad, count_params,
                    elapsed_timer, global_norm, load_checkpoint, myself,
                    plot_grad_flow, unlock_gpu, worker_init_fn_seed)

tqdm_commons = {'ncols': 100, 'ascii': True, 'leave': True}


# ------------------------------- #
# Additional CLI Parameters
# ------------------------------- #
def params(p):
    p.add_argument('--root', default='data')
    p.add_argument('--config_path', default='config/mmnist_cddc.txt')
    p.add_argument('--ds_name', default='NTURGBD')
    p.add_argument('--model_params', type=str, default='{}')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lr_decay', '--lr-decay', type=float, default=0.1)
    p.add_argument('--opt', type=str, default='RMSprop')
    p.add_argument('--opt_params', type=str, default='{}')
    p.add_argument('--ds_params', type=str, default='{}')
    p.add_argument('--lrsch_params', type=str, default='{}')
    p.add_argument('--gclipv', type=float, default=10)
    p.add_argument('--batch-size', '--batch_size', type=int, default=128)
    p.add_argument('--forward_batch_size', type=int, default=32)
    p.add_argument('--test-batch-size', '--test_batch_size', type=int,
        default=128)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--loss_weights', type=str, default='{}')
    p.add_argument('--acc_name', type=str, default='acc')

    p.add_argument('--num_workers', type=int, default=os.cpu_count())

    p.add_argument('--plot_cm', action='store_true', default=False)
    p.add_argument('--amp', type=int, default=0)
    p.add_argument('--debug_val', '--debug-val',
                   action='store_true', default=False)
    p.add_argument('--debug_runs', type=int, default=5)
    return p


class TVTMachinePrototype(metaclass=ABCMeta):
    """
    Train, validation, and test machine.
    This class should be inherited.
    Class methods not starting with "_" should be overriden.
    Other methods MIGHT just work fine.
    """
    def __init__(self, args):
        self.args = args

        self.summary_writer = SummaryWriter(args.summary_to)
        self.global_step = 0
        self.current_epoch = 0

        # ------------------------------- #
        # Process Arguments
        # ------------------------------- #
        if type(args.ds_params) is str:
            self.ds_params = json.loads(args.ds_params)
            self.model_params = json.loads(args.model_params)

            self.lrsch_params = json.loads(args.lrsch_params)
            self.loss_weights = json.loads(args.loss_weights)
        
        else:
            self.ds_params = args.ds_params
            self.model_params = args.model_params

            self.lrsch_params = args.lrsch_params
            self.loss_weights = args.loss_weights

        self.amp = bool(self.args.amp)
        self.device = args.device
        self.parallel = args.parallel
        self.device_list = [int(i) for i in args.device_list]
        if bool(self.parallel):
            self.device = 'cuda:' + str(self.device_list[0])

        self.debug_mode = bool(args.debug_mode)

        # ------------------------------- #
        # Initialize Components
        # ------------------------------- #
        self.dataloaders = self.create_dataloader()
        self.model = self.create_model()
        self.optimizer = self._create_optimizer()
        self.lr_schdlr = self._create_lrsch()
        self.scaler = GradScaler()
        self.ss = ShouldSaveModel(init_step=args.start_epoch-1)
        self.es = EarlyStop(patience=args.patience)
        self.loss_fn = self.create_loss_fn()

        # ------------------------------- #
        # Load Component States (optional)
        # ------------------------------- #
        if args.load_weight_from or args.from_model:
            _state_dict = load_checkpoint(
                args.from_model if args.from_model else args.load_weight_from,
                map_location=self.device)
            self._load_model(_state_dict)
            if args.from_model:
                self._load_states(_state_dict)
        self._freeze_model()
        
        # ------------------------------- #
        # Handle GPU Parallelization
        # ------------------------------- #
        if args.parallel:
            self._parallelization()
        
        # ------------------------------- #
        # Time Recorder
        # ------------------------------- #
        self.time_record = RunningAverage(3)

    @abstractmethod
    def create_dataloader(self) -> Dict[str, DataLoader]:
        """
        Returns: a dictionary with keys like 'train', 'val', 'test', or as desired.
        Example:
        ```
        if self.args.ds_name == 'nturgbd':
            ds_class = NTURGBD
            ds_root = self.args.root+'/nturgbd/processed/'
        elif self.args.ds_name == 'kinetics-skeleton':
            ds_class = KineticsSkeleton
            ds_root = self.args.root+'/kinetics-skeleton/processed/'
        else:
            ds_class = None
            raise NotImplementedError(f'{self.args.ds_name} not supported.')
        train_dataset = ds_class(root=ds_root, split='train', debug=self.debug_mode, **self.dataset_params)
        test_dataset = ds_class(root=ds_root, split='test', debug=self.debug_mode, **self.dataset_params)
    
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=True)

        test_dataloader  = DataLoader(
            test_dataset, batch_size=self.args.test_batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=False)

        dataloaders = {'train': train_dataloader, 'test': test_dataloader}

        ```
        """
        pass

    @abstractmethod
    def create_model(self) -> Module:
        """
        Example:
        ```
        model = Model(**self.model_params).to(self.device)
        logging.getLogger(myself()).info(
            f'Number of params: {count_params(model)}.')
        return model
        ```
        """
        pass

    @abstractmethod
    def create_loss_fn(self):
        pass
    
    def _create_optimizer(self):
        model_parameters = self.model.parameters()
        optimizer = utils.get_callable('.'.join(['torch', 'optim', self.args.opt]))(
            model_parameters, **self.args.opt_params.kwargs)
        # if self.args.opt == 'SGD':
        #     optimizer = torch.optim.SGD(
        #         model_parameters, **self.args.opt_params.kwargs)
        # elif self.args.opt == 'Adam':
        #     # optimizer = torch.optim.Adam(
        #     #     model_parameters,
        #     #     # eps=1e-2 / args.batch_size ** 2,
        #     #     lr=self.args.lr, weight_decay=self.args.weight_decay)
        #     optimizer = torch.optim.Adam(
        #         model_parameters, **self.args.opt_params.kwargs
        #     )
        # elif self.args.opt == 'RMSprop':
        #     optimizer = torch.optim.RMSprop(
        #         model_parameters,
        #         momentum=0.9,
        #         # eps=1e-2 / args.batch_size ** 2,
        #         lr=self.args.lr, weight_decay=self.args.weight_decay)
        # else:
        #     raise NotImplementedError(f'Optimizer {self.args.opt} unsupported.')
        return optimizer

    def _create_lrsch(self):
        lr_schdlr = utils.get_callable('.'.join(['torch', 'optim', 'lr_scheduler',  self.args.lrsch]))(self.optimizer, *self.lrsch_params.args, **self.lrsch_params.kwargs)
        return lr_schdlr
    
    def _save_checkpoint(self, path):
        # special treatment for model weights because DataParallel wraps it into
        # model.module.
        model_state_dict = self.model.state_dict()
        model_weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in model_state_dict.items()
        ])
        states_dict = {
            'model': model_weights,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_schdlr.state_dict(),
            'grad_scaler': self.scaler.state_dict(),
            'checkpoint_epoch': self.current_epoch,
            'initial_lr': self.args.opt_params.kwargs.lr,
            'global_step': self.global_step,
            'shouldsave': self.ss.state_dict(),
            'earlystop': self.es.state_dict(),
        }
        torch.save(states_dict, path)

    def _load_model(self, state_dict, no_ignore=False):
        model_weights = state_dict['model']
        model_state_dict = OrderedDict(
                [[k.split('module.')[-1],
                  v.to(self.device)] for k, v in model_weights.items()])
        self.model = self.model.to(self.device)
        try:
            if no_ignore:  # in post-training test, no module should be ignored
                ignored_n = []
                ignored_ft = []
            else:
                ignored_n = self.args.ignored_names
                ignored_ft = self.args.ignored_filter_names
            self.model.load_my_state_dict(
                model_state_dict,
                ignored_names=ignored_n, ignored_filter_names=ignored_ft)
        except torch.nn.modules.module.ModuleAttributeError:
            self.model.module.load_my_state_dict(model_state_dict)
    
    def _freeze_model(self):
        # partially freeze
        for n, p in self.model.named_parameters():
            if n in self.args.freeze_names:
                p.requires_grad = False
                logging.getLogger(myself()).warning(f'Frozen: {n}.')
            for ft in self.args.freeze_filter_names:
                if ft in n:
                    p.requires_grad = False
                    logging.getLogger(myself()).warning(f'Frozen: {n}.')
                    break

    def _load_states(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_schdlr.load_state_dict(state_dict['lr_scheduler'])
        self.scaler.load_state_dict(state_dict['grad_scaler'])
        self.global_step = state_dict['global_step']
        self.current_epoch = state_dict['checkpoint_epoch'] + 1
        self.ss.load_state_dict(state_dict['shouldsave'])
        self.es.load_state_dict(state_dict['earlystop'])

    def _parallelization(self):
        self.model = torch.nn.DataParallel(
            self.model,
            device_ids=self.device_list,
            output_device=self.device_list[0])
    
    def train_epoch(self):
        self.model.train()

        # train_loss_avg = RunningAverage(len(self.dataloaders['train']))
        t = tqdm(
            self.dataloaders['train'], disable=(not self.args.pbar), **tqdm_commons)
        
        all_loss_values = {}
        for batch_id, data in enumerate(t):
            
            with elapsed_timer() as elapsed:
                loss_values = self.train_batch(data)
            batch_time = elapsed()
            
            for k in loss_values.keys():
                if k not in all_loss_values: all_loss_values[k] = []
                all_loss_values[k].append(loss_values[k])
            # train_info.update({'loss': train_loss_avg.add(np.mean(loss_values['loss']))})

            self.global_step += 1

            if batch_id == (self.args.debug_runs-1):
                logging.getLogger(myself()).info(f'At epoch {self.current_epoch}, batch_id {batch_id}: '
                    f'alloc_mem={torch.cuda.memory_stats()["allocated_bytes.all.peak"]}, '
                    f'res_mem={torch.cuda.memory_stats()["reserved_bytes.all.peak"]}. '
                    f'ETA (this epoch): {batch_time * (len(t)-6):.1f}s.')
                if self.debug_mode:
                    plot_grad_flow(self.model)
                    break
        
        train_info = {'phase': 'train'}
        for k in all_loss_values:
            train_info.update({k: np.mean(all_loss_values[k])})
        return train_info
    
    def _gradient_accumulate_splits(self, batch_size):
        real_batch_size = self.args.forward_batch_size
        if real_batch_size <= batch_size:
            splits = batch_size // real_batch_size
            assert batch_size % real_batch_size == 0, \
                f'Forward batch size {real_batch_size} should be a factor '\
                f'of arg.batch_size {batch_size}!'
        # In very rare cases, the real batch size will be larger than batch size
        # because e.g. augmented samples for contrastive learning must be passed
        # with the original samples. In this case we just disable splitting.
        else:
            splits = 1
        return splits
    
    def train_batch(self, data):
        X, y = data
        X, y = X.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()

        # ####### Gradient Accumulation for Smaller Batches ####### #
        splits = self._gradient_accumulate_splits(len(X))
        
        loss_values = []
        with torch.autograd.set_detect_anomaly(self.debug_mode):
            for j in range(splits):
                left = j * self.args.forward_batch_size
                right = left + self.args.forward_batch_size
                Xb, yb = X[left:right], y[left:right]

                with autocast(self.amp):
                    y_ = self.model(Xb)
                    losses, lsw = self.loss_fn(y_, yb.long(), adjs=None)

                if self.global_step % 50 == 0:
                    for k in losses.keys():
                        self.summary_writer.add_scalar(
                            f'loss/{k}', losses[k].item(),
                            global_step=self.global_step)
                
                if self.global_step % 100 == 0:
                    self.summary_writer.add_scalar(
                        'loss/grad_norm', global_norm(self.model),
                        global_step=self.global_step)
                    layer_avg_grad, _ = collect_layer_grad(
                        self.model.named_parameters())
                    for k, v in layer_avg_grad.items():
                        self.summary_writer.add_scalar(
                            f'grad/norm_{k}', v, global_step=self.global_step)
                self.summary_writer.flush()
            
                loss, losses = apply_loss_weights(losses, self.loss_weights)
                loss_values.append(loss.item())

                if loss != loss:
                    raise ValueError('Loss goes to nan.')
            
                if self.amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
            if self.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        unlock_gpu()
        del y_, loss, losses

        return loss_values
    
    def valtest(self, phase, data_loader, topk=5):
        self.model = self.model.to(self.device)
        self.model.eval()
        t = tqdm(data_loader, disable=(not self.args.pbar), **tqdm_commons)

        preds = []
        preds_topk = []
        labels = []
        loss_values = []

        tested_total = 0
        with torch.no_grad():
            for batch_id, data in enumerate(t):
                X, y = data
                B, C, T, nJ, nB = X.shape

                tested_total += X.shape[0]
                X, y = X.to(self.device), y.to(self.device).long()
                y_ = self.model(X)

                losses, lsw = self.loss_fn(y_, y.long(), adjs=None)
                loss, _ = apply_loss_weights(losses, self.loss_weights)
                loss_values.append(loss.item())

                preds.append(y_['cls_lgts'].argmax(-1))
                preds_topk.append(torch.topk(y_['cls_lgts'], topk, -1).indices)
                labels.append(y.long())

                if self.debug_mode and batch_id == self.args.debug_runs:
                    break

        loss = np.mean(loss_values)

        results = {}
        labels = torch.cat(labels, 0)
        preds = torch.cat(preds, 0).reshape(labels.shape)
        results['acc'] = float((preds == labels).sum()) / labels.shape[0]
        preds_topk = torch.cat(preds_topk, 0)
        results['acc_topk'] = float((preds_topk == labels[:, None]).sum()) / labels.shape[0]

        self.summary_writer.add_scalar(
            f'acc/{phase}/acc', results['acc'],
            global_step=self.global_step)
        self.summary_writer.add_scalar(
            f'acc/{phase}/acc_topk{topk}', results['acc'],
            global_step=self.global_step)

        self.summary_writer.flush()

        if self.args.plot_cm:
            num_classes = data_loader.dataset.num_classes
            cm = confusion_matrix(
                    preds, labels, num_classes=num_classes, normalize=True)
            if not self.args.test_only:
                cm_path = f'{self.args.save_model_to}/{self.args.model_id}/cm.pdf'
            else:
                cm_path = './cm.tmp.pdf'
            plot_confusion_matrix(
                cm_path, cm, labels=list(range(num_classes)), figsize=(30,30))

        test_info = {
            'phase': phase,
            'acc': results['acc'],
            f'acc_top{topk}': results['acc_topk'],
            'test_loss': loss}

        torch.cuda.empty_cache()
        return test_info
    
    def copy_source(self, main_file=None):
        """Backup `main_file`, `config_path`, and the whole `lib` folder."""
        shutil.copy(
            osp.join('.', main_file),
            osp.join(self.args.save_model_to, self.args.model_id, main_file))
        if os.getenv('DISABLE_YACS', 0) == '1':
            os.makedirs(osp.join(self.args.save_model_to, self.args.model_id,
                    *self.args.config_path.split('/')[:-1]))
            shutil.copy(
                osp.join('.', self.args.config_path),
                osp.join(self.args.save_model_to, self.args.model_id,
                    *self.args.config_path.split('/')))
        else:
            self.args.dump(
                stream=open(osp.join(
                    self.args.save_model_to, self.args.model_id, 'config.yaml'),
                    'w')
            )
        copy_tree(
            src='lib',
            dst=osp.join(self.args.save_model_to, self.args.model_id, 'lib'))
    
    def run(self):
        if self.args.test_only:
            best_epoch = self.current_epoch - 1
            test_info = self.valtest('test', self.dataloaders['test'])
            print(f"Best model at epoch {best_epoch}, {test_info}")
        else:
            try:
                for _ in range(self.current_epoch, self.args.max_epoch):
                    logging.getLogger(myself()).info(
                        "*"*10 + f" Epoch {self.current_epoch} starts. " + "*"*10)
                    with elapsed_timer() as elapsed:
                        # ------------------------------- #
                        # Train an epoch
                        # ------------------------------- #
                        train_info = self.train_epoch()
                        logging.getLogger(myself()).info(
                            f"Epoch {self.current_epoch}, {train_info}.")

                        # ------------------------------- #
                        # Validation
                        # ------------------------------- #
                        val_info = self.valtest('val', self.dataloaders['val'])
                        logging.getLogger(myself()).info(
                            f"Epoch {self.current_epoch}, {val_info}")

                        if self.ss.step(
                            loss=None, acc=val_info[self.args.acc_name], criterion=lambda x1, x2: x2):
                            self._save_checkpoint(
                                f'{self.args.save_model_to}/{self.args.model_id}/best.state')
                        self._save_checkpoint(
                            f'{self.args.save_model_to}/{self.args.model_id}/latest.state')
                        
                        if type(self.lr_schdlr) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                            self.lr_schdlr.step(train_info['loss'])
                        else:
                            self.lr_schdlr.step()
                    
                    # ------------------------------- #
                    # Estimate Time
                    # ------------------------------- #
                    epoch_time = self.time_record.add(elapsed())
                    eta_total = epoch_time*(self.args.max_epoch-self.current_epoch-1)
                    eta_time = (datetime.now() + timedelta(seconds=eta_total)).strftime('%Y-%m-%d, %H:%M:%S')
                    logging.getLogger(myself()).info(
                            f"Epoch {self.current_epoch} finished. "
                            f"Elapsed={epoch_time:.1f}s. "
                            f"ETA (all epochs): {eta_time}.")

                    if self.es.step(
                        loss=None, acc=val_info[self.args.acc_name],
                        criterion=lambda x1, x2: x2): break

                    if self.debug_mode: break

                    self.current_epoch += 1
            except ValueError as e:
                logging.getLogger(myself()).error(f'{e}')
                logging.getLogger(myself()).error('Premature training end.')
            finally:
                # ------------------------------- #
                # Post Training Procedures
                # ------------------------------- #
                logging.getLogger(myself()).info('Training ended.')
                state_dict = load_checkpoint(
                    f'{self.args.save_model_to}/{self.args.model_id}/best.state',
                    state_dict_to_load=['model', 'checkpoint_epoch']
                )
                self._load_model(state_dict, no_ignore=True)
                self._load_states(state_dict)
                
                test_info = self.valtest('test', self.dataloaders['test'])
                logging.getLogger(myself()).info(
                        f"Best model at epoch {self.current_epoch-1}, {test_info}")
                test_info.update({'Epoch': self.current_epoch-1})
                with open(osp.join(self.args.save_model_to, self.args.model_id, 'best_perf.json'), 'w') as f:
                    json.dump(test_info, f)
                if self.args.debug_mode:
                    print(torch.cuda.memory_summary())
        
        self.summary_writer.close()
        logging.getLogger(myself()).info(
            f"Operations on {self.args.model_id} completed.")
