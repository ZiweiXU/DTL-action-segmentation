import os

from lib.xutilities.utils import wait_gpu, obtain_gpu, unlock_gpu, release_gpu
if os.environ.get('NO_GPU_WAIT', '0') != '1':
    req_mem = int(os.environ.get('REQ_M', '6000'))
    req_cards = int(os.environ.get('REQ_C', '1'))
    allowed_collisions = int(os.environ.get('ALLOW_COLL', '2'))
    wait_gpu(req_mem=req_mem, req_cards=req_cards, allowed_collisions=allowed_collisions)
    obtain_gpu(lockname='.gpu.usage.json')

from copy import deepcopy
import json
import logging
import os
import os.path as osp
import sys
from typing import Dict
from functools import partial
from typing import Optional
from datetime import timedelta, datetime
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from lib.tl.parser import parse
from lib.eval.metrics import overlap_f1, accuracy, edit_score
from lib.eval.mlad import get_all_conditional_metrics
from lib.xutilities.engine import TVTMachinePrototype, tqdm_commons
from lib.xutilities.utils import init, worker_init_fn_seed, apply_loss_weights,\
    myself, count_params, get_callable, load_checkpoint, elapsed_timer

from lib.transformation import TempDownSamp, ToTensor


class TVTMachine(TVTMachinePrototype):
    def __init__(self, args, **kwargs):
        super(TVTMachine, self).__init__(args)
        self.lg_refiner = kwargs.get('lg_refiner', None)
        self.lg_eva = kwargs.get('lg_eva', None)

        if not self.args.test_only:
            self.copy_source(main_file=osp.basename(__file__))
        
        # sibling optimizer for logic
        lg_opt_params = deepcopy(self.args.opt_params.kwargs)
        # lg_opt_params['weight_decay'] = 0.0
        # self.lg_optimizer = get_callable('.'.join(['torch', 'optim', 'SGD']))(self.model.parameters(), lr=lg_opt_params['lr'])
        self.lg_optimizer = get_callable('.'.join(['torch', 'optim', self.args.opt]))(self.model.parameters(), **lg_opt_params)
    
    def create_dataloader(self) -> Dict[str, DataLoader]:
        dataset_class = get_callable(self.args.ds_name)

        train_dataset = dataset_class(
            transform=Compose([ToTensor(), TempDownSamp(self.args.extra_kwargs.sample_rate)]),
            **self.ds_params['commons'], **self.ds_params['train'])
        val_dataset = dataset_class(
            transform=Compose([ToTensor()]),
            **self.ds_params['commons'], **self.ds_params['val'])
        test_dataset = dataset_class(
            transform=Compose([ToTensor()]),
            **self.ds_params['commons'], **self.ds_params['test'])

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=True)
        val_dataloader  = DataLoader(
            val_dataset, batch_size=self.args.test_batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=False)
        test_dataloader  = DataLoader(
            test_dataset, batch_size=self.args.test_batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn_seed(self.args), drop_last=False)

        return {
            'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader
        }
    
    def create_model(self):
        model_class = get_callable(self.args.model_name)

        model = model_class(**self.model_params).to(self.device)
        logging.getLogger(myself()).info(
            f'Number of params: {count_params(model)}.')
        return model
    
    def create_loss_fn(self):
        return get_callable(self.args.loss_func)
    
    def __collect_gradient(self, model=None):
        named_grad = {}
        model = self.model if model is None else model
        for k, p in model.named_parameters():
            if p.grad is not None:
                named_grad[k] = p.grad.clone()
        return named_grad
    
    def __apply_gradient(self, gradient_dict, model=None):
        """Apply gradient in gradient_dict to model through ADDITION."""
        model = self.model if model is None else model
        for k, p in self.model.named_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if k in gradient_dict:
                p.grad += gradient_dict[k]
    
    def train_batch(self, data):
        meta_enabled = self.model_params.get('meta_enabled', False)
        X, Y, mask = data['feature'], data['label'], None
        if 'batch_gen' in self.args.ds_name:
            mask = data['mask']
            mask = mask.to(self.device)
        
        self.optimizer.zero_grad()
        loss_values = {}

        with torch.autograd.set_detect_anomaly(self.debug_mode):
            Xb, Yb = X, Y
            Xb = Xb.to(self.device).float()
            Yb = Yb.to(self.device).long()
            Ymb = None
            if meta_enabled:
                Ymb = data['meta_target'].to(self.device).long()

            with autocast(self.amp):
                out = self.model(Xb, mask)
                Yb_ = out['output']

                Ym_ = None
                if meta_enabled:
                    Ym_ = out['meta_output']
                losses, _ = self.loss_fn(y=Yb_, ym=Ym_, Y=Yb, Ym=Ymb, loss_weights=self.loss_weights, mask=mask, lg_eva=self.lg_eva, machine=self)
            
            for k in losses.keys():
                loss_values[k] = float(losses[k])

            for k in losses.keys():
                self.summary_writer.add_scalar(
                    f'loss/{k}', float(losses[k]),
                    global_step=self.global_step)

            loss, losses = apply_loss_weights(losses, self.loss_weights)
            loss_values['loss'] = float(loss)

            if loss != loss:
                raise ValueError('Loss goes to nan.')

            # control logic loss
            if losses.get('lg', 0) > 0:
                if self.args.extra_kwargs.get('adaTL', False):
                    raise NotImplementedError
                else:  # non-adaTL
                    # Collect task gradient
                    task_losses = 0
                    for k, v in losses.items():
                        if k != 'lg': task_losses += v
                    self.model.zero_grad()
                    task_losses.backward(retain_graph=True)
                    task_gradient = self.__collect_gradient()

                    # Collect lg gradient
                    self.model.zero_grad()
                    losses['lg'].backward()
                    torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), self.args.extra_kwargs.tl_g_clipv)
                    lg_gradient = self.__collect_gradient()

                    # apply gradient
                    self.model.zero_grad()
                    self.__apply_gradient(task_gradient)
                    self.__apply_gradient(lg_gradient)

                    self.optimizer.step()
            else:
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
        del Yb_, loss, losses

        return loss_values
    
    def valtest(self, phase, data_loader, topk=1):
        self.model = self.model.to(self.device)
        self.model.eval()
        t = tqdm(data_loader, disable=(not self.args.pbar), **tqdm_commons)
        sample_rate = self.args.extra_kwargs.sample_rate
        
        loss_values = []
        lg_loss_values = []

        preds = []
        labels = []

        tested_total = 0
        with torch.no_grad():
            for batch_id, data in enumerate(t):
                X, Y, Ymask, _ = data['feature'], data['label'], None, None
                labels.append(Y.squeeze(0))
                
                if 'batch_gen' in self.args.ds_name:
                    Ymask = data['mask']
                    Ymask = Ymask.to(self.device)[..., ::sample_rate]

                X = X.to(self.device).float()
                Y = Y.to(self.device).long()

                tested_total += X.shape[0]

                # downsample X by sample_rate
                X = X[..., ::sample_rate]
                with torch.no_grad():
                    out = self.model(X, mask=torch.ones_like(X))
                Y_ = out['output']

                # set 'lg' to 0 to disable TL during evaluation
                _val_loss_weight = {k: v for k, v in self.loss_weights.items() if k != 'lg'}
                losses, _ = self.loss_fn(
                    y=Y_, ym=None, Y=Y[..., ::sample_rate], Ym=None,
                    loss_weights=_val_loss_weight, mask=Ymask, lg_eva=self.lg_eva, machine=self)
                loss, _ = apply_loss_weights(losses, self.loss_weights)
                loss_values.append(loss.item())

                # if multi-stage, pick last stage
                if 'mstcn' in self.args.model_name or 'ASFormer' in self.args.model_name:
                    Y_ = Y_[-1]
                # upsample Y_ by sample_rate
                Y_ = F.interpolate(Y_, size=Y.shape[1], mode='nearest')

                preds.append(Y_.argmax(1).squeeze(0).cpu())

                if self.debug_mode and batch_id == (self.args.debug_runs-1):
                    break

        loss = np.mean(loss_values)
        results = {}

        # action: standard metrics
        results['f1_10'] = overlap_f1(
            preds, labels, n_classes=self.args.extra_kwargs.num_classes, bg_classes=None, overlap=0.1)
        results['f1_25'] = overlap_f1(
            preds, labels, n_classes=self.args.extra_kwargs.num_classes, bg_classes=None, overlap=0.25)
        results['f1_50'] = overlap_f1(
            preds, labels, n_classes=self.args.extra_kwargs.num_classes, bg_classes=None, overlap=0.50)
        results['f_acc'] = accuracy(preds, labels).item()
        results['edit'] = edit_score(preds, labels)
        
        # action: time aware metrics
        preds_one_hot = [F.one_hot(i, num_classes=self.args.extra_kwargs.num_classes) for i in preds]
        labels_one_hot = [F.one_hot(i, num_classes=self.args.extra_kwargs.num_classes) for i in labels]
        prec500, re500, map500, fs500, _ = get_all_conditional_metrics(preds_one_hot, labels_one_hot, t=5000)
        results['prec500'], results['re500'], results['map500'], results['fs500'] = prec500, re500, map500, fs500
        
        # total score
        results['total_score'] = results['f1_10'] + results['f1_25'] + results['f1_50'] + results['f_acc'] + results['edit']
        results['test_loss'] = float(loss)

        for k in results.keys():
            self.summary_writer.add_scalar(
                f'metrics/{phase}/{k}', results[k],
                global_step=self.global_step)

        self.summary_writer.flush()

        test_info = {'phase': phase}
        test_info.update(results)

        torch.cuda.empty_cache()
            
        return test_info
    
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
    
    def _load_states(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_schdlr.load_state_dict(state_dict['lr_scheduler'])
        self.scaler.load_state_dict(state_dict['grad_scaler'])
        self.global_step = state_dict['global_step']
        self.current_epoch = state_dict['checkpoint_epoch'] + 1
        self.ss.load_state_dict(state_dict['shouldsave'])
        self.es.load_state_dict(state_dict['earlystop'])
    
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
                        if (self.current_epoch % self.args.val_every == 0) or (self.current_epoch == self.args.max_epoch - 1):
                            val_info = self.valtest('val', self.dataloaders['val'])
                            logging.getLogger(myself()).info(
                                f"Epoch {self.current_epoch}, {val_info}")

                            if self.ss.step(
                                current_step=self.current_epoch,
                                loss=None, acc=val_info[self.args.acc_name], criterion=lambda x1, x2: x2):
                                self._save_checkpoint(
                                    f'{self.args.save_model_to}/{self.args.model_id}/best.state')
                            self._save_checkpoint(
                                f'{self.args.save_model_to}/{self.args.model_id}/latest.state')
                            
                            if self.es.step(
                                loss=None, acc=val_info[self.args.acc_name],
                                criterion=lambda x1, x2: x2): break
                        
                        if type(self.lr_schdlr) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                            self.lr_schdlr.step(train_info['ce'] + train_info.get('sm', 0.0))
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

                    if self.debug_mode: break

                    self.current_epoch += 1
            except ValueError as e:
                logging.getLogger(myself()).error(f'{e}')
                logging.error('Premature ending. Test with the last best model.')
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


if __name__ == '__main__':
    import warnings; warnings.simplefilter("ignore")
    try:
        # dirty initialisation jobs
        args = init(pytorch_deterministic=True)
        
        # config logic evaluator, if extra_kwargs.rule_path is specified
        lg_eva :Optional[partial] = None
        if 'rule_path' in args.extra_kwargs:
            # pre-process lg rules
            SUBCLASSES = [line.split()[1] for line in open(args.extra_kwargs.mapping_path).readlines()]
            _classes = list(map(str.lower, SUBCLASSES))
            if args.model_params.get('meta_enabled', False):
                METACLASSES = [line.split()[1] for line in open(args.extra_kwargs.meta_mapping_path).readlines()]
                _classes += list(map(str.lower, METACLASSES))
            with open(args.extra_kwargs.rule_path) as f:
                rule_expr = f.read()
            sys.setrecursionlimit(10000)
            rule_eva = parse(rule_expr)
            ap_map = lambda x: _classes.index(x)
            lg_eva = partial(rule_eva, ap_map=ap_map, rho=args.extra_kwargs.rho)
    
        # go
        TVTMachine(args, lg_eva=lg_eva).run()

    finally:
        if os.environ.get('NO_GPU_WAIT', '0') != '1':
            release_gpu(lockname='.gpu.usage.json')
