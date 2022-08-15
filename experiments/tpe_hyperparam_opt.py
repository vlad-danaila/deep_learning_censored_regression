import math

import optuna
import torch as t
import logging
from experiments.constants import ABS_ERR
from deep_tobit.loss import Reparametrized_Scaled_Tobit_Loss, Scaled_Tobit_Loss, \
    Heteroscedastic_Reparametrized_Scaled_Tobit_Loss, Heteroscedastic_Scaled_Tobit_Loss
from deep_tobit.util import distinguish_censored_versus_observed_data
from experiments.synthetic.models import DenseNetwork, get_scale_network
from experiments.util import get_scale, get_device, dump_json
from experiments.synthetic.plot import plot_epochs
from experiments.train import train_network_mae_mse_gll, train_network_tobit_fixed_std, train_network_tobit_dyn_std
from experiments.constants import NB_TRIALS, TPE_STARTUP_TRIALS, SEED
from os.path import join
import os

PREVIOUS_BEST = 'PREVIOUS_BEST'
CHECKPOINT = 'CHECKPOINT'

def propose_conf(trial: optuna.trial.Trial):
    return {
        'max_lr': trial.suggest_uniform('max_lr', 1e-5, 1e-2),
        'epochs': trial.suggest_int('epochs', 5, 20),
        'batch': trial.suggest_int('batch', 32, 512),
        'pct_start': 0.45,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': trial.suggest_int('div_factor', 2, 10),
        'final_div_factor': 1e4,
        'weight_decay': 0
    }

def save_checkpoint_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    prev_best = study.user_attrs.get(PREVIOUS_BEST)
    checkpoint = study.user_attrs.get(CHECKPOINT)
    checkpoint_file = checkpoint + '.tar'
    best_checkpoint_file = checkpoint + ' best.tar'
    if trial.value < prev_best:
        study.set_user_attr(PREVIOUS_BEST, trial.value)
        if os.path.exists(best_checkpoint_file):
            os.remove(best_checkpoint_file)
        os.rename(checkpoint_file, best_checkpoint_file)

def tpe_opt_hyperparam(root_folder, checkpoint, train_callback):
    # TODO add prunner, don't forget to include in optuna.create_study(..., pruner = pruner) or just raise optuna.TrialPruned()
    study_name = f'study {checkpoint}'
    sampler = optuna.samplers.TPESampler(multivariate = True, n_startup_trials = TPE_STARTUP_TRIALS, seed=SEED)
    study = optuna.create_study(sampler=sampler, study_name = study_name, direction = 'minimize',
                                storage = f'sqlite:///{root_folder}/{checkpoint}.db', load_if_exists = True)
    study.set_user_attr(PREVIOUS_BEST, math.inf)
    study.set_user_attr(CHECKPOINT, f'{root_folder}/{checkpoint}')
    study.optimize(train_callback, n_trials = NB_TRIALS, callbacks=[save_checkpoint_callback])
    logging.info(study.best_params)
    dump_json(study.best_params, f'{root_folder}/{checkpoint} hyperparam.json')

def get_objective_fn_mae_mse(dataset_train, dataset_val, bound_min, bound_max, checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True):
    def objective_fn(trial: optuna.trial.Trial):
        conf = propose_conf(trial)
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = True, num_workers = 0)
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
        loss_fn = criterion()
        params = model.parameters()
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_mae_mse_gll(bound_min, bound_max,
                model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best[ABS_ERR]
    return objective_fn

def get_objective_fn_gll(dataset_train, dataset_val, bound_min, bound_max, checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True):
    def objective_fn(trial: optuna.trial.Trial):
        conf = propose_conf(trial)
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = True, num_workers = 0)
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
        scale = get_scale() # could be sigma or gamma
        loss_fn = criterion(scale)
        params = [
            {'params': model.parameters()},
            {'params': scale}
        ]
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_mae_mse_gll(bound_min, bound_max,
                model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best[ABS_ERR]
    return objective_fn

def get_objective_fn_tobit_fixed_std(dataset_train, dataset_val, bound_min, bound_max, checkpoint, model_fn = DenseNetwork, plot = False, log = True, truncated_low = None, truncated_high = None, isReparam = False):
    def objective_fn(trial: optuna.trial.Trial):
        conf = propose_conf(trial)
        censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
        model = model_fn()
        loader_train = t.utils.data.DataLoader(dataset_train, batch_size = conf['batch'], shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
        loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
        scale = get_scale()
        if isReparam:
            loss_fn = Reparametrized_Scaled_Tobit_Loss(scale, get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
        else:
            loss_fn = Scaled_Tobit_Loss(scale, get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
        params = [
            {'params': model.parameters()},
            {'params': scale}
        ]
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_tobit_fixed_std(bound_min, bound_max,
                                                                         model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best[ABS_ERR]
    return objective_fn

def get_objective_fn_tobit_dyn_std(dataset_train, dataset_val, bound_min, bound_max, checkpoint, model_fn = DenseNetwork, scale_model_fn = get_scale_network, plot = False, log = True,
                                     truncated_low = None, truncated_high = None, is_reparam = False):
    def objective_fn(trial: optuna.trial.Trial):
        conf = propose_conf(trial)
        censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
        loader_train = t.utils.data.DataLoader(dataset_train, batch_size = conf['batch'], shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
        loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
        model = model_fn()
        scale_model = scale_model_fn()
        if is_reparam:
            loss_fn = Heteroscedastic_Reparametrized_Scaled_Tobit_Loss(get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
        else:
            loss_fn = Heteroscedastic_Scaled_Tobit_Loss(get_device(), truncated_low = truncated_low, truncated_high = truncated_high)
        params = [
            {'params': model.parameters()},
            {'params': scale_model.parameters()}
        ]
        optimizer = t.optim.SGD(params, lr = conf['max_lr'] / conf['div_factor'], momentum = conf['max_momentum'], weight_decay = conf['weight_decay'])
        scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = conf['max_lr'],
            steps_per_epoch = len(loader_train),
            epochs = conf['epochs'],
            pct_start = conf['pct_start'],
            anneal_strategy = conf['anneal_strategy'],
            base_momentum = conf['base_momentum'],
            max_momentum = conf['max_momentum'],
            div_factor = conf['div_factor'],
            final_div_factor = conf['final_div_factor']
        )
        train_metrics, val_metrics, best = train_network_tobit_dyn_std(bound_min, bound_max,
                                                                       model, scale_model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'],
                                                                       len(dataset_val), conf['epochs'], grad_clip = conf['grad_clip'], log = log, is_reparam=is_reparam)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best[ABS_ERR]
    return objective_fn