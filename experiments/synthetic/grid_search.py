import sys
import torch as t
from deep_tobit.util import to_torch, to_numpy, normalize, unnormalize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import random
import numpy as np
import sklearn as sk
import sklearn.metrics
import math
from sklearn.model_selection import ParameterGrid
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
import os
import numpy.random
import collections
from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.models import DenseNetwork
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.train import train_network, eval_network
from experiments.synthetic.plot import *

"""# Grid Search Setup"""

def grid_search(grid_config, train_callback, checkpoint_name, nb_iterations = 1, conf_validation = None):
    configs = ParameterGrid(grid_config)
    configs_len = len(configs)
    counter = 0
    checkpoint_file = checkpoint_name + '.tar'
    grid_checkpoint_file = 'grid ' + checkpoint_file
    try:
        resume_grid_search = t.load(GRID_RESULTS_FILE)
    except FileNotFoundError:
        resume_grid_search = None

    results = {}
    best = [math.inf, math.inf, -math.inf]
    if resume_grid_search is not None and 'best' in resume_grid_search:
        best_conf = resume_grid_search['best']
        print('Best previous configuration', best_conf)
        best = resume_grid_search[str(best_conf)]
        print(f'Best previous metrics abs err = {best[ABS_ERR]}, R2 = {best[R_SQUARED]}')
        results = resume_grid_search

    for conf in ParameterGrid(grid_config):
        counter += 1

        if resume_grid_search is not None and str(conf) in resume_grid_search:
            print('Allready evaluated configuration', conf)
            continue

        if not conf_validation(conf):
            print('Skipping over configuration', conf)
            results[str(conf)] = 'invalid'
            continue

        print('-' * 5, 'grid search {}/{}'.format(counter, configs_len), '-' * 5)
        print('Config:', conf)

        best_from_iterations = [math.inf, math.inf, -math.inf]

        for i in range(nb_iterations):
            if nb_iterations != 1:
                print('Iteration', i + 1)
            metrics = train_callback(conf)

            # if metrics[R_SQUARED] > best[R_SQUARED]:
            if metrics[ABS_ERR] < best[ABS_ERR]:
                best_from_iterations = metrics

            # if metrics[R_SQUARED] > best[R_SQUARED]:
            if metrics[ABS_ERR] < best[ABS_ERR] and not (math.isnan(metrics[LOSS] or math.isnan(metrics[ABS_ERR]) or math.isnan(metrics[R_SQUARED]))):
                best = metrics
                results['best'] = conf
                if os.path.exists(grid_checkpoint_file):
                    os.remove(grid_checkpoint_file)
                os.rename(checkpoint_file, grid_checkpoint_file)
        else:
            results[str(conf)] = best_from_iterations
            t.save(results, GRID_RESULTS_FILE)

    return best

def train_and_evaluate_UNcensored(checkpoint, criterion, model_fn = DenseNetwork, plot = False, log = True, is_gamma = False, loader_train_fn = None, loader_val_fn = None):
    def grid_callback(dataset_train, dataset_val, bound_min, bound_max, conf):
        model = model_fn()
        if not loader_train_fn:
            loader_train = t.utils.data.DataLoader(dataset_train, conf['batch'], shuffle = False, num_workers = 0)
        else:
            loader_train = loader_train_fn(conf['batch'])
        if not loader_val_fn:
            loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
        else:
            loader_val = loader_val_fn(len(dataset_val))
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
        train_metrics, val_metrics, best = train_network(bound_min, bound_max,
            model, loss_fn, optimizer, scheduler, loader_train, loader_val, checkpoint, conf['batch'], len(dataset_val), conf['epochs'], log = log)
        if plot:
            plot_epochs(train_metrics, val_metrics)
        return best
    return grid_callback

def config_validation(conf):
    return conf['div_factor'] <= conf['final_div_factor'] and conf['max_momentum'] >= conf['base_momentum']

"""# Plot Selected(With Grid) Model"""

def plot_and_evaluate_model_UNcensored(dataset_val, dataset_test, checkpoint_name, criterion, isGrid = True, model_fn = DenseNetwork, is_gamma = False, loader_val = None):
    model = model_fn()
    checkpoint = t.load(('grid ' if isGrid else '') + checkpoint_name + '.tar')
    model.load_state_dict(checkpoint['model'])
    plot_beta(label = 'true distribution')
    # plot_dataset(dataset_test, size = .3, label = 'test data')
    plot_dataset(dataset_val, size = .3, label = 'validation data')
    plot_net(model)
    loss_fn = criterion()
    plt.xlabel('input (standardized)')
    plt.ylabel('outcome (standardized)')
    plt.ylim((-2.5, 2.5))
    lgnd = plt.legend()
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    lgnd.legendHandles[2]._sizes = [10]
    plt.savefig('{}.pdf'.format(checkpoint_name), dpi = 300, format = 'pdf')
    plt.savefig('{}.svg'.format(checkpoint_name), dpi = 300, format = 'svg')
    plt.savefig('{}.png'.format(checkpoint_name), dpi = 200, format = 'png')
    plt.close()

    if not loader_val:
        loader_val = t.utils.data.DataLoader(dataset_val, len(dataset_val), shuffle = False, num_workers = 0)
    val_metrics = eval_network(model, loader_val, loss_fn, len(dataset_val))
    print('Absolute error - validation', val_metrics[ABS_ERR])
    print('R2 - validation', val_metrics[R_SQUARED])

    loader_test = t.utils.data.DataLoader(dataset_test, len(dataset_test), shuffle = False, num_workers = 0)
    test_metrics = eval_network(model, loader_test, loss_fn, len(dataset_test), is_eval_bounded = False)
    print('Absolute error - test', test_metrics[ABS_ERR])
    print('R2 - test', test_metrics[R_SQUARED])

def real_y_std():
    real_x_mean, real_x_std, real_y_mean, real_y_std = calculate_mean_std()
    return real_y_std