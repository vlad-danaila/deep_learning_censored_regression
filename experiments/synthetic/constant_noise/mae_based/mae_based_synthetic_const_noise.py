"""Imports"""

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
from experiments.synthetic.grid_search import train_and_evaluate_UNcensored, plot_and_evaluate_model_UNcensored

"""Constants"""

CHECKPOINT_MAE = 'mae simple model'
CHECKPOINT_BOUNDED_MAE = 'mae cens model'
CHECKPOINT_BOUNDED_MAE_WITH_PENALTY = 'mae cens trunc model'

"""Reproducible experiments"""

set_random_seed()

"""# Datasets"""

x_mean, x_std, y_mean, y_std = calculate_mean_std(lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
print('x mean =', x_mean, 'x std =', x_std, 'y mean =', y_mean, 'y std =', y_std)

dataset_train = TruncatedBetaDistributionDataset(x_mean, x_std, y_mean, y_std, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
dataset_val = TruncatedBetaDistributionValidationDataset(x_mean, x_std, y_mean, y_std, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND, nb_samples = 1000)
dataset_test = TruncatedBetaDistributionValidationDataset(x_mean, x_std, y_mean, y_std)

"""# Training"""

bound_min = normalize(CENSOR_LOW_BOUND, y_mean, y_std)
bound_max = normalize(CENSOR_HIGH_BOUND, y_mean, y_std)
zero_normalized = normalize(0, y_mean, y_std)

"""# MAE

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_UNcensored(CHECKPOINT_MAE, t.nn.L1Loss, plot = False, log = False)

"""Train once with default settings"""

conf = {
    'max_lr': 3e-2,
    'epochs': 10,
    'batch': 100,
    'pct_start': 0.3,
    'anneal_strategy': 'linear',
    'base_momentum': 0.85,
    'max_momentum': 0.95,
    'div_factor': 3,
    'final_div_factor': 1e4,
    'weight_decay': 0
}
train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)

"""Grid search"""

# grid_config = [{
#     'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
#     'epochs': [10, 20],
#     'batch': [100, 200],
#     'pct_start': [0.45],
#     'anneal_strategy': ['linear'],
#     'base_momentum': [0.85],
#     'max_momentum': [0.95],
#     'div_factor': [10, 5, 2],
#     'final_div_factor': [1e4],
#     'weight_decay': [0]
# }]
# grid_best = grid_search(grid_config, train_and_evaluate_net, CHECKPOINT_MAE, conf_validation = config_validation)

"""Load the best model"""

plot_and_evaluate_model_UNcensored(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                   dataset_val, dataset_test, CHECKPOINT_MAE, t.nn.L1Loss, isGrid = False)

# plot_and_evaluate_model_UNcensored(CHECKPOINT_MAE, t.nn.L1Loss, isGrid = True)

# grid_results = t.load(GRID_RESULTS_FILE)
# best_config = grid_results['best']
# best_metrics = grid_results[str(best_config)]
# print(best_config)
# print(best_metrics)

"""# Bounded MAE"""

mae = t.nn.L1Loss()

def bounded_loss(y_pred, y):
  y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
  return mae(y_pred, y)

"""### Learning Rate Range Test"""

# lr_range_test_UNcensored(lambda: bounded_loss, batch_size = 100, epochs = 2, start_lr = 1e-2, end_lr = 1e-1, log_view = False, plt_file_name = 'bounded_mae')

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_UNcensored(CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, plot = False, log = False)

# conf = {
#     'max_lr': 3e-2,
#     'epochs': 10,
#     'batch': 100,
#     'pct_start': 0.3,
#     'anneal_strategy': 'linear',
#     'base_momentum': 0.85,
#     'max_momentum': 0.95,
#     'div_factor': 3,
#     'final_div_factor': 1e4,
#     'weight_decay': 0
# }
# train_and_evaluate_net(conf)

# grid_config = [{
#     'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
#     'epochs': [10, 20],
#     'batch': [100, 200],
#     'pct_start': [0.45],
#     'anneal_strategy': ['linear'],
#     'base_momentum': [0.85],
#     'max_momentum': [0.95],
#     'div_factor': [10, 5, 2],
#     'final_div_factor': [1e4],
#     'weight_decay': [0]
# }]
# grid_best = grid_search(grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MAE, conf_validation = config_validation)

# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, isGrid = False)
# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, isGrid = True)

# grid_results = t.load(GRID_RESULTS_FILE)
# best_config = grid_results['best']
# best_metrics = grid_results[str(best_config)]
# print(best_config)
# print(best_metrics)

"""# Bounded MAE With Penalty"""

def below_zero_mae_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mae(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mae_penalty(y_pred)

"""### Learning Rate Range Test"""

# lr_range_test_UNcensored(lambda: bounded_loss_with_penalty, batch_size = 100, epochs = 2, start_lr = 1e-2, end_lr = 1e-1, log_view = False, plt_file_name = 'bounded_mae_with_penalty')

train_and_evaluate_net = train_and_evaluate_UNcensored(CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, plot = False, log = False)

# conf = {
#     'max_lr': 2e-2,
#     'epochs': 10,
#     'batch': 100,
#     'pct_start': 0.3,
#     'anneal_strategy': 'linear',
#     'base_momentum': 0.85,
#     'max_momentum': 0.95,
#     'div_factor': 2,
#     'final_div_factor': 1e4,
#     'weight_decay': 0
# }
# train_and_evaluate_net(conf)

# grid_config = [{
#     'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
#     'epochs': [10, 20],
#     'batch': [100, 200],
#     'pct_start': [0.45],
#     'anneal_strategy': ['linear'],
#     'base_momentum': [0.85],
#     'max_momentum': [0.95],
#     'div_factor': [10, 5, 2],
#     'final_div_factor': [1e4],
#     'weight_decay': [0]
# }]
# grid_best = grid_search(grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, conf_validation = config_validation)

# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = False)
# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = True)

# grid_results = t.load(GRID_RESULTS_FILE)
# best_config = grid_results['best']
# best_metrics = grid_results[str(best_config)]
# print(best_config)
# print(best_metrics)