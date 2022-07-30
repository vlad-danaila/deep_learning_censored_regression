from experiments.constants import GRID_RESULTS_FILE
from experiments.util import set_random_seed
from experiments.real.pm25.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.real.pm25.grid_eval import plot_and_evaluate_model_tobit_fixed_std
from experiments.grid_train import train_and_evaluate_tobit_dyn_std
from experiments.real.models import get_model, linear_model
from experiments.util import get_device

"""Constants"""
ROOT_DEEP_TOBIT_TRUNCATED = 'experiments/real/pm25/tobit_based/scaled_dynamic_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_TRUNCATED = 'scaled truncated deep tobit model'

"""Reproducible experiments"""

set_random_seed()




"""# Scaled Deep Tobit With Truncation"""

train_and_evaluate_net = train_and_evaluate_tobit_dyn_std(ROOT_DEEP_TOBIT_TRUNCATED + '/' + CHECKPOINT_DEEP_TOBIT_TRUNCATED,
                                                            model_fn = DenseNetwork, plot = False, log = False, truncated_low = zero_normalized)

def train_once_deep_tobit_WITH_trunc():
    conf = {
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'batch': 100,
        'div_factor': 5,
        'epochs': 20,
        'final_div_factor': 10000.0,
        'grad_clip': 1e-2,
        'max_lr': 5e-3,
        'max_momentum': 0.95,
        'pct_start': 0.45,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_TRUNCATED, CHECKPOINT_DEEP_TOBIT_TRUNCATED, model_fn = DenseNetwork, isGrid = False)

def grid_search_deep_tobit_WITH_trunc():
    grid_config = [{
        'max_lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'epochs': [10, 20],
        'batch': [100, 200],
        'pct_start': [0.45],
        'anneal_strategy': ['linear'],
        'base_momentum': [0.85],
        'max_momentum': [0.95],
        'div_factor': [10, 5, 2],
        'final_div_factor': [1e4],
        'weight_decay': [0],
        'grad_clip': [1e-2, 1e-1, 1, 10, 100]
    }]
    grid_best = grid_search(ROOT_DEEP_TOBIT_TRUNCATED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_TRUNCATED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_WITH_trunc_dyn_std():
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_TRUNCATED, CHECKPOINT_DEEP_TOBIT_TRUNCATED, model_fn = DenseNetwork, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_TRUNCATED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

