from experiments.constants import GRID_RESULTS_FILE
from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_tobit_dyn_std, plot_dataset_and_net
from experiments.grid_train import train_and_evaluate_tobit_dyn_std
from experiments.real.models import get_model, linear_model, get_scale_network
from experiments.util import get_device

"""Constants"""
ROOT_DEEP_TOBIT_SCALED_TRUNCATED = 'experiments/real/pm25/tobit_based/scaled_dynamic_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED = 'heteroscedastic scaled truncated deep tobit model'

"""Reproducible experiments"""

set_random_seed()




"""# Scaled Deep Tobit With Truncation"""

train_and_evaluate_net = train_and_evaluate_tobit_dyn_std(ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED,
    model_fn = lambda: get_model(INPUT_SIZE), scale_model_fn = lambda: get_scale_network(INPUT_SIZE), plot = False, log = False, truncated_low = zero_normalized)

def train_once_deep_tobit_WITH_trunc():
    conf = {
        'max_lr': 5e-4,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 5,
        'final_div_factor': 1e4,
        'weight_decay': 0,
        'grad_clip': 1000
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, isGrid = False)

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
    grid_best = grid_search(ROOT_DEEP_TOBIT_SCALED_TRUNCATED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_WITH_trunc_dyn_std():
    plot_and_evaluate_model_tobit_dyn_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_deep_tobit_WITH_trunc_dyn_std():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_SCALED_TRUNCATED}/grid {CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED}.tar')
    scale_model = get_scale_network(INPUT_SIZE)
    scale_model.load_state_dict(checkpoint['sigma'])
    scale_model.eval()
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df), scale_model=scale_model)