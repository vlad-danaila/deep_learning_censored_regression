from experiments.constants import GRID_RESULTS_FILE
from experiments.util import set_random_seed, load_checkpoint
from experiments.real.pm25.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.real.pm25.grid_eval import plot_and_evaluate_model_tobit_fixed_std, plot_dataset_and_net
from experiments.grid_train import train_and_evaluate_tobit_fixed_std
from experiments.real.models import get_model, linear_model

"""Constants"""
ROOT_DEEP_TOBIT_REPARAMETRIZED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED = 'reparametrized deep tobit model'

ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED = 'reparametrized truncated deep tobit model'

ROOT_LINEAR_TOBIT_REPARAMETRIZED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED = 'reparametrized linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'experiments/real/pm25/tobit_based/reparam_fixed_std/liniar_tobit_cens_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED = 'reparametrized truncated linear tobit model'

"""Reproducible experiments"""

set_random_seed()




"""# Reparametrized Deep Tobit"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_DEEP_TOBIT_REPARAMETRIZED + '/' + CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED,
    plot = False, log = False, model_fn = lambda: get_model(INPUT_SIZE), isReparam=True)

def train_once_deep_tobit_NO_trunc_reparam():
    conf = {
        'max_lr': 1e-4,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 5,
        'final_div_factor': 1e4,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, isGrid = False)

def grid_search_deep_tobit_NO_trunc_reparam():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_DEEP_TOBIT_REPARAMETRIZED, dataset_train, dataset_val, bound_min, bound_max,
                          grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_NO_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_REPARAMETRIZED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_REPARAMETRIZED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_deep_tobit_NO_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED}/grid {CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED}.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))




"""# Reparametrized Deep Tobit With Truncation"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED + '/' + CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED,
                                                            model_fn = lambda: get_model(INPUT_SIZE), plot = False, log = False, truncated_low = zero_normalized, isReparam=True)

def train_once_deep_tobit_WITH_trunc_reparam():
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
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, isGrid = False)

def grid_search_deep_tobit_WITH_trunc_reparam():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_WITH_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_deep_tobit_WITH_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}/grid {CHECKPOINT_DEEP_TOBIT_REPARAMETRIZED_TRUNCATED}.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))





"""# Reparametrized Linear Tobit"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_LINEAR_TOBIT_REPARAMETRIZED + '/' + CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED,
                                                            model_fn = lambda: linear_model(INPUT_SIZE), plot = False, log = False, isReparam=True)

def train_once_linear_tobit_NO_trunc_reparam():
    conf = {
        'max_lr': 1e-4,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 5,
        'final_div_factor': 1e4,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, model_fn = linear_model, isGrid = False)

def grid_search_linear_tobit_NO_trunc_reparam():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_LINEAR_TOBIT_REPARAMETRIZED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, conf_validation = config_validation)
    return grid_best

def eval_linear_tobit_NO_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED, model_fn = linear_model, isGrid = True)
    grid_results = t.load(ROOT_LINEAR_TOBIT_REPARAMETRIZED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_linear_tobit_NO_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_LINEAR_TOBIT_REPARAMETRIZED}/grid {CHECKPOINT_LINEAR_TOBIT_REPARAMETRIZED}.tar')
    plot_dataset_and_net(checkpoint, linear_model(INPUT_SIZE), test_df(df))






"""# Reparametrized Linear Tobit With Truncation"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED + '/' + CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED,
                                                            model_fn = lambda: linear_model(INPUT_SIZE), plot = False, log = False, truncated_low = zero_normalized, isReparam=True)

def train_once_linear_tobit_WITH_trunc_reparam():
    conf = {
        'max_lr': 5e-5,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 5,
        'final_div_factor': 1e4,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, model_fn = linear_model, isGrid = False)

def grid_search_linear_tobit_WITH_trunc_reparam():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, dataset_train, dataset_val, bound_min, bound_max,
        grid_config, train_and_evaluate_net, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, conf_validation = config_validation)
    return grid_best

def eval_linear_tobit_WITH_trunc_reparam():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED, model_fn = linear_model, isGrid = True)
    grid_results = t.load(ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_linear_tobit_WITH_trunc_reparam():
    checkpoint = load_checkpoint(f'{ROOT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}/grid {CHECKPOINT_LINEAR_TRUNCATED_TOBIT_REPARAMETRIZED}.tar')
    plot_dataset_and_net(checkpoint, linear_model(INPUT_SIZE), test_df(df))

