from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.real.pm25.dataset import *
from experiments.synthetic.grid_search import train_and_evaluate_mae_mse, plot_and_evaluate_model_mae_mse, grid_search, config_validation, get_grid_search_space

"""Constants"""
ROOT_MAE = 'experiments/real/pm25/mae_based/mae_simple'
CHECKPOINT_MAE = 'mae model'

ROOT_BOUNDED_MAE = 'experiments/real/pm25/mae_based/mae_cens_NO_trunc'
CHECKPOINT_BOUNDED_MAE = 'mae bounded model'

ROOT_BOUNDED_MAE_WITH_PENALTY = 'experiments/real/pm25/mae_based/mae_cens_WITH_trunc'
CHECKPOINT_BOUNDED_MAE_WITH_PENALTY = 'mae bounded with penalty model'

"""Reproducible experiments"""

set_random_seed()

"""# Datasets"""

dataset_train = TruncatedBetaDistributionDataset(x_mean, x_std, y_mean, y_std, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND)
dataset_val = TruncatedBetaDistributionValidationDataset(x_mean, x_std, y_mean, y_std, lower_bound = CENSOR_LOW_BOUND, upper_bound = CENSOR_HIGH_BOUND, nb_samples = 1000)
dataset_test = TruncatedBetaDistributionValidationDataset(x_mean, x_std, y_mean, y_std)

"""# Training"""

bound_min = normalize(CENSOR_LOW_BOUND, y_mean, y_std)
bound_max = normalize(CENSOR_HIGH_BOUND, y_mean, y_std)
zero_normalized = normalize(0, y_mean, y_std)

"""# MAE"""

train_and_evaluate_net = train_and_evaluate_mae_mse(ROOT_MAE + '/' + CHECKPOINT_MAE, t.nn.L1Loss, plot = False, log = False)

"""Train once with default settings"""
def train_once_mae_simple():
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
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                    dataset_val, dataset_test, ROOT_MAE, CHECKPOINT_MAE, t.nn.L1Loss, isGrid = False)

"""Grid search"""
def grid_search_mae_simple():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_MAE, dataset_train, dataset_val, bound_min, bound_max, grid_config,
                            train_and_evaluate_net, CHECKPOINT_MAE, conf_validation = config_validation)
    return grid_best

def eval_mae_simple():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
                                    dataset_val, dataset_test, ROOT_MAE, CHECKPOINT_MAE, t.nn.L1Loss, isGrid = True)
    grid_results = t.load(ROOT_MAE + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)





"""# Bounded MAE"""

mae = t.nn.L1Loss()

def bounded_loss(y_pred, y):
  y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
  return mae(y_pred, y)

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_mae_mse(ROOT_BOUNDED_MAE + '/' + CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, plot = False, log = False)

def train_once_mae_cens_NO_trunc():
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
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, isGrid = False)

def grid_search_mae_cens_NO_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_BOUNDED_MAE, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MAE, conf_validation = config_validation)
    return grid_best

def eval_mae_cens_NO_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                    ROOT_BOUNDED_MAE, CHECKPOINT_BOUNDED_MAE, lambda: bounded_loss, isGrid = True)
    grid_results = t.load(ROOT_BOUNDED_MAE + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)





"""# Bounded MAE With Penalty"""

def below_zero_mae_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mae(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mae_penalty(y_pred)

train_and_evaluate_net = train_and_evaluate_mae_mse(ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + CHECKPOINT_BOUNDED_MAE_WITH_PENALTY,
                                                    lambda: bounded_loss_with_penalty, plot = False, log = False)

def train_once_mae_cens_WITH_trunc():
    conf = {
        'max_lr': 2e-2,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 2,
        'final_div_factor': 1e4,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, ROOT_BOUNDED_MAE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = False)

def grid_search_mae_cens_WITH_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_BOUNDED_MAE_WITH_PENALTY, dataset_train, dataset_val, bound_min, bound_max,
                grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, conf_validation = config_validation)
    return grid_best

def eval_mae_cens_WITH_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, ROOT_BOUNDED_MAE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MAE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = True)
    grid_results = t.load(ROOT_BOUNDED_MAE_WITH_PENALTY + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

eval_mae_simple()
eval_mae_cens_NO_trunc()
eval_mae_cens_WITH_trunc()