from experiments.constants import GRID_RESULTS_FILE
from experiments.util import set_random_seed
from experiments.real.bike_sharing.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.real.bike_sharing.grid_eval import plot_and_evaluate_model_mae_mse
from experiments.grid_train import train_and_evaluate_mae_mse
from experiments.real.models import get_model

"""Constants"""

CHECKPOINT_MSE = 'mse model'
ROOT_MSE = 'experiments/real/bike_sharing/mse_based/mse_simple'

CHECKPOINT_BOUNDED_MSE = 'mse bounded model'
ROOT_BOUNDED_MSE = 'experiments/real/bike_sharing/mse_based/mse_cens_NO_trunc'

CHECKPOINT_BOUNDED_MSE_WITH_PENALTY = 'mse bounded with penalty model'
ROOT_BOUNDED_MSE_WITH_PENALTY = 'experiments/real/bike_sharing/mse_based/mse_cens_WITH_trunc'

"""Reproducible experiments"""

set_random_seed()



"""# MSE"""

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_mae_mse(ROOT_MSE + '/' + CHECKPOINT_MSE,
    t.nn.MSELoss, plot = False, log = False, model_fn = lambda: get_model(INPUT_SIZE))

"""Train once with default settings"""
def train_once_mse_simple():
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
  plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df),
                                  dataset_val, dataset_test, ROOT_MSE, CHECKPOINT_MSE, t.nn.MSELoss, isGrid = False)

"""Grid search"""
def grid_search_mse_simple():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_MSE, dataset_train, dataset_val, bound_min, bound_max,
            grid_config, train_and_evaluate_net, CHECKPOINT_MSE, conf_validation = config_validation)
    return grid_best

def eval_mse_simple():
  plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_MSE,
                                  CHECKPOINT_MSE, t.nn.MSELoss, isGrid = True)
  grid_results = t.load(ROOT_MSE + '/' + GRID_RESULTS_FILE)
  best_config = grid_results['best']
  best_metrics = grid_results[str(best_config)]
  print(best_config)
  print(best_metrics)





"""# Bounded MSE"""

mse = t.nn.MSELoss()

def bounded_loss(y_pred, y):
  y_pred = t.clamp(y_pred, min = bound_min, max = bound_max)
  return mse(y_pred, y)

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_mae_mse(ROOT_BOUNDED_MSE + '/' + CHECKPOINT_BOUNDED_MSE,
    lambda: bounded_loss, plot = False, log = False, model_fn = lambda: get_model(INPUT_SIZE))

def train_once_mse_cens_NO_trunc():
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
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MSE,
                                    CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, isGrid = False)

def grid_search_mse_cens_NO_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_BOUNDED_MSE, dataset_train, dataset_val, bound_min, bound_max,
        grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MSE, conf_validation = config_validation)
    return grid_best

def eval_mse_cens_NO_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MSE,
                                    CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, isGrid = True)
    grid_results = t.load(ROOT_BOUNDED_MSE + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)





"""# Bounded MSE With Penalty"""

def below_zero_mse_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mse(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mse_penalty(y_pred)

train_and_evaluate_net = train_and_evaluate_mae_mse(ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + CHECKPOINT_BOUNDED_MSE_WITH_PENALTY,
    lambda: bounded_loss_with_penalty, plot = False, log = False, model_fn = lambda: get_model(INPUT_SIZE))

def train_once_mse_cens_WITH_trunc():
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
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MSE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = False)

def grid_search_mse_cens_WITH_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_BOUNDED_MSE_WITH_PENALTY, dataset_train, dataset_val, bound_min, bound_max,
        grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, conf_validation = config_validation)
    return grid_best

def eval_mse_cens_WITH_trunc():
    plot_and_evaluate_model_mae_mse(bound_min, bound_max, test_df(df), dataset_val, dataset_test, ROOT_BOUNDED_MSE_WITH_PENALTY,
                                    CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = True)
    grid_results = t.load(ROOT_BOUNDED_MSE_WITH_PENALTY + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)


