from experiments.constants import GRID_RESULTS_FILE
from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.heteroscedastic.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.synthetic.grid_eval import plot_and_evaluate_model_tobit_fixed_std
from experiments.grid_train import train_and_evaluate_tobit_fixed_std
from deep_tobit.util import normalize, distinguish_censored_versus_observed_data
from experiments.synthetic.models import DenseNetwork
from experiments.synthetic.grid_eval import plot_dataset_and_net

"""Constants"""
ROOT_DEEP_TOBIT_SCALED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED = 'scaled deep tobit model'

ROOT_DEEP_TOBIT_SCALED_TRUNCATED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED = 'scaled truncated deep tobit model'

ROOT_LINEAR_TOBIT_SCALED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_SCALED = 'scaled linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_SCALED = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/liniar_tobit_cens_WITH_trunc'
CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED = 'scaled truncated linear tobit model'

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

"""# Common Tobit Setup"""

censored_collate_fn = distinguish_censored_versus_observed_data(bound_min, bound_max)
uncensored_collate_fn = distinguish_censored_versus_observed_data(-math.inf, math.inf)

tobit_loader_train = t.utils.data.DataLoader(dataset_train, batch_size = 100, shuffle = True, num_workers = 0, collate_fn = censored_collate_fn)
tobit_loader_val = t.utils.data.DataLoader(dataset_val, batch_size = len(dataset_val), shuffle = False, num_workers = 0, collate_fn = censored_collate_fn)
tobit_loader_test = t.utils.data.DataLoader(dataset_test, batch_size = len(dataset_test), shuffle = False, num_workers = 0, collate_fn = uncensored_collate_fn)




"""# Scaled Deep Tobit"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_DEEP_TOBIT_SCALED + '/' + CHECKPOINT_DEEP_TOBIT_SCALED, plot = False, log = False)

def train_once_deep_tobit_NO_trunc():
  conf = {
      'max_lr': 2e-4,
      'epochs': 10,
      'batch': 100,
      'pct_start': 0.3,
      'anneal_strategy': 'linear',
      'base_momentum': 0.85,
      'max_momentum': 0.95,
      'div_factor': 4,
      'final_div_factor': 1e4,
      'weight_decay': 0
  }
  train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
  plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                          ROOT_DEEP_TOBIT_SCALED, CHECKPOINT_DEEP_TOBIT_SCALED, model_fn = DenseNetwork, isGrid = False)

def grid_search_deep_tobit_NO_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_DEEP_TOBIT_SCALED, dataset_train, dataset_val, bound_min, bound_max,
                          grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_SCALED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_NO_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED, CHECKPOINT_DEEP_TOBIT_SCALED, model_fn = DenseNetwork, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_SCALED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_deep_tobit_NO_trunc():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_SCALED}/grid {CHECKPOINT_DEEP_TOBIT_SCALED}.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)





"""# Scaled Deep Tobit With Truncation"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED,
                                                            model_fn = DenseNetwork, plot = False, log = False, truncated_low = zero_normalized)

def train_once_deep_tobit_WITH_trunc():
    conf = {
        'max_lr': 2e-4,
        'epochs': 10,
        'batch': 100,
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 4,
        'final_div_factor': 1e4,
        'weight_decay': 0
    }
    train_and_evaluate_net(dataset_train, dataset_val, bound_min, bound_max, conf)
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, model_fn = DenseNetwork, isGrid = False)

def grid_search_deep_tobit_WITH_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_DEEP_TOBIT_SCALED_TRUNCATED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, conf_validation = config_validation)
    return grid_best

def eval_deep_tobit_WITH_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_DEEP_TOBIT_SCALED_TRUNCATED, CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED, model_fn = DenseNetwork, isGrid = True)
    grid_results = t.load(ROOT_DEEP_TOBIT_SCALED_TRUNCATED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_deep_tobit_WITH_trunc():
    checkpoint = t.load(f'{ROOT_DEEP_TOBIT_SCALED_TRUNCATED}/grid {CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED}.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)






"""# Scaled Linear Tobit"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_LINEAR_TOBIT_SCALED + '/' + CHECKPOINT_LINEAR_TOBIT_SCALED,
                                                            model_fn = lambda: t.nn.Linear(1, 1), plot = False, log = False)

def train_once_linear_tobit_NO_trunc():
    conf = {
        'max_lr': 5e-3,
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
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_SCALED, CHECKPOINT_LINEAR_TOBIT_SCALED, model_fn = lambda: t.nn.Linear(1, 1), isGrid = False)

def grid_search_linear_tobit_NO_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_LINEAR_TOBIT_SCALED, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_LINEAR_TOBIT_SCALED, conf_validation = config_validation)
    return grid_best

def eval_linear_tobit_NO_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_LINEAR_TOBIT_SCALED, CHECKPOINT_LINEAR_TOBIT_SCALED, model_fn = lambda: t.nn.Linear(1, 1), isGrid = True)
    grid_results = t.load(ROOT_LINEAR_TOBIT_SCALED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_linear_tobit_NO_trunc():
    checkpoint = t.load(f'{ROOT_LINEAR_TOBIT_SCALED}/grid {CHECKPOINT_LINEAR_TOBIT_SCALED}.tar')
    plot_dataset_and_net(checkpoint, t.nn.Linear(1, 1), x_mean, x_std, y_mean, y_std, dataset_val)






"""# Scaled Linear Tobit With Truncation"""

train_and_evaluate_net = train_and_evaluate_tobit_fixed_std(ROOT_LINEAR_TRUNCATED_TOBIT_SCALED + '/' + CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED,
                                                            model_fn = lambda: t.nn.Linear(1, 1), plot = False, log = False, truncated_low = zero_normalized)

def train_once_linear_tobit_WITH_trunc():
    conf = {
        'max_lr': 5e-3,
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
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, model_fn = lambda: t.nn.Linear(1, 1), isGrid = False)

def grid_search_linear_tobit_WITH_trunc():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, dataset_train, dataset_val, bound_min, bound_max,
        grid_config, train_and_evaluate_net, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, conf_validation = config_validation)
    return grid_best

def eval_linear_tobit_WITH_trunc():
    plot_and_evaluate_model_tobit_fixed_std(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                                            ROOT_LINEAR_TRUNCATED_TOBIT_SCALED, CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED, model_fn = lambda: t.nn.Linear(1, 1), isGrid = True)
    grid_results = t.load(ROOT_LINEAR_TRUNCATED_TOBIT_SCALED + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)

def plot_linear_tobit_WITH_trunc():
    checkpoint = t.load(f'{ROOT_LINEAR_TRUNCATED_TOBIT_SCALED}/grid {CHECKPOINT_LINEAR_TRUNCATED_TOBIT_SCALED}.tar')
    plot_dataset_and_net(checkpoint, t.nn.Linear(1, 1), x_mean, x_std, y_mean, y_std, dataset_val)