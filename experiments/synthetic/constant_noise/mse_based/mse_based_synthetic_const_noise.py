from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.grid_search import train_and_evaluate_UNcensored, plot_and_evaluate_model_UNcensored, grid_search, config_validation

"""Constants"""

CHECKPOINT_MSE = 'mse model'
ROOT_MSE = 'experiments/synthetic/constant_noise/mse_based/mse_simple'

CHECKPOINT_BOUNDED_MSE = 'mse bounded model'
ROOT_BOUNDED_MSE = 'experiments/synthetic/constant_noise/mse_based/mse_cens_NO_trunc'

CHECKPOINT_BOUNDED_MSE_WITH_PENALTY = 'mse bounded with penalty model'
ROOT_BOUNDED_MSE_WITH_PENALTY = 'experiments/synthetic/constant_noise/mse_based/mse_cens_WITH_trunc'

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

"""# MSE"""

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_UNcensored(ROOT_MSE + '/' + CHECKPOINT_MSE, t.nn.MSELoss, plot = False, log = False)

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
  plot_and_evaluate_model_UNcensored(bound_min, bound_max, x_mean, x_std, y_mean, y_std,
              dataset_val, dataset_test, ROOT_MSE, CHECKPOINT_MSE, t.nn.MSELoss, isGrid = False)

"""Grid search"""
def grid_search_mse_simple():
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
      'weight_decay': [0]
  }]
  grid_best = grid_search(ROOT_MSE, dataset_train, dataset_val, bound_min, bound_max,
            grid_config, train_and_evaluate_net, CHECKPOINT_MSE, conf_validation = config_validation)
  return grid_best

def eval_mse_simple():
  plot_and_evaluate_model_UNcensored(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test, ROOT_MSE,
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

"""### Learning Rate Range Test"""

# lr_range_test_UNcensored(lambda: bounded_loss, batch_size = 100, epochs = 2, start_lr = 1e-2, end_lr = 1e-1, log_view = False, plt_file_name = 'bounded_mse')

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_UNcensored(CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, plot = False, log = False)

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
# grid_best = grid_search(grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MSE, conf_validation = config_validation)
# print(grid_best)

# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, isGrid = False)
# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MSE, lambda: bounded_loss, isGrid = True)

# grid_results = t.load(GRID_RESULTS_FILE)
# best_config = grid_results['best']
# best_metrics = grid_results[str(best_config)]
# print(best_config)
# print(best_metrics)

"""# Bounded MSE With Penalty"""

def below_zero_mse_penalty(y_pred):
  y_below_zero = t.clamp(y_pred, min = -math.inf, max = zero_normalized)
  return mse(y_below_zero, t.full_like(y_below_zero, zero_normalized))

def bounded_loss_with_penalty(y_pred, y):
  return bounded_loss(y_pred, y) + below_zero_mse_penalty(y_pred)

"""### Learning Rate Range Test"""

# lr_range_test_UNcensored(lambda: bounded_loss_with_penalty, batch_size = 100, epochs = 2, start_lr = 1e-2, end_lr = 1e-1, log_view = False, plt_file_name = 'bounded_mse_with_penalty')

train_and_evaluate_net = train_and_evaluate_UNcensored(CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, plot = False, log = False)

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
# grid_best = grid_search(grid_config, train_and_evaluate_net, CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, conf_validation = config_validation)
# print(grid_best)

# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = False)
# plot_and_evaluate_model_UNcensored(CHECKPOINT_BOUNDED_MSE_WITH_PENALTY, lambda: bounded_loss_with_penalty, isGrid = True)

# grid_results = t.load(GRID_RESULTS_FILE)
# best_config = grid_results['best']
# best_metrics = grid_results[str(best_config)]
# print(best_config)
# print(best_metrics)