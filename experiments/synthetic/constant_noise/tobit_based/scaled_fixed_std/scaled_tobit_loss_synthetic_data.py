from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.constant_noise.dataset import *
from experiments.synthetic.grid_search import train_and_evaluate_gll, plot_and_evaluate_model_gll, grid_search, config_validation
from deep_tobit.util import to_torch, to_numpy, normalize, unnormalize, distinguish_censored_versus_observed_data
from deep_tobit.loss import Scaled_Tobit_Loss

"""Constants"""
ROOT_DEEP_TOBIT_SCALED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/deep_tobit_cens_NO_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED = 'scaled deep tobit model'

ROOT_DEEP_TOBIT_SCALED_TRUNCATED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/deep_tobit_cens_WITH_trunc'
CHECKPOINT_DEEP_TOBIT_SCALED_TRUNCATED = 'scaled truncated deep tobit model'

ROOT_LINEAR_TOBIT_SCALED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/liniar_tobit_cens_NO_trunc'
CHECKPOINT_LINEAR_TOBIT_SCALED = 'scaled linear tobit model'

ROOT_LINEAR_TRUNCATED_TOBIT_SCALED = 'experiments/synthetic/constant_noise/tobit_based/scaled_fixed_std/liniar_tobit_cens_WITH_trunc'
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


"""# PDF Log-Likelihood"""

class GausianLogLikelihoodLoss(t.nn.Module):

  def __init__(self, sigma):
    super(GausianLogLikelihoodLoss, self).__init__()
    self.sigma = sigma
    self.epsilon = t.tensor(1e-40, dtype = t.float32, requires_grad = False)

  def forward(self, y_pred: t.Tensor, y_true: t.Tensor):
    sigma = t.abs(self.sigma)
    return t.sum(t.log(sigma + self.epsilon) + (((y_true - y_pred)/sigma) ** 2) / 2)

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_gll(ROOT_GLL + '/' + CHECKPOINT_GLL, GausianLogLikelihoodLoss, plot = False, log = False)

def train_once_gll():
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
  plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                              ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, isGrid = False)

def grid_search_gll():
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
  grid_best = grid_search(ROOT_GLL, dataset_train, dataset_val, bound_min, bound_max,
                          grid_config, train_and_evaluate_net, CHECKPOINT_GLL, conf_validation = config_validation)
  return grid_best

def eval_gll_scaled():
  plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                              ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, isGrid = True)
  grid_results = t.load(ROOT_GLL + '/' + GRID_RESULTS_FILE)
  best_config = grid_results['best']
  best_metrics = grid_results[str(best_config)]
  print(best_config)
  print(best_metrics)

eval_gll_scaled()