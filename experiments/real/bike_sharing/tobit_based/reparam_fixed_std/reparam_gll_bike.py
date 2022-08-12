from experiments.constants import GRID_RESULTS_FILE
from experiments.util import set_random_seed
from experiments.real.bike_sharing.dataset import *
from experiments.grid_search import grid_search, config_validation, get_grid_search_space
from experiments.real.bike_sharing.grid_eval import plot_and_evaluate_model_gll
from experiments.grid_train import train_and_evaluate_gll
from experiments.real.models import get_model
from experiments.util import get_device

"""Constants"""
ROOT_GLL = 'experiments/real/bike_sharing/tobit_based/reparam_fixed_std/gll'
CHECKPOINT_GLL = 'gausian log likelihood model'

"""Reproducible experiments"""

set_random_seed()

"""# PDF Log-Likelihood"""

class GausianLogLikelihoodLoss(t.nn.Module):

  def __init__(self, gamma):
    super(GausianLogLikelihoodLoss, self).__init__()
    self.gamma = gamma
    self.epsilon = t.tensor(1e-40, dtype = t.float32, requires_grad = False, device = get_device())

  def forward(self, y_pred: t.Tensor, y_true: t.Tensor):
    gamma = t.abs(self.gamma)
    return -t.sum(t.log(gamma + self.epsilon) - ((gamma * y_true - y_pred) ** 2) / 2)

"""### Grid Search"""

train_and_evaluate_net = train_and_evaluate_gll(ROOT_GLL + '/' + CHECKPOINT_GLL, GausianLogLikelihoodLoss,
                                                plot = False, log = False, model_fn = lambda: get_model(INPUT_SIZE))

"""Train once with default settings"""
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
    plot_and_evaluate_model_gll(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                    ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, isGrid = False)

def grid_search_gll():
    grid_config = get_grid_search_space()
    grid_best = grid_search(ROOT_GLL, dataset_train, dataset_val, bound_min, bound_max,
                            grid_config, train_and_evaluate_net, CHECKPOINT_GLL, conf_validation = config_validation)
    return grid_best

def eval_gll_reparam():
    plot_and_evaluate_model_gll(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                                    ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, isGrid = True)
    grid_results = t.load(ROOT_GLL + '/' + GRID_RESULTS_FILE)
    best_config = grid_results['best']
    best_metrics = grid_results[str(best_config)]
    print(best_config)
    print(best_metrics)


