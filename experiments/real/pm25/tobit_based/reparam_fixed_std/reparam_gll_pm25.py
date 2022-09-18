from experiments.util import set_random_seed, load_checkpoint, get_device
from experiments.real.pm25.dataset import *
from experiments.real.pm25.eval_optimized import plot_and_evaluate_model_gll, plot_dataset_and_net
from experiments.real.models import get_model
from experiments.tpe_hyperparam_opt import get_objective_fn_gll, tpe_opt_hyperparam

"""Constants"""
ROOT_GLL = 'experiments/real/pm25/tobit_based/reparam_fixed_std/gll'
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

objective_gll = get_objective_fn_gll(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_GLL}/{CHECKPOINT_GLL}', GausianLogLikelihoodLoss, input_size= INPUT_SIZE)

def tpe_opt_gll_reparam():
    return tpe_opt_hyperparam(ROOT_GLL, CHECKPOINT_GLL, objective_gll)

def eval_gll_reparam():
  plot_and_evaluate_model_gll(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                              ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, is_optimized= True)

def plot_gll_reparam():
    checkpoint = load_checkpoint(f'{ROOT_GLL}/{CHECKPOINT_GLL} best.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))