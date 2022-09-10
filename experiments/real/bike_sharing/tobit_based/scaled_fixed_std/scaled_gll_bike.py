from experiments.util import set_random_seed, load_checkpoint, get_device
from experiments.real.bike_sharing.dataset import *
from experiments.real.bike_sharing.eval_optimized import plot_and_evaluate_model_gll, plot_dataset_and_net
from experiments.real.models import get_model
from experiments.tpe_hyperparam_opt import get_objective_fn_gll, tpe_opt_hyperparam

"""Constants"""
ROOT_GLL = 'experiments/real/bike_sharing/tobit_based/scaled_fixed_std/gll'
CHECKPOINT_GLL = 'gausian log likelihood model'

"""Reproducible experiments"""

set_random_seed()

"""# PDF Log-Likelihood"""

class GausianLogLikelihoodLoss(t.nn.Module):

    def __init__(self, sigma):
        super(GausianLogLikelihoodLoss, self).__init__()
        self.sigma = sigma
        self.epsilon = t.tensor(1e-40, dtype = t.float32, requires_grad = False, device = get_device())

    def forward(self, y_pred: t.Tensor, y_true: t.Tensor):
        sigma = t.abs(self.sigma)
        return t.sum(t.log(sigma + self.epsilon) + (((y_true - y_pred)/sigma) ** 2) / 2)

objective_gll = get_objective_fn_gll(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_GLL}/{CHECKPOINT_GLL}', GausianLogLikelihoodLoss, model_fn = lambda: get_model(INPUT_SIZE))

def tpe_opt_gll_scaled():
    return tpe_opt_hyperparam(ROOT_GLL, CHECKPOINT_GLL, objective_gll)

def eval_gll_scaled():
  plot_and_evaluate_model_gll(bound_min, bound_max, test_df(df), dataset_val, dataset_test,
                              ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, is_optimized= True)

def plot_gll_scaled():
    checkpoint = load_checkpoint(f'{ROOT_GLL}/{CHECKPOINT_GLL} best.tar')
    plot_dataset_and_net(checkpoint, get_model(INPUT_SIZE), test_df(df))