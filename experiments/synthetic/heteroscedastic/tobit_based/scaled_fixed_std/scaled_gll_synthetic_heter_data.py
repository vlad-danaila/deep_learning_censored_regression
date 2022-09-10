from experiments.synthetic.constants import *
from experiments.util import set_random_seed
from experiments.synthetic.heteroscedastic.dataset import *
from experiments.synthetic.eval_optimized import plot_and_evaluate_model_gll
from experiments.synthetic.eval_optimized import plot_dataset_and_net
from experiments.synthetic.models import DenseNetwork
from experiments.tpe_hyperparam_opt import get_objective_fn_gll, tpe_opt_hyperparam

"""Constants"""
ROOT_GLL = 'experiments/synthetic/heteroscedastic/tobit_based/scaled_fixed_std/gll'
CHECKPOINT_GLL = 'gausian log likelihood model'

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

objective_gll = get_objective_fn_gll(
    dataset_train, dataset_val, bound_min, bound_max, f'{ROOT_GLL}/{CHECKPOINT_GLL}', GausianLogLikelihoodLoss, plot = False, log = False)

def tpe_opt_gll_scaled():
    return tpe_opt_hyperparam(ROOT_GLL, CHECKPOINT_GLL, objective_gll)

def eval_gll_scaled():
  plot_and_evaluate_model_gll(bound_min, bound_max, x_mean, x_std, y_mean, y_std, dataset_val, dataset_test,
                              ROOT_GLL, CHECKPOINT_GLL, GausianLogLikelihoodLoss, is_optimized= True)

def plot_gll_scaled():
    checkpoint = t.load(f'{ROOT_GLL}/{CHECKPOINT_GLL} best.tar')
    plot_dataset_and_net(checkpoint, DenseNetwork(), x_mean, x_std, y_mean, y_std, dataset_val)